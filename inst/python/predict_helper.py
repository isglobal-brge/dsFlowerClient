"""Predict helper for dsFlowerClient.

Loads a saved model in native format and runs inference on input data.
Outputs JSON predictions to stdout for R to consume via processx.

Usage:
  python predict_helper.py --model <path> --data <csv> --type response|prob
                           [--framework pytorch]
                           [--template <template_name>]
                           [--bounds-b64 <public_bounds>]
"""

import argparse
import base64
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


_MAX_INPUT_ABS = 1.0e6
_MAX_ACTIVATION_ABS = 1.0e6
_MAX_OUTPUT_ABS = 30.0


def _finite_torch(value, limit):
    import torch
    return torch.clamp(
        torch.nan_to_num(value, nan=0.0, posinf=float(limit),
                         neginf=-float(limit)),
        min=-float(limit), max=float(limit))


# Per-template output semantics for the linear / MLP PyTorch families. The
# forward pass is a generic linear (or MLP) stack, but the head's meaning differs
# by modality -- applying sigmoid+threshold to a regression/count/survival head
# (the old behaviour) returns plausible-but-wrong values. Templates not listed
# default to classification (binary / multiclass).
_PT_OUTPUT = {
    "pytorch_logreg":            "binary",
    "pytorch_multiclass":        "multiclass",
    "pytorch_mlp":               "multiclass",
    "pytorch_multilabel":        "multilabel",
    "pytorch_linear_regression": "regression",
    "pytorch_ridge":             "regression",
    "pytorch_lasso":             "regression",
    "pytorch_elasticnet":        "regression",
    "pytorch_poisson":           "count",
    "pytorch_coxph":             "risk",
    "pytorch_cause_specific_cox": "risk",
    "pytorch_lognormal_aft":     "aft",
}
# Architectures the generic linear/MLP forward CANNOT reproduce (conv kernels,
# BatchNorm, recurrence, temporal convolution, user-supplied nets): predicting
# with the flat weight stack would be silently wrong, so we refuse rather than
# mislead. Matched as substrings of the template name to tolerate variants
# (pytorch_resnet18, pytorch_resnet, ...).
_PT_UNSUPPORTED_KINDS = (
    "resnet", "densenet", "unet", "vision", "lstm", "tcn", "native",
)


def _is_unsupported_pt(template):
    t = (template or "").lower()
    return any(k in t for k in _PT_UNSUPPORTED_KINDS)


def _linear_forward(weights, X_t, output_limit=_MAX_OUTPUT_ABS):
    W = weights[0].float()
    b = weights[1].float()
    return _finite_torch(X_t @ W.T + b, output_limit).squeeze(-1)


def _mlp_forward(weights, X_t, output_limit=_MAX_OUTPUT_ABS):
    import torch
    h = X_t
    n_layers = len(weights) // 2
    for i in range(n_layers):
        W = weights[i * 2].float()
        b = weights[i * 2 + 1].float()
        limit = output_limit if i == n_layers - 1 else _MAX_ACTIVATION_ABS
        h = _finite_torch(h @ W.T + b, limit)
        if i < n_layers - 1:
            h = _finite_torch(torch.relu(h), _MAX_ACTIVATION_ABS)
    return h.squeeze(-1)


def _load_model_spec_module():
    """Load the same data-only model builder bundled in the trusted runner."""
    path = (Path(__file__).resolve().parents[1] / "flower_app" /
            "dsflower_runner" / "model_spec.py")
    if not path.is_file():
        raise RuntimeError("bundled declarative model builder is unavailable")
    module_spec = importlib.util.spec_from_file_location(
        "_dsflower_predict_model_spec", str(path))
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("could not load the declarative model builder")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _decode_model_spec(raw):
    if not isinstance(raw, str) or len(raw) > 256 * 1024:
        raise ValueError("encoded model spec is missing or oversized")
    decoded = base64.b64decode(raw, validate=True)
    if len(decoded) > 128 * 1024:
        raise ValueError("decoded model spec is oversized")
    value = json.loads(decoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model spec must decode to an object")
    return value


def _ordinal_probabilities(logits):
    """Convert K-1 cumulative logits to a finite K-class distribution."""
    import torch
    q = torch.sigmoid(logits)
    if q.dim() == 1:
        q = q.unsqueeze(-1)
    # Finite samples can violate monotonicity slightly.  Projection is public
    # post-processing and makes the resulting class probabilities coherent.
    q = torch.cummin(q, dim=-1).values
    parts = [1.0 - q[:, :1]]
    if q.shape[1] > 1:
        parts.append(q[:, :-1] - q[:, 1:])
    parts.append(q[:, -1:])
    probs = torch.clamp(torch.cat(parts, dim=-1), min=0.0, max=1.0)
    return probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1.0e-12)


def _apply_loss_semantics(logits, loss_name, pred_type):
    import torch
    loss_name = str(loss_name)
    if loss_name in ("mse", "huber", "quantile"):
        return logits.squeeze(-1).detach().numpy().tolist()
    if loss_name in ("poisson_nll", "negbin_nll", "gamma_nll"):
        return torch.exp(torch.clamp(logits.squeeze(-1), -30.0, 30.0)).detach().numpy().tolist()
    if loss_name == "multilabel_bce":
        probs = torch.sigmoid(logits)
        return (probs if pred_type == "prob" else (probs > 0.5).int()).detach().numpy().tolist()
    if loss_name == "ordinal":
        probs = _ordinal_probabilities(logits)
        return (probs if pred_type == "prob" else torch.argmax(probs, dim=-1)).detach().numpy().tolist()
    if loss_name in ("cross_entropy", "hinge") or (
            logits.dim() > 1 and logits.shape[-1] > 1):
        probs = torch.softmax(logits, dim=-1)
        return (probs if pred_type == "prob" else torch.argmax(probs, dim=-1)).detach().numpy().tolist()
    flat = logits.squeeze(-1)
    probs = torch.sigmoid(flat)
    return (probs if pred_type == "prob" else (flat > 0).int()).detach().numpy().tolist()


def _load_state_dict_safely(model_path):
    """Load tensor weights without enabling arbitrary pickle execution."""
    import torch
    try:
        checkpoint = torch.load(
            model_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Safe checkpoint loading requires PyTorch weights_only support"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("PyTorch checkpoint must contain a state_dict mapping")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("PyTorch checkpoint state_dict must be a non-empty mapping")
    if any(not isinstance(key, str) or not torch.is_tensor(value)
           for key, value in state_dict.items()):
        raise ValueError("PyTorch state_dict accepts only named tensor values")
    return state_dict


def predict_pytorch_spec(model_path, X, pred_type, spec_b64, loss_name,
                         num_classes=2, num_labels=2):
    """Rebuild the exact declarative architecture, then load its state_dict."""
    import torch
    builder = _load_model_spec_module()
    public_spec = _decode_model_spec(spec_b64)
    cfg = {"num-classes": int(num_classes), "num-labels": int(num_labels)}
    out_dim = builder.output_width(str(loss_name), cfg)
    model = builder.build_from_spec(
        public_spec, in_dim=int(X.shape[1]), out_dim=int(out_dim),
        num_labels=int(num_labels),
        output_limit=builder.output_limit_for_loss(str(loss_name)))
    state_dict = _load_state_dict_safely(model_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        logits = _finite_torch(logits, builder.output_limit_for_loss(str(loss_name)))
    return _apply_loss_semantics(logits, loss_name, pred_type)


def predict_pytorch(model_path, X, pred_type, template=None, spec_b64=None,
                    loss_name=None, num_classes=2, num_labels=2):
    """Predict with a PyTorch checkpoint, with template-aware output semantics."""
    import torch
    from collections import OrderedDict

    if spec_b64 is not None:
        if not loss_name:
            raise ValueError("declarative prediction requires the pinned loss name")
        return predict_pytorch_spec(
            model_path, X, pred_type, spec_b64, loss_name,
            num_classes=num_classes, num_labels=num_labels)

    if _is_unsupported_pt(template):
        print(json.dumps({"error": (
            f"Prediction for template '{template}' needs its native architecture "
            "(conv / recurrent / temporal layers); the generic linear predictor "
            "would be wrong. Use federated evaluation, or load the model with its "
            "ClientApp Net.")}), file=sys.stderr)
        sys.exit(2)

    state_dict = _load_state_dict_safely(model_path)
    if isinstance(state_dict, OrderedDict):
        weights = list(state_dict.values())
    else:
        keys = sorted(state_dict.keys(), key=lambda k: int(k) if k.isdigit() else k)
        weights = [state_dict[k] for k in keys]

    X_t = torch.tensor(X, dtype=torch.float32)
    behavior = _PT_OUTPUT.get(template or "", None)

    # AFT: state_dict is [weight, bias, log_scale]; the point prediction is the
    # linear mean (predicted log-time); return predicted time = exp(mu).
    if behavior == "aft":
        mu = _linear_forward(weights[:2], X_t)
        return torch.exp(mu).detach().numpy().tolist()

    output_limit = _MAX_ACTIVATION_ABS if behavior == "regression" else _MAX_OUTPUT_ABS
    logits = (_linear_forward(weights, X_t, output_limit)
              if len(weights) == 2 else _mlp_forward(weights, X_t, output_limit))

    if behavior == "regression":
        return logits.detach().numpy().tolist()                       # raw predicted value
    if behavior == "count":
        return torch.exp(logits).detach().numpy().tolist()            # Poisson rate = exp(log-rate)
    if behavior == "risk":
        return logits.detach().numpy().tolist()                       # Cox log-hazard / risk score
    if behavior == "multilabel":
        p = torch.sigmoid(logits)                                     # independent per-label probs
        if pred_type == "prob":
            return p.detach().numpy().tolist()
        return (p > 0.5).int().detach().numpy().tolist()

    # classification (binary / multiclass) -- default
    if logits.dim() > 1 and logits.shape[-1] > 1:
        if pred_type == "prob":
            return torch.softmax(logits, dim=-1).detach().numpy().tolist()
        return torch.argmax(logits, dim=-1).detach().numpy().tolist()
    if pred_type == "prob":
        return torch.sigmoid(logits).detach().numpy().tolist()
    return (logits > 0).int().detach().numpy().tolist()


def _decode_b64_json(raw):
    import base64
    return json.loads(base64.b64decode(str(raw), validate=True).decode("utf-8"))


def _apply_feature_preprocessing(X, bounds_b64=None, norm_b64=None):
    """Repeat training preprocessing: public clip+affine first, legacy mean/SD otherwise."""
    if bounds_b64:
        bounds = _decode_b64_json(bounds_b64)
        lower = np.asarray(bounds.get("lower", []), dtype=np.float64)
        upper = np.asarray(bounds.get("upper", []), dtype=np.float64)
        if lower.shape != (X.shape[1],) or upper.shape != (X.shape[1],):
            raise ValueError(
                "public bounds length (%d/%d) != newdata feature count (%d)"
                % (lower.size, upper.size, X.shape[1]))
        if (not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
                or not np.all(lower < upper)
                or np.any(np.abs(lower) > _MAX_INPUT_ABS)
                or np.any(np.abs(upper) > _MAX_INPUT_ABS)):
            raise ValueError(
                "public bounds must be finite, within [-1e6, 1e6], with lower < upper")
        center = (lower + upper) / 2.0
        scale = (upper - lower) / 2.0
        safe = np.where(np.isfinite(X), X, center)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            transformed = (np.clip(safe, lower, upper) - center) / scale
        transformed = np.nan_to_num(
            transformed, nan=0.0, posinf=_MAX_ACTIVATION_ABS,
            neginf=-_MAX_ACTIVATION_ABS)
        return np.clip(
            transformed, -_MAX_ACTIVATION_ABS,
            _MAX_ACTIVATION_ABS).astype(np.float32)

    if not norm_b64:
        safe = np.where(np.isfinite(X), X, 0.0)
        return np.clip(safe, -_MAX_INPUT_ABS, _MAX_INPUT_ABS).astype(np.float32)
    norm = _decode_b64_json(norm_b64)
    mean = np.asarray(norm["means"], dtype=np.float64)
    sd = np.asarray(norm["sds"], dtype=np.float64)
    if mean.shape != (X.shape[1],) or sd.shape != (X.shape[1],):
        raise ValueError(
            "standardization stats length (%d/%d) != newdata feature count (%d)"
            % (mean.size, sd.size, X.shape[1]))
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    safe = np.clip(np.where(np.isfinite(X), X, 0.0),
                   -_MAX_INPUT_ABS, _MAX_INPUT_ABS).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed = (safe - mean) / sd
    transformed = np.nan_to_num(
        transformed, nan=0.0, posinf=_MAX_ACTIVATION_ABS,
        neginf=-_MAX_ACTIVATION_ABS)
    return np.clip(
        transformed, -_MAX_ACTIVATION_ABS,
        _MAX_ACTIVATION_ABS).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--type", default="response", choices=["response", "prob"])
    parser.add_argument("--framework", choices=["pytorch"], default=None)
    parser.add_argument("--template", default=None)
    parser.add_argument("--spec-b64", dest="spec_b64", default=None)
    parser.add_argument("--loss-name", dest="loss_name", default=None)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=2)
    parser.add_argument("--num-labels", dest="num_labels", type=int, default=2)
    parser.add_argument("--bounds-b64", dest="bounds_b64", default=None)
    parser.add_argument("--norm-b64", dest="norm_b64", default=None)
    args = parser.parse_args()

    # Resolve the public artifact kind before selecting its preprocessing path.
    framework = args.framework
    if framework is None:
        if args.model.endswith(".pt"):
            framework = "pytorch"
        else:
            print(json.dumps({"error": "Cannot detect framework from model file"}),
                  file=sys.stderr)
            sys.exit(1)

    # Read data
    df = pd.read_csv(args.data)
    X = df.values.astype(np.float32)

    # Public bounds take precedence for new models; mean/SD remains a legacy path.
    try:
        X = _apply_feature_preprocessing(X, args.bounds_b64, args.norm_b64)
    except (KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)

    # Predict
    if framework == "pytorch":
        preds = predict_pytorch(
            args.model, X, args.type, args.template,
            spec_b64=args.spec_b64, loss_name=args.loss_name,
            num_classes=args.num_classes, num_labels=args.num_labels)
    else:
        print(json.dumps({"error": f"Unknown framework: {framework}"}),
              file=sys.stderr)
        sys.exit(1)

    # Output JSON
    json.dump(preds, sys.stdout)


if __name__ == "__main__":
    main()
