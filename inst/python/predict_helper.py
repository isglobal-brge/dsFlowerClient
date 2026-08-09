"""Predict helper for dsFlowerClient.

Loads a saved model in native format and runs inference on input data.
Outputs JSON predictions to stdout for R to consume via processx.

Usage:
  python predict_helper.py --model <path> --data <csv> --type response|prob
                           [--framework pytorch]
                           --spec-b64 <model_spec> --loss-name <loss_name>
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


def _finite_torch(value, limit):
    import torch
    return torch.clamp(
        torch.nan_to_num(value, nan=0.0, posinf=float(limit),
                         neginf=-float(limit)),
        min=-float(limit), max=float(limit))


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


def _decode_b64_json(raw):
    import base64
    return json.loads(base64.b64decode(str(raw), validate=True).decode("utf-8"))


def _apply_feature_preprocessing(X, bounds_b64=None):
    """Repeat the public clip-and-affine transform, or preserve raw features."""
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

    safe = np.where(np.isfinite(X), X, 0.0)
    return np.clip(safe, -_MAX_INPUT_ABS, _MAX_INPUT_ABS).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--type", default="response", choices=["response", "prob"])
    parser.add_argument("--framework", choices=["pytorch"], default=None)
    parser.add_argument("--spec-b64", dest="spec_b64", default=None)
    parser.add_argument("--loss-name", dest="loss_name", default=None)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=2)
    parser.add_argument("--num-labels", dest="num_labels", type=int, default=2)
    parser.add_argument("--bounds-b64", dest="bounds_b64", default=None)
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

    try:
        X = _apply_feature_preprocessing(X, args.bounds_b64)
    except (KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)

    # Predict
    if framework == "pytorch":
        if not args.spec_b64 or not args.loss_name:
            print(json.dumps({"error": (
                "Prediction requires saved declarative model_spec and loss_name "
                "metadata; Tier-2 and other no-spec artifacts require federated "
                "validation.")}), file=sys.stderr)
            sys.exit(2)
        preds = predict_pytorch_spec(
            args.model, X, args.type, args.spec_b64, args.loss_name,
            num_classes=args.num_classes, num_labels=args.num_labels)
    else:
        print(json.dumps({"error": f"Unknown framework: {framework}"}),
              file=sys.stderr)
        sys.exit(1)

    # Output JSON
    json.dump(preds, sys.stdout)


if __name__ == "__main__":
    main()
