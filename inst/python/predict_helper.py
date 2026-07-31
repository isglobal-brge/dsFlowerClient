"""Predict helper for dsFlowerClient.

Loads a saved model in native format and runs inference on input data.
Outputs JSON predictions to stdout for R to consume via processx.

Usage:
  python predict_helper.py --model <path> --data <csv> --type response|prob
                           [--framework pytorch|xgboost]
                           [--template <template_name>]
                           [--bounds-b64 <public_bounds>]
"""

import argparse
import json
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


def predict_pytorch(model_path, X, pred_type, template=None):
    """Predict with a PyTorch checkpoint, with template-aware output semantics."""
    import torch
    from collections import OrderedDict

    if _is_unsupported_pt(template):
        print(json.dumps({"error": (
            f"Prediction for template '{template}' needs its native architecture "
            "(conv / recurrent / temporal layers); the generic linear predictor "
            "would be wrong. Use federated evaluation, or load the model with its "
            "ClientApp Net.")}), file=sys.stderr)
        sys.exit(2)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
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


def predict_xgboost(model_path, X, pred_type):
    """Predict with an XGBoost model."""
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(model_path)
    dmat = xgb.DMatrix(X)
    probs = booster.predict(dmat)
    if pred_type == "prob":
        return probs.tolist()
    return (probs > 0.5).astype(int).tolist()


def _xgb_leaf(splits, leaves, row):
    """Traverse one custom tree to its leaf value for a single sample."""
    node = 0
    for _ in range(128):  # depth guard
        key = str(node)
        if key in splits:
            s = splits[key]
            node = int(s["left"]) if row[int(s["feature"])] <= float(s["threshold"]) \
                else int(s["right"])
        else:
            return float(leaves.get(key, 0.0))
    return 0.0


def _is_dpgbdt_booster(model):
    """True for the DP-GBDT (S-GBDT) booster format: complete random-split trees
    serialized as {feat, thr, w} arrays + a top-level depth/base_margin. This is a
    DIFFERENT shape from the splits/leaves dict format predict_xgboost_custom reads."""
    if "depth" not in model or "base_margin" not in model:
        return False
    trees = model.get("trees", [])
    return bool(trees) and isinstance(trees[0], dict) and "feat" in trees[0] and "w" in trees[0]


def predict_dpgbdt(model_path, X, pred_type):
    """Predict with the DP-GBDT booster (binary:logistic). Mirrors
    dp_gbdt.predict_margin: F(x) = base_margin + sum_t w_t[leaf_t(x)] over the
    complete random-split trees; leaf weights ALREADY include the learning rate
    (Newton step), so we do NOT re-apply it. Routing follows the complete-tree rule
    (root 0; child of k is 2k+1/2k+2; go right iff x[feat] >= thr)."""
    with open(model_path) as f:
        model = json.load(f)
    depth = int(model["depth"])
    base = float(model.get("base_margin", 0.0))
    n = X.shape[0]
    F = np.full(n, base, dtype=np.float64)
    rows = np.arange(n)
    for tree in model.get("trees", []):
        feat = np.asarray(tree["feat"], dtype=np.int64)
        thr = np.asarray(tree["thr"], dtype=np.float64)
        w = np.asarray(tree["w"], dtype=np.float64)
        node = np.zeros(n, dtype=np.int64)
        for _ in range(depth):
            go_right = X[rows, feat[node]] >= thr[node]
            node = 2 * node + 1 + go_right.astype(np.int64)
        leaf = node - ((1 << depth) - 1)
        F = F + w[leaf]
    p = 1.0 / (1.0 + np.exp(-np.clip(F, -60.0, 60.0)))
    if pred_type == "prob":
        return p.tolist()
    return (p > 0.5).astype(int).tolist()


def predict_xgboost_custom(model_path, X, pred_type):
    """Predict with dsFlower's federated XGBoost JSON (binary + multiclass).

    The model is an additive ensemble of per-class trees: score[c] is the sum of
    learning_rate * leaf over the trees tagged with class c, then sigmoid (binary)
    or softmax (multiclass).
    """
    with open(model_path) as f:
        model = json.load(f)
    lr = float(model.get("learning_rate", 0.3))
    n_outputs = int(model.get("n_outputs", 1))
    multiclass = bool(model.get("multiclass", False)) or n_outputs > 1
    objective = str(model.get("objective", "binary:logistic"))
    trees = model.get("trees", [])

    n = X.shape[0]
    scores = np.zeros((n, max(1, n_outputs)), dtype=np.float64)
    for tree in trees:
        cls = int(tree.get("class_idx", 0)) % scores.shape[1]
        splits = tree.get("splits", {})
        leaves = tree.get("leaves", {})
        for i in range(n):
            scores[i, cls] += lr * _xgb_leaf(splits, leaves, X[i])

    if multiclass:
        z = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=1, keepdims=True)
        if pred_type == "prob":
            return probs.tolist()
        return np.argmax(probs, axis=1).tolist()

    if objective.startswith("binary") or objective.endswith("logistic"):
        p = 1.0 / (1.0 + np.exp(-scores[:, 0]))
        if pred_type == "prob":
            return p.tolist()
        return (p > 0.5).astype(int).tolist()

    # regression
    return scores[:, 0].tolist()


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
    parser.add_argument("--framework", default=None)
    parser.add_argument("--template", default=None)
    parser.add_argument("--bounds-b64", dest="bounds_b64", default=None)
    parser.add_argument("--norm-b64", dest="norm_b64", default=None)
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.data)
    X = df.values.astype(np.float32)

    # Public bounds take precedence for new models; mean/SD remains a legacy path.
    try:
        X = _apply_feature_preprocessing(X, args.bounds_b64, args.norm_b64)
    except (KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)

    # Auto-detect framework from model file extension
    framework = args.framework
    if framework is None:
        if args.model.endswith(".pt"):
            framework = "pytorch"
        elif args.model.endswith(".xgb.json") or args.model.endswith(".xgb"):
            framework = "xgboost"
        else:
            print(json.dumps({"error": "Cannot detect framework from model file"}),
                  file=sys.stderr)
            sys.exit(1)

    # Predict
    if framework == "pytorch":
        preds = predict_pytorch(args.model, X, args.type, args.template)
    elif framework == "xgboost":
        if args.model.endswith(".json"):
            with open(args.model) as _f:
                _m = json.load(_f)
            if _is_dpgbdt_booster(_m):           # DP-GBDT {feat,thr,w} format
                preds = predict_dpgbdt(args.model, X, args.type)
            else:                                 # splits/leaves custom format
                preds = predict_xgboost_custom(args.model, X, args.type)
        else:
            preds = predict_xgboost(args.model, X, args.type)
    else:
        print(json.dumps({"error": f"Unknown framework: {framework}"}),
              file=sys.stderr)
        sys.exit(1)

    # Output JSON
    json.dump(preds, sys.stdout)


if __name__ == "__main__":
    main()
