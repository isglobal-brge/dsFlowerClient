"""Predict helper for dsFlowerClient.

Loads a saved model in native format and runs inference on input data.
Outputs JSON predictions to stdout for R to consume via processx.

Usage:
  python predict_helper.py --model <path> --data <csv> --type response|prob
                           [--framework sklearn|pytorch|xgboost]
                           [--template <template_name>]
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd


def predict_sklearn(model_path, X, pred_type):
    """Predict with a sklearn model (joblib)."""
    import joblib
    model = joblib.load(model_path)
    if pred_type == "prob":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            # Binary: return P(class=1)
            if probs.shape[1] == 2:
                return probs[:, 1].tolist()
            return probs.tolist()
        # Fallback: decision function -> sigmoid
        dec = model.decision_function(X)
        return (1 / (1 + np.exp(-dec))).tolist()
    return model.predict(X).tolist()


def predict_pytorch(model_path, X, pred_type, template=None):
    """Predict with a PyTorch checkpoint."""
    import torch
    from collections import OrderedDict

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    shapes = checkpoint.get("shapes", [])

    # Reconstruct weights as list of tensors
    if isinstance(state_dict, OrderedDict):
        weights = list(state_dict.values())
    else:
        keys = sorted(state_dict.keys(), key=lambda k: int(k) if k.isdigit() else k)
        weights = [state_dict[k] for k in keys]

    X_t = torch.tensor(X, dtype=torch.float32)

    # Simple linear model (2 params: weight + bias)
    if len(weights) == 2:
        W = weights[0].float()
        b = weights[1].float()
        logits = X_t @ W.T + b
        logits = logits.squeeze(-1)
        if pred_type == "prob":
            return torch.sigmoid(logits).detach().numpy().tolist()
        return (logits > 0).int().detach().numpy().tolist()

    # MLP (alternating weight/bias pairs)
    h = X_t
    n_layers = len(weights) // 2
    for i in range(n_layers):
        W = weights[i * 2].float()
        b = weights[i * 2 + 1].float()
        h = h @ W.T + b
        if i < n_layers - 1:
            h = torch.relu(h)

    logits = h.squeeze(-1)
    if pred_type == "prob":
        if logits.dim() > 1 and logits.shape[-1] > 1:
            return torch.softmax(logits, dim=-1).detach().numpy().tolist()
        return torch.sigmoid(logits).detach().numpy().tolist()
    if logits.dim() > 1 and logits.shape[-1] > 1:
        return torch.argmax(logits, dim=-1).detach().numpy().tolist()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--type", default="response", choices=["response", "prob"])
    parser.add_argument("--framework", default=None)
    parser.add_argument("--template", default=None)
    args = parser.parse_args()

    # Read data
    df = pd.read_csv(args.data)
    X = df.values.astype(np.float32)

    # Auto-detect framework from model file extension
    framework = args.framework
    if framework is None:
        if args.model.endswith(".joblib"):
            framework = "sklearn"
        elif args.model.endswith(".pt"):
            framework = "pytorch"
        elif args.model.endswith(".xgb.json") or args.model.endswith(".xgb"):
            framework = "xgboost"
        else:
            print(json.dumps({"error": "Cannot detect framework from model file"}),
                  file=sys.stderr)
            sys.exit(1)

    # Predict
    if framework == "sklearn":
        preds = predict_sklearn(args.model, X, args.type)
    elif framework == "pytorch":
        preds = predict_pytorch(args.model, X, args.type, args.template)
    elif framework == "xgboost":
        preds = predict_xgboost(args.model, X, args.type)
    else:
        print(json.dumps({"error": f"Unknown framework: {framework}"}),
              file=sys.stderr)
        sys.exit(1)

    # Output JSON
    json.dump(preds, sys.stdout)


if __name__ == "__main__":
    main()
