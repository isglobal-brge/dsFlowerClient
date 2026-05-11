#!/usr/bin/env python3
"""Centralized baselines for dsFlower method-validation demos.

The script trains a local model on the pooled validation dataset and emits a
small JSON summary. It is intentionally dependency-light and mirrors the losses
used by the server templates closely enough for path validation.
"""

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd


def _fail(message):
    print(message, file=sys.stderr)
    return 2


def _parse_json(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON parameter payload: {exc}") from exc


def _as_float(value, default):
    if value is None:
        return default
    return float(value)


def _as_int(value, default):
    if value is None:
        return default
    return int(value)


def _hidden_layers(value, default=None):
    if value is None or value == "":
        return [] if default is None else list(default)
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _binary_log_loss(y, prob, eps=1e-15):
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)))


def _binary_metrics(y, prob):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    return {
        "loss": _binary_log_loss(y, prob),
        "accuracy": float(np.mean(pred == y)),
    }


def _hinge_loss(y, scores):
    y_signed = np.where(np.asarray(y) > 0, 1.0, -1.0)
    return float(np.mean(np.maximum(0.0, 1.0 - y_signed * scores)))


def _sklearn_baseline(method, x, y, params, seed):
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.metrics import accuracy_score, log_loss

    if method == "sklearn_logreg":
        model = LogisticRegression(
            penalty=params.get("penalty", "l2"),
            C=_as_float(params.get("C"), 1.0),
            max_iter=_as_int(params.get("max_iter"), 100),
            random_state=seed,
        )
        model.fit(x, y)
        prob = model.predict_proba(x)
        return {
            "metric_name": "log_loss",
            "loss": float(log_loss(y, prob, labels=np.unique(y))),
            "accuracy": float(accuracy_score(y, model.predict(x))),
        }

    if method == "sklearn_ridge":
        model = RidgeClassifier(alpha=_as_float(params.get("alpha"), 1.0))
        model.fit(x, y)
        scores = model.decision_function(x)
        return {
            "metric_name": "ridge_decision_mse",
            "loss": float(np.mean((scores - y) ** 2)),
            "accuracy": float(accuracy_score(y, model.predict(x))),
        }

    if method in ("sklearn_sgd", "sklearn_svm", "sklearn_elastic_net"):
        loss = params.get("loss", "hinge" if method == "sklearn_svm" else "log_loss")
        penalty = params.get("penalty", "elasticnet" if method == "sklearn_elastic_net" else "l2")
        kwargs = {
            "loss": loss,
            "alpha": _as_float(params.get("alpha"), 0.0001),
            "learning_rate": params.get("lr_schedule", "optimal"),
            "penalty": penalty,
            "random_state": seed,
            "max_iter": _as_int(params.get("max_iter"), 1000),
            "tol": 1e-4,
        }
        if kwargs["learning_rate"] in ("constant", "invscaling", "adaptive"):
            kwargs["eta0"] = _as_float(params.get("eta0"), 0.01)
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = _as_float(params.get("l1_ratio"), 0.15)
        model = SGDClassifier(**kwargs)
        model.fit(x, y)
        pred = model.predict(x)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(x)
            loss_value = float(log_loss(y, prob, labels=np.unique(y)))
            metric_name = "log_loss"
        else:
            scores = model.decision_function(x)
            loss_value = _hinge_loss(y, scores)
            metric_name = "hinge_loss"
        return {
            "metric_name": metric_name,
            "loss": loss_value,
            "accuracy": float(accuracy_score(y, pred)),
        }

    raise ValueError(f"Unsupported sklearn method: {method}")


def _torch_import():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, DataLoader, TensorDataset


def _make_mlp(nn, input_dim, hidden_layers, output_dim=1):
    layers = []
    prev = input_dim
    for hidden in hidden_layers:
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.ReLU())
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def _train_loop(torch, loader, model, optimizer, loss_fn, epochs, step_fn=None):
    model.train()
    for _ in range(max(1, int(epochs))):
        for batch in loader:
            optimizer.zero_grad()
            loss = step_fn(batch) if step_fn is not None else loss_fn(batch)
            loss.backward()
            optimizer.step()


def _cox_loss(torch, log_risk, time, event):
    order = torch.argsort(time, descending=True)
    log_risk = log_risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(log_risk, dim=0)
    loss = -torch.sum((log_risk - log_cumsum) * event)
    n_events = event.sum()
    if float(n_events.item()) > 0:
        loss = loss / n_events
    return loss


def _aft_loss(torch, mu, log_scale, log_time, event):
    sigma = torch.exp(log_scale) + 1e-8
    z = (log_time - mu) / sigma
    log_phi = -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)
    log_surv = torch.log(0.5 * torch.erfc(z / math.sqrt(2)) + 1e-15)
    ll_uncensored = log_phi - log_scale - log_time
    ll = event * ll_uncensored + (1.0 - event) * log_surv
    return -ll.mean()


def _competing_risks_loss(torch, log_risks, time, event_type, n_causes):
    order = torch.argsort(time, descending=True)
    log_risks = log_risks[order]
    event_type = event_type[order]
    total = torch.tensor(0.0, device=log_risks.device)
    n_terms = 0
    for cause in range(n_causes):
        is_event = (event_type == (cause + 1)).float()
        n_events = is_event.sum()
        if float(n_events.item()) == 0.0:
            continue
        lr = log_risks[:, cause]
        log_cumsum = torch.logcumsumexp(lr, dim=0)
        total = total + (-torch.sum((lr - log_cumsum) * is_event) / n_events)
        n_terms += 1
    if n_terms > 0:
        total = total / n_terms
    return total


def _torch_binary_baseline(method, x, y, params, seed, rounds):
    torch, nn, DataLoader, TensorDataset = _torch_import()
    torch.manual_seed(seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32)
    batch_size = _as_int(params.get("batch_size"), 32)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)
    lr = _as_float(params.get("learning_rate"), 0.01)
    epochs = max(1, int(rounds) * _as_int(params.get("local_epochs"), 1))

    if method == "pytorch_logreg":
        model = nn.Linear(x.shape[1], 1)
    elif method == "pytorch_mlp":
        model = _make_mlp(nn, x.shape[1], _hidden_layers(params.get("hidden_layers"), [64, 32]), 1)
    else:
        raise ValueError(f"Unsupported binary torch method: {method}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def step(batch):
        xb, yb = batch
        return criterion(model(xb), yb.unsqueeze(1))

    _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)

    model.eval()
    with torch.no_grad():
        logits = model(x_t).squeeze(1)
        loss = float(criterion(logits.unsqueeze(1), y_t.unsqueeze(1)).item())
        probs = torch.sigmoid(logits).cpu().numpy()
    metrics = _binary_metrics(y, probs)
    metrics["loss"] = loss
    metrics["metric_name"] = "binary_cross_entropy"
    return metrics


def _torch_linear_regression(x, y, params, seed, rounds):
    torch, nn, DataLoader, TensorDataset = _torch_import()
    torch.manual_seed(seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, y_t),
                        batch_size=_as_int(params.get("batch_size"), 32),
                        shuffle=True)
    model = nn.Linear(x.shape[1], 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_as_float(params.get("learning_rate"), 0.01))
    epochs = max(1, int(rounds) * _as_int(params.get("local_epochs"), 1))

    def step(batch):
        xb, yb = batch
        return criterion(model(xb), yb.unsqueeze(1))

    _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)
    model.eval()
    with torch.no_grad():
        pred = model(x_t).squeeze(1)
        mse = float(criterion(pred, y_t).item())
    return {"metric_name": "mse", "loss": mse, "mse": mse}


def _torch_multiclass(x, y, params, seed, rounds):
    torch, nn, DataLoader, TensorDataset = _torch_import()
    torch.manual_seed(seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))
    n_classes = _as_int(params.get("n_classes"), int(np.max(y)) + 1)
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y.astype(np.int64), dtype=torch.long)
    loader = DataLoader(TensorDataset(x_t, y_t),
                        batch_size=_as_int(params.get("batch_size"), 32),
                        shuffle=True)
    model = _make_mlp(nn, x.shape[1], _hidden_layers(params.get("hidden_layers"), []), n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_as_float(params.get("learning_rate"), 0.01))
    epochs = max(1, int(rounds) * _as_int(params.get("local_epochs"), 1))

    def step(batch):
        xb, yb = batch
        return criterion(model(xb), yb)

    _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)
    model.eval()
    with torch.no_grad():
        logits = model(x_t)
        loss = float(criterion(logits, y_t).item())
        pred = logits.argmax(dim=1).cpu().numpy()
    return {
        "metric_name": "cross_entropy",
        "loss": loss,
        "accuracy": float(np.mean(pred == y)),
    }


def _torch_multilabel(x, y, params, seed, rounds):
    torch, nn, DataLoader, TensorDataset = _torch_import()
    torch.manual_seed(seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, y_t),
                        batch_size=_as_int(params.get("batch_size"), 32),
                        shuffle=True)
    n_labels = y.shape[1] if y.ndim > 1 else _as_int(params.get("n_labels"), 1)
    model = _make_mlp(nn, x.shape[1], _hidden_layers(params.get("hidden_layers"), [64, 32]), n_labels)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_as_float(params.get("learning_rate"), 0.01))
    epochs = max(1, int(rounds) * _as_int(params.get("local_epochs"), 1))

    def step(batch):
        xb, yb = batch
        return criterion(model(xb), yb)

    _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)
    model.eval()
    with torch.no_grad():
        logits = model(x_t)
        loss = float(criterion(logits, y_t).item())
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
    return {
        "metric_name": "multilabel_bce",
        "loss": loss,
        "accuracy": float(np.mean(pred == y.astype(int))),
    }


def _torch_poisson(x, y, params, seed, rounds):
    torch, nn, DataLoader, TensorDataset = _torch_import()
    torch.manual_seed(seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, y_t),
                        batch_size=_as_int(params.get("batch_size"), 32),
                        shuffle=True)
    model = _make_mlp(nn, x.shape[1], _hidden_layers(params.get("hidden_layers"), []), 1)
    criterion = nn.PoissonNLLLoss(log_input=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=_as_float(params.get("learning_rate"), 0.01))
    epochs = max(1, int(rounds) * _as_int(params.get("local_epochs"), 1))

    def step(batch):
        xb, yb = batch
        return criterion(model(xb).squeeze(1), yb)

    _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)
    model.eval()
    with torch.no_grad():
        loss = float(criterion(model(x_t).squeeze(1), y_t).item())
    return {"metric_name": "poisson_nll", "loss": loss}


class _AFTModel:
    pass


def _torch_survival(method, x, time, event, params, seed, rounds):
    torch, nn, DataLoader, TensorDataset = _torch_import()
    torch.manual_seed(seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))
    x_t = torch.tensor(x, dtype=torch.float32)
    time_t = torch.tensor(time.astype(np.float32), dtype=torch.float32)
    event_t = torch.tensor(event.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, time_t, event_t),
                        batch_size=_as_int(params.get("batch_size"), 32),
                        shuffle=True)
    epochs = max(1, int(rounds) * _as_int(params.get("local_epochs"), 1))
    lr = _as_float(params.get("learning_rate"), 0.01)

    if method == "pytorch_coxph":
        model = nn.Linear(x.shape[1], 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        def step(batch):
            xb, tb, eb = batch
            return _cox_loss(torch, model(xb).squeeze(1), tb, eb)

        _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)
        model.eval()
        with torch.no_grad():
            loss = float(_cox_loss(torch, model(x_t).squeeze(1), time_t, event_t).item())
        return {"metric_name": "cox_partial_likelihood", "loss": loss}

    if method == "pytorch_lognormal_aft":
        class AFT(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, 1)
                self.log_scale = nn.Parameter(torch.tensor(0.0))

            def forward(self, xb):
                return self.linear(xb).squeeze(1), self.log_scale

        model = AFT(x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        def step(batch):
            xb, tb, eb = batch
            mu, log_scale = model(xb)
            return _aft_loss(torch, mu, log_scale, torch.log(tb + 1e-8), eb)

        _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)
        model.eval()
        with torch.no_grad():
            mu, log_scale = model(x_t)
            loss = float(_aft_loss(torch, mu, log_scale, torch.log(time_t + 1e-8), event_t).item())
        return {"metric_name": "lognormal_aft_nll", "loss": loss}

    if method == "pytorch_cause_specific_cox":
        n_causes = _as_int(params.get("n_causes"), 2)

        class CauseSpecific(nn.Module):
            def __init__(self, input_dim, causes):
                super().__init__()
                self.heads = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(causes)])

            def forward(self, xb):
                return torch.cat([head(xb) for head in self.heads], dim=1)

        model = CauseSpecific(x.shape[1], n_causes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        event_long = event_t.long()
        loader = DataLoader(TensorDataset(x_t, time_t, event_long),
                            batch_size=_as_int(params.get("batch_size"), 32),
                            shuffle=True)

        def step(batch):
            xb, tb, eb = batch
            return _competing_risks_loss(torch, model(xb), tb, eb, n_causes)

        _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)
        model.eval()
        with torch.no_grad():
            loss = float(_competing_risks_loss(torch, model(x_t), time_t, event_long, n_causes).item())
        return {"metric_name": "cause_specific_cox_loss", "loss": loss}

    raise ValueError(f"Unsupported survival method: {method}")


def _torch_lstm(x, y, params, seed, rounds):
    torch, nn, DataLoader, TensorDataset = _torch_import()
    torch.manual_seed(seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))

    class LSTMClassifier(nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, xb):
            out, _ = self.lstm(xb)
            return self.head(out[:, -1, :])

    x_seq = x.reshape((x.shape[0], x.shape[1], 1)).astype(np.float32)
    x_t = torch.tensor(x_seq, dtype=torch.float32)
    y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, y_t),
                        batch_size=_as_int(params.get("batch_size"), 32),
                        shuffle=True)
    model = LSTMClassifier(
        _as_int(params.get("hidden_size"), 64),
        _as_int(params.get("num_layers"), 2),
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_as_float(params.get("learning_rate"), 0.001))
    epochs = max(1, int(rounds) * _as_int(params.get("local_epochs"), 1))

    def step(batch):
        xb, yb = batch
        return criterion(model(xb), yb.unsqueeze(1))

    _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)
    model.eval()
    with torch.no_grad():
        logits = model(x_t).squeeze(1)
        loss = float(criterion(logits.unsqueeze(1), y_t.unsqueeze(1)).item())
        prob = torch.sigmoid(logits).cpu().numpy()
    metrics = _binary_metrics(y, prob)
    metrics["loss"] = loss
    metrics["metric_name"] = "binary_cross_entropy"
    return metrics


def _torch_tcn(x, y, params, seed, rounds):
    torch, nn, DataLoader, TensorDataset = _torch_import()
    torch.manual_seed(seed)
    torch.set_num_threads(_as_int(params.get("threads"), 1))

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, xb):
            residual = xb
            out = self.relu(self.bn1(self.conv1(xb)))
            out = out[:, :, :xb.size(2)]
            out = self.bn2(self.conv2(out))
            out = out[:, :, :xb.size(2)]
            if self.downsample is not None:
                residual = self.downsample(xb)
            return self.relu(out + residual)

    class TCN(nn.Module):
        def __init__(self, n_channels, kernel_size, n_layers, hidden_dim=64):
            super().__init__()
            layers = []
            in_ch = n_channels
            for i in range(n_layers):
                layers.append(Block(in_ch, hidden_dim, kernel_size, 2 ** i))
                in_ch = hidden_dim
            self.network = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dim, 1)

        def forward(self, xb):
            out = self.network(xb)
            return self.head(out.mean(dim=2))

    n_channels = _as_int(params.get("n_channels"), 1)
    seq_len = x.shape[1] // n_channels
    x_seq = x.reshape((x.shape[0], n_channels, seq_len)).astype(np.float32)
    x_t = torch.tensor(x_seq, dtype=torch.float32)
    y_t = torch.tensor(y.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_t, y_t),
                        batch_size=_as_int(params.get("batch_size"), 32),
                        shuffle=True)
    model = TCN(
        n_channels=n_channels,
        kernel_size=_as_int(params.get("kernel_size"), 3),
        n_layers=_as_int(params.get("n_layers"), 4),
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_as_float(params.get("learning_rate"), 0.001))
    epochs = max(1, int(rounds) * _as_int(params.get("local_epochs"), 1))

    def step(batch):
        xb, yb = batch
        return criterion(model(xb), yb.unsqueeze(1))

    _train_loop(torch, loader, model, optimizer, None, epochs, step_fn=step)
    model.eval()
    with torch.no_grad():
        logits = model(x_t).squeeze(1)
        loss = float(criterion(logits.unsqueeze(1), y_t.unsqueeze(1)).item())
        prob = torch.sigmoid(logits).cpu().numpy()
    metrics = _binary_metrics(y, prob)
    metrics["loss"] = loss
    metrics["metric_name"] = "binary_cross_entropy"
    return metrics


def _xgboost_baseline(x, y, params, seed):
    import xgboost as xgb
    dtrain = xgb.DMatrix(x, label=y)
    objective = params.get("objective", "binary:logistic")
    xgb_params = {
        "objective": objective,
        "max_depth": _as_int(params.get("max_depth"), 3),
        "eta": _as_float(params.get("eta"), 0.3),
        "lambda": _as_float(params.get("reg_lambda"), 1.0),
        "seed": seed,
        "verbosity": 0,
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=_as_int(params.get("n_trees"), 10))
    pred = model.predict(dtrain)
    return {
        "metric_name": "log_loss",
        "loss": _binary_log_loss(y, pred),
        "accuracy": float(np.mean((pred >= 0.5).astype(int) == y.astype(int))),
    }


def run(args):
    params_payload = args.model_params
    if args.model_params_file:
        with open(args.model_params_file, encoding="utf-8") as fh:
            params_payload = fh.read()
    params = _parse_json(params_payload)
    features = [x for x in args.features.split(",") if x]
    targets = [x for x in args.target.split(",") if x]
    df = pd.read_csv(args.data)
    x = df[features].to_numpy(dtype=np.float32)
    method = args.method

    if method.startswith("sklearn_"):
        y = df[targets[0]].to_numpy()
        result = _sklearn_baseline(method, x, y, params, args.seed)
    elif method in ("pytorch_logreg", "pytorch_mlp"):
        y = df[targets[0]].to_numpy(dtype=np.float32)
        result = _torch_binary_baseline(method, x, y, params, args.seed, args.rounds)
    elif method == "pytorch_linear_regression":
        y = df[targets[0]].to_numpy(dtype=np.float32)
        result = _torch_linear_regression(x, y, params, args.seed, args.rounds)
    elif method == "pytorch_multiclass":
        y = df[targets[0]].to_numpy(dtype=np.int64)
        result = _torch_multiclass(x, y, params, args.seed, args.rounds)
    elif method == "pytorch_multilabel":
        y = df[targets].to_numpy(dtype=np.float32)
        result = _torch_multilabel(x, y, params, args.seed, args.rounds)
    elif method == "pytorch_poisson":
        y = df[targets[0]].to_numpy(dtype=np.float32)
        result = _torch_poisson(x, y, params, args.seed, args.rounds)
    elif method in ("pytorch_coxph", "pytorch_lognormal_aft"):
        result = _torch_survival(
            method,
            x,
            df[targets[0]].to_numpy(dtype=np.float32),
            df[targets[1]].to_numpy(dtype=np.float32),
            params,
            args.seed,
            args.rounds,
        )
    elif method == "pytorch_cause_specific_cox":
        result = _torch_survival(
            method,
            x,
            df[targets[0]].to_numpy(dtype=np.float32),
            df[targets[1]].to_numpy(dtype=np.float32),
            params,
            args.seed,
            args.rounds,
        )
    elif method == "pytorch_lstm":
        y = df[targets[0]].to_numpy(dtype=np.float32)
        result = _torch_lstm(x, y, params, args.seed, args.rounds)
    elif method == "pytorch_tcn":
        y = df[targets[0]].to_numpy(dtype=np.float32)
        result = _torch_tcn(x, y, params, args.seed, args.rounds)
    elif method == "xgboost":
        y = df[targets[0]].to_numpy(dtype=np.float32)
        result = _xgboost_baseline(x, y, params, args.seed)
    else:
        raise ValueError(f"Unsupported method: {method}")

    result.update({
        "method": method,
        "status": "ok",
        "n_samples": int(len(df)),
        "n_features": int(len(features)),
    })
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model-params", default="{}")
    parser.add_argument("--model-params-file", default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        return _fail(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
