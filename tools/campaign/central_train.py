#!/usr/bin/env python
"""Central (non-federated, non-DP) baseline trainer for campaign cells.

Fits the same model class as the federated contract on the pooled training
split and writes per-row test probabilities. Used for contracts whose central
twin is not expressible as an R glm (e.g. pytorch_mlp). Deterministic given
--seed. Runs inside the campaign client venv (torch available).
"""
import argparse
import csv

import torch
from torch import nn


def read_csv(path, with_target):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    if with_target:
        assert header[-1] == "target", "train.csv must end with a target column"
        x = [[float(v) for v in r[:-1]] for r in body]
        y = [float(r[-1]) for r in body]
        return torch.tensor(x), torch.tensor(y)
    x = [[float(v) for v in r] for r in body]
    return torch.tensor(x), None


def build_mlp(n_in, hidden):
    layers, prev = [], n_in
    for width in hidden:
        layers += [nn.Linear(prev, width), nn.ReLU()]
        prev = width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", default="64,32")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x, y = read_csv(args.train, with_target=True)
    xt, _ = read_csv(args.test, with_target=False)

    mean, std = x.mean(0), x.std(0).clamp_min(1e-8)
    x, xt = (x - mean) / std, (xt - mean) / std

    # Internal validation split for early stopping: the central baseline is a
    # well-trained (not overfitted) fit of the same model class, so the gap
    # column isolates the DP-federation cost rather than an optimisation
    # artefact of the ceiling itself.
    n = x.shape[0]
    perm0 = torch.randperm(n)
    n_val = max(1, int(round(0.2 * n)))
    val_idx, tr_idx = perm0[:n_val], perm0[n_val:]
    xv, yv = x[val_idx], y[val_idx]
    x, y = x[tr_idx], y[tr_idx]

    hidden = [int(v) for v in args.hidden.split(",") if v.strip()]
    model = build_mlp(x.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    def val_auc():
        model.eval()
        with torch.no_grad():
            s = model(xv).squeeze(-1)
        model.train()
        order = torch.argsort(s)
        ranks = torch.empty_like(order, dtype=torch.float)
        ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float)
        pos = yv == 1
        n1, n0 = int(pos.sum()), int((~pos).sum())
        if n1 == 0 or n0 == 0:
            return 0.5
        return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    best_auc, best_state, patience_left = -1.0, None, 20
    n_tr = x.shape[0]
    for _ in range(args.epochs):
        perm = torch.randperm(n_tr)
        for start in range(0, n_tr, args.batch):
            idx = perm[start:start + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(x[idx]).squeeze(-1), y[idx])
            loss.backward()
            opt.step()
        auc = val_auc()
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            patience_left = 20
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(xt).squeeze(-1))
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prob"])
        for p in prob.tolist():
            w.writerow([f"{p:.10f}"])


if __name__ == "__main__":
    main()
