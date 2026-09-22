#!/usr/bin/env python3
"""One sealed-test scoring pass, followed by complete paired evidence assembly."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import rankdata
from opacus.accountants import PRVAccountant

from central_twins import verify_captures
from prepare_public_data import load_split


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, p):
    assert p.shape == (len(y), 6) and np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(1), 1.0, atol=1e-6)
    aucs = []
    for label in range(6):
        positive = y == label
        n1, n0 = positive.sum(), (~positive).sum()
        assert n1 > 0 and n0 > 0
        aucs.append(float((rankdata(p[:, label])[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)))
    return {"macro_auc": float(np.mean(aucs)), "accuracy": float(np.mean(p.argmax(1) == y)),
            "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)).mean()),
            "auc_per_class": aucs}


def mean_sd(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


@lru_cache(None)
def independent_accounting(sigma, epsilon):
    accountant = PRVAccountant()
    accountant.history = [(sigma, 1.0, 5)]
    delta0 = 1e-6 / (1 + math.exp(epsilon / 2))
    epsilon0 = accountant.get_epsilon(delta=delta0)
    assert 2 * epsilon0 <= epsilon and delta0 * (1 + math.exp(epsilon0)) <= 1e-6
    return {"accountant": "PRVAccountant", "full_horizon_steps": 5,
            "epsilon_replace_one": 2 * epsilon0, "delta_replace_one": delta0 * (1 + math.exp(epsilon0))}


def main(root, out):
    tools = Path(__file__).resolve().parent
    protocol = read(tools / "protocol.json")
    audit = read(root / "prepared/audit.json")
    runtime = read(root / "runtime.json")
    runs = []
    initial_by_seed = {}
    for epsilon in protocol["epsilon_order"]:
        for seed in protocol["seeds"]:
            run = root / "runs" / f"pytorch_lstm-eps{epsilon}-seed{seed}"
            status = read(run / "federation-status.json")
            assert status["status"] == "trained_unscored" and status["cleanup_ok"]
            assert read(run / "central/status.json")["status"] == "trained_unscored"
            initial = read(run / "public-capture/public-initial.json")
            previous = initial_by_seed.setdefault(seed, initial["tensor_sha256"])
            assert previous == initial["tensor_sha256"]
            captures, _ = verify_captures(root, run, initial["config"])
            for capture in captures:
                m = capture["mechanism"]
                assert m["accounting_population"] == 7 and m["total_steps"] == 5
                independent_accounting(m["noise_multiplier"], epsilon)
            assert sha(Path(status["output_dir"]) / "model.pt") == status["model_sha256"]
            assert sha(run / "central/model.pt") == read(run / "central/status.json")["model_sha256"]
            runs.append((epsilon, seed, run, status, initial, captures))
    # The marker is opened exclusively before the first test read. No retraining follows.
    with (root / "test-scoring-started.json").open("x") as stream:
        json.dump({"started_at": datetime.now(timezone.utc).isoformat(),
                   "protocol_sha256": sha(tools / "protocol.json"),
                   "models": [s[3]["model_sha256"] for s in runs]}, stream, indent=2)
    assert sha(root / "data_cache/uci-har-240.zip") == audit["archive_sha256"]
    x, y, subjects = load_split(root / "data_cache/uci-har-240.zip", "test")
    assert len(y) == 2947 and len(np.unique(subjects)) == 9
    assert not set(subjects) & set(audit["train_subjects"])
    x = x.reshape(len(y), -1)
    sys.path.insert(0, str(root / "src/dsFlowerClient/inst/python"))
    from predict_helper import _apply_feature_preprocessing, predict_pytorch_spec
    import torch
    torch.set_num_threads(2)
    prior = np.asarray(audit["train_class_counts"], dtype=float)
    prior /= prior.sum()
    trivial = metrics(y, np.tile(prior, (len(y), 1)))
    grouped = {e: [] for e in protocol["epsilon_order"]}
    for epsilon, seed, run, status, initial, captures in runs:
        started = time.monotonic()
        cfg = initial["config"]
        values = _apply_feature_preprocessing(x, cfg.get("feature-bounds-b64"))
        branch_metrics = {}
        for name, path in (("federated_dp", Path(status["output_dir"]) / "model.pt"), ("central", run / "central/model.pt")):
            p = np.concatenate([np.asarray(predict_pytorch_spec(str(path), values[i:i+256], "prob",
                    cfg["model-spec-b64"], "cross_entropy", num_classes=6)) for i in range(0, len(values), 256)])
            branch_metrics[name] = metrics(y, p)
        rep = {"seed": seed, **branch_metrics, "trivial": trivial,
               "gap_macro_auc": branch_metrics["federated_dp"]["macro_auc"] - branch_metrics["central"]["macro_auc"],
               "federated_model_sha256": status["model_sha256"],
               "central_training": read(run / "central/status.json"),
               "node_contract": read(run / "node-contract.json"), "node_round_captures": captures,
               "independent_accounting": independent_accounting(captures[0]["mechanism"]["noise_multiplier"], epsilon),
               "initial_tensor_sha256": initial["tensor_sha256"], "effective_config": cfg,
               "elapsed_s": {"federated_training": status["elapsed_s"],
                             "central_training": read(run / "central/status.json")["elapsed_s"],
                             "scoring": time.monotonic() - started},
               "diagnostic_pass": branch_metrics["federated_dp"]["macro_auc"] > 0.5 and branch_metrics["federated_dp"]["accuracy"] > trivial["accuracy"]}
        (run / "scores.json").write_text(json.dumps(rep, indent=2) + "\n")
        grouped[epsilon].append(rep)
        print(epsilon, seed, branch_metrics, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    cell_summaries = []
    for epsilon, reps in grouped.items():
        summary = {branch: {metric: mean_sd([r[branch][metric] for r in reps])
                           for metric in ("macro_auc", "accuracy", "log_loss")}
                   for branch in ("federated_dp", "central", "trivial")}
        summary["gap_macro_auc"] = mean_sd([r["gap_macro_auc"] for r in reps])
        diagnostic = summary["federated_dp"]["macro_auc"]["mean"] > .5 and summary["federated_dp"]["accuracy"]["mean"] > trivial["accuracy"]
        cell = {"schema": "dsflower-sequence-campaign-v1", "status": "executed", "contract": "pytorch_lstm",
                "epsilon": epsilon, "runtime": runtime, "protocol": protocol, "protocol_sha256": sha(tools / "protocol.json"),
                "dataset": {**audit, "n_test_windows": len(y), "n_test_subjects": 9,
                            "test_subjects": sorted(map(int, np.unique(subjects))), "n_total": len(y) + audit["n_train_windows"]},
                "per_replicate": reps, "summary": summary,
                "acceptance_diagnostic": {"applicable": epsilon == 8, "passed": diagnostic,
                                          "per_seed_pass": [r["diagnostic_pass"] for r in reps]},
                "limitations": [protocol["patient_preprocessing"], "Only 21 independent training privacy units; fixed split, SD reflects training replicates only.",
                                "Central uses the same subject-pooling preprocessing; it is not a fully trained window-level HAR reference.",
                                "No pooled-DP twin scheduled; each epsilon is a separate per-training contract, not a composed campaign guarantee."],
                "scored_at": datetime.now(timezone.utc).isoformat()}
        name = f"pilot_uci_har_pytorch_lstm_eps{epsilon}.json"
        (out / name).write_text(json.dumps(cell, indent=2, allow_nan=False) + "\n")
        cell_summaries.append({"file": name, "sha256": sha(out / name), "epsilon": epsilon,
                               "summary": summary, "acceptance_diagnostic": cell["acceptance_diagnostic"]})
    (out / "summary.json").write_text(json.dumps({"schema": "dsflower-sequence-summary-v1", "status": "executed",
        "contract": "pytorch_lstm", "dataset": audit["name"], "cells": cell_summaries,
        "protocol_sha256": sha(tools / "protocol.json"), "runtime": runtime}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    main(args.root, args.out)
