#!/usr/bin/env python3
"""Verify all corrected artifacts, then score the sealed test split once."""
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
from opacus.accountants import PRVAccountant

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_public_data import load_split
from score_and_assemble import mean_sd, metrics
from dsflower_runner import client_app, dp_harness


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


@lru_cache(None)
def independent_accounting(sigma, sample_rate, steps, epsilon, delta):
    accountant = PRVAccountant()
    accountant.history = [(sigma, sample_rate, steps)]
    delta0 = delta / (1 + math.exp(epsilon / 2))
    epsilon0 = accountant.get_epsilon(delta=delta0)
    achieved_delta = delta0 * (1 + math.exp(epsilon0))
    assert 2 * epsilon0 <= epsilon and achieved_delta <= delta
    return {"accountant": "PRVAccountant", "full_horizon_steps": steps,
            "sample_rate": sample_rate, "noise_multiplier": sigma,
            "epsilon_replace_one": 2 * epsilon0,
            "delta_replace_one": achieved_delta}


def verify_run(root, protocol, audit, data, unit, epsilon, seed):
    run = root / "r3/runs" / f"{unit}-eps{epsilon}-seed{seed}"
    status = read(run / "federation-status.json")
    assert status["status"] == "trained_unscored" and status["cleanup_ok"]
    assert status["seed"] == seed and status["epsilon"] == epsilon
    assert status["unit"] == unit and status["contract"] == "pytorch_lstm"
    checkpoint = Path(status["output_dir"]) / "model.pt"
    assert sha(checkpoint) == status["model_sha256"]
    execution = read(run / "execution-status.json")
    assert execution["status"] == "trained_unscored"
    assert execution["protocol_sha256"] == sha(root / "r3/protocol.json")
    initial = read(run / "public-capture/public-initial.json")
    assert initial["seed"] == seed
    cfg = initial["config"]
    assert cfg["loss-name"] == "cross_entropy" and cfg["num-classes"] == 6
    bounds = client_app._effective_feature_bounds(cfg)
    public_scale = np.tile([1., 1., 1., 1., 1., 1., 2., 2., 2.], 128)
    assert np.array_equal(bounds["lower"], -public_scale)
    assert np.array_equal(bounds["upper"], public_scale)
    x = client_app._apply_feature_bounds(data["X"], cfg)
    y = data["y"].astype(np.float32)
    subjects = data["subjects"]
    expected = {}
    for site in audit["site_subjects"]:
        select = np.isin(subjects, site)
        xp, yp = x[select], y[select]
        if unit == "subject":
            xp, yp = client_app._pool_by_patient(
                xp, yp, subjects[select], "cross_entropy")
            xp = client_app._totalize_private_features(xp)
        expected[(digest(xp), digest(yp))] = (int(select.sum()), len(yp))
    assert len(expected) == 3
    captures = [read(path) for path in sorted(
        (run / "public-capture").glob("accountant-*.json"))]
    assert len(captures) == 15
    seen, accounting = set(), []
    schedule = protocol["model_params"]
    for capture in captures:
        key = (capture["features_sha256"], capture["targets_sha256"])
        assert key in expected
        rows, units = expected[key]
        assert capture["source_rows"] == rows
        assert capture["round"] in range(1, 6)
        assert (key, capture["round"]) not in seen
        seen.add((key, capture["round"]))
        pins = capture["training_pins"]
        assert pins["num_rounds"] == protocol["rounds"] == 5
        assert pins["batch_size"] == schedule["batch_size"]
        assert pins["local_epochs"] == schedule["local_epochs"]
        assert pins["learning_rate"] == schedule["learning_rate"]
        assert pins["optimizer"]["name"] == schedule["optimizer"]
        assert pins["round_index"] == capture["round"]
        privacy = capture["privacy_config"]
        assert privacy["epsilon"] == epsilon
        assert privacy["clipping_norm"] == protocol["clipping_norm"] == 1
        assert privacy["delta"] == protocol["delta"] == 1e-6
        expected_mechanism = dp_harness.effective_dpsgd_mechanism(
            epsilon, protocol["delta"], protocol["clipping_norm"], units,
            schedule["batch_size"], schedule["local_epochs"], 5)
        mechanism = capture["mechanism"]
        assert mechanism == expected_mechanism
        round_steps = mechanism["steps_per_epoch"] * schedule["local_epochs"]
        assert capture["observed_round_steps"] == round_steps
        assert capture["accountant_type"] == "PRVAccountant"
        assert capture["accountant_history"] == [[
            mechanism["noise_multiplier"], mechanism["sample_rate"], round_steps]]
        accounting.append(independent_accounting(
            mechanism["noise_multiplier"], mechanism["sample_rate"],
            mechanism["total_steps"], epsilon, protocol["delta"]))
    assert len(seen) == 15
    return {"unit": unit, "epsilon": epsilon, "seed": seed, "run": run,
            "status": status, "execution": execution, "checkpoint": checkpoint,
            "initial": initial, "captures": captures, "accounting": accounting}


def main(root, out, verify_only):
    tools = Path(__file__).resolve().parent
    protocol_path = root / "r3/protocol.json"
    assert sha(protocol_path) == sha(tools / "protocol.json")
    protocol = read(protocol_path)
    assert sorted(protocol["epsilon_order"]) == [1, 4, 8]
    assert len(protocol["seeds"]) == len(set(protocol["seeds"])) == 3
    audit = read(root / "r3/prepared/audit.json")
    data_path = root / "r3/prepared/train.npz"
    assert sha(data_path) == audit["train_npz_sha256"]
    data = np.load(data_path)
    assert len(data["y"]) == 7352 and len(np.unique(data["subjects"])) == 21
    assert len(audit["site_subjects"]) == 3
    assert all(len(site) == 7 for site in audit["site_subjects"])
    assert len(set(sum(audit["site_subjects"], []))) == 21
    runs = [verify_run(root, protocol, audit, data, unit, epsilon, seed)
            for unit in ("subject", "window")
            for epsilon in protocol["epsilon_order"] for seed in protocol["seeds"]]
    central = {}
    for seed in protocol["seeds"]:
        path = root / "r3/central" / f"seed{seed}"
        status = read(path / "status.json")
        assert status["status"] == "trained_unscored"
        assert status["seed"] == seed and status["test_accessed"] is False
        assert sha(path / "model.pt") == status["model_sha256"]
        initial_hashes = {tuple(run["initial"]["tensor_sha256"]) for run in runs
                          if run["seed"] == seed}
        assert initial_hashes == {tuple(status["initial_tensor_sha256"])}
        central[seed] = {"path": path / "model.pt", "training": status}
    runtime_path = root / "r3/runtime.json"
    runtime = read(runtime_path if runtime_path.exists() else root / "runtime.json")
    print(json.dumps({"status": "verified_unscored", "runs": len(runs),
                      "central_models": len(central), "test_accessed": False}), flush=True)
    if verify_only:
        return
    sys.path.insert(0, str(root / "src/dsFlowerClient/inst/python"))
    from predict_helper import _apply_feature_preprocessing, predict_pytorch_spec
    import torch
    torch.set_num_threads(2)
    # Exclusive marker precedes the first test feature or target access.
    with (root / "r3/test-scoring-started.json").open("x") as stream:
        json.dump({"started_at": datetime.now(timezone.utc).isoformat(),
                   "protocol_sha256": sha(protocol_path),
                   "federated_models": [run["status"]["model_sha256"] for run in runs],
                   "central_models": [value["training"]["model_sha256"]
                                      for value in central.values()]}, stream, indent=2)
    archive = root / "data_cache/uci-har-240.zip"
    assert sha(archive) == audit["archive_sha256"]
    x, y, subjects = load_split(archive, "test")
    assert len(y) == 2947 and len(np.unique(subjects)) == 9
    assert not set(subjects) & set(audit["train_subjects"])
    cfg = runs[0]["initial"]["config"]
    values = _apply_feature_preprocessing(x.reshape(len(y), -1), cfg["feature-bounds-b64"])
    prior = np.bincount(data["y"], minlength=6).astype(float)
    prior /= prior.sum()
    trivial = metrics(y, np.tile(prior, (len(y), 1)))

    def score(path, config):
        probabilities = np.asarray(predict_pytorch_spec(
            str(path), values, "prob", config["model-spec-b64"],
            "cross_entropy", num_classes=6))
        result = metrics(y, probabilities)
        return result, {
            "predicted_class_counts": np.bincount(probabilities.argmax(1), minlength=6).tolist(),
            "mean_class_probability": probabilities.mean(0).tolist(),
            "max_probability_span_across_windows": float(np.ptp(probabilities, axis=0).max()),
            "macro_auc_above_chance": result["macro_auc"] > .5,
            "accuracy_above_majority": result["accuracy"] > trivial["accuracy"],
            "annotation_only": True}

    for seed, value in central.items():
        config = next(run["initial"]["config"] for run in runs if run["seed"] == seed)
        value["metrics"], value["diagnostics"] = score(value["path"], config)
    grouped = {(unit, epsilon): [] for unit in ("subject", "window")
               for epsilon in protocol["epsilon_order"]}
    for run in runs:
        started = time.monotonic()
        federated, diagnostics = score(run["checkpoint"], run["initial"]["config"])
        seed = run["seed"]
        rep = {"seed": seed, "federated_dp": federated,
               "central": central[seed]["metrics"], "trivial": trivial,
               "gap_macro_auc": federated["macro_auc"] - central[seed]["metrics"]["macro_auc"],
               "federated_model_sha256": run["status"]["model_sha256"],
               "central_training": central[seed]["training"],
               "initial_tensor_sha256": run["initial"]["tensor_sha256"],
               "effective_config": run["initial"]["config"],
               "node_contract": read(run["run"] / "node-contract.json"),
               "node_round_captures": run["captures"],
               "independent_accounting": run["accounting"],
               "elapsed_s": {"federated_training": run["status"]["elapsed_s"],
                             "scoring": time.monotonic() - started},
               "utility_diagnostics": {"federated_dp": diagnostics,
                                       "central": central[seed]["diagnostics"]}}
        with (run["run"] / "scores.json").open("x") as stream:
            json.dump(rep, stream, indent=2, allow_nan=False)
            stream.write("\n")
        grouped[(run["unit"], run["epsilon"])].append(rep)
        print(run["unit"], run["epsilon"], seed, federated, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for (unit, epsilon), reps in grouped.items():
        summary = {branch: {metric: mean_sd([rep[branch][metric] for rep in reps])
                            for metric in ("macro_auc", "accuracy", "log_loss")}
                   for branch in ("central", "federated_dp", "trivial")}
        summary["gap_macro_auc"] = mean_sd([rep["gap_macro_auc"] for rep in reps])
        interpretation = (
            "Released patient contract: per-subject clipping after feature averaging and modal-label pooling; "
            "7 subjects/site, 21 total. Original-window per-subject loss is unsupported by this contract. "
            "The central comparator learns all original windows, so its gap also includes the pooling/task mismatch."
            if unit == "subject" else
            "Window-level mechanism measurement on subject-disjoint sites. Each row/window is the privacy unit; "
            "this does not provide subject-level protection.")
        cell = {"schema": "dsflower-sequence-campaign-r3-v1", "status": "executed",
                "contract": "pytorch_lstm", "unit": unit, "epsilon": epsilon,
                "protocol": protocol, "protocol_sha256": sha(protocol_path),
                "runtime": runtime, "interpretation": interpretation,
                "dataset": {**audit, "test_accessed": True, "n_test_windows": len(y),
                            "preparation_test_accessed": False,
                            "n_privacy_units_per_site": [7, 7, 7] if unit == "subject" else audit["n_per_site"],
                            "n_test_subjects": 9, "test_subjects": sorted(map(int, np.unique(subjects)))},
                "per_replicate": reps, "summary": summary,
                "diagnostics": {"annotation_only": True,
                    "central_learned": summary["central"]["macro_auc"]["mean"] > .5
                                       and summary["central"]["accuracy"]["mean"] > trivial["accuracy"],
                    "federated_auc_above_chance": summary["federated_dp"]["macro_auc"]["mean"] > .5,
                    "federated_accuracy_above_majority": summary["federated_dp"]["accuracy"]["mean"] > trivial["accuracy"]},
                "limitations": [interpretation, "Fixed split; SD measures training replicates only.",
                    "Each epsilon is a separate per-training contract, not a composed campaign guarantee.",
                    "AUC and accuracy use windows; the central comparator is shared once per seed across corrected cells."],
                "scored_at": datetime.now(timezone.utc).isoformat()}
        filename = f"har_{unit}_pytorch_lstm_eps{epsilon}.json"
        with (out / filename).open("x") as stream:
            json.dump(cell, stream, indent=2, allow_nan=False)
            stream.write("\n")
        cells.append({"file": filename, "sha256": sha(out / filename), "unit": unit,
                      "epsilon": epsilon, "summary": summary,
                      "interpretation": interpretation, "diagnostics": cell["diagnostics"]})
    with (out / "summary.json").open("x") as stream:
        json.dump({"schema": "dsflower-sequence-summary-r3-v1", "status": "executed",
                   "contract": "pytorch_lstm", "dataset": audit["name"], "cells": cells,
                   "protocol_sha256": sha(protocol_path), "runtime": runtime},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    main(args.root, args.out or args.root / "r3/evidence", args.verify_only)
