#!/usr/bin/env python3
"""Confirm three DP-selected schedules on the unchanged R4 inner patient split."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from dsflower_runner import dp_harness, params, validation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r4"))
from diagnose import controls, metrics, save, sha
from stage_training import stage


def staged_collection(root, out):
    """Reuse the verified raw-image staging; create a fresh copy if absent."""
    previous = root / "r4/diagnosis"
    staged = Path("/tmp/cells-vision-r4/inner-training")
    if not staged.exists():
        staged = Path("/tmp/cells-vision-r5/inner-training")
        audit = stage(previous / "collections", staged)
    else:
        audit = json.loads((staged / "staging-audit.json").read_text())
        assert audit == json.loads((previous / "training-staging.json").read_text())
        assert audit["n_patients"] == 681 and not audit["test_accessed"]
        for site in audit["sites"]:
            directory = staged / f"site{site['site']}"
            for name, hashes in site["file_hashes"].items():
                assert sha(directory / name) == hashes["staged_sha256"]
            with (directory / "content_hash_index.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    image = Path(row["uri"])
                    assert (directory / "images").resolve() in image.resolve().parents
                    assert sha(image) == row["content_hash"]
    save(out / "training-staging.json", dict(audit=audit, reused=staged.name == "inner-training"
        and "cells-vision-r4" in str(staged), verified_at=datetime.now(timezone.utc).isoformat(),
        test_accessed=False))
    return staged


def cached_inner(root, staged):
    previous = root / "r4/diagnosis"
    audit = json.loads((previous / "audit.json").read_text())
    cache = previous / "training-features.npz"
    assert sha(cache) == audit["diagnosis_cache_sha256"]
    with np.load(cache) as data:
        x, y, ids = data["x"], data["y"], data["ids"]
    with np.load(previous / "inner-indices.npz") as data:
        train, val = data["train"], data["validation"]
    assert x.shape == (852, 512) and len(train) == 681 and len(val) == 171
    assert not set(train) & set(val)
    lookup = {str(patient): i for i, patient in enumerate(ids)}
    expected, all_indices = set(), []
    # Re-extraction in the actual inner manifest order preserves its precise
    # backbone batching numerics; reordered outer cached tensors are not exact.
    with np.load(root / "r5/inner-features.npz") as data:
        inner_xs = [data[f"x{i}"] for i in range(3)]
        inner_ys = [data[f"y{i}"] for i in range(3)]
    for site in range(1, 4):
        with (staged / f"site{site}" / "samples.csv").open(newline="") as stream:
            patient_ids = list(dict.fromkeys(row["subject_id"] for row in csv.DictReader(stream)))
        indices = np.asarray([lookup[patient] for patient in patient_ids])
        assert len(indices) == 227
        sx, sy = inner_xs[site - 1], inner_ys[site - 1]
        assert sx.shape == (227, 512) and np.array_equal(sy, y[indices])
        all_indices.extend(indices.tolist())
        expected.add((hashlib.sha256(sx.tobytes()).hexdigest(),
                      hashlib.sha256(sy.tobytes()).hexdigest()))
    assert set(all_indices) == set(train) and len(set(all_indices)) == 681
    return x[val], y[val], expected, audit


def run_federation(tools, root, job_path):
    process = subprocess.Popen(["Rscript", str(tools / "run_federated.R"), str(root), str(job_path)],
                               start_new_session=True)
    try:
        code = process.wait(timeout=2400)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # All children in the dedicated federation process group are stopped.
        import os
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, process.args)


def confirm(root, out, tools, candidate, epsilon, staged, vx, vy, expected, audit):
    name = f"{candidate['id']}-eps{epsilon}"
    run = out / "federations" / name
    protocol = json.loads((tools.parent / "r4/diagnosis-protocol.json").read_text())
    model_params = dict(protocol["model_params"])
    for key in ("learning_rate", "local_epochs", "batch_size", "optimizer", "weight_decay"):
        model_params[key] = candidate[key]
    if candidate["optimizer"] == "sgd":
        model_params["momentum"] = candidate.get("momentum", 0)
    job = dict(run_dir=str(run), collection_root=str(staged), epsilon=epsilon,
        seed=protocol["diagnosis"]["seed"], model_params=model_params,
        n_patients_per_site=[227] * 3, split_sha256=audit["inner_train_ids_sha256"],
        candidate_id=candidate["id"])
    job_path = out / f"job-{name}.json"
    assert not run.exists(), "Retain existing attempts; do not overwrite a federation."
    save(job_path, job)
    print("INNER_FEDERATION_START", name, datetime.now(timezone.utc).isoformat(), flush=True)
    run_federation(tools, root, job_path)
    status = json.loads((run / "federation-status.json").read_text())
    assert status["status"] == "trained_unscored" and status["cleanup_ok"]
    capture_dir = run / "public-capture"
    initial = json.loads((capture_dir / "public-initial.json").read_text())
    previous_initial = json.loads((root / "r4/diagnosis/federations/adam_lr0.003_e20_b32-startup-retry/public-capture/public-initial.json").read_text())
    assert initial["seed"] == job["seed"]
    assert initial["tensor_sha256"] == previous_initial["tensor_sha256"]
    cfg = initial["config"]
    captures = [json.loads(path.read_text()) for path in sorted(capture_dir.glob("accountant-*.json"))]
    assert len(captures) == 15 and not list(capture_dir.glob("failure-*.json"))
    mechanism = dp_harness.effective_dpsgd_mechanism(epsilon, 1e-6, 1, 227,
        candidate["batch_size"], candidate["local_epochs"], 5)
    round_steps = candidate["local_epochs"] * math.ceil(227 / candidate["batch_size"])
    for capture in captures:
        pins = capture["training_pins"]
        assert all(pins[key] == candidate[key] for key in ("learning_rate", "batch_size", "local_epochs"))
        assert pins["num_rounds"] == 5
        assert pins["optimizer"]["name"] == candidate["optimizer"]
        assert pins["optimizer"]["weight_decay"] == candidate["weight_decay"]
        assert pins["optimizer"]["l1_penalty"] == 0
        assert pins["scheduler"]["name"] == "none"
        if candidate["optimizer"] == "sgd":
            assert pins["optimizer"]["momentum"] == candidate.get("momentum", 0)
        assert capture["observed_round_steps"] == round_steps
        assert capture["mechanism"] == mechanism
        assert mechanism["total_steps"] == 5 * round_steps
        assert capture["privacy_config"]["epsilon"] == epsilon
        assert capture["privacy_config"]["delta"] == 1e-6
        assert capture["privacy_config"]["clipping_norm"] == 1
    observed = {(c["features_sha256"], c["targets_sha256"], c["round"]) for c in captures}
    assert observed == {(a, b, r) for a, b in expected for r in range(1, 6)}, "Cached training features differ from actual node tensors."
    model = params.load_user_model(cfg, 512, "cross_entropy")
    model.load_state_dict(torch.load(Path(status["output_dir"]) / "model.pt", map_location="cpu", weights_only=True))
    probability = np.asarray(validation.neural_predictions(model, vx, "cross_entropy"))[:, 1]
    row = dict(candidate=candidate, inner_validation=metrics(vy, probability),
        n_fit_patients=681, n_validation_patients=171, actual_federation=True,
        epsilon=epsilon, delta=1e-6, clipping_norm=1, rounds=5, sites=3,
        mechanism=mechanism, feature_parity=True, elapsed_s=status["elapsed_s"],
        test_accessed=False, federation_status_sha256=sha(run / "federation-status.json"),
        model_sha256=status["model_sha256"], run_dir=str(run), initialization_seed=job["seed"])
    save(run / "inner-validation.json", row)
    print("INNER_VALIDATION", json.dumps(row), flush=True)
    return row


def main(root):
    controls()
    tools = Path(__file__).resolve().parent
    out = root / "r5"
    shortlist = json.loads((out / "shortlist.json").read_text())
    assert len(shortlist) == 3 and len({row["id"] for row in shortlist}) == 3
    assert not (out / "dp-confirmation.json").exists()
    staged = staged_collection(root, out)
    vx, vy, expected, audit = cached_inner(root, staged)
    results = []
    start = time.monotonic()
    for candidate in shortlist:
        results.append(confirm(root, out, tools, candidate, 8, staged, vx, vy, expected, audit))
        save(out / "dp-confirmation.json", results)
    selected = sorted(results, key=lambda row: (-row["inner_validation"]["auc"], row["candidate"]["id"]))[0]
    if selected["inner_validation"]["auc"] < 0.60:
        for epsilon in (4, 1):
            results.append(confirm(root, out, tools, selected["candidate"], epsilon, staged, vx, vy, expected, audit))
            save(out / "dp-confirmation.json", results)
    selection = dict(selected=selected, all_candidates=results, test_accessed=False,
        selected_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic() - start,
        selection_rule="Maximum actual federated-DP patient inner-validation AUC at epsilon 8; tie candidate id ascending.",
        budget_dependence_run=selected["inner_validation"]["auc"] < 0.60)
    save(out / "selection.json", selection)
    print("SCHEDULE_SELECTED", json.dumps(selected), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
