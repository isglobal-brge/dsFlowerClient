#!/usr/bin/env python3
"""Fit declared R5 twins using freshly extracted TRAIN features, held in memory."""
import argparse
import hashlib
import hmac
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from dsflower_runner import client_app, dp_harness, params

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "r4"))
from diagnose import controls, extract, ids_hash, logistic
# prepare.py imports segmentation preparation and changes sys.path; bind this
# helper to the vision driver explicitly before importing the original twins.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from central_twins import arrays_hash, digest, nonprivate_round, private_key


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def logistic_arrays(vector, template):
    """Equivalent two-logit CE parameterization of the converged affine logit."""
    assert [array.shape for array in template] == [(2, 512), (2,)]
    weight = np.stack([-vector[:-1] / 2, vector[:-1] / 2])
    bias = np.asarray([-vector[-1] / 2, vector[-1] / 2])
    return [weight.astype(template[0].dtype), bias.astype(template[1].dtype)]


def main(root, run, features=None):
    controls()
    assert not (root / "r5_impl/scoring-lock.json").exists(), "Never retrain a scored cell"
    assert not (run / "twins-status.json").exists(), "Twins already completed"
    job = read(run / "job.json")
    federation = read(run / "federation-status.json")
    assert federation["status"] == "trained_unscored" and federation["cleanup_ok"]
    assert job["n_patients_per_site"] == [284, 284, 284]
    initial = read(run / "public-capture/public-initial.json")
    cfg, seed = initial["config"], initial["seed"]
    assert seed == job["seed"] and cfg["backbone"] == "resnet18"
    assert cfg["image-size"] == 224 and cfg["loss-name"] == "cross_entropy"
    # Standalone use re-extracts. The foreground matrix may supply the same fresh
    # final-phase extraction in memory; no diagnosis cache is ever consulted.
    if features is None:
        features = extract(Path(job["collection_root"]), cfg)
    xs, ys, ids, image_labels, feature_hashes = features
    assert [x.shape for x in xs] == [(284, 512)] * 3
    assert [y.shape for y in ys] == [(284,)] * 3
    assert len(ids) == len(set(ids)) == 852
    x, y = np.concatenate(xs), np.concatenate(ys)
    training_hashes = [(digest(a), digest(b)) for a, b in zip(xs, ys)]
    expected = {pair: index for index, pair in enumerate(training_hashes)}
    assert len(expected) == 3
    captures = [read(path) for path in (run / "public-capture").glob("accountant-*.json")]
    assert len(captures) == 15
    assert not list((run / "public-capture").glob("failure-*.json"))
    epsilon, schedule = job["epsilon"], job["model_params"]
    mechanism = dp_harness.effective_dpsgd_mechanism(
        epsilon, 1e-6, 1.0, 284, schedule["batch_size"], schedule["local_epochs"], 5)
    seen = set()
    pins = dict(captures[0]["training_pins"], round_index=1)
    for capture in captures:
        pair = (capture["features_sha256"], capture["targets_sha256"])
        assert pair in expected, "Fresh final features differ from actual node tensors"
        coordinate = (expected[pair], capture["round"])
        assert coordinate not in seen
        seen.add(coordinate)
        current = capture["training_pins"]
        assert dict(current, round_index=1) == pins
        assert current["num_rounds"] == 5
        assert current["round_index"] == capture["round"]
        for key in ("learning_rate", "local_epochs", "batch_size"):
            assert current[key] == schedule[key]
        assert current["optimizer"]["name"] == schedule["optimizer"]
        assert current["optimizer"]["momentum"] == schedule["momentum"]
        assert current["optimizer"]["weight_decay"] == schedule.get("weight_decay", 0)
        assert current["optimizer"]["l1_penalty"] == schedule.get("l1_penalty", 0) == 0
        assert current["scheduler"]["name"] == schedule.get("scheduler", "none") == "none"
        privacy = capture["privacy_config"]
        assert privacy["epsilon"] == epsilon and privacy["delta"] == 1e-6
        assert privacy["clipping_norm"] == 1 and capture["mechanism"] == mechanism
        assert capture["observed_round_steps"] == mechanism["steps_per_epoch"] * pins["local_epochs"]
    assert seen == {(site, round_index) for site in range(3) for round_index in range(1, 6)}
    with np.load(run / "public-capture/public-initial-arrays.npz") as stored:
        public_arrays = [stored[str(i)] for i in range(len(stored.files))]
    assert [digest(array) for array in public_arrays] == initial["tensor_sha256"]
    pooled_mechanism = dp_harness.effective_dpsgd_mechanism(
        epsilon, 1e-6, 1.0, 852, 852, pins["local_epochs"], 5)
    common = dict(status="trained_unscored", n_patients=852, n_images=len(image_labels),
        test_accessed=False, training_prevalence=float(np.mean(image_labels)),
        patient_training_prevalence=float(np.mean(y)), training_ids_sha256=ids_hash(ids),
        training_tensor_sha256=[list(pair) for pair in training_hashes], feature_hashes=feature_hashes,
        feature_policy="Fresh final-phase extraction; shared only in process memory; no feature cache read or written.",
        model_spec_b64=cfg["model-spec-b64"])
    results = {}
    for branch in ("central", "nonprivate_federated", "pooled_dp"):
        out = (root / "r5_impl/twins/central" if branch == "central" else
               root / "r5_impl/twins" / str(seed) / branch if branch == "nonprivate_federated" else
               run / branch)
        if (out / "status.json").exists():
            assert branch != "pooled_dp", "Pooled DP must execute once per cell replicate"
            status = read(out / "status.json")
            for key, value in common.items():
                assert status[key] == value, f"Reusable {branch} changed: {key}"
            if branch == "nonprivate_federated":
                assert status["initial_tensor_sha256"] == initial["tensor_sha256"]
                assert status["training_pins"] == pins
            assert sha(Path(status["artifact"])) == status["model_sha256"]
            results[branch] = status
            continue
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        details = {}
        if branch == "central":
            vector, convergence = logistic(x, y)
            arrays = logistic_arrays(vector, public_arrays)
            model = params.load_user_model(cfg, 512, "cross_entropy")
            params.set_torch_params(model, arrays)
            details = dict(solver=convergence, seed=None, training_pins=None,
                initial_tensor_sha256=None, mechanism=None,
                interpretation="Converged C=1 L2 logistic head on all 852 pooled patient features; unpenalized intercept; no privacy.")
        else:
            secret = private_key(out)
            arrays = [array.copy() for array in public_arrays]
            for round_index in range(1, 6):
                round_pins = dict(pins, round_index=round_index,
                    batch_size=852 if branch == "pooled_dp" else 284)
                round_input_hash = arrays_hash(arrays)
                updated = []
                site_data = list(zip(xs, ys)) if branch == "nonprivate_federated" else [(x, y)]
                for site, (sx, sy) in enumerate(site_data):
                    model = params.load_user_model(cfg, 512, "cross_entropy").cuda()
                    params.set_torch_params(model, arrays)
                    master = hmac.new(secret,
                        f"vision-r5_impl|{seed}|{branch}|{round_index}|{site}|{round_input_hash}".encode(),
                        hashlib.sha256).digest()
                    if branch == "nonprivate_federated":
                        next_arrays = nonprivate_round(model, sx, sy, round_pins, cfg, master)
                    else:
                        next_arrays, count = client_app._dp_fit(model, sx, sy,
                            dict(epsilon=epsilon, delta=1e-6, clipping_norm=1., n_samples=852),
                            round_pins, 852, cfg, master, pooled_mechanism["noise_multiplier"])
                        assert count == 852
                    updated.append(next_arrays)
                    del model
                arrays = (updated[0] if branch == "pooled_dp" else
                    [sum(update[i] * len(sy) for update, (_, sy) in zip(updated, site_data))
                     / sum(len(sy) for _, sy in site_data) for i in range(len(arrays))])
            details = dict(seed=seed, training_pins=dict(pins, batch_size=852 if branch == "pooled_dp" else 284),
                initial_tensor_sha256=initial["tensor_sha256"],
                mechanism=pooled_mechanism if branch == "pooled_dp" else None,
                steps_per_site=pooled_mechanism["total_steps"] if branch == "pooled_dp" else mechanism["total_steps"],
                interpretation=("Three-site FedAvg with identical initialization and selected schedule; same Poisson sampling and expected-batch divisor, with clipping and noise both removed."
                    if branch == "nonprivate_federated" else
                    "Unchanged released _dp_fit on 852 pooled patient records; same five-round schedule; freshly calibrated pooled DP geometry."))
        checkpoint = out / "arrays.npz"
        np.savez(checkpoint, **{str(i): array for i, array in enumerate(arrays)})
        status = dict(common, branch=branch, elapsed_s=time.monotonic() - started,
            artifact=str(checkpoint), model_sha256=sha(checkpoint), **details)
        save(out / "status.json", status)
        results[branch] = status
        print("TRAINED_TWIN", branch, seed, epsilon, status["elapsed_s"], flush=True)
    save(run / "twins-status.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.run)
