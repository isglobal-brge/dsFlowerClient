#!/usr/bin/env python3
"""Freeze every R5 artifact, then score the single held-out cohort once."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np


BRANCHES = ("central", "nonprivate_federated", "pooled_dp")


def metrics(y, probability):
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    assert len(y) == len(p) and set(y) == {0, 1}
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    differences = p[y == 1, None] - p[None, y == 0]
    auc = ((differences > 0).sum() + .5 * (differences == 0).sum()) / differences.size
    safe = np.clip(p, 1e-15, 1 - 1e-15)
    return dict(auc=float(auc), accuracy=float(((p >= .5) == y).mean()),
                brier=float(np.mean((p - y) ** 2)),
                log_loss=float(-np.mean(y * np.log(safe) + (1 - y) * np.log1p(-safe))))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


@lru_cache(None)
def independent_accounting(sigma, sample_rate, steps, epsilon, delta):
    from opacus.accountants import PRVAccountant
    accountant = PRVAccountant()
    accountant.history = [(sigma, sample_rate, steps)]
    delta0 = delta / (1 + math.exp(epsilon / 2))
    epsilon0 = accountant.get_epsilon(delta=delta0)
    achieved_delta = delta0 * (1 + math.exp(epsilon0))
    assert 2 * epsilon0 <= epsilon and achieved_delta <= delta
    return dict(accountant="PRVAccountant", full_horizon_steps=steps,
                sample_rate=sample_rate, noise_multiplier=sigma,
                epsilon_replace_one=2 * epsilon0, delta_replace_one=achieved_delta)


def verify_run(root, protocol, epsilon, seed, frozen):
    from dsflower_runner import dp_harness
    run = root / "r5_impl/runs" / f"pytorch_resnet18-eps{epsilon}-seed{seed}"
    federation_path = run / "federation-status.json"
    federation = read(federation_path)
    assert federation["status"] == "trained_unscored" and federation["cleanup_ok"]
    assert federation["seed"] == seed and federation["epsilon"] == epsilon
    assert federation["contract"] == "pytorch_resnet18"
    artifact = Path(federation["output_dir"])
    assert sha(artifact / "model.pt") == federation["model_sha256"]
    initial_path = run / "public-capture/public-initial.json"
    initial = read(initial_path)
    assert initial["seed"] == seed
    cfg = initial["config"]
    assert cfg["backbone"] == "resnet18" and cfg["image-size"] == 224
    assert cfg["loss-name"] == "cross_entropy" and cfg["num-classes"] == 2
    captures = sorted((run / "public-capture").glob("accountant-*.json"))
    assert len(captures) == 15 and not list((run / "public-capture").glob("failure-*.json"))
    schedule = protocol["model_params"]
    mechanism = dp_harness.effective_dpsgd_mechanism(
        epsilon, protocol["delta"], protocol["clipping_norm"], 284,
        schedule["batch_size"], schedule["local_epochs"], 5)
    seen = set()
    for path in captures:
        capture = read(path)
        key = (capture["features_sha256"], capture["targets_sha256"], capture["round"])
        assert key not in seen and capture["round"] in range(1, 6)
        seen.add(key)
        pins = capture["training_pins"]
        assert pins["num_rounds"] == 5 and pins["round_index"] == capture["round"]
        assert pins["batch_size"] == schedule["batch_size"]
        assert pins["local_epochs"] == schedule["local_epochs"]
        assert pins["learning_rate"] == schedule["learning_rate"]
        assert pins["optimizer"]["name"] == schedule["optimizer"]
        assert pins["optimizer"]["momentum"] == schedule["momentum"]
        assert pins["optimizer"]["weight_decay"] == schedule.get("weight_decay", 0)
        assert pins["optimizer"]["l1_penalty"] == schedule.get("l1_penalty", 0)
        assert pins["scheduler"]["name"] == schedule.get("scheduler", "none")
        privacy = capture["privacy_config"]
        assert privacy["epsilon"] == epsilon and privacy["delta"] == 1e-6
        assert privacy["clipping_norm"] == 1
        assert capture["mechanism"] == mechanism
        round_steps = mechanism["steps_per_epoch"] * schedule["local_epochs"]
        assert capture["observed_round_steps"] == round_steps
        assert capture["accountant_type"] == "PRVAccountant"
        assert capture["accountant_history"] == [[
            mechanism["noise_multiplier"], mechanism["sample_rate"], round_steps]]
    site_hashes = {(key[0], key[1]) for key in seen}
    assert len(site_hashes) == 3
    assert seen == {(x, y, r) for x, y in site_hashes for r in range(1, 6)}
    accounting = independent_accounting(mechanism["noise_multiplier"],
        mechanism["sample_rate"], mechanism["total_steps"], epsilon, 1e-6)
    twins_path = run / "twins-status.json"
    twins = read(twins_path)
    assert set(BRANCHES).issubset(twins)
    for branch in BRANCHES:
        status = twins[branch]
        assert status["status"] == "trained_unscored" and status["test_accessed"] is False
        assert sha(Path(status["artifact"])) == status["model_sha256"]
    pooled = twins["pooled_dp"]
    expected_pooled = dp_harness.effective_dpsgd_mechanism(epsilon, 1e-6, 1., 852, 852, 2, 5)
    assert pooled["mechanism"] == expected_pooled
    assert pooled["training_pins"]["batch_size"] == 852
    independent_accounting(expected_pooled["noise_multiplier"], 1., 10, epsilon, 1e-6)
    paths = [federation_path, artifact / "model.pt", artifact / "metadata.json",
             initial_path, run / "public-capture/public-initial-arrays.npz", twins_path,
             *captures, *[Path(twins[b]["artifact"]) for b in BRANCHES]]
    for optional in (run / "node-contract.json", run / "execution-status.json"):
        if optional.exists():
            paths.append(optional)
    for path in paths:
        frozen[str(path)] = sha(path)
    return dict(run=run, artifact=artifact, initial=initial, twins=twins,
                federation=federation, accounting=accounting,
                epsilon=epsilon, seed=seed)


def predict(driver, artifact, paths_file, output):
    started = time.monotonic()
    assert not output.exists()
    subprocess.run(["Rscript", str(driver), str(artifact), str(paths_file), str(output)], check=True)
    with output.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["benign", "malignant"]
        probability = np.asarray([float(row["malignant"]) for row in reader])
    return probability, time.monotonic() - started


def main(root, verify_only):
    tools = Path(__file__).resolve().parent
    protocol_path = tools / "protocol.json"
    protocol = read(protocol_path)
    assert protocol["split_seed"] == 20260919
    assert sorted(protocol["epsilon_order"]) == [1, 4, 8]
    assert protocol["seeds"] == [20260919, 20260920, 20260921]
    assert protocol["rounds"] == 5 and protocol["sites"] == 3
    assert protocol["delta"] == 1e-6 and protocol["clipping_norm"] == 1
    lock = root / "r5_impl/scoring-lock.json"
    assert not lock.exists(), "R5 scoring already started; never repeat a scored cell"
    frozen = {}
    runs = [verify_run(root, protocol, epsilon, seed, frozen)
            for epsilon in protocol["epsilon_order"] for seed in protocol["seeds"]]
    # Every epsilon reuses the same initialization and noiseless twins for a seed.
    for seed in protocol["seeds"]:
        same_seed = [run for run in runs if run["seed"] == seed]
        assert len({tuple(run["initial"]["tensor_sha256"]) for run in same_seed}) == 1
        for branch in ("central", "nonprivate_federated"):
            assert len({run["twins"][branch]["model_sha256"] for run in same_seed}) == 1
    print(json.dumps(dict(status="verified_unscored", federations=len(runs),
                          test_accessed=False)), flush=True)
    if verify_only:
        return
    prediction_driver = tools.parent / "predict.R"
    for path in [protocol_path, Path(__file__), prediction_driver]:
        frozen[str(path)] = sha(path)
    # This exclusive marker precedes opening ANY held-out split, metadata or image.
    save(lock, dict(started_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(protocol_path), artifacts=frozen,
        split_seed=protocol["split_seed"], epsilon_order=protocol["epsilon_order"],
        scoring_unit="image", secondary_unit="patient-mean features and modal label",
        configuration_changes_after_scoring_forbidden=True))
    started = time.monotonic()
    import torch
    from dsflower_runner import client_app, params, validation, vision
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    split_path = root / "prepared/busbra" / f"split-{protocol['split_seed']}.json"
    split = read(split_path)
    expected_hash = protocol["split"]["sha256_by_seed"][str(protocol["split_seed"])]
    assert sha(split_path) == expected_hash
    test_ids = set(split["test"])
    assert len(test_ids) == 212 and len(set(split["train"])) == 852
    assert not test_ids.intersection(split["train"])
    training_ids_sha256 = hashlib.sha256("\n".join(sorted(split["train"])).encode()).hexdigest()
    assert read(root / "r4/diagnosis/audit.json")["training_ids_sha256"] == training_ids_sha256
    assert all(run["twins"][branch]["training_ids_sha256"] == training_ids_sha256
               for run in runs for branch in BRANCHES)
    with (root / "data/busbra/BUSBRA/bus_data.csv").open() as stream:
        rows = [row for row in csv.DictReader(stream)
                if "busbra:" + str(int(row["Case"])) in test_ids]
    rows.sort(key=lambda row: row["ID"])
    y = np.asarray([protocol["class_mapping"][row["Pathology"].lower()] for row in rows])
    patient_ids = ["busbra:" + str(int(row["Case"])) for row in rows]
    assert set(patient_ids) == test_ids
    paths = [str(root / "data/busbra/BUSBRA/Images" / (row["ID"] + ".png")) for row in rows]
    directory = root / "r5_impl/scores"
    directory.mkdir(parents=True, exist_ok=False)
    paths_file = directory / "paths.json"
    save(paths_file, paths)
    encoder, size, is3d, device = vision.prepare_backbone("resnet18",
        protocol["model"]["extractor_profile"], 512, 224)
    features = np.concatenate([vision.extract_features_from_paths(
        encoder, paths[i:i + 1024], size, is3d, device) for i in range(0, len(paths), 1024)])
    features = client_app._totalize_private_features(features)
    patient_features, patient_y = client_app._pool_by_patient(
        features, y.astype(np.float32), patient_ids, "cross_entropy")
    patient_features = client_app._totalize_private_features(patient_features)
    assert patient_features.shape == (212, 512)
    cache, records = {}, []

    def direct_metrics(model, values, targets):
        probability = np.asarray(validation.neural_predictions(model, values, "cross_entropy"))[:, 1]
        return metrics(targets, probability)

    for seed in protocol["seeds"]:
        seed_runs = [run for run in runs if run["seed"] == seed]
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {run["epsilon"]: pool.submit(predict, prediction_driver, run["artifact"],
                    paths_file, directory / f"federated-eps{run['epsilon']}-seed{seed}.csv")
                    for run in seed_runs}
            predictions = {epsilon: job.result() for epsilon, job in jobs.items()}
        for run in seed_runs:
            epsilon, cfg = run["epsilon"], run["initial"]["config"]
            probability, prediction_seconds = predictions[epsilon]
            image_results = dict(federated_dp=metrics(y, probability))
            federated_model = params.load_user_model(cfg, 512, "cross_entropy")
            state = torch.load(run["artifact"] / "model.pt", map_location="cpu", weights_only=True)
            federated_model.load_state_dict(state.get("state_dict", state), strict=True)
            patient_results = dict(federated_dp=direct_metrics(federated_model, patient_features, patient_y))
            for branch in BRANCHES:
                status = run["twins"][branch]
                key = (branch, status["model_sha256"])
                if key not in cache:
                    with np.load(status["artifact"]) as arrays:
                        model = params.load_user_model(cfg, 512, "cross_entropy")
                        params.set_torch_params(model, [arrays[str(i)] for i in range(len(arrays.files))])
                    cache[key] = (direct_metrics(model, features, y),
                                  direct_metrics(model, patient_features, patient_y))
                image_results[branch], patient_results[branch] = cache[key]
            prevalence = run["twins"]["central"]["training_prevalence"]
            image_results["trivial"] = metrics(y, np.repeat(prevalence, len(y)))
            patient_prevalence = run["twins"]["central"]["patient_training_prevalence"]
            patient_results["trivial"] = metrics(patient_y, np.repeat(patient_prevalence, len(patient_y)))
            majority_rate = float(max(np.mean(y), 1 - np.mean(y)))
            record = dict(status="scored", epsilon=epsilon, seed=seed,
                split_seed=protocol["split_seed"], split_sha256=expected_hash,
                metrics=image_results, scoring_unit="image",
                patient_metrics=patient_results,
                patient_metric_definition="Frozen features averaged within held-out patient, modal label; training-patient prevalence trivial forecast. Direct fixed-numerics head inference.",
                n_train_patients=852, n_test_patients=212, n_test_images=len(y),
                test_prevalence=float(np.mean(y)), test_majority_rate=majority_rate,
                patient_test_prevalence=float(np.mean(patient_y)),
                training_prevalence=prevalence, patient_training_prevalence=patient_prevalence,
                gap=image_results["federated_dp"]["auc"] - image_results["central"]["auc"],
                patient_gap=patient_results["federated_dp"]["auc"] - patient_results["central"]["auc"],
                acceptance_diagnostic=(image_results["federated_dp"]["auc"] > .5 and
                    image_results["federated_dp"]["accuracy"] > majority_rate + 1e-12),
                diagnostic_annotation_only=True, prediction_elapsed_s=prediction_seconds,
                independent_accounting=run["accounting"],
                test_prediction_sha256=sha(directory / f"federated-eps{epsilon}-seed{seed}.csv"))
            save(run["run"] / "scores.json", record)
            records.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
    # Recheck frozen bytes after scoring; no declaration or trained model can drift.
    assert all(sha(Path(path)) == digest for path, digest in frozen.items())
    save(directory / "results.json", dict(status="scored", records=records,
        scoring_unit="image", split_seed=protocol["split_seed"],
        protocol_sha256=sha(protocol_path), scoring_lock_sha256=sha(lock),
        elapsed_s=time.monotonic() - started,
        finished_at=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    main(args.root, args.verify_only)
