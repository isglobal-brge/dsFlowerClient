#!/usr/bin/env python3
"""Exact public benchmark twins using the trusted decoder/loss/DP training loop.

The nonprivate diagnostics are isolated here; no runtime privacy off switch is
introduced. Public tensors stay on the benchmark pod and never enter extdata.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import stat
import time
from types import SimpleNamespace

import numpy as np
import torch
from dsflower_runner import client_app, dp_harness, params, segmentation, seeding, task
from segmentation_metrics import metrics, trivial_masks
from assemble_evidence import validate_captures, independent_accounting

BENCHMARK_KEY_ROOT = Path("/tmp") / ("dsflower-segmentation-benchmark-%d" % os.getuid())


def array_hash(values):
    return hashlib.sha256(b"".join(np.asarray(a).tobytes() for a in values)).hexdigest()


def source_row_counts(samples_path, sites):
    with samples_path.open(newline="") as handle:
        counts = Counter(row["subject_id"] for row in csv.DictReader(handle))
    if any(subject not in counts for ids in sites for subject in ids):
        raise ValueError("public split contains a subject absent from cached source rows")
    return [sum(counts[subject] for subject in ids) for ids in sites]


def private_key(directory):
    # Public benchmark only: /workspace may be FUSE and ignore chmod. OS /tmp
    # supports owner-only files; this key persists across retries on this pod,
    # but is not promised to survive pod recreation and is never archived.
    key_directory = BENCHMARK_KEY_ROOT / hashlib.sha256(str(directory.resolve()).encode()).hexdigest()
    for parent in (BENCHMARK_KEY_ROOT, key_directory):
        parent.mkdir(mode=0o700, exist_ok=True)
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("benchmark key directory must be owned by this user with mode 0700")
    path = key_directory / "benchmark-secret"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(os.urandom(32))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("benchmark key must be owned by this user with mode 0600")
        value = handle.read(33)
    if len(value) != 32:
        raise ValueError("benchmark secret has invalid length")
    return value


def validate_twin_pins(cfg, manifest, pins):
    """Bind cached feature semantics and all active optimizer pins to federation."""
    segmentation.validate_config(cfg)
    segmentation.validate_config(manifest)
    for config in (cfg, manifest):
        segmentation.validate_decoder_spec(json.loads(base64.b64decode(
            config["model-spec-b64"], validate=True)))
    for key in segmentation.PIN_KEYS:
        # Alpha changes the loss only; the preregistered BCE branch shares features.
        if key != "segmentation-alpha" and cfg[key] != manifest[key]:
            raise ValueError("cached feature semantics differ from captured federation: " + key)
    expected = {"batch-size": 16, "local-epochs": 2, "num-server-rounds": 5,
                "learning-rate": .01, "optimizer-name": "sgd", "scheduler-name": "none"}
    defaults = {"weight-decay": 0., "l1-penalty": 0., "optimizer-momentum": 0.,
                "optimizer-nesterov": False}
    for config in (cfg, manifest):
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError("captured/cached schedule differs from preregistration: " + key)
        for key, value in defaults.items():
            if config.get(key, value) != value:
                raise ValueError("captured/cached optimizer differs from preregistration: " + key)
        if any(key.startswith("scheduler-") and key != "scheduler-name" for key in config):
            raise ValueError("preregistered schedule does not admit extra scheduler controls")
    if (pins["batch_size"] != 16 or pins["local_epochs"] != 2 or pins["num_rounds"] != 5
            or pins["learning_rate"] != .01 or pins["scheduler"]["name"] != "none"
            or pins["loss_name"] != "segmentation_bce_dice" or pins["n_classes"] != 2):
        raise ValueError("effective twin pins differ from captured federation")
    optimizer = {"name": "sgd", "weight_decay": 0., "l1_penalty": 0.,
                 "momentum": 0., "nesterov": False}
    if any(pins["optimizer"].get(key) != value for key, value in optimizer.items()):
        raise ValueError("effective twin optimizer differs from captured federation")


def nonprivate_round(model, X, y, pins, cfg, master):
    """Same Poisson geometry, expected divisor and schedule, without DP clipping/noise."""
    device = next(model.parameters()).device
    optimizer = client_app._build_optimizer(model, pins)
    loss = segmentation.loss_factory(cfg)
    steps = math.ceil(len(X) / pins["batch_size"])
    expected = max(1, len(X) // steps)
    sampler = dp_harness._SecurePoissonBatchSampler(
        num_samples=len(X), steps=steps,
        rng=seeding.np_rng(seeding.sub_seed(master, "sample")))
    seeding.seed_torch(seeding.sub_seed(master, "train"))
    model.train()
    for _ in range(pins["local_epochs"]):
        for indices in sampler:
            optimizer.zero_grad()
            xb, yb = torch.from_numpy(X[indices]).to(device), torch.from_numpy(y[indices]).to(device)
            output = model(xb)
            objective = loss(output, yb) * len(indices) / expected
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.nan_to_num_(nan=0, posinf=dp_harness.MAX_PARAMETER_ABS,
                                          neginf=-dp_harness.MAX_PARAMETER_ABS)
                    parameter.clamp_(-dp_harness.MAX_PARAMETER_ABS, dp_harness.MAX_PARAMETER_ABS)
    return params.get_torch_params(model)


def run_twins(X, y, subjects, split, cfg, initial, pins, epsilon, secret, output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = {str(subject): i for i, subject in enumerate(subjects)}
    def indices(ids):
        return np.array([index[subject] for subject in sorted(ids)])
    train = indices(split["train"])
    test = indices(split["test"])
    sites = [indices(ids) for ids in split["sites"]]
    semantic = json.dumps({"cfg": cfg, "split": split, "epsilon": epsilon,
                           "tensors": array_hash([X, y]), "initial": array_hash(initial)}, sort_keys=True).encode()
    started = time.monotonic()
    result = {}
    for branch in ("pooled_nonprivate", "pooled_dp", "federated_nonprivate"):
        arrays = [a.copy() for a in initial]
        groups = sites if branch == "federated_nonprivate" else [train]
        mechanism = dp_harness.effective_dpsgd_mechanism(
            epsilon, 1e-5, 1., len(train), pins["batch_size"], pins["local_epochs"], pins["num_rounds"])
        for round_index in range(1, pins["num_rounds"] + 1):
            local_arrays = []
            for site, rows in enumerate(groups):
                model = params.load_user_model(cfg, segmentation.FEATURE_DIM, "segmentation_bce_dice").to(device)
                params.set_torch_params(model, arrays)
                round_pins = dict(pins, round_index=round_index)
                master = hmac.new(secret, semantic + f"|{branch}|{site}|{round_index}".encode()
                                  + array_hash(arrays).encode(), hashlib.sha256).digest()
                if branch == "pooled_dp":
                    local, _ = client_app._dp_fit(model, X[rows], y[rows],
                        {"epsilon": epsilon, "delta": 1e-5, "clipping_norm": 1., "n_samples": len(rows)},
                        round_pins, len(rows), cfg, master, mechanism["noise_multiplier"])
                else:
                    local = nonprivate_round(model, X[rows], y[rows], round_pins, cfg, master)
                local_arrays.append(local)
            arrays = [np.mean(np.stack(parts), axis=0).astype(np.float32)
                      for parts in zip(*local_arrays)]
        # Match the released-artifact local predictor's CPU decoder arithmetic.
        model = params.load_user_model(cfg, segmentation.FEATURE_DIM, "segmentation_bce_dice")
        params.set_torch_params(model, arrays)
        model.eval()
        with torch.no_grad():
            probability = np.concatenate([model(torch.from_numpy(X[test[start:start + 16]])).sigmoid().numpy()
                                         for start in range(0, len(test), 16)])
        destination = output / (branch + ".npz")
        np.savez(destination, **{str(i): a for i, a in enumerate(arrays)})
        result[branch] = {"metrics": metrics(probability, y[test, :1]),
                          "artifact_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                          "initial_tensor_sha256": array_hash(initial)}
        if branch == "pooled_dp":
            result[branch]["mechanism"] = mechanism
            result[branch]["independent_accounting"] = independent_accounting(
                mechanism["noise_multiplier"], mechanism["sample_rate"], mechanism["total_steps"], epsilon)
    result["trivial"] = trivial_masks(y[test, :1])
    result["elapsed_s"] = time.monotonic() - started
    result["peak_cuda_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--epsilon", type=int, choices=(1, 4, 8), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gates = json.loads(args.gates.read_text())
    if any(gates.get(f"segmentation_6_1_{i}") is not True for i in range(1, 8)):
        raise ValueError("all seven blocking segmentation mechanism/API gates must pass before scoring")
    initial_meta = json.loads((args.capture / "public-initial.json").read_text())
    cfg = initial_meta["config"]
    segmentation.validate_config(cfg)
    for key, expected in {"batch-size": 16, "local-epochs": 2, "num-server-rounds": 5,
                          "learning-rate": .01}.items():
        if cfg.get(key) != expected:
            raise ValueError("captured public initialization differs from primary preregistration: " + key)
    data = np.load(args.features / "public-subject-tensors.npz", allow_pickle=False)
    initial_npz = np.load(args.capture / "public-initial-arrays.npz", allow_pickle=False)
    initial = [initial_npz[str(i)] for i in range(len(initial_npz.files))]
    if [hashlib.sha256(a.tobytes()).hexdigest() for a in initial] != initial_meta["tensor_sha256"]:
        raise ValueError("public initial arrays do not match capture digest")
    split = json.loads(args.split.read_text())
    expected_alpha = 1. if split.get("variant") == "bce" else .5
    if cfg["segmentation-alpha"] != expected_alpha:
        raise ValueError("loss alpha differs from preregistered variant")
    if split["seed"] != initial_meta["seed"]:
        raise ValueError("public split and initialization seeds must match")
    # Match effective tensors from actual node rounds, not only nominal transforms.
    lookup = {str(subject): i for i, subject in enumerate(data["subjects"])}
    expected_site_hashes = {}
    expected_source_rows = {}
    site_source_rows = source_row_counts(args.features / "samples.csv", split["sites"])
    for ids, source_rows in zip(split["sites"], site_source_rows):
        rows = [lookup[s] for s in sorted(ids)]
        hashes = (hashlib.sha256(data["X"][rows].tobytes()).hexdigest(),
                  hashlib.sha256(data["y"][rows].tobytes()).hexdigest())
        expected_site_hashes[hashes] = len(rows)
        expected_source_rows[hashes] = source_rows
    captures = [json.loads(p.read_text()) for p in args.capture.glob("accountant-*.json")]
    site_accounting = validate_captures(captures, list(map(len, split["sites"])),
                                        args.epsilon, expected_site_hashes, expected_source_rows)
    pins = task.load_run_pins(SimpleNamespace(node_config={"manifest-dir": str(args.features)}))
    feature_manifest = json.loads((args.features / "manifest.json").read_text())
    validate_twin_pins(cfg, feature_manifest, pins)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    result = run_twins(data["X"], data["y"], data["subjects"], split, cfg, initial, pins,
                       args.epsilon, private_key(args.out), args.out)
    result.update({"schema": "dsflower-segmentation-public-twins-v1", "status": "executed",
                   "seed": split["seed"], "epsilon": args.epsilon,
                   "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
                   "n_train": len(split["train"]), "n_per_site": list(map(len, split["sites"])),
                   "federated_accounting": site_accounting,
                   "torch": torch.__version__})
    result["prediction_device"] = "cpu"
    (args.out / "twins.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
