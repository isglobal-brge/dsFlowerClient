#!/usr/bin/env python3
"""Train public pooled twins; verify exact node tensors before any test access."""
import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import stat
from pathlib import Path
import time

import numpy as np
import torch
from dsflower_runner import client_app, dp_harness, params, seeding, vision

BENCHMARK_KEY_ROOT = Path("/tmp") / ("dsflower-vision-benchmark-%d" % os.getuid())


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


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def arrays_hash(arrays):
    return hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()


def train_features(root, seed, cfg):
    cache = root / "twins" / str(seed)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "training-features.npz"
    if path.exists():
        with np.load(path) as data:
            return [data[f"x{i}"] for i in range(3)], [data[f"y{i}"] for i in range(3)], data["image_labels"]
    encoder, size, is3d, device = vision.prepare_backbone(cfg["backbone"],
        cfg["vision-extractor-profile"], cfg["num-features"], cfg["image-size"])
    xs, ys, image_labels, values = [], [], [], {}
    for i in range(3):
        site = root / "prepared/vision" / str(seed) / f"site{i+1}"
        with (site / "samples.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        paths = [str(site / "images" / row["relative_path"]) for row in rows]
        labels = np.array([0 if row["pathology"] == "benign" else 1 for row in rows], dtype=np.float32)
        x = vision.extract_features_from_paths(encoder, paths, size, is3d, device)
        x = client_app._totalize_private_features(x)
        x, y = client_app._pool_by_patient(x, labels, [row["subject_id"] for row in rows], "cross_entropy")
        x = client_app._totalize_private_features(x)
        assert x.shape == (284, 512) and y.shape == (284,)
        xs.append(x)
        ys.append(y)
        image_labels.extend(labels)
        values.update({f"x{i}": x, f"y{i}": y})
    image_labels = np.asarray(image_labels, dtype=np.float32)
    np.savez(path, **values, image_labels=image_labels)
    return xs, ys, image_labels


def nonprivate_round(model, x, y, pins, cfg, master):
    """Same Poisson sampling and expected-batch divisor, without clipping/noise."""
    device = next(model.parameters()).device
    optimizer = client_app._build_optimizer(model, pins)
    criterion = dp_harness.loss_from_allowlist("cross_entropy", cfg)
    steps = math.ceil(len(x) / pins["batch_size"])
    divisor = max(1, len(x) // steps)
    sampler = dp_harness._SecurePoissonBatchSampler(num_samples=len(x), steps=steps,
        rng=seeding.np_rng(seeding.sub_seed(master, "sample")))
    model.train()
    for _ in range(pins["local_epochs"]):
        for indices in sampler:
            optimizer.zero_grad()
            output = model(torch.from_numpy(x[indices]).to(device))
            loss = (criterion(output, torch.from_numpy(y[indices]).long().to(device)) * len(indices) / divisor
                    if indices else output.sum() * 0)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.clamp_(-dp_harness.MAX_PARAMETER_ABS, dp_harness.MAX_PARAMETER_ABS)
    return params.get_torch_params(model)


def main(root, run):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    initial = json.loads((run / "public-capture/public-initial.json").read_text())
    cfg, seed = initial["config"], initial["seed"]
    assert cfg["backbone"] == "resnet18" and cfg["image-size"] == 224
    xs, ys, image_labels = train_features(root, seed, cfg)
    captures = [json.loads(p.read_text()) for p in (run / "public-capture").glob("accountant-*.json")]
    assert len(captures) == 15 and not list((run / "public-capture").glob("failure-*.json"))
    expected = {(digest(x), digest(y)): i for i, (x, y) in enumerate(zip(xs, ys))}
    seen = set()
    for capture in captures:
        key = (capture["features_sha256"], capture["targets_sha256"])
        assert key in expected, "Central features differ from actual node training tensors"
        pair = (expected[key], capture["round"])
        assert pair not in seen
        seen.add(pair)
        assert capture["observed_round_steps"] == 9
        assert capture["privacy_config"]["delta"] == 1e-6
        assert capture["privacy_config"]["clipping_norm"] == 1
        assert capture["mechanism"]["total_steps"] == 45
    assert seen == {(i, r) for i in range(3) for r in range(1, 6)}
    pins = captures[0]["training_pins"]
    assert pins["num_rounds"] == 5 and pins["local_epochs"] == 1 and pins["batch_size"] == 32
    assert pins["learning_rate"] == .001 and pins["optimizer"]["name"] == "sgd"
    assert pins["scheduler"]["name"] == "none"
    x, y = np.concatenate(xs), np.concatenate(ys)
    stored = np.load(run / "public-capture/public-initial-arrays.npz")
    public_arrays = [stored[str(i)] for i in range(len(stored.files))]
    assert [digest(a) for a in public_arrays] == initial["tensor_sha256"]
    epsilon = captures[0]["privacy_config"]["epsilon"]
    mechanism = dp_harness.effective_dpsgd_mechanism(epsilon, 1e-6, 1., 852, 32, 1, 5)
    results = {}
    for branch in ("central", "pooled_dp"):
        out = root / "twins" / str(seed) / branch if branch == "central" else run / branch
        if (out / "status.json").exists():
            status = json.loads((out / "status.json").read_text())
            assert branch == "central" and status["initial_tensor_sha256"] == initial["tensor_sha256"]
            results[branch] = status
            continue
        out.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        secret = private_key(out)
        arrays = [a.copy() for a in public_arrays]
        for round_index in range(1, 6):
            model = params.load_user_model(cfg, 512, "cross_entropy").cuda()
            params.set_torch_params(model, arrays)
            round_pins = dict(pins, round_index=round_index)
            master = hmac.new(secret, f"vision|{seed}|{branch}|{round_index}|".encode()
                              + arrays_hash(arrays).encode(), hashlib.sha256).digest()
            if branch == "central":
                arrays = nonprivate_round(model, x, y, round_pins, cfg, master)
            else:
                arrays, n = client_app._dp_fit(model, x, y,
                    dict(epsilon=epsilon, delta=1e-6, clipping_norm=1., n_samples=852),
                    round_pins, 852, cfg, master, mechanism["noise_multiplier"])
                assert n == 852
        checkpoint = out / "arrays.npz"
        np.savez(checkpoint, **{str(i): a for i, a in enumerate(arrays)})
        status = dict(status="trained_unscored", seed=seed, branch=branch, n_patients=852,
            n_images=len(image_labels), elapsed_s=time.monotonic()-started, steps=135,
            poisson_sample_rate=1/27, training_pins=pins, initial_tensor_sha256=initial["tensor_sha256"],
            model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            artifact=str(checkpoint), test_accessed=False,
            training_prevalence=float(image_labels.mean()),
            mechanism=mechanism if branch == "pooled_dp" else None)
        (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        results[branch] = status
    (run / "twins-status.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: dict(status=v["status"], elapsed_s=v["elapsed_s"]) for k, v in results.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.run)
