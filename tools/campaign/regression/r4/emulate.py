"""Training-only finite-schedule diagnosis; never a private model release or cell.

Analytic linear MSE gradients reproduce the runner's coordinate saturation,
global norm clip, expected-size divisor, local optimizer reset, and unit FedAvg.
Public diagnostic seeds make noise/no-noise comparisons paired. NumPy supplies
independent Bernoulli samples and Gaussian noise with the runner's distributions;
these diagnostic random streams do not claim a privacy guarantee.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
SEEDS = [101, 202, 303, 404, 505]
INNER_SEED = 20260922


def metrics(z, y, w):
    residual = (z @ w - y).astype(np.float64) * 43
    return {"rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mean_residual": float(residual.mean())}


def gradients(z, y, w, coordinate=True, norm=True):
    g = 2 * (z @ w - y)[:, None] * z
    if coordinate:
        g = np.clip(g, -1, 1)
    if norm:
        g *= np.minimum(1, 1 / (np.linalg.norm(g, axis=1) + 1e-6))[:, None]
    return g


def initial(seed):
    torch.manual_seed(seed)
    layer = torch.nn.Linear(20, 1)
    return np.concatenate([layer.weight.detach().numpy().ravel(),
                           layer.bias.detach().numpy()])


def load_data(root, protocol, split_seed, full=False):
    sites, validation = [], []
    indices = []
    lo = np.array(protocol["feature_bounds"]["lower"], dtype=np.float64)
    hi = np.array(protocol["feature_bounds"]["upper"], dtype=np.float64)
    for site in range(1, 4):
        path = root / "data_cache" / f"cdcbmi_seed{split_seed}" / f"site{site}.csv"
        frame = pd.read_csv(path)
        x = frame[protocol["features"]].to_numpy(dtype=np.float32)
        x = ((np.clip(x, lo, hi) - (lo + hi) / 2) / ((hi - lo) / 2)).astype(np.float32)
        z = np.column_stack([x, np.ones(len(x), dtype=np.float32)])
        raw_y = frame.BMI.to_numpy(dtype=np.float32)
        assert np.all((raw_y >= 12) & (raw_y <= 98))
        y = (np.clip(raw_y, 12, 98) - 55) / 43
        order = np.random.default_rng(INNER_SEED + site).permutation(len(y))
        valid_idx, train_idx = order[:2400], order[2400:]
        indices.append({"site": site, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "inner_training_positions_sha256": hashlib.sha256(train_idx.astype('<i8').tobytes()).hexdigest(),
                        "inner_validation_positions_sha256": hashlib.sha256(valid_idx.astype('<i8').tobytes()).hexdigest()})
        sites.append((z, y) if full else (z[train_idx], y[train_idx]))
        if not full:
            validation.append((z[valid_idx], y[valid_idx]))
    train = tuple(np.concatenate([s[i] for s in sites]) for i in range(2))
    valid = train if full else tuple(np.concatenate([s[i] for s in validation]) for i in range(2))
    return sites, train, valid, indices


def run_one(job):
    sites, train, valid, config, seed, sigma = job
    w = initial(seed) if config.get("initialization", "random") == "random" else np.zeros(21, dtype=np.float32)
    coordinate = config.get("coordinate", True)
    norm = config.get("norm", True)
    lr, epochs, batch = config["lr"], config["epochs"], config["batch"]
    trajectory = []
    for round_index in range(5):
        local = []
        for site_index, (z, y) in enumerate(sites):
            steps = math.ceil(len(y) / batch)
            expected = max(1, int(len(y) / steps))
            sample_rng = np.random.default_rng([seed, round_index, site_index, 1])
            noise_rng = np.random.default_rng([seed, round_index, site_index, 2])
            theta = torch.tensor(w, requires_grad=True)
            optimizers = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam,
                          "adamw": torch.optim.AdamW, "rmsprop": torch.optim.RMSprop}
            optimizer = optimizers[config["optimizer"]](
                [theta], lr=lr, weight_decay=config.get("weight_decay", 0.0))
            for _ in range(epochs * steps):
                mask = sample_rng.random(len(y)) < 1 / steps
                g = gradients(z[mask], y[mask], theta.detach().numpy(), coordinate, norm)
                summed = g.sum(axis=0)
                if sigma:
                    summed += noise_rng.normal(0, sigma, summed.shape).astype(np.float32)
                theta.grad = torch.from_numpy(summed / expected)
                optimizer.step()
                with torch.no_grad():
                    theta.clamp_(-1e6, 1e6)
            local.append(theta.detach().numpy().copy())
        w = np.mean(local, axis=0, dtype=np.float32)
        trajectory.append({"round": round_index + 1, "training": metrics(*train, w),
                           "validation": metrics(*valid, w)})
    z, y = train
    raw = 2 * (z @ w - y)[:, None] * z
    saturated = np.clip(raw, -1, 1)
    return {"seed": seed, "sigma": sigma, "weights": w.tolist(),
            "trajectory": trajectory, "training": metrics(*train, w),
            "validation": metrics(*valid, w),
            "final_gradient": {"coordinate_clamped_row_fraction": float(np.mean(np.any(np.abs(raw) > 1, axis=1))),
              "norm_clipped_row_fraction": float(np.mean(np.linalg.norm(saturated, axis=1) > 1)),
              "mean_raw_gradient_norm": float(np.linalg.norm(raw.mean(axis=0))),
              "mean_clipped_gradient_norm": float(np.linalg.norm(gradients(z, y, w, coordinate, norm).mean(axis=0)))}}


def parity():
    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer
    from dsflower_runner.dp_harness import totalize_grad_samples
    differences = []
    rng = np.random.default_rng(73)
    for n in [7, 32, 53]:
        x = rng.uniform(-1, 1, (n, 20)).astype(np.float32)
        y = rng.uniform(-1, 1, n).astype(np.float32)
        layer = torch.nn.Linear(20, 1)
        start = initial(19)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(start[:20][None, :]))
            layer.bias.copy_(torch.tensor(start[20:]))
        model = GradSampleModule(layer)
        optimizer = DPOptimizer(torch.optim.SGD(model.parameters(), lr=.01),
            noise_multiplier=0, max_grad_norm=1, expected_batch_size=32)
        loss = torch.nn.functional.mse_loss(model(torch.from_numpy(x)), torch.from_numpy(y[:, None]))
        loss.backward()
        totalize_grad_samples(optimizer.params, 1)
        optimizer.step()
        actual = np.concatenate([layer.weight.detach().numpy().ravel(), layer.bias.detach().numpy()])
        z = np.column_stack([x, np.ones(n, dtype=np.float32)])
        expected = start - .01 * gradients(z, y, start).sum(axis=0) / 32
        differences.append({"actual_batch": n, "expected_batch": 32,
                            "max_parameter_difference": float(np.max(np.abs(actual - expected)))})
    assert max(row["max_parameter_difference"] for row in differences) < 1e-6
    return differences


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/workspace/cells"))
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--mode", choices=["baseline", "candidates", "controls"], default="baseline")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.runner))
    import opacus
    from dsflower_runner.dp_harness import effective_dpsgd_mechanism
    protocol = json.loads(args.protocol.read_text())
    configs = [{"name": "default", "optimizer": "sgd", "lr": .01, "epochs": 1, "batch": 32}]
    if args.mode == "candidates":
        configs = [
            {"name": "sgd_lr003_e3_b32", "optimizer": "sgd", "lr": .03, "epochs": 3, "batch": 32},
            {"name": "sgd_lr001_e5_b32", "optimizer": "sgd", "lr": .01, "epochs": 5, "batch": 32},
            {"name": "sgd_lr005_e3_b32", "optimizer": "sgd", "lr": .05, "epochs": 3, "batch": 32},
            {"name": "sgd_lr01_e1_b32", "optimizer": "sgd", "lr": .1, "epochs": 1, "batch": 32},
            {"name": "sgd_lr003_e5_b64", "optimizer": "sgd", "lr": .03, "epochs": 5, "batch": 64},
            {"name": "adam_lr0003_e1_b32", "optimizer": "adam", "lr": .003, "epochs": 1, "batch": 32},
            {"name": "adam_lr0001_e3_b32", "optimizer": "adam", "lr": .001, "epochs": 3, "batch": 32},
            {"name": "adamw_lr0001_e3_b32_wd001", "optimizer": "adamw", "lr": .001, "epochs": 3, "batch": 32, "weight_decay": .01},
        ]
    if args.mode == "controls":
        configs = [
            dict(configs[0], name="without_coordinate_clamp", coordinate=False),
            dict(configs[0], name="unclipped_mse", coordinate=False, norm=False),
            dict(configs[0], name="zero_initialization", initialization="zero"),
            dict(configs[0], name="long_clipped_schedule", lr=.01, epochs=20),
        ]
    output = {"purpose": "training-only diagnostic, not an evaluation cell", "inner_seed": INNER_SEED,
              "initialization_seeds": SEEDS, "rounds": 5, "parity": parity(),
              "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                          "opacus": opacus.__version__, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "semantics": "Poisson q=1/ceil(n/B); summed coordinate-clamped and global norm-1 clipped MSE gradients divided by floor(n/ceil(n/B)); fixed unit FedAvg; new optimizer each node-round",
              "splits": []}
    for split_seed in (protocol["seeds"][:1] if args.mode == "controls" else protocol["seeds"]):
        sites, train, valid, indices = load_data(args.root, protocol, split_seed)
        ols = np.linalg.lstsq(train[0].astype(np.float64), train[1], rcond=None)[0]
        trivial = np.zeros(21); trivial[-1] = train[1].mean()
        values = np.linalg.eigvalsh(train[0].astype(np.float64).T @ train[0] / len(train[1]))
        result = {"split_seed": split_seed, "n_train": len(train[1]), "n_valid": len(valid[1]),
                  "indices": indices, "ols": {"training": metrics(*train, ols), "validation": metrics(*valid, ols)},
                  "trivial": {"training": metrics(*train, trivial), "validation": metrics(*valid, trivial)},
                  "gram_eigenvalues": values.tolist(), "runs": []}
        for config in configs:
            mechanisms = {str(eps): effective_dpsgd_mechanism(eps, 1e-6, 1, len(sites[0][1]), config["batch"], config["epochs"], 5)
                          for eps in ([1, 4, 8] if args.mode == "baseline" else [8])}
            sigma = mechanisms["8"]["noise_multiplier"]
            noises = [0, sigma] if args.mode == "baseline" else [0]
            for noise in noises:
                jobs = [(sites, train, valid, config, seed, noise) for seed in SEEDS]
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                    runs = list(pool.map(run_one, jobs))
                entry = {"config": config, "mechanisms": mechanisms, "sigma": noise, "replicates": runs}
                result["runs"].append(entry)
                print(json.dumps({"split_seed": split_seed, "config": config, "sigma": noise,
                    "rmse": [r["validation"]["rmse"] for r in runs]}), flush=True)
        if args.mode == "baseline":
            full_sites, full_train, full_valid, _ = load_data(args.root, protocol, split_seed, full=True)
            mechanisms = {str(eps): effective_dpsgd_mechanism(eps, 1e-6, 1, 12000, 32, 1, 5)
                          for eps in [1, 4, 8]}
            mechanism = mechanisms["8"]
            result["original_geometry_mechanisms"] = mechanisms
            result["full_training_only"] = []
            for noise in [0, mechanism["noise_multiplier"]]:
                jobs = [(full_sites, full_train, full_valid, configs[0], seed, noise) for seed in SEEDS]
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                    runs = list(pool.map(run_one, jobs))
                result["full_training_only"].append({"sigma": noise, "replicates": runs})
        output["splits"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
