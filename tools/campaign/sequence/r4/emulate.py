#!/usr/bin/env python3
"""TRAIN-subject diagnostics; no released model, evaluation cell or TEST reader.

The clipped arm calls the installed runner's _dp_fit with sigma=0. Non-private
arms use the same model, optimizer, ChaCha Poisson sampler and mean divisor.
Public diagnostic streams pair the noiseless arms; actual DP keeps node secrets.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus import GradSampleModule
from scipy.stats import rankdata
from dsflower_runner import client_app, dp_harness, params, seeding, server_app


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metrics(y, prob):
    assert prob.shape == (len(y), 6) and np.isfinite(prob).all()
    assert np.allclose(prob.sum(1), 1, atol=1e-6)
    auc = []
    for k in range(6):
        positive = y == k
        n1, n0 = positive.sum(), (~positive).sum()
        assert min(n1, n0) > 0
        auc.append(float((rankdata(prob[:, k])[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)))
    return {"macro_auc": float(np.mean(auc)), "accuracy": float((prob.argmax(1) == y).mean()),
            "log_loss": float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-15, 1)).mean()),
            "auc_per_class": auc}


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        prob = torch.cat([model(z.cuda()).softmax(1).cpu()
                          for z in torch.from_numpy(x).split(512)]).numpy()
    return metrics(y, prob)


def arrays(model):
    return [a.copy() for a in params.get_torch_params(model)]


def build(cfg, weights=None):
    model = params.load_user_model(cfg, 1152, "cross_entropy")
    if weights is not None:
        params.set_torch_params(model, weights)
    return model.cuda()


def pins_for(schedule):
    return {"loss_name": "cross_entropy", "batch_size": schedule["batch"],
            "local_epochs": schedule["epochs"], "num_rounds": 5, "n_classes": 6,
            "learning_rate": schedule["lr"], "round_index": 1,
            "optimizer": {"name": schedule["optimizer"], "weight_decay": 0.,
                          "l1_penalty": 0., "momentum": 0., "nesterov": False,
                          "beta1": .9, "beta2": .999, "eps": 1e-8,
                          "amsgrad": False, "rmsprop_alpha": .99},
            "scheduler": {"name": "none", "step_size": 1, "gamma": .1, "min_lr": 0.}}


def diagnostic_master(seed, round_index, site):
    # Public replay stream ONLY for non-private diagnostic arms, never a release.
    return hashlib.sha256(f"har-inner-noiseless:{seed}:{round_index}:{site}".encode()).digest()


def nonprivate_local(model, x, y, pins, master, poisson=True):
    n, batch = len(y), pins["batch_size"]
    steps = math.ceil(n / batch)
    divisor = max(1, n // steps)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long()),
                        batch_size=batch, shuffle=not poisson)
    if poisson:
        loader = dp_harness._make_secure_poisson_loader(loader, steps_per_epoch=steps,
            secure_sampling_rng=seeding.np_rng(seeding.sub_seed(master, "sample")))
    optimizer = client_app._build_optimizer(model, pins)
    seeding.seed_torch(seeding.sub_seed(master, "train"))
    model.train()
    for _ in range(pins["local_epochs"]):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb.cuda())
            loss = torch.nn.functional.cross_entropy(output, yb.cuda(), reduction="sum")
            (loss / (divisor if poisson else max(1, len(yb)))).backward()
            optimizer.step()
            with torch.no_grad():
                for p in model.parameters():
                    p.nan_to_num_(nan=0., posinf=1e6, neginf=-1e6).clamp_(-1e6, 1e6)
    return arrays(model)


def central(cfg, initial, x, y, vx, vy, steps_per_round, poisson=False):
    model = build(cfg, initial)
    batch = 256
    # Reproduce R3's CUDA randperm and ordinary final partial minibatch.
    torch.manual_seed(20260922)
    xt, yt = torch.from_numpy(x).cuda(), torch.from_numpy(y).long().cuda()
    history = []
    total = 0
    started = time.monotonic()
    order, cursor = None, 0
    for r in range(1, 6):
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        rng = seeding.np_rng(seeding.sub_seed(diagnostic_master(20260922, r, 9), "sample"))
        model.train()
        for _ in range(steps_per_round):
            if poisson:
                index = np.flatnonzero(rng.bernoulli_mask_one_in(math.ceil(len(y) / batch), len(y)))
                denominator = len(y) // math.ceil(len(y) / batch)
            else:
                if order is None or cursor >= len(y):
                    order, cursor = torch.randperm(len(y), device="cuda"), 0
                index = order[cursor:cursor + batch]
                cursor += batch
                denominator = len(index)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xt[index]), yt[index], reduction="sum")
            (loss / denominator).backward()
            optimizer.step()
            total += 1
        row = {"round": r, "steps": total, **evaluate(model, vx, vy)}
        history.append(row)
        print(json.dumps({"arm": "central", "poisson": poisson,
                          "steps_per_round": steps_per_round, **row}), flush=True)
    return {"history": history, "final": history[-1], "steps": total,
            "poisson": poisson, "optimizer_reset_every_steps": steps_per_round,
            "elapsed_s": time.monotonic() - started}


def quantiles(values):
    q = [0, .01, .1, .5, .9, .95, .99, 1]
    return {str(k): float(v) for k, v in zip(q, np.quantile(values, q))}


def gradient_diagnostic(cfg, weights, x, y, check=False):
    model = build(cfg, weights)
    model.train()
    reference = []
    if check:
        for i in range(4):
            model.zero_grad()
            torch.nn.functional.cross_entropy(model(torch.from_numpy(x[i:i+1]).cuda()),
                torch.as_tensor(y[i:i+1], device="cuda")).backward()
            reference.append(torch.cat([p.grad.flatten() for p in model.parameters()]).cpu())
    model.zero_grad()
    wrapped = GradSampleModule(model, batch_first=True, loss_reduction="mean")
    raw_norms, coordinate_norms, factors = [], [], []
    affected, coordinates, nonfinite = 0, 0, 0
    raw_sum = clipped_sum = None
    max_gradient_error = 0.
    for start in range(0, len(y), 256):
        wrapped.zero_grad()
        xb = torch.from_numpy(x[start:start+256]).cuda()
        yb = torch.from_numpy(y[start:start+256]).long().cuda()
        torch.nn.functional.cross_entropy(wrapped(xb), yb).backward()
        gs = torch.cat([p.grad_sample.reshape(len(yb), -1) for p in wrapped.parameters()], 1)
        if check and start == 0:
            max_gradient_error = float((gs[:4].cpu() - torch.stack(reference)).abs().max())
            assert max_gradient_error < 2e-5
        nonfinite += int((~torch.isfinite(gs)).sum())
        affected += int((gs.abs() > 1).sum())
        coordinates += gs.numel()
        rn = gs.norm(2, dim=1)
        cs = gs.nan_to_num(nan=0., posinf=1., neginf=-1.).clamp(-1., 1.)
        cn = cs.norm(2, dim=1)
        factor = (1. / (cn + 1e-6)).clamp(max=1.)
        clipped = cs * factor[:, None]
        raw_sum = gs.sum(0) if raw_sum is None else raw_sum + gs.sum(0)
        clipped_sum = clipped.sum(0) if clipped_sum is None else clipped_sum + clipped.sum(0)
        raw_norms.extend(rn.detach().cpu().tolist())
        coordinate_norms.extend(cn.detach().cpu().tolist())
        factors.extend(factor.detach().cpu().tolist())
    result = {"n_windows": len(y), "raw_norm_quantiles": quantiles(raw_norms),
              "post_coordinate_norm_quantiles": quantiles(coordinate_norms),
              "norm_multiplier_quantiles": quantiles(factors),
              "fraction_global_clipped": float(np.mean(np.asarray(coordinate_norms) > 1)),
              "coordinate_fraction_clamped": affected / coordinates,
              "nonfinite_gradient_coordinates": nonfinite,
              "mean_gradient_norm_raw": float(raw_sum.norm() / len(y)),
              "mean_gradient_norm_clipped": float(clipped_sum.norm() / len(y)),
              "aggregate_norm_retention": float(clipped_sum.norm() / raw_sum.norm()),
              "aggregate_cosine": float(torch.nn.functional.cosine_similarity(raw_sum, clipped_sum, dim=0)),
              "per_sample_autograd_max_abs_error": max_gradient_error if check else None,
              "grad_sample_module": type(wrapped).__name__,
              "force_functorch": wrapped.force_functorch,
              "trainable_leaf_samplers": [{"module": name, "class": type(module).__name__,
                  "native_opacus_sampler": type(module) in GradSampleModule.GRAD_SAMPLERS}
                  for name, module in model.named_modules()
                  if any(p.requires_grad for p in module.parameters(recurse=False))
                  and not list(module.children())],
              "recurrent_module": type(next(m for m in model.modules() if type(m).__name__ == "DPLSTM")).__name__}
    wrapped.remove_hooks()
    return result


def federated(cfg, initial, data, validation, audit, schedule, *, clipped, out, seed=20260922,
              poisson=True, collect_norms=False):
    x, y, subjects = data
    vx, vy = validation
    weights = copy.deepcopy(initial)
    pins = pins_for(schedule)
    history, geometry, after_one = [], [], None
    for ids in audit["site_subjects"]:
        n = int(np.isin(subjects, ids).sum())
        geometry.append(dp_harness.effective_dpsgd_mechanism(8., 1e-6, 1., n,
            schedule["batch"], schedule["epochs"], 5))
    started = time.monotonic()
    for r in range(1, 6):
        local = []
        for site, ids in enumerate(audit["site_subjects"]):
            mask = np.isin(subjects, ids)
            sx, sy = x[mask], y[mask]
            model = build(cfg, weights)
            pins["round_index"] = r
            master = diagnostic_master(seed, r, site)
            if clipped:
                pcfg = {"epsilon": 8., "delta": 1e-6, "clipping_norm": 1., "n_samples": len(sy)}
                local_arrays, count = client_app._dp_fit(model, sx, sy.astype(np.float32), pcfg,
                    pins, len(sy), cfg, master=master, noise_multiplier=0.)
                assert count == len(sy)
                local.append([a.copy() for a in local_arrays])
            else:
                local.append(nonprivate_local(model, sx, sy, pins, master, poisson=poisson))
            del model
        # Released nodes return num-examples=1; Flower FedAvg gives equal weights.
        weights = [sum(site[k] * (1. / 3.) for site in local) for k in range(len(weights))]
        model = build(cfg, weights)
        row = {"round": r, **evaluate(model, vx, vy)}
        history.append(row)
        print(json.dumps({"schedule": schedule, "clipped": clipped, "poisson": poisson, **row}), flush=True)
        torch.save(model.cpu().state_dict(), out.with_suffix(".pt"))
        del model
        if collect_norms and r == 1:
            after_one = gradient_diagnostic(cfg, weights, x, y)
        result = {"schedule": schedule, "clipped": clipped, "poisson": poisson,
                  "noise_multiplier_executed": 0., "noise_calibration_at_epsilon8": geometry,
                  "sampling_stream": "Public deterministic ChaCha diagnostic stream; not a DP release",
                  "site_weights": [1, 1, 1], "history": history, "final": row,
                  "gradient_after_one_round": after_one, "elapsed_s": time.monotonic() - started}
        save(out, result)
    return result


CANDIDATES = [
    {"name": "adam01_e4_b256", "optimizer": "adam", "lr": .01, "epochs": 4, "batch": 256},
    {"name": "adam003_e4_b256", "optimizer": "adam", "lr": .003, "epochs": 4, "batch": 256},
    {"name": "adam001_e4_b256", "optimizer": "adam", "lr": .001, "epochs": 4, "batch": 256},
    {"name": "adam01_e4_b512", "optimizer": "adam", "lr": .01, "epochs": 4, "batch": 512},
    {"name": "adam003_e4_b512", "optimizer": "adam", "lr": .003, "epochs": 4, "batch": 512},
    {"name": "adam003_e8_b256", "optimizer": "adam", "lr": .003, "epochs": 8, "batch": 256},
    {"name": "adam003_e8_b512", "optimizer": "adam", "lr": .003, "epochs": 8, "batch": 512},
    {"name": "sgd01_e8_b256", "optimizer": "sgd", "lr": .1, "epochs": 8, "batch": 256},
]


def main(root, mode):
    work = root / "r4"
    out = work / "emulation"
    out.mkdir(exist_ok=True)
    cfg = json.loads((work / "config.json").read_text())
    audit = json.loads((work / "prepared/audit.json").read_text())
    fit = np.load(work / "prepared/train.npz")
    val = np.load(work / "prepared/validation.npz")
    x, vx = (client_app._apply_feature_bounds(z["X"], cfg) for z in (fit, val))
    y, subjects, vy = fit["y"], fit["subjects"], val["y"]
    assert len(y) == 5564 and len(vy) == 1788
    assert not set(subjects) & set(val["subjects"])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(20260922)
    model = server_app._build_initial_model(cfg)
    initial = arrays(model)
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in initial]
    twin = build(cfg, initial)
    model = model.cuda().eval()
    twin.eval()
    with torch.no_grad():
        forward_error = float((model(torch.from_numpy(x[:16]).cuda()) -
                               twin(torch.from_numpy(x[:16]).cuda())).abs().max())
    assert forward_error == 0
    seen = []
    recurrence = next(m for m in model.modules() if type(m).__name__ == "RecurrentBlock")
    handle = recurrence.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape))
    with torch.no_grad():
        model(torch.from_numpy(x[:2]).cuda())
    handle.remove()
    assert tuple(seen[0]) == (2, 128, 9)
    del model, twin
    summary_path = work / "emulation.json"
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "test_accessed": False, "initial_tensor_sha256": hashes,
        "server_node_forward_max_abs_error": forward_error,
        "recurrent_input_shape": list(seen[0]), "diagnostic_seed": 20260922}
    assert result["initial_tensor_sha256"] == hashes
    if mode in ("controls", "all"):
        if "initial_gradients" not in result:
            result["initial_gradients"] = gradient_diagnostic(cfg, initial, x, y, check=True)
            save(summary_path, result)
        horizons = sorted(set(math.ceil(n / 256) * 4 for n in audit["n_per_site"]))
        median_horizon = int(np.median([math.ceil(n / 256) * 4 for n in audit["n_per_site"]]))
        result["finite_primary_steps_per_round"] = median_horizon
        for name, horizon, poisson in [("central_full", math.ceil(len(y) / 256) * 4, False),
              *[(f"central_finite_{h*5}", h, False) for h in horizons],
              ("central_finite_poisson", median_horizon, True)]:
            if name not in result:
                result[name] = central(cfg, initial, x, y, vx, vy, horizon, poisson=poisson)
                save(summary_path, result)
        for name, clipped, poisson in [("federated_unclipped", False, True),
                                        ("federated_clipped", True, True),
                                        ("federated_unclipped_shuffled", False, False)]:
            if name not in result:
                result[name] = federated(cfg, initial, (x, y, subjects), (vx, vy), audit,
                    CANDIDATES[0], clipped=clipped, poisson=poisson, out=out / f"{name}.json",
                    collect_norms=poisson)
                save(summary_path, result)
    if mode in ("candidates", "all"):
        save(work / "candidate-schedules.json", CANDIDATES)
        records = result.setdefault("candidates", {})
        for schedule in CANDIDATES:
            name = schedule["name"]
            if name in records:
                continue
            if schedule == CANDIDATES[0] and "federated_clipped" in result:
                records[name] = result["federated_clipped"]
            else:
                records[name] = federated(cfg, initial, (x, y, subjects), (vx, vy), audit,
                    schedule, clipped=True, out=out / f"{name}.json")
            records[name]["full_training_noise_calibration_at_epsilon8"] = [
                dp_harness.effective_dpsgd_mechanism(8., 1e-6, 1., n,
                    schedule["batch"], schedule["epochs"], 5) for n in [2553, 2397, 2402]]
            save(summary_path, result)
        result["candidate_order_by_noiseless_auc"] = sorted(records,
            key=lambda name: records[name]["final"]["macro_auc"], reverse=True)
        save(summary_path, result)
    print("INNER_DIAGNOSIS_COMPLETE", mode, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=["controls", "candidates", "all"], default="all")
    args = parser.parse_args()
    main(args.root, args.mode)
