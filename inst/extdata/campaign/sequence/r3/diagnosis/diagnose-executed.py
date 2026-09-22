#!/usr/bin/env python3
"""TRAIN-only central diagnosis using the unchanged contract model on CUDA."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from dsflower_runner import client_app, params, server_app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_public_data import load_split
from score_and_assemble import metrics


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def bounded_config(cfg):
    cfg = dict(cfg)
    b = np.tile([1.] * 6 + [2.] * 3, 128)
    bounds = {"lower": (-b).tolist(), "upper": b.tolist()}
    cfg["feature-bounds-b64"] = base64.b64encode(json.dumps(bounds).encode()).decode()
    return cfg, bounds


def fit(cfg, x, y, vx, vy, *, seed, lr, epochs, batch, optimizer, reset_every, out=None):
    torch.manual_seed(seed)
    model = server_app._build_initial_model(cfg).cuda()
    initial = [hashlib.sha256(a.tobytes()).hexdigest() for a in params.get_torch_params(model)]
    xt, yt = torch.from_numpy(x).cuda(), torch.from_numpy(y).long().cuda()
    vt = torch.from_numpy(vx).cuda()
    history = []
    start = time.monotonic()
    steps = 0
    for epoch in range(epochs):
        if epoch % reset_every == 0:
            opt = (torch.optim.Adam if optimizer == "adam" else torch.optim.SGD)(model.parameters(), lr=lr)
        model.train()
        losses = []
        order = torch.randperm(len(x), device="cuda")
        for index in order.split(batch):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(xt[index]), yt[index])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            steps += 1
        model.eval()
        with torch.no_grad():
            prob = torch.cat([model(chunk).softmax(1) for chunk in vt.split(512)]).cpu().numpy()
        row = {"epoch": epoch + 1, "training_loss": float(np.mean(losses)), **metrics(vy, prob)}
        history.append(row)
        print(json.dumps({"lr": lr, "optimizer": optimizer, "n_train": len(x), **row}), flush=True)
    result = {"status": "trained_unscored", "seed": seed, "epochs": epochs,
              "learning_rate": lr, "batch_size": batch, "optimizer": optimizer,
              "optimizer_reset_every_epochs": reset_every, "steps": steps,
              "initial_tensor_sha256": initial, "history": history,
              "elapsed_s": time.monotonic() - start, "test_accessed": False,
              "n_training_windows": len(x)}
    if out:
        out.mkdir(parents=True, exist_ok=False)
        torch.save(model.cpu().state_dict(), out / "model.pt")
        result["model_sha256"] = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
        save(out / "status.json", result)
    return result


def main(root):
    target = root / "r3"
    target.mkdir(exist_ok=True)
    out = target / "diagnosis"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    data = np.load(root / "prepared/train.npz")
    raw, y, subjects = data["X"], data["y"], data["subjects"]
    ax, ay, asub = load_split(root / "data_cache/uci-har-240.zip", "train")
    assert np.array_equal(ax.reshape(len(y), -1), raw)
    assert np.array_equal(ay, y) and np.array_equal(asub, subjects)
    old = json.loads((root / "runs/pytorch_lstm-eps1-seed20260922/public-capture/public-initial.json").read_text())
    cfg, bounds = bounded_config(old["config"])
    spec = json.loads(base64.b64decode(cfg["model-spec-b64"]))
    assert spec["nodes"][0]["shape"] == [128, 9]
    assert [spec["nodes"][i]["op"] for i in range(3)] == ["reshape", "lstm", "linear"]
    ids = np.unique(subjects)
    val_ids = ids[::5]
    val = np.isin(subjects, val_ids)
    train = ~val
    public = client_app._apply_feature_bounds(raw, cfg)
    old_scaled = client_app._apply_feature_bounds(raw, old["config"])
    b = np.asarray([1.] * 6 + [2.] * 3)
    assert np.array_equal(public.reshape(-1, 128, 9), np.clip(ax, -b, b).astype(np.float64) / b)
    # Capture the actual model's recurrent input; no alternate architecture.
    model = server_app._build_initial_model(cfg)
    seen = []
    recurrent = next(m for m in model.modules() if m.__class__.__name__ == "RecurrentBlock")
    handle = recurrent.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().numpy().copy()))
    with torch.no_grad():
        model(torch.from_numpy(public[:2]))
    handle.remove()
    assert seen[0].shape == (2, 128, 9) and np.array_equal(seen[0], public[:2].reshape(2, 128, 9))
    pooled_x, pooled_y = client_app._pool_by_patient(old_scaled, y.astype(np.float32), subjects, "cross_entropy")
    audit = {"started_at": datetime.now(timezone.utc).isoformat(), "test_accessed": False,
             "train_subjects": ids[train_ids].tolist() if False else ids[~np.isin(ids, val_ids)].tolist(),
             "validation_subjects": val_ids.tolist(), "n_inner_train": int(train.sum()), "n_validation": int(val.sum()),
             "archive_alignment_exact": True, "recurrent_input_shape": list(seen[0].shape),
             "layout_exact": True, "prior_was_affine_scaled": True,
             "public_bounds": bounds, "channel_abs_bounds": b.tolist(),
             "train_clipped_fraction_per_channel": (np.abs(ax) > b).mean((0, 1)).tolist(),
             "subject_modal_class_counts": np.bincount(pooled_y.astype(int), minlength=6).tolist(),
             "old_effective_subjects": len(pooled_x), "old_steps": 5,
             "selection_rule": "Largest final inner-validation macro AUC among Adam lr .003 and .01; 20 epochs, batch 256, optimizer reset every four epochs; tie lower learning rate."}
    save(out / "audit.json", audit)
    records = {}
    # Frozen, small TRAIN-only schedule comparison; do not use test outcomes.
    for lr in (.003, .01):
        records[f"public_adam_{lr}"] = fit(cfg, public[train], y[train], public[val], y[val],
            seed=20260922, lr=lr, epochs=20, batch=256, optimizer="adam", reset_every=4)
        save(out / "experiments.json", records)
    selected = max((.003, .01), key=lambda lr: records[f"public_adam_{lr}"]["history"][-1]["macro_auc"])
    # Schedule and scaling ablations share the same subject split and initialization.
    records["old_bounds_default_window"] = fit(cfg, old_scaled[train], y[train], old_scaled[val], y[val],
        seed=20260922, lr=.001, epochs=5, batch=32, optimizer="sgd", reset_every=1)
    records["old_bounds_corrected_window"] = fit(cfg, old_scaled[train], y[train], old_scaled[val], y[val],
        seed=20260922, lr=selected, epochs=20, batch=256, optimizer="adam", reset_every=4)
    px, py = client_app._pool_by_patient(old_scaled[train], y[train].astype(np.float32), subjects[train], "cross_entropy")
    records["old_bounds_default_pooled"] = fit(cfg, px, py, old_scaled[val], y[val],
        seed=20260922, lr=.001, epochs=5, batch=32, optimizer="sgd", reset_every=1)
    records["public_corrected_pooled"] = fit(cfg, *client_app._pool_by_patient(public[train], y[train].astype(np.float32), subjects[train], "cross_entropy"), public[val], y[val],
        seed=20260922, lr=selected, epochs=20, batch=256, optimizer="adam", reset_every=4)
    save(out / "experiments.json", records)
    save(out / "selection.json", {"learning_rate": selected, "optimizer": "adam", "local_epochs": 4,
        "rounds": 5, "batch_size": 256, "n_tokens": 128, "n_features": 9, "hidden": 32,
        "n_classes": 6, "scheduler": "none", "test_accessed": False, "selection_rule": audit["selection_rule"]})
    print("DIAGNOSIS_COMPLETE", selected, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    main(p.parse_args().root)
