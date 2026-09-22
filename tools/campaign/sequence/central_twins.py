#!/usr/bin/env python3
"""Unscored noiseless twin of the released subject-pooled sequence model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from dsflower_runner import client_app, dp_harness, params, server_app


def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def training_tensors(root, cfg):
    data = np.load(root / "prepared/train.npz")
    x = client_app._apply_feature_bounds(data["X"], cfg)
    return x, data["y"], data["subjects"]


def verify_captures(root, run, cfg):
    audit = json.loads((root / "prepared/audit.json").read_text())
    x, y, subjects = training_tensors(root, cfg)
    expected = {}
    for site in audit["site_subjects"]:
        select = np.isin(subjects, site)
        xp, yp = client_app._pool_by_patient(x[select], y[select], subjects[select], "cross_entropy")
        xp = client_app._totalize_private_features(xp)
        expected[(digest(xp), digest(yp))] = int(select.sum())
    captures = [json.loads(p.read_text()) for p in (run / "public-capture").glob("accountant-*.json")]
    assert len(captures) == 15, f"Expected 15 successful node-round captures, got {len(captures)}"
    seen = set()
    for capture in captures:
        key = (capture["features_sha256"], capture["targets_sha256"])
        assert key in expected and capture["source_rows"] == expected[key]
        assert (key, capture["round"]) not in seen
        seen.add((key, capture["round"]))
        assert capture["round"] in range(1, 6)
        mechanism = capture["mechanism"]
        assert mechanism["sample_rate"] == 1.0
        assert capture["observed_round_steps"] == 1
        assert capture["privacy_config"]["clipping_norm"] == 1.0
    assert len(seen) == 15
    return captures, (x, y, subjects)


def main(root, run):
    out = run / "central"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    initial = json.loads((run / "public-capture/public-initial.json").read_text())
    cfg = initial["config"]
    captures, (x, y, subjects) = verify_captures(root, run, cfg)
    x, y = client_app._pool_by_patient(x, y, subjects, "cross_entropy")
    x = client_app._totalize_private_features(x)
    assert len(x) == 21
    pins = captures[0]["training_pins"]
    assert pins["num_rounds"] == 5 and pins["local_epochs"] == 1 and pins["batch_size"] == 32
    torch.set_num_threads(2)
    torch.manual_seed(initial["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = server_app._build_initial_model(cfg)
    stored = np.load(run / "public-capture/public-initial-arrays.npz")
    arrays = [stored[str(i)] for i in range(len(stored.files))]
    params.set_torch_params(model, arrays)
    model = model.cuda()
    x_tensor = torch.from_numpy(x).cuda()
    y_tensor = torch.from_numpy(y).long().cuda()
    criterion = dp_harness.loss_from_allowlist("cross_entropy", cfg)
    losses = []
    for round_index in range(1, 6):
        optimizer = client_app._build_optimizer(model, pins)
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for p in model.parameters():
                p.clamp_(-dp_harness.MAX_PARAMETER_ABS, dp_harness.MAX_PARAMETER_ABS)
        assert torch.isfinite(loss)
        losses.append(float(loss.detach().cpu()))
    checkpoint = out / "model.pt"
    torch.save({name: p.detach().cpu() for name, p in torch.nn.Module.named_parameters(model)}, checkpoint)
    # Exercise the unchanged local predictor on TRAIN inputs before test access.
    sys.path.insert(0, str(root / "src/dsFlowerClient/inst/python"))
    from predict_helper import predict_pytorch_spec
    smoke = np.asarray(predict_pytorch_spec(str(checkpoint), x[:2], "prob",
                       cfg["model-spec-b64"], "cross_entropy", num_classes=6))
    assert smoke.shape == (2, 6) and np.isfinite(smoke).all() and np.allclose(smoke.sum(1), 1)
    result = {"status": "trained_unscored", "seed": initial["seed"], "n_subjects": 21,
              "training_loss": losses, "elapsed_s": time.monotonic() - started,
              "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "initial_tensor_sha256": initial["tensor_sha256"],
              "optimizer": pins["optimizer"], "rounds": 5, "steps": 5,
              "learning_rate": pins["learning_rate"], "test_accessed": False,
              "poisson_sample_rate": 1.0, "subject_modal_class_counts": np.bincount(y, minlength=6).tolist()}
    (out / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    main(args.root, args.run)
