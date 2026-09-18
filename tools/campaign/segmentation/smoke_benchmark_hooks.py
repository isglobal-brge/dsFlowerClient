#!/usr/bin/env python3
"""Synthetic initialization/accountant instrumentation check; no utility scores.

Run with benchmark_hooks and the runner parent on PYTHONPATH plus
F_SEG_PUBLIC_BENCHMARK=1, F_SEG_INIT_SEED=20260919 and a fresh F_SEG_CAPTURE_DIR.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from feature_smoke import config
from dsflower_runner import server_app, client_app, dp_harness, params


def main():
    directory = Path(os.environ["F_SEG_CAPTURE_DIR"])
    if list(directory.glob("accountant-*.json")):
        raise ValueError("use an empty capture directory for this synthetic smoke")
    cfg = config()
    torch.set_num_threads(2)
    first, _ = server_app._initial_arrays(cfg, "neural")
    second, _ = server_app._initial_arrays(cfg, "neural")
    assert all(np.array_equal(x, y) for x, y in zip(params.get_torch_params(first), params.get_torch_params(second)))
    assert (directory / "public-initial-arrays.npz").is_file()
    model = params.load_user_model(cfg, 32768, "segmentation_bce_dice")
    pins = {"loss_name": "segmentation_bce_dice", "batch_size": 2, "local_epochs": 1,
            "num_rounds": 2, "n_classes": 2, "round_index": 1, "learning_rate": .01,
            "optimizer": {"name": "sgd", "weight_decay": 0., "l1_penalty": 0.,
                          "momentum": 0., "nesterov": False}, "scheduler": {"name": "none"}}
    mechanism = dp_harness.effective_dpsgd_mechanism(8, 1e-5, 1, 4, 2, 1, 2)
    x = np.zeros((4, 32768), np.float32)
    y = np.zeros((4, 2, 128, 128), np.float32)
    y[:, 1] = 1
    arrays, n = client_app._dp_fit(model, x, y,
        {"epsilon": 8, "delta": 1e-5, "clipping_norm": 1, "n_samples": 4},
        pins, 4, cfg, b"synthetic-public-fixture-seed-001", mechanism["noise_multiplier"])
    files = list(directory.glob("accountant-*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["observed_round_steps"] == 2
    assert len(arrays) == 6 and n == 4
    print("passed: deterministic public initial arrays; exactly two logical accountant steps; six decoder tensors")


if __name__ == "__main__":
    main()
