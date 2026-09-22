#!/usr/bin/env python3
"""Unscored synthetic GPU check of the unchanged recurrent training path."""
import base64
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from dsflower_runner import client_app, dp_harness, params

root = Path(sys.argv[1])
torch.set_num_threads(2)
torch.manual_seed(20260922)
assert torch.cuda.is_available()
records = []
for kind, lr in (("lstm", .001), ("gru", .01)):
    spec = {"kind": "graph", "output": "out", "nodes": [
        {"name": "x", "op": "reshape", "in": ["@in"], "shape": [128, 9]},
        {"name": "h", "op": kind, "in": ["x"], "hidden": 32},
        {"name": "out", "op": "linear", "in": ["h"], "out": "@out"}]}
    cfg = {"model-spec-b64": base64.b64encode(json.dumps(spec).encode()).decode(),
           "num-features": 1152, "num-classes": 6, "loss-name": "cross_entropy"}
    pins = {"loss_name": "cross_entropy", "batch_size": 32, "local_epochs": 1, "num_rounds": 5,
            "round_index": 1, "n_classes": 6, "learning_rate": lr,
            "optimizer": {"name": "sgd", "weight_decay": 0, "l1_penalty": 0,
                          "momentum": 0, "nesterov": False}, "scheduler": {"name": "none"}}
    pcfg = {"epsilon": 1.0, "delta": 1e-6, "clipping_norm": 1.0, "n_samples": 7}
    started = time.monotonic()
    phase = "model_load"
    try:
        model = params.load_user_model(cfg, 1152, "cross_entropy")
        phase = "dp_fit"
        x = np.random.default_rng(20260922).normal(0, .1, (7, 1152)).astype(np.float32)
        y = np.arange(7, dtype=np.int64) % 6
        mechanism = dp_harness.effective_dpsgd_mechanism(1., 1e-6, 1., 7, 32, 1, 5)
        arrays, count = client_app._dp_fit(model, x, y, pcfg, pins, 7, cfg,
                        os.urandom(32), mechanism["noise_multiplier"])
        assert count == 7 and all(np.isfinite(a).all() for a in arrays)
        records.append({"contract": "pytorch_" + kind, "status": "passed", "phase": phase,
                        "elapsed_s": time.monotonic() - started, "mechanism": mechanism})
        break  # GRU is eligible only if LSTM cannot execute.
    except Exception as error:
        records.append({"contract": "pytorch_" + kind, "status": "failed", "phase": phase,
                        "exception_type": type(error).__name__, "error": str(error),
                        "traceback": traceback.format_exc(), "elapsed_s": time.monotonic() - started})
result = {"schema": "dsflower-sequence-preflight-v1", "synthetic_only": True,
          "test_accessed": False, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "attempts": records}
(root / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
sys.exit(0 if records[-1]["status"] == "passed" else 1)
