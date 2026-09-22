#!/usr/bin/env python3
"""Train the declared noiseless central reference on all 7,352 TRAIN windows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from dsflower_runner import client_app
from diagnose import bounded_config, fit, save


def main(root):
    work = root / "r3"
    if (work / "test-scoring-started.json").exists():
        raise RuntimeError("Scoring started; refuse training")
    tools = Path(__file__).resolve().parent
    protocol = json.loads((tools / "protocol.json").read_text())
    assert (work / "protocol.json").read_bytes() == (tools / "protocol.json").read_bytes()
    selected = protocol["model_params"]
    data = np.load(root / "prepared/train.npz")
    old = json.loads((root / "runs/pytorch_lstm-eps1-seed20260922/public-capture/public-initial.json").read_text())
    cfg, bounds = bounded_config(old["config"])
    audit = json.loads((root / "prepared/audit.json").read_text())
    audit.update(bounds=bounds, channel_bounds={"lower": [-1.] * 6 + [-2.] * 3,
                 "upper": [1.] * 6 + [2.] * 3},
                 bounds_source="Fixed public design constants, not empirical extrema",
                 test_accessed=False)
    prepared = work / "prepared"
    prepared.mkdir(exist_ok=False)
    for name in ("train.csv", "train.npz"):
        (prepared / name).symlink_to(root / "prepared" / name)
    save(prepared / "audit.json", audit)
    x = client_app._apply_feature_bounds(data["X"], cfg)
    y = data["y"]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for seed in protocol["seeds"]:
        destination = work / "central" / f"seed{seed}"
        result = fit(cfg, x, y, x[::8], y[::8], seed=seed,
            lr=selected["learning_rate"], epochs=protocol["rounds"] * selected["local_epochs"],
            batch=selected["batch_size"], optimizer=selected["optimizer"],
            reset_every=selected["local_epochs"], out=destination)
        result["history_evaluation"] = "TRAIN monitor every eighth window; not validation or test"
        result["protocol_sha256"] = hashlib.sha256((tools / "protocol.json").read_bytes()).hexdigest()
        result["privacy"] = "No DP, no gradient clipping, no federation; shuffled ordinary minibatches"
        result["comparison"] = "Same architecture/init, bounds and nominal epochs; central uses all original windows"
        save(destination / "status.json", result)
        print("CENTRAL_COMPLETE", seed, result["model_sha256"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    main(p.parse_args().root)
