#!/usr/bin/env python3
"""Load every frozen model through the unchanged predictor using two TRAIN rows."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    work = root / "r3"
    if (work / "test-scoring-started.json").exists():
        raise RuntimeError("Test scoring has started; refuse prediction smoke check")
    protocol = json.loads((work / "protocol.json").read_text())
    sys.path.insert(0, str(root / "src/dsFlowerClient/inst/python"))
    from predict_helper import _apply_feature_preprocessing, predict_pytorch_spec
    torch.set_num_threads(2)
    data = np.load(work / "prepared/train.npz")
    models = []
    configs = {}
    for unit in ("subject", "window"):
        for epsilon in protocol["epsilon_order"]:
            for seed in protocol["seeds"]:
                run = work / "runs" / f"{unit}-eps{epsilon}-seed{seed}"
                status = json.loads((run / "federation-status.json").read_text())
                cfg = json.loads((run / "public-capture/public-initial.json").read_text())["config"]
                configs[seed] = cfg
                models.append((run.name, Path(status["output_dir"]) / "model.pt", status, cfg))
    for seed in protocol["seeds"]:
        directory = work / "central" / f"seed{seed}"
        status = json.loads((directory / "status.json").read_text())
        models.append((f"central-seed{seed}", directory / "model.pt", status, configs[seed]))
    assert len(models) == 21
    bounds = models[0][3]["feature-bounds-b64"]
    values = _apply_feature_preprocessing(data["X"][:2], bounds)
    records = []
    for name, checkpoint, status, cfg in models:
        assert status["status"] == "trained_unscored"
        assert cfg["feature-bounds-b64"] == bounds
        assert sha(checkpoint) == status["model_sha256"]
        probabilities = np.asarray(predict_pytorch_spec(
            str(checkpoint), values, "prob", cfg["model-spec-b64"], "cross_entropy", num_classes=6))
        assert probabilities.shape == (2, 6) and np.isfinite(probabilities).all()
        assert (probabilities >= 0).all() and np.allclose(probabilities.sum(1), 1., atol=1e-6)
        records.append({"model": name, "model_sha256": status["model_sha256"],
                        "probabilities_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest()})
    with (work / "prediction-smoke.json").open("x") as stream:
        json.dump({"status": "passed", "test_accessed": False, "training_rows": [0, 1],
                   "protocol_sha256": sha(work / "protocol.json"), "models": records}, stream, indent=2)
        stream.write("\n")
    print("Prediction smoke passed for all 21 frozen models on two TRAIN rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
