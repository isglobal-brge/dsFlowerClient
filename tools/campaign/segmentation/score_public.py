#!/usr/bin/env python3
"""Score only analyst-local public test probabilities produced by ds.flower.predict."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from segmentation_metrics import metrics, trivial_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    data = np.load(args.features / "public-subject-tensors.npz", allow_pickle=False)
    lookup = {str(subject): i for i, subject in enumerate(data["subjects"])}
    mask = data["y"][[lookup[s] for s in sorted(split["test"])], :1]
    probability = np.loadtxt(args.probabilities, delimiter=",").reshape(-1, 1, 128, 128)
    result = {"schema": "dsflower-segmentation-channel-b-v1", "status": "executed",
              "seed": split["seed"], "metrics": metrics(probability, mask),
              "trivial": trivial_masks(mask),
              "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
              "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
