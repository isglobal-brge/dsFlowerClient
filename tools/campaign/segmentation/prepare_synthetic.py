#!/usr/bin/env python3
"""Deterministic, unscored public fixture for the real three-node gate."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(root):
    root = root.resolve()
    images, masks = root / "images", root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rows, subjects = [], [f"s{i:03d}" for i in range(51)]
    yy, xx = np.indices((128, 128))
    for i, subject in enumerate(subjects):
        image = np.stack([(xx + i * 3) % 256, (yy * 2) % 256, (xx + yy) % 256], axis=-1).astype("uint8")
        Image.fromarray(image).save(images / f"{subject}.png")
        mask = ((xx - 64) ** 2 + (yy - 64) ** 2 < (16 + i % 8) ** 2).astype("uint8") * 255
        empty = i % 16 == 0
        if empty:
            mask[:] = 0
        Image.fromarray(mask).save(masks / f"{subject}.png")
        # One invalid selected mask per site, retained in the subject census.
        mask_path = "missing.png" if i < 48 and i % 16 == 1 else f"{subject}.png"
        row = {"subject_id": subject, "image_id": "a", "relative_path": f"{subject}.png",
               "mask_path": mask_path, "mask_empty": "TRUE" if empty else "FALSE"}
        rows.append(row)
        if i < 48 and i % 16 == 2:
            second = np.zeros((128, 128), dtype="uint8")
            second[10:20, 10:20] = 255
            Image.fromarray(second).save(masks / f"{subject}-second.png")
            rows.append(dict(row, mask_path=f"{subject}-second.png"))
            rows.append(dict(row, image_id="z", mask_path="missing.png"))
    with (root / "samples.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {"dataset": "synthetic", "public_subjects": 51, "source_rows": len(rows),
             "image_root": str(images), "mask_root": str(masks),
             "scoring_performed": False}
    split = {"seed": 20260919, "train": subjects[:48], "test": subjects[48:],
             "sites": [subjects[start:start + 16] for start in (0, 16, 32)]}
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (root / "split-20260919.json").write_text(json.dumps(split, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out)
