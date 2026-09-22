#!/usr/bin/env python3
"""Download official HAR archive and prepare TRAIN ONLY; test stays in the zip."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
CHANNELS = [f"{kind}_{axis}" for kind in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"]


def dataset_zip(path):
    outer = zipfile.ZipFile(path)
    nested = [n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")]
    return zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer


def load_split(archive, split):
    with dataset_zip(archive) as z:
        prefix = next(n[:-len(f"{split}/y_{split}.txt")] for n in z.namelist()
                      if n.endswith(f"{split}/y_{split}.txt"))
        def read(name, dtype=float):
            return np.loadtxt(io.BytesIO(z.read(prefix + name)), dtype=dtype)
        x = np.stack([read(f"{split}/Inertial Signals/{c}_{split}.txt") for c in CHANNELS], axis=-1)
        y = read(f"{split}/y_{split}.txt", int) - 1
        subjects = read(f"{split}/subject_{split}.txt", int)
    assert x.shape == (len(y), 128, 9) and np.isfinite(x).all()
    assert set(np.unique(y)) == set(range(6))
    return x.astype(np.float32), y, subjects


def main(root):
    cache = root / "data_cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "uci-har-240.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (cache / "CHECKSUMS.sha256").write_text(f"{sha}  {archive.name}\n")
    x, y, subjects = load_split(archive, "train")
    ids = sorted(map(int, np.unique(subjects)))
    assert len(ids) == 21 and len(y) == 7352
    sites = [ids[s::3] for s in range(3)]
    lower, upper = x.min(axis=(0, 1)), x.max(axis=(0, 1))
    pad = (upper - lower) * 0.1
    lower, upper = lower - pad, upper + pad
    features = [f"t{t:03d}_{c}" for t in range(128) for c in CHANNELS]
    frame = pd.DataFrame(x.reshape(len(y), -1), columns=features)
    frame["target"], frame["subject"] = y, subjects
    prepared = root / "prepared"
    prepared.mkdir(exist_ok=True)
    frame.to_csv(prepared / "train.csv", index=False)
    np.savez(prepared / "train.npz", X=x.reshape(len(y), -1), y=y, subjects=subjects)
    audit = {"name": "UCI Human Activity Recognition Using Smartphones", "uci_id": 240,
             "source_url": URL, "landing_url": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
             "archive_sha256": sha, "citation": "Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN.",
             "dataset_doi": "10.24432/C54S4K", "licence": "CC BY 4.0 (UCI repository)",
             "n_train_windows": len(y), "n_train_subjects": len(ids), "site_subjects": sites,
             "n_per_site": [int(np.isin(subjects, site).sum()) for site in sites],
             "n_privacy_units_per_site": [7, 7, 7], "train_subjects": ids,
             "features": features, "channels": CHANNELS,
             "shape": [128, 9], "bounds": {"lower": np.tile(lower, 128).tolist(), "upper": np.tile(upper, 128).tolist()},
             "channel_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
             "train_class_counts": np.bincount(y, minlength=6).tolist(),
             "train_npz_sha256": hashlib.sha256((prepared / "train.npz").read_bytes()).hexdigest(),
             "test_accessed": False}
    (prepared / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("archive_sha256", "n_train_windows", "site_subjects", "n_per_site", "test_accessed")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    main(p.parse_args().root)
