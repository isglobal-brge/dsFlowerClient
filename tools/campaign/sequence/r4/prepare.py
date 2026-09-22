#!/usr/bin/env python3
"""Prepare the frozen R3 subject-disjoint inner split; never read official TEST."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_public_data import load_split


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(root):
    work = root / "r4"
    destination = work / "prepared"
    destination.mkdir(parents=True, exist_ok=False)
    source = root / "prepared"
    audit = json.loads((source / "audit.json").read_text())
    data = np.load(source / "train.npz", allow_pickle=False)
    x, y, subjects = data["X"], data["y"], data["subjects"]
    assert x.shape == (7352, 1152) and x.dtype == np.float32
    archive = root / "data_cache/uci-har-240.zip"
    assert digest(archive) == audit["archive_sha256"]
    raw, labels, ids = load_split(archive, "train")
    assert np.array_equal(x, raw.reshape(7352, 1152, order="C"))
    assert np.array_equal(y, labels) and np.array_equal(subjects, ids)
    val_subjects = [1, 8, 17, 25, 30]
    assert np.unique(subjects)[::5].tolist() == val_subjects
    val = np.isin(subjects, val_subjects)
    fit = ~val
    assert int(fit.sum()) == 5564 and int(val.sum()) == 1788
    sites = [[s for s in site if s not in val_subjects]
             for site in audit["site_subjects"]]
    assert sorted(sum(sites, [])) == np.unique(subjects[fit]).tolist()
    assert len(set(sum(sites, []))) == 16
    for name, selection in (("train", fit), ("validation", val)):
        np.savez(destination / f"{name}.npz", X=x[selection], y=y[selection],
                 subjects=subjects[selection])
    frame = pd.DataFrame(x[fit], columns=audit["features"])
    frame["target"], frame["subject"] = y[fit], subjects[fit]
    frame.to_csv(destination / "train.csv", index=False)
    channel_bounds = np.asarray([1.] * 6 + [2.] * 3)
    bound = np.tile(channel_bounds, 128)
    bounds = {"lower": (-bound).tolist(), "upper": bound.tolist()}
    repository = Path(__file__).resolve().parents[4]
    config_source = repository / "inst/extdata/campaign/sequence/har_window_pytorch_lstm_eps8.json"
    # Reuse the released contract's recorded R-built configuration only.
    config = json.loads(config_source.read_text())["per_replicate"][0]["effective_config"]
    spec = json.loads(base64.b64decode(config["model-spec-b64"]))
    assert spec["nodes"] == [
        {"name": "x", "op": "reshape", "in": ["@in"], "shape": [128, 9]},
        {"name": "h", "op": "lstm", "in": ["x"], "hidden": 32},
        {"name": "out", "op": "linear", "in": ["h"], "out": "@out"}]
    recorded_bounds = json.loads(base64.b64decode(config["feature-bounds-b64"]))
    assert {key: recorded_bounds[key] for key in bounds} == bounds
    assert recorded_bounds["features"] == audit["features"]
    required = {"batch-size": 256, "learning-rate": .01, "local-epochs": 4,
                "optimizer-name": "adam", "num-server-rounds": 5,
                "num-features": 1152, "num-classes": 6,
                "loss-name": "cross_entropy", "strategy": "fedavg"}
    assert all(config[key] == value for key, value in required.items())
    config["results-dir"] = str(work / "emulation")
    save(work / "config.json", config)
    old_sites = audit["site_subjects"]
    audit.update({
        "diagnosis": "R4 training-only subject-disjoint inner split",
        "source_train_npz_sha256": digest(source / "train.npz"),
        "original_site_subjects": old_sites,
        "site_subjects": sites,
        "n_per_site": [int(np.isin(subjects[fit], site).sum()) for site in sites],
        "n_privacy_units_per_site": [int(np.isin(subjects[fit], site).sum()) for site in sites],
        "site_class_counts": [np.bincount(y[np.isin(subjects, site)], minlength=6).tolist()
                              for site in sites],
        "train_subjects": np.unique(subjects[fit]).tolist(),
        "validation_subjects": val_subjects,
        "n_train_windows": int(fit.sum()), "n_train_subjects": 16,
        "n_validation_windows": int(val.sum()), "n_validation_subjects": 5,
        "n_inner_train": int(fit.sum()), "n_validation": int(val.sum()),
        "train_class_counts": np.bincount(y[fit], minlength=6).tolist(),
        "validation_class_counts": np.bincount(y[val], minlength=6).tolist(),
        "bounds": bounds,
        "channel_bounds": {"lower": (-channel_bounds).tolist(), "upper": channel_bounds.tolist()},
        "channel_abs_bounds": channel_bounds.tolist(),
        "bounds_source": "Fixed public R3 design constants, no empirical extrema or moments",
        "train_clipped_fraction_per_channel": (np.abs(raw[fit]) > channel_bounds).mean((0, 1)).tolist(),
        "layout": "C-order flattened token-major [N,128,9]; raw windows, bounds applied by contract once",
        "archive_train_alignment_exact": True,
        "train_npz_sha256": digest(destination / "train.npz"),
        "validation_npz_sha256": digest(destination / "validation.npz"),
        "train_csv_sha256": digest(destination / "train.csv"),
        "config_source": str(config_source.relative_to(repository)),
        "config_source_sha256": digest(config_source),
        "config_sha256": digest(work / "config.json"),
        "test_accessed": False})
    save(destination / "audit.json", audit)
    print(json.dumps({key: audit[key] for key in (
        "site_subjects", "n_per_site", "n_inner_train", "n_validation", "test_accessed")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
