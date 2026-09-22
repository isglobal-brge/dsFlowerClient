#!/usr/bin/env python3
"""Pooled, noiseless forest with the native contract's public size and depth."""
import argparse
import csv
import json
import sys

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def read_frame(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    for key in ("train", "test", "schema", "out", "details", "contract", "runner"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    args = parser.parse_args()
    with open(args.schema) as handle:
        schema = json.load(handle)
    train, test = read_frame(args.train), read_frame(args.test)

    def features(frame):
        # Match the native float32 clipping and public cut geometry. No fitting.
        return np.column_stack([
            np.searchsorted(np.array(cuts, dtype=np.float32),
                            np.clip(frame[name].astype(np.float32), lo, hi),
                            side="left")
            for name, lo, hi, cuts in zip(schema["features"], schema["bounds"]["lower"],
                                         schema["bounds"]["upper"], schema["cuts"])
        ])

    params = schema["model_params"]
    forest_class = (RandomForestClassifier if args.contract == "random_forest"
                    else ExtraTreesClassifier)
    max_features = (int(np.ceil(np.sqrt(len(schema["features"]))))
                    if args.contract == "random_forest" else 1.0)
    model = forest_class(n_estimators=params["n_estimators"],
                         max_depth=params["max_depth"], max_features=max_features,
                         random_state=args.seed, n_jobs=1)
    model.fit(features(train), train["target"].astype(int))
    probabilities = model.predict_proba(features(test))[:, list(model.classes_).index(1)]
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prob"])
        writer.writerows((value,) for value in probabilities)
    # Public accountant recomputation, never a replacement training mechanism.
    sys.path.insert(0, args.runner)
    from dsflower_runner import forest_accounting
    if args.contract == "random_forest":
        accounting = forest_accounting.random_forest_release(
            "binary_classification", max_features, params["max_depth"], args.epsilon, 1e-6)
    else:
        accounting = forest_accounting.joint_leaf_release(
            "binary_classification", params["n_estimators"], args.epsilon, 1e-6)
    with open(args.details, "w") as handle:
        json.dump({"implementation": forest_class.__name__, "sklearn": sklearn.__version__,
                   "params": model.get_params(), "accounting_source":
                   "Recomputed from installed, unmodified dsFlower forest_accounting.py; public inputs only",
                   "native_accounting": accounting}, handle, indent=2)


if __name__ == "__main__":
    main()
