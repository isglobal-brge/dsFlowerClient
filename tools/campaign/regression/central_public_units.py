"""Public-unit OLS on training rows; BMI scoring only after the fit is frozen."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from central_train import metrics, transform


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_rows(train, protocol):
    x = transform(train, protocol)
    bounds = protocol["original_target_bounds"]
    y = np.clip(train[protocol["target"]].to_numpy(dtype=float),
                bounds["lower"], bounds["upper"])
    target_transform = protocol["public_target_transform"]
    y = ((y - target_transform["center"]) / target_transform["scale"]).astype(np.float32)
    design = np.column_stack([np.ones(len(train)), x]).astype(float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
    fit = dict(method="OLS", intercept=True, observation_unit="row",
               n_training_rows=len(train), n_features=x.shape[1], rank=int(rank),
               singular_values=singular_values.tolist(),
               coefficients=coefficients.tolist(), numpy_version=np.__version__,
               pandas_version=pd.__version__, target_transform=target_transform)
    patient_column = protocol.get("patient_column")
    if patient_column:
        fit["n_subjects"] = int(train[patient_column].nunique())
    return coefficients, fit, float(train[protocol["target"]].mean())


def score_rows(test, protocol, coefficients, fit, training_mean):
    design = np.column_stack([np.ones(len(test)), transform(test, protocol)]).astype(float)
    truth = test[protocol["target"]].to_numpy(dtype=float)
    target_transform = protocol["public_target_transform"]
    prediction = target_transform["center"] + target_transform["scale"] * (design @ coefficients)
    return dict(central=metrics(truth, prediction),
                trivial=metrics(truth, np.full(len(test), training_mean)),
                training_mean=training_mean, central_fit=fit)


def main():
    parser = argparse.ArgumentParser()
    for name in ("train", "test", "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Baseline already exists; refusing to overwrite it.")
    protocol = json.loads(args.protocol.read_text())
    train = pd.read_csv(args.train)
    coefficients, fit, training_mean = fit_rows(train, protocol)
    # The held-out frame is first opened after all training calculations finish.
    test = pd.read_csv(args.test)
    result = score_rows(test, protocol, coefficients, fit, training_mean)
    result["provenance"] = dict(
        train_sha256=sha256(args.train), test_sha256=sha256(args.test),
        protocol_sha256=sha256(args.protocol), script_sha256=sha256(__file__),
        transform_source_sha256=sha256(Path(__file__).with_name("central_train.py")),
        fit_completed_before_test_load=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
