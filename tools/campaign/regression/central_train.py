"""Noiseless OLS twin of public preprocessing and patient mean pooling."""
import argparse
import json

import numpy as np
import pandas as pd


def metrics(y, prediction):
    residual = np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float)
    return dict(rmse=float(np.sqrt(np.mean(residual**2))),
                mae=float(np.mean(np.abs(residual))),
                r2=float(1 - np.sum(residual**2) / np.sum((y - np.mean(y))**2)))


def transform(frame, protocol):
    lower = np.asarray(protocol["feature_bounds"]["lower"], dtype=float)
    upper = np.asarray(protocol["feature_bounds"]["upper"], dtype=float)
    # The canonical loader uses float32 inputs before its bounded transform.
    x = frame[protocol["features"]].to_numpy(dtype=np.float32).astype(float)
    return ((np.clip(x, lower, upper) - (lower + upper) / 2) /
            ((upper - lower) / 2)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    for name in ("train", "test", "protocol", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    with open(args.protocol) as source:
        protocol = json.load(source)
    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    x = transform(train, protocol)
    target = protocol["target"]
    y = np.clip(train[target].to_numpy(dtype=np.float32),
                protocol["target_bounds"]["lower"], protocol["target_bounds"]["upper"])
    ids = train[protocol["patient_column"]].to_numpy()
    subjects = pd.unique(ids)
    pooled_x = np.array([x[ids == subject].astype(float).mean(axis=0)
                         for subject in subjects], dtype=np.float32)
    pooled_y = np.array([y[ids == subject].astype(float).mean()
                         for subject in subjects], dtype=np.float32)
    design = np.column_stack([np.ones(len(subjects)), pooled_x]).astype(float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, pooled_y, rcond=None)
    test_design = np.column_stack([np.ones(len(test)), transform(test, protocol)]).astype(float)
    prediction = test_design @ coefficients
    truth = test[target].to_numpy(dtype=float)
    train_mean = float(train[target].mean())
    result = dict(central=metrics(truth, prediction),
                  trivial=metrics(truth, np.full(len(test), train_mean)),
                  training_mean=train_mean,
                  central_fit=dict(method="OLS", intercept=True, n_subjects=len(subjects),
                                   rank=int(rank), singular_values=singular_values.tolist(),
                                   coefficients=coefficients.tolist(),
                                   numpy_version=np.__version__, pandas_version=pd.__version__))
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
