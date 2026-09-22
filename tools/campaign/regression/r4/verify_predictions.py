#!/usr/bin/env python3
"""Independently score the one released diagnostic linear model; no fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    work = args.root / "r4" / "federated_default"
    inputs = args.root / "r4" / "inner_seed20260820"
    diagnostic = json.loads((work / "diagnostic.json").read_text())
    artifact = Path(diagnostic["federated"]["artifact_dir"])
    protocol = json.loads(
        (Path(__file__).resolve().parent.parent / "cdcbmi_public_units_protocol.json").read_text()
    )
    state = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    assert len(state) == 2, "Expected only one linear weight and bias"
    weights = [value for value in state.values() if tuple(value.shape) == (1, 20)]
    biases = [value for value in state.values() if tuple(value.shape) == (1,)]
    assert len(weights) == len(biases) == 1
    weight, bias = weights[0].numpy(), biases[0].numpy()
    frames = {
        "training": pd.concat(
            [pd.read_csv(inputs / f"site{site}.csv") for site in range(1, 4)],
            ignore_index=True,
        ),
        "validation": pd.read_csv(inputs / "validation.csv"),
    }
    lower = np.asarray(protocol["feature_bounds"]["lower"], dtype=np.float64)
    upper = np.asarray(protocol["feature_bounds"]["upper"], dtype=np.float64)
    result = {"method": "manual public feature transform and released linear coefficients"}
    for name, frame in frames.items():
        values = frame[protocol["features"]].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all()
        transformed = (
            (np.clip(values, lower, upper) - (lower + upper) / 2) / ((upper - lower) / 2)
        ).astype(np.float32)
        public = (transformed @ weight.T + bias).reshape(-1)
        bmi = 55 + 43 * public.astype(np.float64)
        exported = pd.read_csv(work / f"{name}_predictions.csv")
        assert len(exported) == len(frame)
        assert np.array_equal(exported["BMI"].to_numpy(), frame["BMI"].to_numpy())
        public_error = float(np.max(np.abs(public - exported["public_prediction"])))
        bmi_error = float(np.max(np.abs(bmi - exported["BMI_prediction"])))
        residual = bmi - frame["BMI"].to_numpy()
        result[name] = {
            "n": len(frame),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mean_residual": float(np.mean(residual)),
            "max_public_prediction_error": public_error,
            "max_BMI_prediction_error": bmi_error,
            "prediction_path_matches": public_error < 2e-6 and bmi_error < 1e-4,
        }
        assert result[name]["prediction_path_matches"], "Released prediction paths differ"
    (work / "prediction_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
