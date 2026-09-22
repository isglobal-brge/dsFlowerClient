"""Fit finite-schedule and OLS references before guarded test scoring."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from emulate import load_data, run_one
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from central_public_units import fit_rows, score_rows, sha256
from central_train import transform, metrics

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["fit", "score"])
p.add_argument("root", type=Path)
p.add_argument("seed", type=int)
p.add_argument("work", type=Path)
p.add_argument("selection", type=Path)
a = p.parse_args()
protocol_path = Path(__file__).resolve().parent.parent / "cdcbmi_public_units_protocol.json"
protocol = json.loads(protocol_path.read_text())
selected = json.loads(a.selection.read_text())["selected_params"]
prepared = a.root / "data_cache" / f"cdcbmi_seed{a.seed}"
frozen = a.work / "comparators_frozen.json"
if a.mode == "fit":
    assert not frozen.exists()
    start = time.monotonic()
    train = pd.read_csv(prepared / "train.csv")
    coefficients, ols, mean = fit_rows(train, protocol)
    sites, training, _, _ = load_data(a.root, protocol, a.seed, full=True)
    config = dict(optimizer=selected["optimizer"], lr=selected["learning_rate"],
                  epochs=selected["local_epochs"], batch=selected["batch_size"])
    result = run_one((sites, training, training, config, a.seed, 0))
    frozen.write_text(json.dumps(dict(coefficients=coefficients.tolist(), ols=ols,
        training_mean=mean, nonprivate=result, config=config,
        initialization_seed=a.seed, elapsed_s=time.monotonic()-start,
        train_sha256=sha256(prepared / "train.csv")), indent=2)+"\n")
else:
    output = a.work / "baselines.json"
    assert not output.exists()
    f = json.loads(frozen.read_text())
    test = pd.read_csv(prepared / "test.csv")
    result = score_rows(test, protocol, np.array(f["coefficients"]), f["ols"], f["training_mean"])
    z = np.column_stack([transform(test, protocol), np.ones(len(test), dtype=np.float32)])
    result["nonprivate_federated"] = metrics(test.BMI.to_numpy(), 55+43*(z @ np.array(f["nonprivate"]["weights"], dtype=np.float32)))
    result["nonprivate_fit"] = f
    result["provenance"] = dict(train_sha256=f["train_sha256"], test_sha256=sha256(prepared / "test.csv"),
        frozen_sha256=sha256(frozen), fit_completed_before_test_load=True,
        script_sha256=sha256(__file__))
    output.write_text(json.dumps(result, indent=2)+"\n")
