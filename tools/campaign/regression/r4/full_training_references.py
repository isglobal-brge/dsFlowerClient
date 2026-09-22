"""OLS and mean-only references on existing training rows, never sealed test rows."""

import argparse
import json
from pathlib import Path

from emulate import load_data, metrics
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/cells"))
    args = parser.parse_args()
    protocol = json.loads(
        (Path(__file__).resolve().parent.parent / "cdcbmi_public_units_protocol.json").read_text())
    result = {"purpose": "diagnostic references on original training rows only", "splits": []}
    for seed in protocol["seeds"]:
        _, train, _, _ = load_data(args.root, protocol, seed, full=True)
        z, y = train
        weights, _, rank, _ = np.linalg.lstsq(z.astype(np.float64), y, rcond=None)
        trivial = np.zeros(z.shape[1])
        trivial[-1] = y.astype(np.float64).mean()
        result["splits"].append({
            "split_seed": seed, "n_training": len(y), "ols_rank": int(rank),
            "ols": metrics(z, y, weights), "trivial": metrics(z, y, trivial)})
    for name in ("ols", "trivial"):
        result[name + "_mean_rmse"] = float(np.mean([
            split[name]["rmse"] for split in result["splits"]]))
    output = args.root / "r4" / "full_training_references.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
