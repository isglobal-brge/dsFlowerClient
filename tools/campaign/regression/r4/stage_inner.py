"""Stage the fixed training-only inner split for the single real DP diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("/workspace/cells"))
args = p.parse_args()
source = args.root / "data_cache" / "cdcbmi_seed20260820"
output = args.root / "r4" / "inner_seed20260820"
output.mkdir(parents=True, exist_ok=True)
valid = []
audit = []
for site in range(1, 4):
    path = source / f"site{site}.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 12000
    order = np.random.default_rng(20260922 + site).permutation(len(frame))
    train_idx, valid_idx = order[2400:], order[:2400]
    frame.iloc[train_idx].to_csv(output / f"site{site}.csv", index=False)
    valid.append(frame.iloc[valid_idx])
    audit.append({"site": site, "n_train": len(train_idx), "n_validation": len(valid_idx),
                  "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "inner_training_positions_sha256": hashlib.sha256(train_idx.astype('<i8').tobytes()).hexdigest(),
                  "inner_validation_positions_sha256": hashlib.sha256(valid_idx.astype('<i8').tobytes()).hexdigest()})
pd.concat(valid).to_csv(output / "validation.csv", index=False)
(output / "staging.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit))
