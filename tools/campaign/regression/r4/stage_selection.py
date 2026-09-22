"""Stage the fixed training-only inner split for the real DP schedule selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("/workspace/cells"))
p.add_argument("--seed", type=int, required=True)
args = p.parse_args()
source = args.root / "data_cache" / f"cdcbmi_seed{args.seed}"
output = args.root / "r4" / f"selection_inner_seed{args.seed}"
output.mkdir(parents=True, exist_ok=True)
split = json.loads((source / "split.json").read_text())
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
    source_rows = np.asarray(split["sites"][site-1]["source_rows"], dtype="<i8")
    assert not set(source_rows) & set(split["test"]["source_rows"])
    audit.append({"training_source_ids_sha256": hashlib.sha256(source_rows[train_idx].tobytes()).hexdigest(),
                  "validation_source_ids_sha256": hashlib.sha256(source_rows[valid_idx].tobytes()).hexdigest(),"site": site, "n_train": len(train_idx), "n_validation": len(valid_idx),
                  "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "inner_training_positions_sha256": hashlib.sha256(train_idx.astype('<i8').tobytes()).hexdigest(),
                  "inner_validation_positions_sha256": hashlib.sha256(valid_idx.astype('<i8').tobytes()).hexdigest()})
pd.concat(valid).to_csv(output / "validation.csv", index=False)
(output / "staging.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit))
