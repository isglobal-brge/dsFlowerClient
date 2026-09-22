"""Select once by mean real federated-DP inner-validation RMSE."""
import datetime
import json
from pathlib import Path
import statistics
import sys
root = Path(sys.argv[1])
output = Path(sys.argv[2])
assert not output.exists()
assert not list((root.parent / "data_cache").glob("cdcbmi_seed*/test.csv"))
rows = []
for name in ["sgd_lr003_e5_b64", "sgd_lr001_e5_b32"]:
    replicates = [json.loads((root / f"selection_{name}_{seed}" / "selection_replicate.json").read_text())
                  for seed in [20260820, 20260821, 20260822]]
    values = [r["federated"]["metrics"]["validation"]["rmse"] for r in replicates]
    assert all(not r["sealed_test_opened"] for r in replicates)
    rows.append(dict(candidate=name, model_params=replicates[0]["model_params"],
                     rmse_mean=statistics.mean(values), rmse_sd=statistics.stdev(values),
                     validation_rmse=values, replicates=replicates))
selected = min(rows, key=lambda r: r["rmse_mean"])
result = dict(schema="dsflower-regression-r4-selection-v1",
    selected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    criterion="Smallest mean real federated-DP inner-validation RMSE across the three original training partitions; no test rows used",
    selected=selected["candidate"], selected_params=selected["model_params"],
    epsilon=8, delta=1e-6, rounds=5, clipping_norm=1, privacy_unit="row",
    n_inner_training=28800, n_inner_validation=7200, n_sites=3,
    split_method="Reuse diagnosis: independently permute each original 12000-row site with NumPy PCG64 seed 20260922+site (1-based); prefix 2400 validation, suffix 9600 training. Site allocation inherited from original stratified cohort; inner holdout is random within each site, not exactly restratified.",
    hash_encoding="Position and source-row arrays in order, little-endian signed int64 bytes, SHA-256; source rows one-based, positions zero-based",
    candidates=rows, sealed_test_opened=False,
    limitations="Overlapping outer training partitions; three unpaired random initializations, descriptive SD only. Privacy accounting is per fit, not a composed selection release.")
output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({r["candidate"]: r["validation_rmse"] for r in rows}))
print("SELECTED",result["selected"])
