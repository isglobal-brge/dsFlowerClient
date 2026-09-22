"""Summarize retained training-only diagnostics without accessing any dataset."""
import hashlib
import json
from pathlib import Path
import numpy as np

folder = Path(__file__).resolve().parent / "results"
read = lambda name: json.loads((folder / name).read_text())
baseline, candidates, controls = (read(name) for name in
    ["baseline.json", "candidates.json", "controls.json"])


def stats(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)),
                minimum=float(x.min()), maximum=float(x.max()))


def rmse(replicates, key="validation"):
    return [r[key]["rmse"] for r in replicates]


summary = {"description": "Descriptive results; overlapping splits and paired initialization seeds are not independent replicates.",
           "inner_ols": stats([s["ols"]["validation"]["rmse"] for s in baseline["splits"]]),
           "inner_trivial": stats([s["trivial"]["validation"]["rmse"] for s in baseline["splits"]])}
for source, name in [("runs", "inner_default"), ("full_training_only", "original_geometry_training")]:
    pairs = [(s[source][0]["replicates"], s[source][1]["replicates"]) for s in baseline["splits"]]
    for a, b in pairs:
        assert [r["seed"] for r in a] == [r["seed"] for r in b]
    plain = [r for a, _ in pairs for r in a]
    noisy = [r for _, b in pairs for r in b]
    summary[name] = {"no_noise": stats(rmse(plain)), "epsilon8_noise": stats(rmse(noisy)),
                     "paired_rmse_change": stats(np.array(rmse(noisy)) - rmse(plain)),
                     "paired_final_weight_rms_difference": float(np.sqrt(np.mean(
                         (np.array([r["weights"] for r in noisy]) -
                          np.array([r["weights"] for r in plain])) ** 2)))}
ranked = []
for index, entry in enumerate(candidates["splits"][0]["runs"]):
    rows = [s["runs"][index] for s in candidates["splits"]]
    assert all(row["config"] == entry["config"] and row["sigma"] == 0 for row in rows)
    runs = [r for row in rows for r in row["replicates"]]
    ranked.append({"config": entry["config"], "validation": stats(rmse(runs)),
                   "training": stats(rmse(runs, "training")),
                   "split_means": [float(np.mean(rmse(row["replicates"]))) for row in rows],
                   "epsilon8_inner_noise_multiplier": entry["mechanisms"]["8"]["noise_multiplier"]})
summary["ranked_candidates"] = sorted(ranked, key=lambda row: row["validation"]["mean"])
summary["controls_first_split"] = [{"config": row["config"], "validation": stats(rmse(row["replicates"]))}
                                   for row in controls["splits"][0]["runs"]]
for result in [baseline, candidates, controls]:
    assert all(row["max_parameter_difference"] < 1e-6 for row in result["parity"])
    for split in result["splits"]:
        assert split["n_train"] == 28800 and split["n_valid"] == 7200
        for entry in split["runs"]:
            assert len(entry["replicates"]) == 5
            assert all(len(r["trajectory"]) == 5 for r in entry["replicates"])
diagnostic = read("diagnostic.json")
summary["real_federated_default"] = {"baselines": diagnostic["baselines"],
    "metrics": diagnostic["federated"]["metrics"], "model_sha256": diagnostic["federated"]["model_sha256"],
    "prediction_verification": read("prediction_verification.json")}
assert diagnostic["federated"]["n_clients"] == 3
assert diagnostic["federated"]["n_rounds_run"] == 5
assert diagnostic["federated"]["n_failures"] == 0
assert diagnostic["federated"]["cleanup_ok"]
for key in ["training", "validation"]:
    assert summary["real_federated_default"]["prediction_verification"][key]["prediction_path_matches"]
summary["input_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(folder.glob("*.json")) if path.name != "summary.json"}
(folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
