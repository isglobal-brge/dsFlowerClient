"""Validate and summarize completed immutable regression evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    cells = []
    baselines = {}
    for path in sorted(args.directory.glob("pilot_*.json")):
        evidence = json.loads(path.read_text())
        assert evidence["schema"] == "dsflower-campaign-v2"
        assert evidence["versions"]["dsflower"] == evidence["versions"]["dsflowerclient"] == "0.5.0"
        assert len(evidence["versions"]["runner_sha256"]) == 64
        assert len(evidence["per_replicate"]) == 3
        for rep in evidence["per_replicate"]:
            split = rep["split"]
            groups = [set(x) for x in split["site_subjects"]]
            assert len(set.union(*groups)) == sum(map(len, groups)) == 34
            assert len(split["test_subjects"]) == 8
            assert not set.union(*groups) & set(split["test_subjects"])
            assert split["n_train"] + split["n_test"] == 5875
            assert sum(split["n_per_site"]) == split["n_train"]
            assert rep["history"] == dict(n_clients=3, n_failures=0, n_rounds=5, cleanup_ok=True)
            for node in rep["node_privacy"]:
                assert node["policy"]["dp_unit"] == "patient"
                assert node["policy"]["patient_column"] == "subject#"
                assert node["policy"]["per_training_epsilon"] == evidence["privacy"]["epsilon"]
                assert node["policy"]["per_training_delta"] == 1e-6
                assert node["clipping_norm"] == 1
                assert node["staged_configurations"]
            for branch in ("central", "federated_dp", "trivial"):
                assert all(math.isfinite(rep[branch][metric]) for metric in ("rmse", "mae", "r2"))
            assert math.isclose(rep["gap_rmse"], rep["federated_dp"]["rmse"] - rep["central"]["rmse"])
            baseline = {key: rep[key] for key in ("central", "trivial", "split")}
            assert baselines.setdefault(rep["seed"], baseline) == baseline
        for branch, key in (("central", "central_mean_sd"), ("federated_dp", "federated_mean_sd"), ("trivial", "trivial_mean_sd")):
            for metric in ("rmse", "mae", "r2"):
                values = [rep[branch][metric] for rep in evidence["per_replicate"]]
                assert math.isclose(statistics.mean(values), evidence["summary"][key][metric]["mean"])
                assert math.isclose(statistics.stdev(values), evidence["summary"][key][metric]["sd"])
        cells.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          contract=evidence["contract"], epsilon=evidence["privacy"]["epsilon"],
                          n=5875, n_subjects=42, summary=evidence["summary"]))
    assert sorted(cell["epsilon"] for cell in cells) == [1, 4, 8]
    acceptance = next(cell for cell in cells if cell["epsilon"] == 8)["summary"]["diagnostic_pass"]
    summary = dict(schema="dsflower-campaign-v2", track="regression", cells=cells,
                   acceptance_epsilon=8, diagnostic_pass=acceptance,
                   interpretation="Representative subject-private regression measurement; no tuning or test-driven rerun.",
                   alternative="Not run: ridge is reserved for failure to execute the linear contract.")
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
