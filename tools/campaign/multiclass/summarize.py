#!/usr/bin/env python3
"""Validate completed evidence and summarize without retraining or rescoring."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("evidence", type=Path)
args = ap.parse_args()
rows = []
paired = None
tooling = None
for epsilon in (1, 4, 8):
    path = args.evidence / f"pilot_ctg_pytorch_multiclass_eps{epsilon}.json"
    cell = json.loads(path.read_text())
    assert cell["schema"] == "dsflower-campaign-v2"
    assert cell["contract"] == "pytorch_multiclass"
    assert cell["versions"]["dsflower"] == cell["versions"]["dsflowerclient"] == "0.5.0"
    assert cell["dataset"]["n_total"] == 2126
    assert cell["privacy"]["epsilon"] == epsilon
    if tooling is None:
        tooling = cell["tooling_sha256"]
    else:
        assert tooling == cell["tooling_sha256"], "Scoring or protocol changed across cells"
    for name, expected_hash in cell["tooling_sha256"].items():
        source = Path(__file__).resolve().parent / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
    reps = cell["per_replicate"]
    assert [r["seed"] for r in reps] == [20260820, 20260821, 20260822]
    current = [(r["split"], r["central"], r["trivial"], r["feature_bounds"]) for r in reps]
    if paired is None:
        paired = current
    else:
        assert paired == current, "Split, bounds or baselines changed across epsilon"
    for rep in reps:
        assert rep["history"]["n_clients"] == 3
        assert rep["history"]["n_rounds"] == 5
        assert rep["history"]["n_failures"] == 0
        assert rep["cleanup_ok"] and rep["central"]["convergence"] == 0
        metadata = rep["released_metadata"]
        assert metadata["model_params"] == cell["model_params_resolved"]
        assert metadata["target_levels"] == ["1", "2", "3"]
        assert metadata["available_rounds"] == [1, 2, 3, 4, 5]
        assert metadata["privacy"] == "server-enforced-dp"
        assert all(n > 0 for site in rep["split"]["class_counts_sites"] for n in site.values())
        assert sum(rep["split"]["n_per_site"]) == rep["split"]["n_train"]
        assert rep["split"]["n_train"] + rep["split"]["n_test"] == 2126
        for node in rep["node_privacy"].values():
            policy = node["reported_policy"]
            assert policy["per_training_epsilon"] == epsilon
            assert policy["per_training_delta"] == 1e-6
            assert policy["dp_unit"] == "row" and policy["adjacency"] == "replace_one"
            assert node["clipping_norm"] == 1
            assert node["runner_sha256"] == cell["versions"]["runner_sha256"]
        assert math.isclose(rep["gap_macro_auc"], rep["federated_dp"]["macro_auc"] - rep["central"]["macro_auc"], abs_tol=1e-14)
    for branch, key in (("central", "central_mean_sd"), ("federated_dp", "federated_mean_sd"), ("trivial", "trivial_mean_sd")):
        for metric in ("macro_auc", "acc", "logloss"):
            values = [r[branch][metric] for r in reps]
            assert all(math.isfinite(v) for v in values)
            expected = cell["summary"][key][metric]
            assert math.isclose(expected["mean"], statistics.mean(values), abs_tol=1e-14)
            assert math.isclose(expected["sd"], statistics.stdev(values), abs_tol=1e-14)
    gaps = [r["gap_macro_auc"] for r in reps]
    gap = cell["summary"]["gap_mean_sd"]["macro_auc"]
    assert math.isclose(gap["mean"], statistics.mean(gaps), abs_tol=1e-14)
    assert math.isclose(gap["sd"], statistics.stdev(gaps), abs_tol=1e-14)
    rows.append(dict(epsilon=epsilon, evidence_file=path.name,
                     evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     summary=cell["summary"],
                     wall_clock_total_s=sum(r["wall_clock"]["total_s"] for r in reps)))
acceptance = dict(rows[-1]["summary"]["diagnostic"])
acceptance["passed"] = (acceptance["mean_accuracy_above_majority"] and
                        acceptance["mean_macro_auc_above_half"])
summary = dict(schema="dsflower-campaign-v2", track="multiclass", contract="pytorch_multiclass",
               dataset=cell["dataset"], host=cell["host"], versions=cell["versions"],
               protocol=cell["protocol"], tooling_sha256=tooling, cells=rows,
               acceptance_epsilon8=acceptance,
               validation="All three cells: paired splits/bounds/baselines; 3 sites, 5 rounds, zero failures; node policies and runner hashes; means and sample SD independently recomputed.")
(args.evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for row in rows:
    s = row["summary"]
    print(f"epsilon={row['epsilon']} central={s['central_mean_sd']['macro_auc']['mean']:.6f} "
          f"fed={s['federated_mean_sd']['macro_auc']['mean']:.6f} "
          f"accuracy={s['federated_mean_sd']['acc']['mean']:.6f} "
          f"gap={s['gap_mean_sd']['macro_auc']['mean']:.6f} +/- {s['gap_mean_sd']['macro_auc']['sd']:.6f}")
print("Evidence validation passed; wrote summary.json")
