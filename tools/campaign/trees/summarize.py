#!/usr/bin/env python3
"""Validate completed cells and recompute frozen descriptive diagnostics."""
import hashlib
import json
from pathlib import Path
import statistics
import sys


root = Path(sys.argv[1])
paths = sorted(root.glob("pilot_*.json"))
cells = []
for path in paths:
    value = json.loads(path.read_text())
    assert value["schema"] == "dsflower-campaign-v2" and value["status"] == "completed"
    assert value["rounds"] == 1 and value["rounds_reason"] == "native-tree schedule"
    assert value["versions"]["dsflower"] == value["versions"]["dsflowerclient"] == "0.5.0"
    reps = value["per_replicate"]
    assert [rep["seed"] for rep in reps] == [20260820, 20260821, 20260822]
    for rep in reps:
        assert rep["history"]["n_clients"] == 3 and rep["history"]["n_failures"] == 0
        assert rep["history"]["n_rounds"] == 1 and rep["cleanup_ok"]
        assert sum(rep["n_per_site"]) == rep["n_train"]
        for node in rep["node_privacy"]:
            p = node["reported"]
            assert p["privacy_clipping_norm"] == 1 and p["privacy_unit"] == "row"
            assert p["privacy_per_training_epsilon"] == value["privacy"]["epsilon"]
            assert p["privacy_per_training_delta"] == 1e-6
            assert p["runner_sha256"] == value["versions"]["runner_sha256"]
    for branch, key in (("central", "central_mean_sd"), ("federated_dp", "federated_mean_sd"),
                        ("trivial", "trivial_mean_sd"), ("gap", "delta_mean_sd")):
        for metric in ("auc", "acc", "brier", "logloss"):
            numbers = [rep[branch][metric] for rep in reps]
            observed = value["summary"][key][metric]
            assert abs(statistics.mean(numbers) - observed["mean"]) < 1e-12
            assert abs(statistics.stdev(numbers) - observed["sd"]) < 1e-12
            if branch == "gap":
                assert all(abs(rep["gap"][metric] - (rep["federated_dp"][metric] -
                           rep["central"][metric])) < 1e-12 for rep in reps)
    s = value["summary"]
    floor = (s["federated_mean_sd"]["acc"]["mean"] >= s["trivial_mean_sd"]["acc"]["mean"] - .02
             or s["federated_mean_sd"]["auc"]["mean"] > .6)
    assert floor == value["diagnostic"]["utility_floor_pass"]
    red_conditions = (s["central_mean_sd"]["auc"]["mean"] < .95 and
                      abs(s["delta_mean_sd"]["auc"]["mean"]) < .005)
    cells.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "contract": value["contract"], "dataset": value["dataset"]["name"],
                  "n": value["dataset"]["n_total"], "epsilon": value["privacy"]["epsilon"],
                  "rounds": 1, "seeds": value["split"]["seeds"], "summary": s,
                  "utility_floor_pass": floor,
                  "near_zero_cost_flag": bool(red_conditions and value["dataset"]["n_total"] * value["privacy"]["epsilon"] < 2000),
                  "minimum_site_near_zero_cost_flag": bool(red_conditions and min(value["dataset"]["n_per_site"]) * value["privacy"]["epsilon"] < 2000),
                  "elapsed_s": value["elapsed_s"]})

series = []
for contract, dataset in sorted({(c["contract"], c["dataset"]) for c in cells}):
    group = sorted([c for c in cells if (c["contract"], c["dataset"]) == (contract, dataset)],
                   key=lambda c: c["epsilon"])
    comparisons = []
    for first, second in zip(group, group[1:]):
        a, b = first["summary"]["delta_mean_sd"]["auc"], second["summary"]["delta_mean_sd"]["auc"]
        comparisons.append({"epsilon_from": first["epsilon"], "epsilon_to": second["epsilon"],
                            "gap_change": b["mean"] - a["mean"], "allowed_worsening": max(a["sd"], b["sd"]),
                            "pass": b["mean"] - a["mean"] >= -max(a["sd"], b["sd"])})
    series.append({"contract": contract, "dataset": dataset, "epsilon_comparisons": comparisons,
                   "epsilon_monotonicity_pass": all(c["pass"] for c in comparisons) if comparisons else None,
                   "seed_dispersion_check": "not applicable: original diagnostic is specific to heart",
                   "gap_sd_lowest_epsilon": group[0]["summary"]["delta_mean_sd"]["auc"]["sd"],
                   "gap_sd_highest_epsilon": group[-1]["summary"]["delta_mean_sd"]["auc"]["sd"]})

# Comparator and split identities must be unchanged across epsilon.
for contract, dataset in {(c["contract"], c["dataset"]) for c in cells}:
    group = [json.loads((root / c["file"]).read_text()) for c in cells
             if (c["contract"], c["dataset"]) == (contract, dataset)]
    for value in group[1:]:
        for a, b in zip(group[0]["per_replicate"], value["per_replicate"]):
            for key in ("seed", "central", "trivial", "n_per_site", "split_row_ids_sha256",
                        "central_train_csv_sha256", "feature_bounds", "feature_cuts"):
                assert a[key] == b[key], (contract, dataset, key)

failures = [{"file": path.name, "evidence": json.loads(path.read_text())}
            for path in sorted(root.glob("failure_*.json"))]
result = {"schema": "dsflower-campaign-summary-v2", "track": "trees",
          "status": "blocked" if failures and not cells else "measured",
          "cells": cells,
          "series_diagnostics": series, "failures": failures,
          "utility_floor_epsilon8_pass": (all(c["utility_floor_pass"] for c in cells if c["epsilon"] == 8)
                                          if any(c["epsilon"] == 8 for c in cells) else None),
          "epsilon_monotonicity_pass": (all(s["epsilon_monotonicity_pass"] for s in series
                                            if s["epsilon_monotonicity_pass"] is not None)
                                         if any(s["epsilon_monotonicity_pass"] is not None for s in series) else None),
          "near_zero_cost_scan_pass": not any(c["near_zero_cost_flag"] for c in cells) if cells else None,
          "completed_cells": len(cells),
          "unmeasured_primary_cells": [{"contract": "random_forest", "dataset": dataset, "epsilon": epsilon}
              for dataset in ("breast", "cdc9k") for epsilon in (1, 4, 8)
              if not any(c["contract"] == "random_forest" and c["dataset"] == dataset
                         and c["epsilon"] == epsilon for c in cells)],
          "provisioning": {"reused": True, "initial_elapsed_s": 97, "reprovisioning_elapsed_s": 0},
          "interpretation": "Descriptive public benchmark evidence; gaps include algorithm, federation and DP costs. Three replicates do not establish general utility or prove privacy."}
(root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
for c in cells:
    s = c["summary"]
    print(f'{c["contract"]:14} {c["dataset"]:7} eps={c["epsilon"]} '
          f'central={s["central_mean_sd"]["auc"]["mean"]:.6f} '
          f'fed={s["federated_mean_sd"]["auc"]["mean"]:.6f} '
          f'gap={s["delta_mean_sd"]["auc"]["mean"]:+.6f}±{s["delta_mean_sd"]["auc"]["sd"]:.6f} '
          f'floor={"PASS" if c["utility_floor_pass"] else "FAIL"}')
