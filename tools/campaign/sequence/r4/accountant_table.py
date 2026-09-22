#!/usr/bin/env python3
"""Rank TRAIN-only noiseless candidates and their calibrated gradient noise."""
import argparse
import hashlib
import json
import math
from pathlib import Path


PARAMETERS = 5702  # DPLSTM: 4*32*(9+32+2), plus the 32-to-6 linear head.
FULL_POPULATIONS = [2553, 2397, 2402]


def noise_summary(mechanisms):
    sites = []
    for mechanism in mechanisms:
        assert mechanism["clipping_norm"] == 1
        sigma = mechanism["noise_multiplier"]
        divisor = mechanism["expected_batch_size"]
        mean_sd = sigma / divisor
        sites.append({
            "population": mechanism["accounting_population"],
            "sample_rate": mechanism["sample_rate"],
            "steps": mechanism["total_steps"],
            "noise_multiplier": sigma,
            "expected_batch_size": divisor,
            "mean_gradient_coordinate_noise_sd": mean_sd,
            "mean_gradient_noise_l2_rms": math.sqrt(PARAMETERS) * mean_sd,
        })
    return {
        "sites": sites,
        "max_site_coordinate_noise_sd": max(s["mean_gradient_coordinate_noise_sd"] for s in sites),
        "max_site_noise_l2_rms": max(s["mean_gradient_noise_l2_rms"] for s in sites),
    }


def main(root, skip_accountant):
    work = root / "r4"
    source = work / "emulation.json"
    emulation = json.loads(source.read_text())
    assert emulation["test_accessed"] is False
    candidates = []
    for name, record in emulation["candidates"].items():
        assert record["clipped"] and record["noise_multiplier_executed"] == 0
        assert record["history"][-1]["round"] == 5
        candidates.append({
            "name": name, "schedule": record["schedule"],
            "macro_auc": record["final"]["macro_auc"],
            "accuracy": record["final"]["accuracy"],
            "inner": noise_summary(record["noise_calibration_at_epsilon8"]),
            "full": noise_summary(record["full_training_noise_calibration_at_epsilon8"]),
        })
    planned = json.loads((work / "candidate-schedules.json").read_text())
    assert {row["name"] for row in candidates} == {row["name"] for row in planned}
    assert len(candidates) == 8
    for row in candidates:
        row["rank_noiseless_auc"] = 1 + sum(other["macro_auc"] > row["macro_auc"] for other in candidates)
        for scope in ("inner", "full"):
            value = row[scope]["max_site_coordinate_noise_sd"]
            row[f"rank_{scope}_noise"] = 1 + sum(
                other[scope]["max_site_coordinate_noise_sd"] < value for other in candidates)
    candidates.sort(key=lambda row: (-row["macro_auc"], row["name"]))
    same_q = []
    if not skip_accountant:
        from dsflower_runner import dp_harness
        for epochs in (1, 2, 4, 8, 12):
            mechanisms = [dp_harness.effective_dpsgd_mechanism(
                8., 1e-6, 1., n, 256, epochs, 5) for n in FULL_POPULATIONS]
            assert all(m["sample_rate"] == .1 for m in mechanisms)
            same_q.append({"local_epochs": epochs, "batch_size": 256,
                           "rounds": 5, "noise": noise_summary(mechanisms),
                           "mechanisms": mechanisms})
    result = {
        "diagnostic_only": True, "test_accessed": False,
        "emulation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "parameter_count": PARAMETERS, "sqrt_parameter_count": math.sqrt(PARAMETERS),
        "epsilon": 8, "delta": 1e-6, "clipping_norm": 1, "rounds": 5,
        "interpretation": (
            "Noiseless scores use public deterministic diagnostic streams and have no DP guarantee. "
            "Noise ranks use the worst site's per-step mean-gradient coordinate SD, sigma/expected_batch_size. "
            "sqrt(P)*sigma/expected_batch_size is the Gaussian vector's root-mean-square L2 norm before "
            "Adam/SGD transformation, not a predicted parameter displacement or validation AUC. "
            "Lower gradient noise alone is not the optimization objective; the objective is epsilon-8 "
            "federated-DP inner-validation macro-AUC. Equal noise values share a competition rank."),
        "candidates": candidates,
        "order_by_noiseless_auc": [row["name"] for row in candidates],
        "order_by_inner_noise": [row["name"] for row in sorted(candidates,
            key=lambda row: (row["inner"]["max_site_coordinate_noise_sd"], -row["macro_auc"], row["name"]))],
        "order_by_full_noise": [row["name"] for row in sorted(candidates,
            key=lambda row: (row["full"]["max_site_coordinate_noise_sd"], -row["macro_auc"], row["name"]))],
        "same_q_full_training_accountant": same_q,
    }
    (work / "candidate-ranking.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("AUC-rank candidate                 AUC      inner-sd  full-sd   inner-L2 full-L2 noise-ranks(inner/full)")
    for row in candidates:
        inner, full = row["inner"], row["full"]
        print(f'{row["rank_noiseless_auc"]:8d} {row["name"]:25s} {row["macro_auc"]:.6f} '
              f'{inner["max_site_coordinate_noise_sd"]:.6f}  {full["max_site_coordinate_noise_sd"]:.6f}  '
              f'{inner["max_site_noise_l2_rms"]:.4f}   {full["max_site_noise_l2_rms"]:.4f} '
              f'{row["rank_inner_noise"]}/{row["rank_full_noise"]}')
    if same_q:
        print("Full TRAIN, q=0.1, batch=256, rounds=5: epochs steps sigma max-site-gradient-sd")
        for row in same_q:
            site = row["noise"]["sites"][0]
            print(f'{row["local_epochs"]:6d} {site["steps"]:5d} {site["noise_multiplier"]:.6f} '
                  f'{row["noise"]["max_site_coordinate_noise_sd"]:.6f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skip-accountant", action="store_true",
                        help="Only rank cached candidate mechanisms; do not compute the same-q table")
    args = parser.parse_args()
    main(args.root, args.skip_accountant)
