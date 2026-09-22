"""PRV calibration and reference noise scales; no model training."""

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import opacus
from opacus.accountants import PRVAccountant
from opacus.accountants.utils import get_noise_multiplier


@lru_cache(maxsize=None)
def mechanism(epsilon, batch, epochs):
    steps_per_epoch = math.ceil(12000 / batch)
    q = 1 / steps_per_epoch
    steps = 5 * epochs * steps_per_epoch
    epsilon0 = epsilon / 2
    delta0 = 1e-6 / (1 + math.exp(epsilon0))
    sigma = get_noise_multiplier(
        target_epsilon=epsilon0, target_delta=delta0, sample_rate=q,
        steps=steps, accountant="prv")
    return {"noise_multiplier": sigma, "sample_rate": q,
            "steps_per_epoch": steps_per_epoch, "total_steps_per_site": steps,
            "expected_batch_size": max(1, int(12000 / steps_per_epoch)),
            "calibration_add_remove_epsilon": epsilon0,
            "calibration_add_remove_delta": delta0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/cells"))
    args = parser.parse_args()
    result = {
        "purpose": "accountant-only regression diagnosis; no training",
        "accountant": "prv", "opacus_version": opacus.__version__,
        "geometry": {"n_per_site": 12000, "batch_size": 32,
                     "sample_rate": 1 / 375, "steps_per_site": 1875,
                     "sites": 3, "rounds": 5, "local_epochs": 1,
                     "learning_rate": 0.01, "clipping_norm": 1.0},
        "interpretation": (
            "Free random-walk scales ignore data-gradient contraction and are "
            "reference scales, not actual trained parameter error or RMSE predictions. "
            "Averaged-three-site scale assumes independent site noise and unchanged "
            "linear averaging."),
        "budgets": [], "candidate_original_geometry": []}
    for epsilon in (1, 4, 8):
        spec = mechanism(epsilon, 32, 1)
        sigma = spec["noise_multiplier"]
        accountant = PRVAccountant()
        accountant.history = [(sigma, spec["sample_rate"], spec["total_steps_per_site"])]
        epsilon0 = accountant.get_epsilon(delta=spec["calibration_add_remove_delta"])
        converted_delta = (1 + math.exp(epsilon0)) * spec["calibration_add_remove_delta"]
        grad_std = sigma / spec["expected_batch_size"]
        update_std = 0.01 * grad_std
        site_rw = update_std * math.sqrt(spec["total_steps_per_site"])
        avg_rw = site_rw / math.sqrt(3)
        entry = dict(spec, target_replace_one_epsilon=epsilon,
                     target_replace_one_delta=1e-6,
                     achieved_add_remove_epsilon_upper_bound=epsilon0,
                     achieved_replace_one_epsilon_upper_bound_at_target_delta=2 * epsilon0,
                     converted_replace_one_delta=converted_delta,
                     converted_delta_within_target=converted_delta <= 1e-6,
                     per_coordinate_mean_gradient_noise_std=grad_std,
                     per_coordinate_update_noise_std=update_std,
                     free_random_walk_per_coordinate_std_one_site_public=site_rw,
                     free_random_walk_per_coordinate_std_averaged_three_sites_public=avg_rw,
                     free_random_walk_per_coordinate_std_one_site_BMI=43 * site_rw,
                     free_random_walk_per_coordinate_std_averaged_three_sites_BMI=43 * avg_rw)
        assert converted_delta <= 1e-6 and 2 * epsilon0 <= epsilon
        result["budgets"].append(entry)
    candidates = json.loads((args.root / "r4" / "candidates.json").read_text())
    for entry in candidates["splits"][0]["runs"]:
        config = entry["config"]
        result["candidate_original_geometry"].append({
            "config": config, "n_per_site": 12000, "epsilon": 8,
            "delta": 1e-6, "rounds": 5,
            **mechanism(8, config["batch"], config["epochs"])})
    output = args.root / "r4" / "accounting.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
