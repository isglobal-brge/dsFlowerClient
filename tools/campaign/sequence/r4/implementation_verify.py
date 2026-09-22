#!/usr/bin/env python3
"""R3 contract verification reused for R4; no held-out data is opened."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from opacus.accountants import PRVAccountant

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dsflower_runner import client_app, dp_harness


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


@lru_cache(None)
def independent_accounting(sigma, sample_rate, steps, epsilon, delta):
    accountant = PRVAccountant()
    accountant.history = [(sigma, sample_rate, steps)]
    delta0 = delta / (1 + math.exp(epsilon / 2))
    epsilon0 = accountant.get_epsilon(delta=delta0)
    achieved_delta = delta0 * (1 + math.exp(epsilon0))
    assert 2 * epsilon0 <= epsilon and achieved_delta <= delta
    return {"accountant": "PRVAccountant", "full_horizon_steps": steps,
            "sample_rate": sample_rate, "noise_multiplier": sigma,
            "epsilon_replace_one": 2 * epsilon0,
            "delta_replace_one": achieved_delta}


def verify_run(run, protocol, audit, data, unit, epsilon, seed):
    status = read(run / "federation-status.json")
    assert status["status"] == "trained_unscored" and status["cleanup_ok"]
    assert status["seed"] == seed and status["epsilon"] == epsilon
    assert status["unit"] == unit and status["contract"] == "pytorch_lstm"
    checkpoint = Path(status["output_dir"]) / "model.pt"
    assert sha(checkpoint) == status["model_sha256"]
    execution = read(run / "execution-status.json")
    assert execution["status"] == "trained_unscored"
    assert execution["protocol_sha256"] == sha(run / "protocol.json")
    initial = read(run / "public-capture/public-initial.json")
    assert initial["seed"] == seed
    cfg = initial["config"]
    assert cfg["loss-name"] == "cross_entropy" and cfg["num-classes"] == 6
    bounds = client_app._effective_feature_bounds(cfg)
    public_scale = np.tile([1., 1., 1., 1., 1., 1., 2., 2., 2.], 128)
    assert np.array_equal(bounds["lower"], -public_scale)
    assert np.array_equal(bounds["upper"], public_scale)
    x = client_app._apply_feature_bounds(data["X"], cfg)
    y = data["y"].astype(np.float32)
    subjects = data["subjects"]
    expected = {}
    for site in audit["site_subjects"]:
        select = np.isin(subjects, site)
        xp, yp = x[select], y[select]
        if unit == "subject":
            xp, yp = client_app._pool_by_patient(
                xp, yp, subjects[select], "cross_entropy")
            xp = client_app._totalize_private_features(xp)
        expected[(digest(xp), digest(yp))] = (int(select.sum()), len(yp))
    assert len(expected) == 3
    captures = [read(path) for path in sorted(
        (run / "public-capture").glob("accountant-*.json"))]
    assert len(captures) == 15
    seen, accounting = set(), []
    schedule = protocol["model_params"]
    for capture in captures:
        key = (capture["features_sha256"], capture["targets_sha256"])
        assert key in expected
        rows, units = expected[key]
        assert capture["source_rows"] == rows
        assert capture["round"] in range(1, 6)
        assert (key, capture["round"]) not in seen
        seen.add((key, capture["round"]))
        pins = capture["training_pins"]
        assert pins["num_rounds"] == protocol["rounds"] == 5
        assert pins["batch_size"] == schedule["batch_size"]
        assert pins["local_epochs"] == schedule["local_epochs"]
        assert pins["learning_rate"] == schedule["learning_rate"]
        assert pins["optimizer"]["name"] == schedule["optimizer"]
        assert pins["round_index"] == capture["round"]
        privacy = capture["privacy_config"]
        assert privacy["epsilon"] == epsilon
        assert privacy["clipping_norm"] == protocol["clipping_norm"] == 1
        assert privacy["delta"] == protocol["delta"] == 1e-6
        expected_mechanism = dp_harness.effective_dpsgd_mechanism(
            epsilon, protocol["delta"], protocol["clipping_norm"], units,
            schedule["batch_size"], schedule["local_epochs"], 5)
        mechanism = capture["mechanism"]
        assert mechanism == expected_mechanism
        round_steps = mechanism["steps_per_epoch"] * schedule["local_epochs"]
        assert capture["observed_round_steps"] == round_steps
        assert capture["accountant_type"] == "PRVAccountant"
        assert capture["accountant_history"] == [[
            mechanism["noise_multiplier"], mechanism["sample_rate"], round_steps]]
        accounting.append(independent_accounting(
            mechanism["noise_multiplier"], mechanism["sample_rate"],
            mechanism["total_steps"], epsilon, protocol["delta"]))
    assert len(seen) == 15
    return {"unit": unit, "epsilon": epsilon, "seed": seed, "run": run,
            "status": status, "execution": execution, "checkpoint": checkpoint,
            "initial": initial, "captures": captures, "accounting": accounting}

