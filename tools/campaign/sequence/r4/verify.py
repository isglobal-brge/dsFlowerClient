#!/usr/bin/env python3
"""Verify the real inner-split federation and score TRAIN validation windows."""
import argparse
import base64
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from opacus.accountants import PRVAccountant
from dsflower_runner import client_app, dp_harness, params, server_app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_and_assemble import metrics


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(None)
def independent_accounting(sigma, sample_rate, steps):
    accountant = PRVAccountant()
    accountant.history = [(sigma, sample_rate, steps)]
    delta0 = 1e-6 / (1 + math.exp(8 / 2))
    epsilon0 = accountant.get_epsilon(delta=delta0)
    result = {"accountant": "PRVAccountant", "full_horizon_steps": steps,
              "sample_rate": sample_rate, "noise_multiplier": sigma,
              "epsilon_replace_one": 2 * epsilon0,
              "delta_replace_one": delta0 * (1 + math.exp(epsilon0))}
    assert result["epsilon_replace_one"] <= 8
    assert result["delta_replace_one"] <= 1e-6
    return result


def main(root, emulator_results):
    torch.set_num_threads(2)
    work = root / "r4"
    run = work / "real-contract"
    capture_dir = run / "public-capture"
    prepared = work / "prepared"
    audit = read(prepared / "audit.json")
    for name in ("train", "validation"):
        assert file_digest(prepared / f"{name}.npz") == audit[f"{name}_npz_sha256"]
    train = np.load(prepared / "train.npz", allow_pickle=False)
    validation = np.load(prepared / "validation.npz", allow_pickle=False)
    assert train["X"].shape == (5564, 1152)
    assert validation["X"].shape == (1788, 1152)
    assert set(train["subjects"]).isdisjoint(set(validation["subjects"]))
    assert np.unique(validation["subjects"]).tolist() == [1, 8, 17, 25, 30]
    status = read(run / "federation-status.json")
    assert status["status"] == "trained_unscored" and status["cleanup_ok"]
    assert status["seed"] == 20260922 and status["epsilon"] == 8
    assert status["unit"] == "window" and status["contract"] == "pytorch_lstm"
    checkpoint = Path(status["output_dir"]) / "model.pt"
    assert file_digest(checkpoint) == status["model_sha256"]
    history = read(Path(status["output_dir"]) / "history.json")
    assert [row["round"] for row in history] == [1, 2, 3, 4, 5]
    assert all(row["n_failures"] == 0 and row["available"] for row in history)
    initial = read(capture_dir / "public-initial.json")
    assert initial["seed"] == 20260922
    cfg = initial["config"]
    planned = read(work / "config.json")
    for key, value in planned.items():
        if key == "results-dir":
            continue
        if key.endswith("-b64"):
            assert json.loads(base64.b64decode(cfg[key])) == json.loads(base64.b64decode(value)), key
        else:
            assert cfg[key] == value, key
    bounds = client_app._effective_feature_bounds(cfg)
    bound = np.tile([1.] * 6 + [2.] * 3, 128)
    assert np.array_equal(bounds["lower"], -bound) and np.array_equal(bounds["upper"], bound)
    x = client_app._apply_feature_bounds(train["X"], cfg)
    y = train["y"].astype(np.float32)
    expected = {}
    for index, site in enumerate(audit["site_subjects"]):
        select = np.isin(train["subjects"], site)
        assert int(select.sum()) == audit["n_per_site"][index]
        expected[(digest(x[select]), digest(y[select]))] = (index + 1, int(select.sum()))
    assert len(expected) == 3
    captures = [read(path) for path in sorted(capture_dir.glob("accountant-*.json"))]
    assert len(captures) == 15 and not list(capture_dir.glob("failure-*.json"))
    seen, accounting = set(), {}
    for capture in captures:
        key = (capture["features_sha256"], capture["targets_sha256"])
        assert key in expected, "Node input differs from prepared inner-training windows"
        site, population = expected[key]
        round_index = capture["round"]
        assert round_index in range(1, 6) and (site, round_index) not in seen
        seen.add((site, round_index))
        assert capture["source_rows"] == population
        pins, privacy = capture["training_pins"], capture["privacy_config"]
        assert pins["num_rounds"] == 5 and pins["batch_size"] == 256
        assert pins["local_epochs"] == 4 and pins["learning_rate"] == .01
        assert pins["optimizer"]["name"] == "adam" and pins["round_index"] == round_index
        assert privacy["epsilon"] == 8 and privacy["delta"] == 1e-6
        assert privacy["clipping_norm"] == 1
        mechanism = dp_harness.effective_dpsgd_mechanism(8, 1e-6, 1, population, 256, 4, 5)
        assert capture["mechanism"] == mechanism
        round_steps = mechanism["steps_per_epoch"] * 4
        assert capture["observed_round_steps"] == round_steps
        assert capture["accountant_type"] == "PRVAccountant"
        assert capture["accountant_history"] == [[
            mechanism["noise_multiplier"], mechanism["sample_rate"], round_steps]]
        accounting[site] = {"site": site, "mechanism": mechanism,
                            "independent": independent_accounting(
                                mechanism["noise_multiplier"], mechanism["sample_rate"],
                                mechanism["total_steps"])}
    assert len(seen) == 15
    arrays = np.load(capture_dir / "public-initial-arrays.npz", allow_pickle=False)
    assert [digest(arrays[str(i)]) for i in range(len(arrays.files))] == initial["tensor_sha256"]
    torch.manual_seed(20260922)
    model = server_app._build_initial_model(planned)
    assert [digest(array) for array in params.get_torch_params(model)] == initial["tensor_sha256"]
    emulator = read(emulator_results)
    assert emulator["initial_tensor_sha256"] == initial["tensor_sha256"]
    # Use the installed analyst-local predictor, with preprocessing exactly once.
    helper_dir = root / "Rlib/dsFlowerClient/python"
    assert (helper_dir / "predict_helper.py").is_file()
    sys.path.insert(0, str(helper_dir))
    from predict_helper import _apply_feature_preprocessing, predict_pytorch_spec
    vx = _apply_feature_preprocessing(validation["X"], cfg["feature-bounds-b64"])
    assert np.array_equal(vx, client_app._apply_feature_bounds(validation["X"], cfg))
    assert np.array_equal(vx, (np.clip(validation["X"], -bound, bound) / bound).astype(np.float32))
    seen_input = []
    recurrent = next(module for module in model.modules()
                     if module.__class__.__name__ == "RecurrentBlock")
    handle = recurrent.register_forward_pre_hook(
        lambda _module, arguments: seen_input.append(arguments[0].detach().numpy().copy()))
    with torch.no_grad():
        model(torch.from_numpy(vx[:2]))
    handle.remove()
    assert seen_input[0].shape == (2, 128, 9)
    assert np.array_equal(seen_input[0], vx[:2].reshape(2, 128, 9, order="C"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("state_dict", state), strict=True)
    model.eval()
    with torch.no_grad():
        direct = model(torch.from_numpy(vx)).softmax(1).numpy()
    local = np.asarray(predict_pytorch_spec(
        str(checkpoint), vx, "prob", cfg["model-spec-b64"], "cross_entropy", num_classes=6),
        dtype=np.float32)
    assert local.shape == (1788, 6) and np.allclose(local, direct, rtol=0, atol=1e-7)
    response = np.asarray(predict_pytorch_spec(
        str(checkpoint), vx[:64], "response", cfg["model-spec-b64"], "cross_entropy", num_classes=6))
    assert np.array_equal(response, local[:64].argmax(1))
    result = metrics(validation["y"], local)
    direct_metrics = metrics(validation["y"], direct)
    assert result == direct_metrics
    save(work / "real-metrics.json", {
        "status": "inner_validation_scored", "test_accessed": False,
        "seed": 20260922, "epsilon": 8, "delta": 1e-6, "clipping_norm": 1,
        "unit": "window", "n_inner_training_windows": 5564, "n_validation_windows": 1788,
        "validation_subjects": audit["validation_subjects"],
        "model_sha256": status["model_sha256"], "metrics": result,
        "initial_tensor_sha256": initial["tensor_sha256"]})
    save(work / "verification.json", {
        "status": "passed", "test_accessed": False,
        "five_rounds_three_sites_zero_failures": True,
        "captured_node_rounds": 15, "node_inputs_match_prepared_inner_training": True,
        "config_matches_recorded_r3_contract": True,
        "initial_arrays_match_emulator": True,
        "initial_tensor_sha256": initial["tensor_sha256"],
        "accounting_per_site": list(accounting.values()),
        "prediction_helper_path": str(helper_dir / "predict_helper.py"),
        "prediction_helper_sha256": file_digest(helper_dir / "predict_helper.py"),
        "prediction_probabilities_match_direct_model": True,
        "prediction_max_abs_error": float(np.abs(local - direct).max()),
        "response_is_argmax_class_0_through_5": True,
        "class_order": [0, 1, 2, 3, 4, 5],
        "preprocessing_matches_training_and_public_bounds": True,
        "recurrent_input_shape": list(seen_input[0].shape),
        "token_major_layout_exact": True,
        "validation_probabilities_sha256": digest(local),
        "federation_status_sha256": file_digest(run / "federation-status.json"),
        "emulator_results_sha256": file_digest(emulator_results)})
    print(json.dumps({"verification": "passed", "metrics": result,
                      "accounting_per_site": list(accounting.values())}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--emulator-results", type=Path)
    args = parser.parse_args()
    main(args.root, args.emulator_results or args.root / "r4/emulation.json")
