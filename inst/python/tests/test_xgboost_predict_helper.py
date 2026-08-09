"""Contracts and KATs for the dependency-free local XGBoost predictor."""

import base64
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest


HELPER_PATH = Path(__file__).resolve().parents[1] / "xgboost_predict_helper.py"
SPEC = importlib.util.spec_from_file_location(
    "_dsflower_xgboost_predict_helper", HELPER_PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def _canonical(value, *, ascii_only=False, sorted_keys=False):
    return json.dumps(
        value, ensure_ascii=ascii_only, allow_nan=False, sort_keys=sorted_keys,
        separators=(",", ":"),
    ).encode("ascii" if ascii_only else "utf-8")


def _next_float32(value):
    value = struct.unpack(">f", struct.pack(">f", float(value)))[0]
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    bits = bits + 1 if value >= 0.0 else bits - 1
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _request_wire(features=("age", "marker"), task="binary"):
    target = ({
        "name": "outcome", "kind": "binary",
        "levels": [
            {"type": "string", "value": "control"},
            {"type": "string", "value": "case"},
        ],
        "lower": 0.0, "upper": 1.0,
    } if task == "binary" else {
        "name": "outcome", "kind": "continuous", "levels": None,
        "lower": -10.0, "upper": 10.0,
    })
    core = {
        "version": 1, "features": list(features),
        "lower": [0.0, -5.0], "upper": [100.0, 5.0],
        "cuts": [[18.0, 40.0, 65.0], [-1.0, 0.0, 1.0]],
        "target": target,
    }
    schema = dict(core, sha256=hashlib.sha256(_canonical(core)).hexdigest())
    request = {
        "contract": "dsflower-native-tree-request-v1",
        "engine": "xgboost", "mode": "native-tight", "task": task,
        "public_schema": schema,
        "parameters": [
            {"name": "learning_rate", "type": "number", "value": 0.25},
            {"name": "max_delta_step", "type": "number", "value": 1.0},
            {"name": "max_depth", "type": "integer", "value": 1},
            {"name": "min_child_weight", "type": "number", "value": 1.0},
            {"name": "min_split_loss", "type": "number", "value": 0.0},
            {"name": "num_boost_round", "type": "integer", "value": 1},
            {"name": "reg_alpha", "type": "number", "value": 0.0},
            {"name": "reg_lambda", "type": "number", "value": 1.0},
        ],
        "resources": {
            "max_features": 2, "max_trees": 1, "max_depth": 1,
            "max_bins": 4, "max_threads": 4, "memory_mb": 4096,
            "timeout_seconds": 900,
        },
    }
    raw = _canonical(request)
    return request, base64.b64encode(raw).decode("ascii"), \
        hashlib.sha256(raw).hexdigest()


def _model(task="binary"):
    leaf = 0.1
    tree = {
        "base_weights": [0.0, 0.0, 0.0],
        "categories": [], "categories_nodes": [],
        "categories_segments": [], "categories_sizes": [],
        "default_left": [1, 0, 0], "id": 0,
        "left_children": [1, -1, -1],
        "loss_changes": [0.0, 0.0, 0.0],
        "parents": [2_147_483_647, 0, 0],
        "right_children": [2, -1, -1],
        "split_conditions": [_next_float32(18.0), -leaf, leaf],
        "split_indices": [0, 0, 0], "split_type": [0, 0, 0],
        "sum_hessian": [0.0, 0.0, 0.0],
        "tree_param": {
            "num_deleted": "0", "num_feature": "2", "num_nodes": "3",
            "size_leaf_vector": "1",
        },
    }
    base_score = "[5E-1]" if task == "binary" else "[0E0]"
    objective = ({
        "name": "binary:logistic",
        "reg_loss_param": {"scale_pos_weight": "1"},
    } if task == "binary" else {
        "name": "reg:squarederror", "reg_loss_param": {
            "scale_pos_weight": "1"},
    })
    return {
        "learner": {
            "attributes": {}, "feature_names": [], "feature_types": [],
            "gradient_booster": {
                "model": {
                    "cats": {"enc": [], "feature_segments": [],
                             "sorted_idx": []},
                    "gbtree_model_param": {
                        "num_parallel_tree": "1", "num_trees": "1"},
                    "iteration_indptr": [0, 1], "tree_info": [0],
                    "trees": [tree],
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": base_score, "boost_from_average": "0",
                "num_class": "0", "num_feature": "2", "num_target": "1",
            },
            "objective": objective,
        },
        "version": [3, 4, 0],
    }


def _bundle(root, features=("age", "marker"), task="binary"):
    request, request_b64, request_sha256 = _request_wire(features, task)
    helper._runner_modules()
    from dsflower_runner.xgboost_sanitizer import sanitize_xgboost_json
    model, _digest = sanitize_xgboost_json(
        _canonical(_model(task), ascii_only=True, sorted_keys=True),
        expected_task=("binary_classification" if task == "binary"
                       else "regression"),
        expected_features=2,
        expected_trees=1,
        expected_max_depth=1,
        public_cuts=((18.0, 40.0, 65.0), (-1.0, 0.0, 1.0)),
        expected_base_score=(0.5 if task == "binary" else 0.0),
        max_total_nodes=3,
        max_artifact_bytes=64 * 1024 * 1024,
        numeric_abs_cap=1.0e12,
        leaf_abs_cap=_next_float32(0.25),
    )
    artifact = _canonical({
        "aggregation": "mean_prediction",
        "contract": "dsflower-xgboost-ensemble-v1",
        "engine": "xgboost", "models": [json.loads(model.decode("ascii"))],
        "public_schema_sha256": request["public_schema"]["sha256"],
        "task": task, "version": 1,
    }, ascii_only=True, sorted_keys=True)
    artifact_path = os.path.join(root, "model.xgboost-ensemble.json")
    with open(artifact_path, "wb") as handle:
        handle.write(artifact)
    profile = {
        "artifact": {
            "format": "dsflower-xgboost-ensemble-json-v1",
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "size_bytes": len(artifact),
        },
        "contract": "dsflower-xgboost-prediction-profile-v1",
        "native_tree_request_b64": request_b64,
        "native_tree_request_sha256": request_sha256,
        "public_schema_sha256": request["public_schema"]["sha256"],
        "task": task, "version": 1,
    }
    profile_path = os.path.join(root, "model.xgboost-ensemble.profile.json")
    with open(profile_path, "wb") as handle:
        handle.write(_canonical(profile, ascii_only=True, sorted_keys=True))
    return artifact_path, profile_path


def _csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class XGBoostPredictHelperTests(unittest.TestCase):
    def test_binary_kat_clamps_and_preserves_missing_direction(self):
        with tempfile.TemporaryDirectory() as root:
            model, profile = _bundle(root)
            data = os.path.join(root, "data.csv")
            output = os.path.join(root, "predictions.json")
            _csv(data, ("age", "marker"), (
                ("-Inf", "0"), ("Inf", "0"), ("NA", "0"),
                ("1000000000", "NaN")))
            predictor, features, task = helper._load_predictor(model, profile)
            helper._predict_csv(
                predictor, features, task, "prob", data, output, 4)
            with open(output, encoding="ascii") as handle:
                value = json.load(handle)
            low = 1.0 / (1.0 + math.exp(0.1))
            high = 1.0 / (1.0 + math.exp(-0.1))
            self.assertEqual(value["task"], "binary")
            self.assertEqual(value["type"], "prob")
            self.assertAlmostEqual(value["predictions"][0], low, places=7)
            self.assertAlmostEqual(value["predictions"][1], high, places=7)
            self.assertAlmostEqual(value["predictions"][2], low, places=7)
            self.assertAlmostEqual(value["predictions"][3], high, places=7)

    def test_profile_artifact_and_columns_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            model, profile = _bundle(root)
            data = os.path.join(root, "data.csv")
            _csv(data, ("marker", "age"), (("0", "20"),))
            predictor, features, task = helper._load_predictor(model, profile)
            with self.assertRaisesRegex(ValueError, "columns"):
                helper._predict_csv(
                    predictor, features, task, "prob", data,
                    os.path.join(root, "out.json"), 1)

            with open(model, "ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                helper._load_predictor(model, profile)

            model, profile = _bundle(root)
            with open(profile, "ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ValueError, "canonical"):
                helper._load_predictor(model, profile)

    def test_unicode_feature_names_and_regression_type(self):
        with tempfile.TemporaryDirectory() as root:
            model, profile = _bundle(root, features=("âge", "marcador"))
            data = os.path.join(root, "unicode.csv")
            output = os.path.join(root, "out.json")
            _csv(data, ("âge", "marcador"), (("20", "0"),))
            predictor, features, task = helper._load_predictor(model, profile)
            helper._predict_csv(
                predictor, features, task, "response", data, output, 1)
            self.assertTrue(os.path.isfile(output))

            model, profile = _bundle(root, task="regression")
            predictor, features, task = helper._load_predictor(model, profile)
            with self.assertRaisesRegex(ValueError, "type"):
                helper._predict_csv(
                    predictor, features, task, "prob", data,
                    os.path.join(root, "regression.json"), 1)

    def test_isolated_cli_is_stdlib_only_and_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            model, profile = _bundle(root)
            data = os.path.join(root, "data.csv")
            output = os.path.join(root, "out.json")
            _csv(data, ("age", "marker"), (("20", "0"),))
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(HELPER_PATH),
                 "--model", model, "--profile", profile, "--data", data,
                 "--output", output, "--type", "prob",
                 "--expected-rows", "1"],
                check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "ok")
            self.assertEqual(result.stderr, "")

    def test_shallow_prediction_benchmark(self):
        rows = 20_000
        with tempfile.TemporaryDirectory() as root:
            model, profile = _bundle(root)
            data = os.path.join(root, "benchmark.csv")
            output = os.path.join(root, "out.json")
            _csv(data, ("age", "marker"), ((index % 100, 0)
                                            for index in range(rows)))
            predictor, features, task = helper._load_predictor(model, profile)
            started = time.perf_counter()
            helper._predict_csv(
                predictor, features, task, "prob", data, output, rows)
            rate = rows / max(time.perf_counter() - started, 1.0e-9)
            self.assertGreater(rate, 10_000.0)


if __name__ == "__main__":
    unittest.main()
