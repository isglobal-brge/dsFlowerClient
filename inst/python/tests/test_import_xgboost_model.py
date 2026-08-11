"""KATs and fail-closed tests for the isolated external XGBoost importer."""

import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HELPER_PATH = Path(__file__).resolve().parents[1] / "import_xgboost_model.py"
SPEC = importlib.util.spec_from_file_location(
    "_dsflower_import_xgboost_model", HELPER_PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)

_KAT_SHA256 = {
    "binary": {
        "manifest": "a32c1a365d6e0cf2dc9e4872ff8b2d1d593393379c5b1939ed9422b26b780466",
        "model": "bb753e1d92b79464aa2f24b583dff715647e616190d395f02b249498941b5519",
        "profile": "4b4767441694bd5f60b41083012d4a5a9e6ab4963a80cb69c2f603b87d1382e9",
    },
    "regression": {
        "manifest": "ca0698d9f503e6652fc1e2df0981b0c141aa81bfa00a2d4c3f011e1e08f1d5c7",
        "model": "a5e5057dbf11d08177db9e113deecd510682d4cd538cfa936d188e1ab1d50de0",
        "profile": "8d30f11a18befec13424fcb491cd89cdf2d292cabe345f26b002ce506002d564",
    },
}


def _canonical(value, *, ascii_only=False, sorted_keys=False):
    return json.dumps(
        value, ensure_ascii=ascii_only, allow_nan=False,
        sort_keys=sorted_keys, separators=(",", ":"),
    ).encode("ascii" if ascii_only else "utf-8")


def _next_float32(value):
    value = struct.unpack(">f", struct.pack(">f", float(value)))[0]
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    bits = bits + 1 if value >= 0.0 else bits - 1
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def _request_wire(task="binary", *, engine="xgboost"):
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
        "version": 1,
        "features": ["age", "marker"],
        "lower": [0.0, -5.0],
        "upper": [100.0, 5.0],
        "cuts": [[18.0, 40.0, 65.0], [-1.0, 0.0, 1.0]],
        "target": target,
    }
    schema = dict(core, sha256=hashlib.sha256(_canonical(core)).hexdigest())
    request = {
        "contract": "dsflower-native-tree-request-v1",
        "engine": engine,
        "mode": "native-tight",
        "task": task,
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
            "max_features": 2,
            "max_trees": 1,
            "max_depth": 1,
            "max_bins": 4,
            "max_threads": 4,
            "memory_mb": 4096,
            "timeout_seconds": 900,
        },
    }
    raw = _canonical(request)
    return (
        request,
        base64.b64encode(raw).decode("ascii"),
        hashlib.sha256(raw).hexdigest(),
    )


def _model(task="binary"):
    tree = {
        "base_weights": [0.0, 0.0, 0.0],
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
        "default_left": [1, 0, 0],
        "id": 0,
        "left_children": [1, -1, -1],
        "loss_changes": [0.0, 0.0, 0.0],
        "parents": [2_147_483_647, 0, 0],
        "right_children": [2, -1, -1],
        "split_conditions": [_next_float32(18.0), -0.1, 0.1],
        "split_indices": [0, 0, 0],
        "split_type": [0, 0, 0],
        "sum_hessian": [0.0, 0.0, 0.0],
        "tree_param": {
            "num_deleted": "0",
            "num_feature": "2",
            "num_nodes": "3",
            "size_leaf_vector": "1",
        },
    }
    objective = ({
        "name": "binary:logistic",
        "reg_loss_param": {"scale_pos_weight": "1"},
    } if task == "binary" else {
        "name": "reg:squarederror",
        "reg_loss_param": {"scale_pos_weight": "1"},
    })
    return {
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "model": {
                    "cats": {
                        "enc": [], "feature_segments": [], "sorted_idx": []},
                    "gbtree_model_param": {
                        "num_parallel_tree": "1", "num_trees": "1"},
                    "iteration_indptr": [0, 1],
                    "tree_info": [0],
                    "trees": [tree],
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": "[5E-1]" if task == "binary" else "[0E0]",
                "boost_from_average": "0",
                "num_class": "0",
                "num_feature": "2",
                "num_target": "1",
            },
            "objective": objective,
        },
        "version": [3, 4, 0],
    }


def _model_bytes(task="binary", value=None):
    if value is None:
        value = _model(task)
    return _canonical(value, ascii_only=True, sorted_keys=True)


def _command(artifact, output, request, request_sha256, python_flags=None):
    if python_flags is None:
        python_flags = ("-I", "-S")
    return [
        sys.executable, *python_flags, str(HELPER_PATH),
        "--artifact", str(artifact),
        "--request", str(request),
        "--request-sha256", request_sha256,
        "--output-dir", str(output),
    ]


def _portable_stderr(value):
    return value.replace(b"\r\n", b"\n")


def _invoke(root, task="binary", *, raw=None, request=None):
    root = Path(root)
    artifact = root / "external model.json"
    output = root / "empty output"
    output.mkdir()
    artifact.write_bytes(_model_bytes(task) if raw is None else raw)
    request, request_b64, request_sha256 = (
        _request_wire(task) if request is None else request)
    request_path = root / "native tree request.json"
    request_path.write_bytes(base64.b64decode(request_b64, validate=True))
    source_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    result = subprocess.run(
        _command(artifact, output, request_path, request_sha256),
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False)
    if result.returncode != 0:
        return result, output, None, request
    manifest = json.loads(result.stdout.decode("ascii"))
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != source_digest:
        raise AssertionError("importer modified its source artifact")
    return result, output, manifest, request


def _runner_modules():
    runner = Path(__file__).resolve().parents[2] / "flower_app"
    sys.path.insert(0, str(runner))
    from dsflower_runner import native_tree_engine
    from dsflower_runner import native_tree_request
    from dsflower_runner import xgboost_predictor
    return native_tree_engine, native_tree_request, xgboost_predictor


class ExternalXGBoostImporterTests(unittest.TestCase):
    def test_binary_and_regression_known_answers_under_isolated_python(self):
        native_tree_engine, native_tree_request, predictor_module = \
            _runner_modules()
        for task in ("binary", "regression"):
            with self.subTest(task=task), tempfile.TemporaryDirectory(
                    prefix="dsFlower import ü ") as root:
                result, output, manifest, request = _invoke(root, task)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stderr, b"")
                self.assertEqual(
                    result.stdout,
                    _canonical(manifest, ascii_only=True, sorted_keys=True))
                self.assertEqual(
                    sorted(path.name for path in output.iterdir()),
                    [helper.MODEL_FILE, helper.PROFILE_FILE])
                self.assertEqual(manifest["contract"], helper.IMPORT_CONTRACT)
                self.assertEqual(manifest["engine"], "xgboost")
                self.assertEqual(manifest["task"], task)
                self.assertEqual(manifest["track"], "native_tree")
                self.assertEqual(manifest["data_kind"], "tabular")
                self.assertEqual(
                    manifest["sanitization"]["privacy_basis"],
                    "external-unverified")
                self.assertTrue(
                    manifest["sanitization"]["contains_unnoised_statistics"])
                self.assertNotIn(b"direct-dp", result.stdout)
                self.assertFalse(
                    manifest["sanitization"]["contains_executable_payload"])
                self.assertNotIn("source", manifest)
                self.assertNotIn("path", manifest)

                model_path = output / helper.MODEL_FILE
                profile_path = output / helper.PROFILE_FILE
                ensemble = model_path.read_bytes()
                profile = profile_path.read_bytes()
                self.assertEqual(
                    json.loads(ensemble)["contract"],
                    helper.EXTERNAL_ENSEMBLE_CONTRACT)
                self.assertEqual(
                    hashlib.sha256(result.stdout).hexdigest(),
                    _KAT_SHA256[task]["manifest"])
                self.assertEqual(
                    hashlib.sha256(ensemble).hexdigest(),
                    _KAT_SHA256[task]["model"])
                self.assertEqual(
                    hashlib.sha256(profile).hexdigest(),
                    _KAT_SHA256[task]["profile"])
                self.assertEqual(
                    manifest["artifact"]["sha256"],
                    hashlib.sha256(ensemble).hexdigest())
                self.assertEqual(
                    manifest["prediction_profile"]["sha256"],
                    hashlib.sha256(profile).hexdigest())
                public_manifest = native_tree_request.public_backend_manifest(
                    request)
                request_b64 = manifest["native_tree_request_b64"]
                request_sha256 = manifest["native_tree_request_sha256"]
                spec = native_tree_engine.validate_prediction_profile(
                    profile, request, request_b64, request_sha256, ensemble)
                self.assertEqual(spec["engine"], "xgboost")
                predictor = predictor_module.parse_xgboost_ensemble(
                    ensemble, public_manifest)
                predictions = predictor.predict([[0.0, 0.0], [20.0, 0.0]])
                if task == "binary":
                    self.assertAlmostEqual(
                        predictions[0], 1.0 / (1.0 + math.exp(0.1)), places=7)
                    self.assertAlmostEqual(
                        predictions[1], 1.0 / (1.0 + math.exp(-0.1)), places=7)
                    self.assertEqual(
                        manifest["target_levels"], ["control", "case"])
                    self.assertIsNone(manifest["target_bounds"])
                else:
                    self.assertAlmostEqual(predictions[0], -0.1, places=7)
                    self.assertAlmostEqual(predictions[1], 0.1, places=7)
                    self.assertIsNone(manifest["target_levels"])
                    self.assertEqual(
                        manifest["target_bounds"],
                        {"lower": -10.0, "upper": 10.0})
                if os.name != "nt":
                    self.assertEqual(
                        stat.S_IMODE(model_path.stat().st_mode), 0o600)
                    self.assertEqual(
                        stat.S_IMODE(profile_path.stat().st_mode), 0o600)

    def test_output_is_byte_deterministic_and_contains_no_state_files(self):
        with tempfile.TemporaryDirectory() as first, \
                tempfile.TemporaryDirectory() as second:
            result_a, output_a, manifest_a, _ = _invoke(first)
            result_b, output_b, manifest_b, _ = _invoke(second)
            self.assertEqual(result_a.returncode, 0)
            self.assertEqual(result_b.returncode, 0)
            self.assertEqual(result_a.stdout, result_b.stdout)
            self.assertEqual(manifest_a, manifest_b)
            for name in (helper.MODEL_FILE, helper.PROFILE_FILE):
                self.assertEqual(
                    (output_a / name).read_bytes(),
                    (output_b / name).read_bytes())
            forbidden = {
                "history.json", "metadata.json", "manifest.json",
                "privacy.sqlite", "model.rds",
            }
            self.assertFalse(forbidden.intersection(
                path.name for path in output_a.iterdir()))

    def test_malformed_artifact_and_tampered_request_fail_with_fixed_error(self):
        bad_artifacts = [
            b"\x80\x04cpickle\nloads\n.",
            b'{"learner":{},"learner":{},"version":[3,4,0]}',
            b'{"learner":NaN,"version":[3,4,0]}',
        ]
        named = _model()
        named["learner"]["feature_names"] = ["age", "marker"]
        bad_artifacts.append(_model_bytes(value=named))
        retired = _model()
        retired["version"] = [3, 3, 0]
        bad_artifacts.append(_model_bytes(value=retired))
        executable = _model()
        executable["callback"] = "os.system"
        bad_artifacts.append(_model_bytes(value=executable))

        for index, raw in enumerate(bad_artifacts):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as root:
                result, output, manifest, _ = _invoke(root, raw=raw)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(
                    _portable_stderr(result.stderr),
                    helper.ERROR_MESSAGE.encode())
                self.assertIsNone(manifest)
                self.assertEqual(list(output.iterdir()), [])

        with tempfile.TemporaryDirectory() as root:
            request, request_b64, _request_sha256 = _request_wire()
            result, output, manifest, _ = _invoke(
                root, request=(request, request_b64, "0" * 64))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, b"")
            self.assertEqual(
                _portable_stderr(result.stderr), helper.ERROR_MESSAGE.encode())
            self.assertIsNone(manifest)
            self.assertEqual(list(output.iterdir()), [])

        with tempfile.TemporaryDirectory() as root:
            result, output, manifest, _ = _invoke(
                root, request=_request_wire(engine="lightgbm"))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(
                _portable_stderr(result.stderr), helper.ERROR_MESSAGE.encode())
            self.assertIsNone(manifest)
            self.assertEqual(list(output.iterdir()), [])

    def test_request_must_be_a_bounded_regular_canonical_file(self):
        _request, request_b64, request_sha256 = _request_wire()
        request_bytes = base64.b64decode(request_b64, validate=True)
        for case in ("tampered", "oversized", "symlink"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                artifact = root / "model.json"
                artifact.write_bytes(_model_bytes())
                output = root / "output"
                output.mkdir()
                request_path = root / "request.json"
                if case == "tampered":
                    request_path.write_bytes(request_bytes + b" ")
                elif case == "oversized":
                    with request_path.open("wb") as handle:
                        handle.seek(helper.MAX_MANIFEST_BYTES)
                        handle.write(b"x")
                else:
                    real_request = root / "real-request.json"
                    real_request.write_bytes(request_bytes)
                    try:
                        os.symlink(real_request, request_path)
                    except (OSError, NotImplementedError):
                        continue
                result = subprocess.run(
                    _command(
                        artifact, output, request_path, request_sha256),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=False)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(
                    _portable_stderr(result.stderr),
                    helper.ERROR_MESSAGE.encode())
                self.assertEqual(list(output.iterdir()), [])

    def test_cli_requires_isolated_python_without_site(self):
        _request, request_b64, request_sha256 = _request_wire()
        request_bytes = base64.b64decode(request_b64, validate=True)
        for python_flags in ((), ("-I",), ("-S",)):
            with self.subTest(flags=python_flags), \
                    tempfile.TemporaryDirectory() as root:
                root = Path(root)
                artifact = root / "model.json"
                artifact.write_bytes(_model_bytes())
                request_path = root / "request.json"
                request_path.write_bytes(request_bytes)
                output = root / "output"
                output.mkdir()
                result = subprocess.run(
                    _command(
                        artifact, output, request_path, request_sha256,
                        python_flags=python_flags),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=False)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(
                    _portable_stderr(result.stderr),
                    helper.ERROR_MESSAGE.encode())
                self.assertEqual(list(output.iterdir()), [])

    def test_oversized_symlink_and_nonempty_destinations_are_rejected(self):
        request, request_b64, request_sha256 = _request_wire()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            artifact = root / "oversized.json"
            with artifact.open("wb") as handle:
                handle.seek(helper.MAX_ARTIFACT_BYTES)
                handle.write(b"x")
            output = root / "output"
            output.mkdir()
            request_path = root / "request.json"
            request_path.write_bytes(base64.b64decode(request_b64))
            result = subprocess.run(
                _command(artifact, output, request_path, request_sha256),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(
                _portable_stderr(result.stderr), helper.ERROR_MESSAGE.encode())
            self.assertEqual(list(output.iterdir()), [])

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            artifact = root / "model.json"
            artifact.write_bytes(_model_bytes())
            output = root / "output"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="ascii")
            request_path = root / "request.json"
            request_path.write_bytes(base64.b64decode(request_b64))
            result = subprocess.run(
                _command(artifact, output, request_path, request_sha256),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(sentinel.read_text(encoding="ascii"), "keep")
            self.assertEqual([path.name for path in output.iterdir()],
                             ["keep.txt"])

        if hasattr(os, "symlink"):
            with tempfile.TemporaryDirectory() as root:
                root = Path(root)
                real = root / "real.json"
                link = root / "link.json"
                real.write_bytes(_model_bytes())
                try:
                    os.symlink(real, link)
                except (OSError, NotImplementedError):
                    return
                output = root / "output"
                output.mkdir()
                request_path = root / "request.json"
                request_path.write_bytes(base64.b64decode(request_b64))
                result = subprocess.run(
                    _command(link, output, request_path, request_sha256),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=False)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(list(output.iterdir()), [])

    def test_second_install_failure_removes_every_file_it_created(self):
        request, request_b64, request_sha256 = _request_wire()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            artifact = root / "model.json"
            artifact.write_bytes(_model_bytes())
            output = root / "output"
            output.mkdir()
            request_path = root / "request.json"
            request_path.write_bytes(base64.b64decode(request_b64))
            real_install = helper._install_exclusive
            calls = 0

            def fail_second(temporary, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected install failure")
                return real_install(temporary, destination)

            with mock.patch.object(
                    helper, "_install_exclusive", side_effect=fail_second), \
                    mock.patch.object(
                        helper, "_reject_forbidden_imports", return_value=None):
                with self.assertRaises(OSError):
                    helper.import_model(
                        str(artifact), str(request_path), request_sha256,
                        str(output))
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
