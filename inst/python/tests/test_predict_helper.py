"""Prediction parity tests for saved declarative PyTorch artifacts."""

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

PY_ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = _load(PY_ROOT / "predict_helper.py", "dsflower_predict_helper")
builder = _load(
    PY_ROOT.parent / "flower_app" / "dsflower_runner" / "model_spec.py",
    "dsflower_predict_model_spec_test")


class _MaliciousCheckpoint:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return os.system, ("touch " + self.marker,)


class PredictionParityTests(unittest.TestCase):
    def test_feature_preprocessing_uses_public_bounds_or_raw_values(self):
        raw = np.asarray([[np.nan, np.inf], [2.0e6, -2.0e6]])
        np.testing.assert_array_equal(
            helper._apply_feature_preprocessing(raw),
            np.asarray([[0.0, 0.0], [1.0e6, -1.0e6]], dtype=np.float32))

        bounds = base64.b64encode(json.dumps({
            "lower": [0.0, -2.0], "upper": [2.0, 2.0],
        }).encode()).decode()
        transformed = helper._apply_feature_preprocessing(
            np.asarray([[-1.0, np.inf], [2.0, -2.0]]), bounds)
        np.testing.assert_array_equal(
            transformed,
            np.asarray([[-1.0, 0.0], [1.0, -1.0]], dtype=np.float32))

    def test_declarative_graph_uses_exact_builder(self):
        spec = {
            "kind": "graph", "output": "out", "nodes": [
                {"name": "h", "op": "linear", "in": ["@in"], "out": 4},
                {"name": "a", "op": "tanh", "in": ["h"]},
                {"name": "out", "op": "linear", "in": ["a"], "out": "@out"},
            ]}
        model = builder.build_from_spec(spec, 3, 2, num_labels=2)
        encoded = base64.b64encode(
            json.dumps(spec, separators=(",", ":")).encode()).decode()
        X = np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(model.state_dict(), handle.name)
            got = helper.predict_pytorch_spec(
                handle.name, X, "prob", encoded, "cross_entropy", 2, 2,
                graph_parameter_format=builder.GRAPH_PARAMETER_FORMAT)
        with torch.no_grad():
            expected = torch.softmax(model(torch.tensor(X)), dim=-1).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)

    def test_main_saved_graph_predictions_and_private_validation_are_identical(self):
        from dsflower_runner import model_spec, params, validation
        for name in ("named", "colliding"):
            with self.subTest(name=name):
                root = PY_ROOT / "tests/fixtures/legacy_graph"
                meta = json.loads((root / (name + ".json")).read_text())
                artifact = root / (name + ".pt")
                self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                 meta["artifact_sha256"])
                self.assertNotIn("graph_parameter_format", meta)
                spec = meta["model_spec"]
                encoded = base64.b64encode(json.dumps(spec).encode()).decode()
                X = np.asarray(meta["inputs"], dtype=np.float32)
                got = helper.predict_pytorch_spec(
                    artifact, X, "prob", encoded, meta["loss_name"])
                np.testing.assert_array_equal(got, meta["predictions"])
                cfg = {"validation-model-track": "neural", "num-features": 2,
                       "num-classes": 2, "loss-name": "bce_logits",
                       "model-spec-b64": encoded,
                       "validation-model-path-b64": base64.b64encode(
                           str(artifact).encode()).decode()}
                arrays = validation.public_model_arrays(cfg)
                model = model_spec.build_from_spec(spec, 2, 1)
                params.set_torch_params(model, arrays)
                with torch.no_grad():
                    actual = torch.sigmoid(model(torch.tensor(X))).squeeze(-1).tolist()
                np.testing.assert_array_equal(actual, meta["predictions"])
                # Newly persisted canonical states must retain this same meaning,
                # including the legacy n0/n1 collision, for both consumers.
                with tempfile.NamedTemporaryFile(suffix=".pt") as saved:
                    torch.save(model.state_dict(), saved.name)
                    version = model_spec.GRAPH_PARAMETER_FORMAT
                    current = helper.predict_pytorch_spec(
                        saved.name, X, "prob", encoded, "bce_logits",
                        graph_parameter_format=version)
                    np.testing.assert_array_equal(current, meta["predictions"])
                    cfg["validation-model-path-b64"] = base64.b64encode(
                        saved.name.encode()).decode()
                    cfg["graph-parameter-format"] = version
                    for left, right in zip(arrays, validation.public_model_arrays(cfg)):
                        np.testing.assert_array_equal(left, right)

    def test_saved_graph_loader_rejects_invalid_versions_keys_and_tensor_contracts(self):
        root = PY_ROOT / "tests/fixtures/legacy_graph"
        meta = json.loads((root / "colliding.json").read_text())
        state = torch.load(root / "colliding.pt", weights_only=True)
        spec = meta["model_spec"]
        key = next(iter(state))
        cases = {
            "missing": {k: v for k, v in state.items() if k != key},
            "extra": dict(state, **{"_mods.unknown.weight": state[key]}),
            "dtype": dict(state, **{key: state[key].double()}),
            "nonfinite": dict(state, **{key: torch.full_like(state[key], float("nan"))}),
            "non_tensor": dict(state, **{key: []}),
        }
        for label, candidate in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                builder.load_saved_state_dict(
                    builder.build_from_spec(spec, 2, 1), candidate, spec, 2, 1)
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            builder.load_saved_state_dict(
                builder.build_from_spec(spec, 2, 1),
                dict(state, **{key: state[key][:1]}), spec, 2, 1)
        for version in ("unknown", "", 1, True, []):
            with self.subTest(version=version), self.assertRaises(ValueError):
                builder.load_saved_state_dict(
                    builder.build_from_spec(spec, 2, 1), state, spec, 2, 1,
                    graph_parameter_format=version)

    def test_graph_migration_preserves_private_validation_magnitude_admission(self):
        from dsflower_runner import validation
        root = PY_ROOT / "tests/fixtures/legacy_graph"
        meta = json.loads((root / "colliding.json").read_text())
        state = torch.load(root / "colliding.pt", weights_only=True)
        key = next(iter(state))
        state[key] = torch.full_like(state[key], 1.0e7)
        encoded = base64.b64encode(json.dumps(meta["model_spec"]).encode()).decode()
        X = np.asarray(meta["inputs"], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".pt") as saved:
            torch.save(state, saved.name)
            # Local prediction retains its existing finite-activation semantics;
            # the stricter private-validation transport bound remains enforced.
            actual = helper.predict_pytorch_spec(
                saved.name, X, "prob", encoded, "bce_logits")
            self.assertTrue(np.isfinite(actual).all())
            cfg = {"validation-model-track": "neural", "num-features": 2,
                   "num-classes": 2, "loss-name": "bce_logits",
                   "model-spec-b64": encoded,
                   "validation-model-path-b64": base64.b64encode(
                       saved.name.encode()).decode()}
            with self.assertRaisesRegex(ValueError, "invalid array"):
                validation.public_model_arrays(cfg)

    def test_ordinal_probabilities_are_coherent(self):
        logits = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])
        probs = helper._ordinal_probabilities(logits).numpy()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        self.assertTrue(np.all(probs >= 0.0))

    def test_quantile_prediction_returns_the_direct_conditional_quantile(self):
        logits = torch.tensor([[1.25], [-0.5]])
        self.assertEqual(
            helper._apply_loss_semantics(logits, "quantile", "response"),
            [1.25, -0.5])

    def test_prediction_never_unpickles_arbitrary_checkpoint_code(self):
        spec = {"kind": "sequential", "layers": [
            {"op": "linear", "in": "@in", "out": "@out"},
        ]}
        encoded = base64.b64encode(
            json.dumps(spec, separators=(",", ":")).encode()).decode()
        X = np.asarray([[1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "malicious.pt"
            marker = Path(tmpdir) / "executed"
            torch.save({"state_dict": _MaliciousCheckpoint(str(marker))}, checkpoint)
            with self.assertRaises(Exception):
                helper.predict_pytorch_spec(
                    str(checkpoint), X, "response", encoded,
                    "bce_logits", 2, 2)
            self.assertFalse(marker.exists())

    def test_cli_rejects_artifacts_without_a_declarative_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "newdata.csv"
            data_path.write_text("x\n1\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(PY_ROOT / "predict_helper.py"),
                 "--model", str(Path(tmpdir) / "model.pt"),
                 "--data", str(data_path), "--framework", "pytorch"],
                check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("model_spec and loss_name", result.stderr)

if __name__ == "__main__":
    unittest.main()
