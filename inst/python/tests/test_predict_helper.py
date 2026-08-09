"""Prediction parity tests for saved declarative and DP-GBDT artifacts."""

import base64
import importlib.util
import json
import os
from pathlib import Path
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
                handle.name, X, "prob", encoded, "cross_entropy", 2, 2)
        with torch.no_grad():
            expected = torch.softmax(model(torch.tensor(X)), dim=-1).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)

    def test_ordinal_probabilities_are_coherent(self):
        logits = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])
        probs = helper._ordinal_probabilities(logits).numpy()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        self.assertTrue(np.all(probs >= 0.0))

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

    def test_tree_bounds_repeat_training_raw_domain_semantics(self):
        bounds = base64.b64encode(json.dumps(
            {"lower": [10.0, -4.0], "upper": [20.0, 8.0]}).encode()).decode()
        X = np.asarray([[np.nan, np.inf], [30.0, -20.0]])
        got = helper._apply_tree_bounds(X, bounds)
        np.testing.assert_allclose(
            got, np.asarray([[15.0, 2.0], [30.0, -20.0]], dtype=np.float32))

    def test_bounded_regression_booster_returns_clipped_margin(self):
        booster = {
            "objective": "reg:squarederror", "depth": 1,
            "base_margin": 5.0, "margin_bounds": [0.0, 10.0],
            "trees": [{"feat": [0], "thr": [0.0], "w": [-20.0, 20.0]}],
        }
        X = np.asarray([[-1.0], [1.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as handle:
            json.dump(booster, handle)
            handle.flush()
            got = helper.predict_dpgbdt(handle.name, X, "response")
        self.assertEqual(got, [0.0, 10.0])


if __name__ == "__main__":
    unittest.main()
