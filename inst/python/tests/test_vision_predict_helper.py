"""Vision-only local prediction tests with no backbone downloads."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock
import unittest

import numpy as np


HELPER_PATH = Path(__file__).resolve().parents[1] / "predict_helper.py"
SPEC = importlib.util.spec_from_file_location(
    "dsflower_vision_predict_helper", str(HELPER_PATH))
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class VisionPredictionTests(unittest.TestCase):
    CONFIG = {
        "backbone": "fixture", "vision-extractor-profile": "v1",
        "num-features": 1, "image-size": 32,
        "loss-name": "cross_entropy", "num-classes": 2, "num-labels": 2,
    }

    def test_cli_reads_bounded_vision_inputs_before_public_preflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "config.json"
            paths = Path(tmpdir) / "paths.json"
            config.write_text("{}", encoding="utf-8")
            # Reject the private-input shape before importing the optional
            # PyTorch vision runtime; this CI job intentionally lacks it.
            paths.write_text("{}", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(HELPER_PATH), "--model", "unused.pt",
                 "--data", str(paths), "--framework", "pytorch_vision",
                 "--config", str(config)],
                check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("error", json.loads(result.stderr))

    def test_preflights_then_batches_2d_and_3d_paths_in_order(self):
        for is_3d in (False, True):
            with self.subTest(is_3d=is_3d):
                calls, chunks = [], []
                validation = SimpleNamespace(
                    public_model_arrays=lambda config: (
                        calls.append("artifact") or
                        [np.zeros(1, dtype=np.float32)]),
                    neural_predictions=lambda model, features, loss:
                        np.column_stack([1.0 - features[:, 0], features[:, 0]]))
                def extract(encoder, paths, image_size, actual_3d, device):
                    self.assertIs(actual_3d, is_3d)
                    calls.append("paths")
                    chunks.append(list(paths))
                    return np.asarray(
                        [int(path) % 2 for path in paths], dtype=np.float32)[:, None]
                vision = SimpleNamespace(
                    prepare_backbone=lambda *args: (
                        calls.append("extractor") or
                        ("encoder", 32, is_3d, "cpu")),
                    extract_features_from_paths=extract)
                model_spec = SimpleNamespace(
                    read_spec=lambda config: calls.append("spec") or {},
                    output_width=lambda *args: 2,
                    output_limit_for_loss=lambda loss: 30.0,
                    build_from_spec=lambda *args, **kwargs: object())
                params = SimpleNamespace(
                    set_torch_params=lambda *args: calls.append("head"))
                with mock.patch.object(
                        helper, "_load_vision_runner_modules",
                        return_value=(validation, vision, model_spec, params)), \
                        mock.patch.object(helper, "_VISION_PREDICT_BATCH_ROWS", 2):
                    got = helper.predict_pytorch_vision(
                        self.CONFIG, ["0", "1", "2", "3", "4"], "response")
                self.assertEqual(got, [0, 1, 0, 1, 0])
                self.assertEqual(chunks, [["0", "1"], ["2", "3"], ["4"]])
                self.assertEqual(calls, [
                    "artifact", "extractor", "spec", "head",
                    "paths", "paths", "paths"])

    def test_failed_public_preflight_never_reaches_private_paths(self):
        touched = []
        unreachable = SimpleNamespace(
            prepare_backbone=lambda *args: touched.append("extractor"),
            extract_features_from_paths=lambda *args: touched.append("paths"))
        invalid_artifact = SimpleNamespace(
            public_model_arrays=mock.Mock(side_effect=ValueError("artifact")))
        with mock.patch.object(
                helper, "_load_vision_runner_modules",
                return_value=(invalid_artifact, unreachable, None, None)):
            with self.assertRaisesRegex(ValueError, "artifact"):
                helper.predict_pytorch_vision({}, ["private-image"], "prob")
        self.assertEqual(touched, [])

        invalid_extractor = SimpleNamespace(
            prepare_backbone=mock.Mock(side_effect=ValueError("extractor")),
            extract_features_from_paths=lambda *args: touched.append("paths"))
        valid_artifact = SimpleNamespace(
            public_model_arrays=lambda config: [np.zeros(1)])
        with mock.patch.object(
                helper, "_load_vision_runner_modules",
                return_value=(valid_artifact, invalid_extractor, None, None)):
            with self.assertRaisesRegex(ValueError, "extractor"):
                helper.predict_pytorch_vision({}, ["private-image"], "prob")
        self.assertEqual(touched, [])


if __name__ == "__main__":
    unittest.main()
