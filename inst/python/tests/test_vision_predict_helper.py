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


class TrainableVisionPredictionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") and
                         importlib.util.find_spec("torchvision"),
                         "full vision runtime is optional")
    def test_full_state_reopens_in_both_modes_without_public_checkpoint(self):
        import base64
        import hashlib
        import torch
        from PIL import Image

        validation, vision, _spec, _params = helper._load_vision_runner_modules()
        from dsflower_runner import initialisation
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        config = {
            "validation-model-track": "neural", "data-kind": "image",
            "backbone": "resnet18", "backbone-mode": "scratch",
            "vision-extractor-profile": vision.extractor_profile_for("resnet18"),
            "num-features": 512, "image-size": 32, "num-classes": 2,
            "num-labels": 2, "loss-name": "cross_entropy",
            "validation-task": "binary", "validation-artifact-format": "pytorch-state-dict-v1",
            "model-spec-b64": base64.b64encode(json.dumps(
                {"layers": [{"op": "linear"}]}).encode()).decode(),
        }
        model = initialisation.build_initial_model(config).eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact, path = Path(tmpdir) / "model.pt", Path(tmpdir) / "image.png"
            torch.save(model.state_dict(), artifact)
            Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(path)
            config.update({
                "validation-model-path-b64": base64.b64encode(str(artifact).encode()).decode(),
                "validation-artifact-size-bytes": artifact.stat().st_size,
                "validation-artifact-sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            })
            with torch.no_grad():
                expected = torch.softmax(model(torch.from_numpy(
                    vision.read_image_2d(str(path), 32)[None])), dim=1).numpy()
            for mode in ("scratch", "finetune"):
                with self.subTest(mode=mode), mock.patch.object(
                        vision, "verified_backbone_bytes",
                        side_effect=AssertionError("released full model needs no public weights")):
                    config["backbone-mode"] = mode
                    got = helper.predict_pytorch_vision(config, [str(path)], "prob")
                    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)
            # Trainable inference matches training's bounded feature map.
            overflow = np.full((1, 512), np.inf, dtype=np.float32)
            overflow[0, :2] = [np.nan, -np.inf]
            bounded = np.clip(np.nan_to_num(overflow, nan=0.0,
                                           posinf=1e6, neginf=-1e6), -1e6, 1e6)
            expected_overflow = validation.neural_predictions(
                model.head, bounded, "cross_entropy")
            with mock.patch.object(vision, "extract_features_from_paths", return_value=overflow):
                got = helper.predict_pytorch_vision(config, [str(path)], "prob")
                np.testing.assert_allclose(got, expected_overflow)
            # Exact state schema remains mandatory even for a correctly pinned file.
            state = model.state_dict()
            original = state[next(iter(state))]
            state[next(iter(state))] = original.reshape(-1)
            torch.save(state, artifact)
            config["validation-artifact-size-bytes"] = artifact.stat().st_size
            config["validation-artifact-sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                helper.load_trainable_vision_model(config, validation, vision)
            config["validation-artifact-sha256"] = "0" * 64
            with mock.patch.object(vision, "read_image_2d",
                                   side_effect=AssertionError("must not read an image")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    helper.predict_pytorch_vision(config, [str(path)], "prob")


if __name__ == "__main__":
    unittest.main()
