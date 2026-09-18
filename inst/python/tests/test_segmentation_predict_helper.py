"""Segmentation local reconstruction tests; no checkpoint downloads required."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "flower_app"))
from dsflower_runner import model_spec, segmentation

SPEC = importlib.util.spec_from_file_location(
    "segmentation_predict_helper", ROOT / "python" / "predict_helper.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
torch.set_num_threads(1)


class SegmentationPredictionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.pt"
        model = model_spec.build_from_spec(
            segmentation.decoder_spec(), 32768, 1, output_shape=(1, 128, 128))
        for parameter in model.parameters():
            parameter.data.zero_()
        torch.save(model.state_dict(), self.path)
        self.cfg = {
            "task-type": "segmentation", "data-kind": "image",
            "loss-name": "segmentation_bce_dice", "num-features": 32768,
            "num-classes": 2, "image-size": 128, "backbone": segmentation.BACKBONE,
            "vision-extractor-profile": segmentation.PROFILE,
            "segmentation-selection": segmentation.SELECTION,
            "segmentation-preprocessing": segmentation.PREPROCESSING,
            "segmentation-checkpoint-sha256": segmentation.CHECKPOINT_SHA256,
            "segmentation-output-shape": "1,128,128",
            "segmentation-alpha": .5, "segmentation-smooth": 1,
            "mask-vocabulary": "0,255",
            "model-spec-b64": base64.b64encode(json.dumps(
                segmentation.decoder_spec()).encode()).decode(),
            "validation-model-path-b64": base64.b64encode(str(self.path).encode()).decode(),
            "validation-artifact-format": "pytorch-state-dict-v1",
            "validation-artifact-size-bytes": self.path.stat().st_size,
            "validation-artifact-sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
        }

    def test_probability_and_fixed_half_threshold(self):
        with mock.patch.object(segmentation, "prepare_encoder", return_value=(None, "cpu")), \
             mock.patch.object(segmentation, "extract_prediction_features", side_effect=
                 lambda encoder, paths, device: np.zeros((len(paths), 32768), np.float32)):
            probabilities = helper.predict_pytorch_vision(self.cfg, ["first.png", "second.png"], "prob")
            responses = helper.predict_pytorch_vision(self.cfg, ["first.png"], "response")
        self.assertEqual(np.asarray(probabilities).shape, (2, 16384))
        np.testing.assert_equal(probabilities, np.full((2, 16384), .5))
        np.testing.assert_equal(responses, np.ones((1, 16384), int))

    def test_encoder_preflight_before_input_reads_and_batches_bound_memory(self):
        events = []
        def prepare(config):
            self.assertFalse(any(key.startswith("validation-") for key in config))
            segmentation.validate_config(config)
            events.append("encoder")
            return None, "cpu"
        def extract(encoder, paths, device):
            events.append(list(paths))
            return np.zeros((len(paths), 32768), np.float32)
        with mock.patch.object(segmentation, "prepare_encoder", side_effect=prepare), \
             mock.patch.object(segmentation, "extract_prediction_features", side_effect=extract):
            result = helper.predict_pytorch_vision(self.cfg, [str(i) + ".png" for i in range(17)], "response")
        self.assertEqual(len(result), 17)
        self.assertEqual(events[0], "encoder")
        self.assertEqual([len(chunk) for chunk in events[1:]], [16, 1])

    def test_malformed_profile_and_artifact_never_read_paths(self):
        for field, value in (("segmentation-checkpoint-sha256", "0" * 64),
                             ("validation-artifact-sha256", "0" * 64),
                             ("segmentation-output-shape", "128,128"),
                             ("validation-artifact-size-bytes", 1)):
            with self.subTest(field=field), \
                 mock.patch.object(segmentation, "extract_prediction_features") as extract:
                cfg = dict(self.cfg, **{field: value})
                with self.assertRaises(ValueError):
                    helper.predict_pytorch_vision(cfg, ["secret/path.png"], "prob")
                extract.assert_not_called()

    def test_unavailable_encoder_never_reads_paths(self):
        with mock.patch.object(segmentation, "prepare_encoder", side_effect=ValueError("missing checkpoint")), \
             mock.patch.object(segmentation, "extract_prediction_features") as extract:
            with self.assertRaisesRegex(ValueError, "missing checkpoint"):
                helper.predict_pytorch_vision(self.cfg, ["secret/path.png"], "prob")
            extract.assert_not_called()

    def test_pixel_output_bound_precedes_runner_loading(self):
        with mock.patch.object(helper, "_load_segmentation_runner") as load:
            with self.assertRaisesRegex(ValueError, "cell ceiling"):
                helper.predict_pytorch_vision(self.cfg, ["image.png"] * 123, "prob")
            load.assert_not_called()

    def test_wrong_decoder_is_rejected_before_paths(self):
        cfg = dict(self.cfg)
        cfg["model-spec-b64"] = base64.b64encode(json.dumps({
            "kind": "sequential", "layers": [{"op": "linear", "out": "@out"}]
        }).encode()).decode()
        with mock.patch.object(segmentation, "extract_prediction_features") as extract:
            with self.assertRaisesRegex(ValueError, "pinned convolutional decoder"):
                helper.predict_pytorch_vision(cfg, ["secret/path.png"], "prob")
            extract.assert_not_called()

    def test_declarative_preflight_uses_spatial_output_only_for_trusted_loss(self):
        payload = {"spec": segmentation.decoder_spec(), "loss_name": "segmentation_bce_dice",
                   "input_dim": 32768, "num_classes": 2, "num_labels": 2}
        path = Path(self.tmp.name) / "contract.json"
        path.write_text(json.dumps(payload))
        command = [sys.executable, str(ROOT / "python" / "validate_model_spec.py"), str(path)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload["loss_name"] = "bce_logits"
        path.write_text(json.dumps(payload))
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid declarative model", result.stderr)


if __name__ == "__main__":
    unittest.main()
