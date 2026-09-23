"""Dependency-free tests for the saved neural artifact preflight helper."""

from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


HELPER_PATH = Path(__file__).resolve().parents[1] / "validate_model_artifact.py"
SPEC = importlib.util.spec_from_file_location(
    "_dsflower_validate_model_artifact", HELPER_PATH)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class ValidateModelArtifactTests(unittest.TestCase):
    @staticmethod
    def _invoke(config, public_model_arrays, prepare_backbone):
        runner = ModuleType("dsflower_runner")
        runner.validation = SimpleNamespace(
            public_model_arrays=public_model_arrays)
        runner.vision = SimpleNamespace(prepare_backbone=prepare_backbone)
        with tempfile.TemporaryDirectory() as directory:
            contract = Path(directory) / "contract.json"
            contract.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.dict(sys.modules, {"dsflower_runner": runner}), \
                    mock.patch.object(
                        sys, "argv", [str(HELPER_PATH), str(contract)]), \
                    redirect_stderr(io.StringIO()) as stderr:
                status = helper._entrypoint()
        return status, stderr.getvalue()

    def test_image_loads_public_arrays_then_preflights_extractor(self):
        events = []
        public = mock.Mock(side_effect=lambda _cfg: events.append("arrays") or [1])
        preflight = mock.Mock(
            side_effect=lambda *_args: events.append("extractor")
            or (object(), 128, True, "cpu"))
        config = {
            "data-kind": "image", "backbone": "densenet121_3d",
            "vision-extractor-profile":
                "dsflower-densenet121-monai-seed0-extractor-v1",
            "num-features": 1024, "image-size": 128,
        }

        status, stderr = self._invoke(config, public, preflight)

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(events, ["arrays", "extractor"])
        public.assert_called_once_with(config)
        preflight.assert_called_once_with(
            "densenet121_3d",
            "dsflower-densenet121-monai-seed0-extractor-v1", 1024, 128)

    def test_tabular_artifact_does_not_construct_an_extractor(self):
        public = mock.Mock(return_value=[1])
        preflight = mock.Mock(side_effect=AssertionError("vision preflight"))

        status, stderr = self._invoke(
            {"num-features": 2}, public, preflight)

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        public.assert_called_once()
        preflight.assert_not_called()

    def test_extractor_failure_returns_nonzero_status(self):
        public = mock.Mock(return_value=[1])
        preflight = mock.Mock(side_effect=RuntimeError("missing MONAI"))

        status, stderr = self._invoke({
            "data-kind": "image", "backbone": "densenet121_3d",
            "vision-extractor-profile":
                "dsflower-densenet121-monai-seed0-extractor-v1",
            "num-features": 1024, "image-size": 128,
        }, public, preflight)

        self.assertEqual(status, 2)
        self.assertIn("invalid validation artifact", stderr)
        public.assert_called_once()
        preflight.assert_called_once()


if __name__ == "__main__":
    unittest.main()


def test_real_saved_segmentation_and_survival_artifact_preflight(tmp_path):
    import base64
    import hashlib
    import subprocess
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "flower_app"))
    from dsflower_runner import model_spec, segmentation

    for loss in ("segmentation_bce_dice", "aft_weibull_nll", "discrete_hazard_nll"):
        config = {"validation-model-track": "neural", "num-classes": 2,
                  "num-labels": 2, "loss-name": loss}
        if loss == "segmentation_bce_dice":
            spec = segmentation.decoder_spec("pointwise")
            config.update({"validation-task": "segmentation", "task-type": "segmentation",
                "data-kind": "image", "backbone": segmentation.BACKBONE,
                "num-features": segmentation.FEATURE_DIM, "image-size": 128,
                "vision-extractor-profile": segmentation.PROFILE,
                "segmentation-alpha": 1., "segmentation-smooth": 1.,
                "mask-vocabulary": "0,255", "segmentation-selection": segmentation.SELECTION,
                "segmentation-preprocessing": segmentation.PREPROCESSING,
                "segmentation-checkpoint-sha256": segmentation.CHECKPOINT_SHA256,
                "segmentation-output-shape": "1,128,128"})
            output = {"output_shape": segmentation.OUTPUT_SHAPE}
        else:
            spec = {"kind": "sequential", "layers": [{"op": "linear", "out": "@out"}]}
            survival = {"schema_version": 1, "time_unit": "days", "time_origin": "baseline",
                        "t_min": 1, "horizon": 10}
            survival.update({"edges": [0, 5, 10]} if loss == "discrete_hazard_nll" else
                            {"time_scale": 1, "distribution": "weibull", "dispersion": 1})
            config.update({"validation-task": "survival", "task-type": "survival", "num-features": 2,
                           "survival-config-b64": base64.b64encode(json.dumps(survival).encode()).decode()})
            output = {}
        config["model-spec-b64"] = base64.b64encode(json.dumps(spec).encode()).decode()
        model = model_spec.build_from_spec(spec, config["num-features"],
            model_spec.output_width(loss, config), num_labels=2,
            output_limit=model_spec.output_limit_for_loss(loss), **output)
        artifact = tmp_path / (loss + ".pt")
        torch.save(model.state_dict(), artifact)
        config["validation-model-path-b64"] = base64.b64encode(str(artifact).encode()).decode()
        if loss == "segmentation_bce_dice":
            config.update({"validation-artifact-format": "pytorch-state-dict-v1",
                           "validation-artifact-sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                           "validation-artifact-size-bytes": artifact.stat().st_size})
        request = tmp_path / "request.json"
        request.write_text(json.dumps(config))
        result = subprocess.run([sys.executable, str(HELPER_PATH), str(request)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        config["num-features"] += 1
        request.write_text(json.dumps(config))
        rejected = subprocess.run([sys.executable, str(HELPER_PATH), str(request)], capture_output=True, text=True)
        assert rejected.returncode != 0
