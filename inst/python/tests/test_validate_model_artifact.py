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
