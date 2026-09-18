"""Independent analytic references for released survival artifact prediction."""
import base64
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

PY_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("survival_predict_helper", PY_ROOT / "predict_helper.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
builder = helper._load_model_spec_module()
MODEL_SPEC = {"kind": "sequential", "layers": [{"op": "linear", "out": "@out"}]}
SPEC_B64 = base64.b64encode(json.dumps(MODEL_SPEC).encode()).decode()


def aft(distribution):
    return dict(schema_version=1, time_unit="days", time_origin="baseline",
                t_min=1, horizon=20, time_scale=5,
                distribution=distribution, dispersion=1)


def hazard():
    return dict(schema_version=1, time_unit="days", time_origin="baseline",
                t_min=1, horizon=10, edges=[0, 5, 10])


class SurvivalPredictionTests(unittest.TestCase):
    def fixture(self, directory, loss, config, bias=0):
        width = builder.output_width(loss, {"survival-config": config})
        model = builder.build_from_spec(MODEL_SPEC, 1, width,
                                       output_limit=builder.output_limit_for_loss(loss))
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.bias.fill_(bias)
        path = Path(directory) / "model.pt"
        torch.save(model.state_dict(), path)
        return str(path)

    def test_aft_artifact_curves_medians_and_ranking_match_analytic_references(self):
        x = np.zeros((2, 1), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            for distribution in ("weibull", "lognormal"):
                loss = "aft_" + distribution + "_nll"
                cfg = aft(distribution)
                path = self.fixture(directory, loss, cfg)
                got = helper.predict_pytorch_spec(path, x, "survival", SPEC_B64,
                        loss, survival_config=cfg, times=[0, 5, 10])
                expected = ([1, math.exp(-1), math.exp(-2)] if distribution == "weibull"
                            else [1, .5, .5*math.erfc(math.log(2)/math.sqrt(2))])
                np.testing.assert_allclose(got["survival"], [expected, expected])
                median = helper.predict_pytorch_spec(path, x, "median", SPEC_B64,
                            loss, survival_config=cfg)
                np.testing.assert_allclose(median, [5*math.log(2)]*2 if distribution == "weibull" else [5]*2)
                risk = helper.predict_pytorch_spec(path, x, "risk", SPEC_B64,
                            loss, survival_config=cfg)
                np.testing.assert_array_equal(risk, [0, 0])

    def test_hazard_grid_interpolation_and_restricted_mean_risk(self):
        cfg = hazard()
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory, "discrete_hazard_nll", cfg)
            x = np.zeros((1, 1), dtype=np.float32)
            got = helper.predict_pytorch_spec(path, x, "survival", SPEC_B64,
                "discrete_hazard_nll", survival_config=cfg, times=[0, 4, 5, 9, 10])
            np.testing.assert_allclose(got["survival"], [[1, 1, .5, .5, .25]])
            risk = helper.predict_pytorch_spec(path, x, "risk", SPEC_B64,
                "discrete_hazard_nll", survival_config=cfg)
            self.assertEqual(risk, [-7.5])
            median = helper.predict_pytorch_spec(path, x, "response", SPEC_B64,
                "discrete_hazard_nll", survival_config=cfg)
            self.assertEqual(median, [5])
            path = self.fixture(directory, "discrete_hazard_nll", cfg, bias=-10)
            median = helper.predict_pytorch_spec(path, x, "median", SPEC_B64,
                "discrete_hazard_nll", survival_config=cfg)
            self.assertEqual(median, [None])
            self.assertEqual(json.dumps(median, allow_nan=False), "[null]")

    def test_missing_or_mismatched_survival_metadata_fails_closed(self):
        cfg = aft("weibull")
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory, "aft_weibull_nll", cfg)
            with self.assertRaisesRegex(ValueError, "saved public configuration"):
                helper.predict_pytorch_spec(path, np.zeros((1,1)), "response", SPEC_B64,
                                           "aft_weibull_nll")
            with self.assertRaisesRegex(ValueError, "does not match"):
                helper.predict_pytorch_spec(path, np.zeros((1,1)), "response", SPEC_B64,
                    "aft_lognormal_nll", survival_config=cfg)

    def test_public_hazard_architecture_preflight_accepts_grid_width(self):
        payload = dict(spec=MODEL_SPEC, loss_name="discrete_hazard_nll", input_dim=1,
                       num_classes=2, num_labels=2, survival_config=hazard())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run([sys.executable, str(PY_ROOT / "validate_model_spec.py"), str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
