"""Real stdlib-only subprocess tests for every native-tree local predictor."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
FLOWER_APP = ROOT / "inst" / "flower_app"
HELPER = ROOT / "inst" / "python" / "native_tree_predict_helper.py"
sys.path.insert(0, str(FLOWER_APP))

from dsflower_runner import native_tree_runtime_probe  # noqa: E402


class NativeTreePredictHelperTests(unittest.TestCase):
    def test_every_engine_reopens_and_predicts_under_isolated_stdlib(self):
        for engine in (
                "xgboost", "extra_trees", "random_forest", "lightgbm",
                "catboost"):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as root:
                release = native_tree_runtime_probe.synthetic_release(engine)
                root = Path(root)
                model = root / "model.json"
                profile = root / "profile.json"
                data = root / "data.csv"
                output = root / "predictions.json"
                model.write_bytes(release["artifact"])
                profile.write_bytes(release["profile"])
                with data.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(release["request"]["public_schema"]["features"])
                    writer.writerows(((-0.75,), (0.75,)))

                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(root / "untrusted")
                probe = subprocess.run(
                    [sys.executable, "-I", "-S", str(HELPER),
                     "--model", str(model), "--profile", str(profile),
                     "--probe"],
                    env=environment, capture_output=True, text=True,
                    check=False)
                self.assertEqual(probe.returncode, 0, probe.stderr)
                self.assertEqual(json.loads(probe.stdout), {
                    "engine": engine, "features": 1, "status": "available",
                    "task": "binary",
                })

                prediction = subprocess.run(
                    [sys.executable, "-I", "-S", str(HELPER),
                     "--model", str(model), "--profile", str(profile),
                     "--data", str(data), "--output", str(output),
                     "--type", "prob", "--expected-rows", "2"],
                    env=environment, capture_output=True, text=True,
                    check=False)
                self.assertEqual(prediction.returncode, 0, prediction.stderr)
                self.assertEqual(prediction.stdout, "ok")
                payload = json.loads(output.read_text(encoding="ascii"))
                self.assertEqual(payload["engine"], engine)
                self.assertEqual(payload["task"], "binary")
                self.assertEqual(payload["type"], "prob")
                self.assertEqual(len(payload["predictions"]), 2)


if __name__ == "__main__":
    unittest.main()
