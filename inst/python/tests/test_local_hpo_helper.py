"""Executable contracts for the ephemeral local Optuna bridge."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[1] / "local_hpo_helper.py"
HAS_OPTUNA = importlib.util.find_spec("optuna") is not None
if HAS_OPTUNA:
    import optuna
    HAS_OPTUNA = optuna.__version__ == "4.8.0"


@unittest.skipUnless(HAS_OPTUNA, "Optuna 4.8.0 is not installed")
class LocalHpoHelperTests(unittest.TestCase):
    def _study(self, cwd):
        process = subprocess.Popen(
            [sys.executable, "-u", "-I", "-X", "utf8", str(HELPER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=cwd,
        )
        request = {
            "protocol": "dsflower-local-hpo-v1",
            "direction": "minimize",
            "seed": 19,
            "n_trials": 12,
            "space": {
                "depth": {
                    "type": "integer", "lower": 2, "upper": 8,
                    "log": False, "step": 1,
                },
                "rate": {
                    "type": "float", "lower": 0.01, "upper": 1.0,
                    "log": True, "step": None,
                },
                "engine": {
                    "type": "categorical",
                    "values": ["xgboost", "random_forest"],
                },
            },
        }
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        trials = []
        for number in range(request["n_trials"]):
            event = json.loads(process.stdout.readline())
            self.assertEqual(event["event"], "trial")
            self.assertEqual(event["number"], number)
            params = event["params"]
            value = (params["rate"] - 0.2) ** 2 + (params["depth"] - 4) ** 2
            trials.append((params, value))
            process.stdin.write(json.dumps({
                "event": "value", "number": number, "value": value,
            }) + "\n")
            process.stdin.flush()
        complete = json.loads(process.stdout.readline())
        process.stdin.close()
        stderr = process.stderr.read()
        self.assertEqual(process.wait(timeout=5), 0, stderr)
        process.stdout.close()
        process.stderr.close()
        self.assertEqual(complete["event"], "complete")
        self.assertEqual(complete["optuna_version"], "4.8.0")
        return trials, complete

    def test_seeded_study_is_reproducible_and_creates_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._study(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            second = self._study(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(first, second)

    def test_storage_configuration_is_not_part_of_the_protocol(self):
        request = {
            "protocol": "dsflower-local-hpo-v1",
            "direction": "minimize", "seed": 0, "n_trials": 1,
            "space": {"x": {
                "type": "float", "lower": 0.0, "upper": 1.0,
                "log": False, "step": None,
            }},
            "storage": "somewhere",
        }
        result = subprocess.run(
            [sys.executable, "-u", "-I", "-X", "utf8", str(HELPER)],
            input=json.dumps(request) + "\n", capture_output=True,
            text=True, check=False,
        )
        event = json.loads(result.stdout)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(event["event"], "error")
        self.assertIn("exactly", event["message"])

    def test_trial_ceiling_is_checked_before_study_creation(self):
        request = {
            "protocol": "dsflower-local-hpo-v1",
            "direction": "minimize", "seed": 0, "n_trials": 1_000_001,
            "space": {"x": {
                "type": "float", "lower": 0.0, "upper": 1.0,
                "log": False, "step": None,
            }},
        }
        result = subprocess.run(
            [sys.executable, "-u", "-I", "-X", "utf8", str(HELPER)],
            input=json.dumps(request) + "\n", capture_output=True,
            text=True, check=False,
        )
        event = json.loads(result.stdout)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(event["event"], "error")
        self.assertIn("integer range", event["message"])


if __name__ == "__main__":
    unittest.main()
