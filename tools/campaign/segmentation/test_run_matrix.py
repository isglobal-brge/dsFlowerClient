import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_matrix


class MatrixArtifactTests(unittest.TestCase):
    def run_fixture(self, root, bad_output=False, single_cell=False):
        logs = root / "logs"
        logs.mkdir()
        gates = root / "gates.json"
        gates.write_text(json.dumps({f"segmentation_6_1_{i}": True for i in range(1, 8)}))
        cell = ("breast", "full", 8, 20260919)
        work = root / "runs-batch16" / "breast-full-eps8-seed20260919"
        calls = []
        extra = ["--cell", "breast-full-eps8-seed20260919"] if single_cell else []

        def run(command, **kwargs):
            status = json.loads((work / "execution-status.json").read_text())
            self.assertEqual(status["status"], "running")
            self.assertEqual(status["phase"], ("federation", "channel_b", "twins")[len(calls)])
            calls.append(command)
            if len(calls) == 1:
                output = work / "artifact" / "generated-run"
                output.mkdir(parents=True)
                (output / "model.pt").write_bytes(b"synthetic-unit-test-model")
                status = {"output_dir": str(root / "another-cell" if bad_output else output)}
                (work / "federation-status.json").write_text(json.dumps(status))
            return SimpleNamespace(returncode=0)

        def path(value):
            return logs if value == "/workspace/logs" else Path(value)

        with patch("run_matrix.Path", side_effect=path), patch("run_matrix.planned_cells", return_value=[cell]), \
             patch("run_matrix.subprocess.run", side_effect=run), \
             patch.dict("os.environ", {"F_SEG_GATES_JSON": str(gates)}), \
             patch("sys.argv", ["run_matrix.py", "--root", str(root), "--workers", "1"] + extra), \
             contextlib.redirect_stdout(io.StringIO()):
            if bad_output:
                with self.assertRaisesRegex(SystemExit, "Matrix includes failures"):
                    run_matrix.main()
            else:
                run_matrix.main()
        return calls, json.loads((work / "execution-status.json").read_text()), work

    def test_scoring_receives_recorded_nested_federation_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls, status, work = self.run_fixture(Path(temporary))
            self.assertEqual(len(calls), 3)
            score = calls[1]
            self.assertEqual(score[score.index("--artifact") + 1],
                             str((work / "artifact" / "generated-run" / "model.pt").resolve()))
            self.assertEqual(status["status"], "executed")

    def test_exact_cell_filter_preserves_all_three_phases(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls, status, _ = self.run_fixture(Path(temporary), single_cell=True)
            self.assertEqual(len(calls), 3)
            self.assertEqual(status["status"], "executed")

    def test_path_resolution_failure_is_preserved_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls, status, _ = self.run_fixture(Path(temporary), bad_output=True)
            self.assertEqual(len(calls), 1)
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["phase"], "channel_b")
            self.assertIn("outside this campaign cell", status["error"])


if __name__ == "__main__":
    unittest.main()
