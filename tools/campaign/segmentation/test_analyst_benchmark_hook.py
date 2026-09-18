import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
RUNNER_PARENT = TOOLS.parents[2] / "inst/flower_app"


class AnalystBenchmarkHookTests(unittest.TestCase):
    def run_python(self, arguments):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "capture"
            env = dict(os.environ, F_SEG_PUBLIC_BENCHMARK="1", F_SEG_INIT_SEED="20260919",
                       F_SEG_CAPTURE_DIR=str(capture), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       PYTHONPATH=os.pathsep.join((str(TOOLS / "benchmark_hooks"),
                                                 str(RUNNER_PARENT), str(TOOLS))))
            env.pop("DSFLOWER_NODE_SECRET_FILE", None)
            result = subprocess.run([sys.executable, *arguments], env=env,
                                    capture_output=True, text=True, timeout=120)
            return result, sorted(p.name for p in capture.glob("*"))

    def test_python_startup_imports_no_torch_numpy_or_runner(self):
        result, captures = self.run_python(["-c", '''import sys
import os
assert "sitecustomize" in sys.modules
assert all(os.environ[name] == "2" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
assert not any(name.split(".")[0] in {"torch", "numpy", "dsflower_runner"} for name in sys.modules)
print("LAZY_STARTUP_PASS")
'''])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("LAZY_STARTUP_PASS", result.stdout)
        self.assertEqual(captures, [])

    def test_original_initialization_and_actual_dp_smoke_still_passes(self):
        result, captures = self.run_python([str(TOOLS / "smoke_benchmark_hooks.py")])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("exactly two logical accountant steps; six decoder tensors", result.stdout)
        self.assertIn("public-initial-arrays.npz", captures)
        self.assertIn("public-initial.json", captures)
        self.assertEqual(sum(name.startswith("accountant-") for name in captures), 1)


if __name__ == "__main__":
    unittest.main()
