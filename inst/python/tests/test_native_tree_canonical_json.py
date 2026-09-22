"""Real native training output and fail-closed canonical JSON subprocess tests."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "inst" / "python" / "native_tree_canonical_json.py"


class NativeTreeCanonicalJsonTests(unittest.TestCase):
    def canonicalize(self, raw):
        return subprocess.run([sys.executable, "-I", "-S", str(HELPER)],
                              input=raw, capture_output=True, check=False)

    def test_real_training_output_round_trips(self):
        fixture = ROOT / "tests" / "testthat" / "fixtures"
        spec = importlib.util.spec_from_file_location(
            "training_fixture", fixture / "generate-native-tree-training.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        artifact, _profile = module.training_release()
        result = self.canonicalize(artifact)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, artifact)
        self.assertEqual(artifact, (fixture / "native-tree-training.json").read_bytes())

    def test_invalid_json_fails_closed(self):
        for raw in (b'', b'{"x":1,"x":2}', b'{"x":NaN}',
                    b'{"x":Infinity}', b'{"x":1e999}', b'{} trailing'):
            with self.subTest(raw=raw):
                result = self.canonicalize(raw)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b'')

    def test_noncanonical_bytes_are_not_reproduced(self):
        for raw in (b'{"z":1,"a":2}', b'{}\n', b'{ "x":0.123456789012345670}'):
            with self.subTest(raw=raw):
                result = self.canonicalize(raw)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotEqual(result.stdout, raw)


if __name__ == "__main__":
    unittest.main()
