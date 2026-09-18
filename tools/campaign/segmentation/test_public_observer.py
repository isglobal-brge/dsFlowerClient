import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


TOOLS = Path(__file__).resolve().parent
HOOK = TOOLS.parents[3] / "dsFlower/inst/python/sitecustomize.py"
spec = importlib.util.spec_from_file_location(
    "segmentation_public_observer", TOOLS / "benchmark_hooks/segmentation_public_observer.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def package_hash(path):
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*.py")):
        digest.update(child.relative_to(path).as_posix().encode() + b"\n")
        digest.update(child.read_bytes() + b"\0")
    return digest.hexdigest()


class PublicObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.secret_dir = self.root / "state"
        self.secret_dir.mkdir(mode=0o700)
        self.config_path = self.secret_dir / observer.CONFIG_NAME
        self.config = dict(public_fixture_only=True, dataset="synthetic", seed=20260919,
                           capture_dir=str(self.root / "run/public-capture"))
        self.write_config()
        self.env = mock.patch.dict(os.environ, {
            "DSFLOWER_NODE_SECRET_FILE": str(self.secret_dir / "secret")})
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))
        self.config_path.chmod(0o600)

    def test_public_config_has_strict_permissions_scope_and_explicit_opt_in(self):
        self.assertEqual(observer.load_config(self.root), self.config)
        for key, value in [("public_fixture_only", False), ("dataset", "private"),
                           ("seed", True), ("capture_dir", str(self.root.parent / "public-capture"))]:
            original = self.config[key]
            self.config[key] = value
            self.write_config()
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                observer.load_config(self.root)
            self.config[key] = original
        self.write_config()
        self.config_path.chmod(0o644)
        with self.assertRaisesRegex(RuntimeError, "file is unsafe"):
            observer.load_config(self.root)
        self.config_path.chmod(0o600)
        self.secret_dir.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, "directory is unsafe"):
            observer.load_config(self.root)

    def test_symlink_config_is_rejected_and_absent_config_does_nothing(self):
        real = self.secret_dir / "actual.json"
        self.config_path.rename(real)
        self.config_path.symlink_to(real)
        with self.assertRaises(OSError):
            observer.load_config(self.root)
        self.config_path.unlink()
        self.assertIsNone(observer.load_config(self.root))

    def test_unguarded_unpinned_or_wrong_finder_order_cannot_attach(self):
        item = observer.PublicObserver(self.config)
        class GuardFinder:
            pass
        guard_finder = GuardFinder()
        guard = types.SimpleNamespace(_IntegrityFinder=GuardFinder,
            _verified_packages={"dsflower_runner"}, _PINNED_MAP={"dsflower_runner": "pin"},
            _CANONICAL_CLIENTAPP_REF="dsflower_runner.client_app:app")
        with mock.patch.dict(sys.modules, {"sitecustomize": guard}), \
                mock.patch.object(sys, "meta_path", [guard_finder, item]):
            self.assertIs(observer.verified_guard(item), guard)
            guard._verified_packages.clear()
            with self.assertRaises(RuntimeError):
                observer.verified_guard(item)
            guard._verified_packages.add("dsflower_runner")
            guard._PINNED_MAP.clear()
            with self.assertRaises(RuntimeError):
                observer.verified_guard(item)
            guard._PINNED_MAP["dsflower_runner"] = "pin"
            sys.meta_path[:] = [item, guard_finder]
            with self.assertRaises(RuntimeError):
                observer.verified_guard(item)
            sys.meta_path[:] = [item]
            with self.assertRaises(RuntimeError):
                observer.verified_guard(item)

    def test_failure_observer_preserves_reply_and_omits_message_and_locals(self):
        observer.load_config(self.root)
        original = mock.Mock(return_value="unchanged")
        sentinel = "DO_NOT_RECORD_MESSAGE_OR_LOCAL"
        with mock.patch.object(observer, "verified_guard"):
            wrapped = observer.observe_fallback(original, self.config, object())
            try:
                raise ValueError(sentinel)
            except ValueError:
                self.assertEqual(wrapped("public-request", flag=True), "unchanged")
        original.assert_called_once_with("public-request", flag=True)
        files = list(Path(self.config["capture_dir"]).glob("failure-*.json"))
        self.assertEqual(len(files), 1)
        self.assertNotIn(sentinel, files[0].read_text())
        record = json.loads(files[0].read_text())
        self.assertEqual(record["exception_type"], "ValueError")
        self.assertEqual(set(record), {"public_fixture_only", "exception_type", "frames"})

    def run_guarded_import(self, *, corrupt=False, wrong_steps=False):
        package = self.root / "dsflower_runner"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "dp_harness.py").write_text('''from types import SimpleNamespace
class PRVAccountant:
    def __init__(self):
        self.history = [[9., .5, 2]]
def make_private_dpsgd(*args, **kwargs):
    assert kwargs == {"secure_rng": True}
    return None, None, None, SimpleNamespace(accountant=PRVAccountant())
def effective_dpsgd_mechanism(epsilon, delta, clip, n, batch, epochs, rounds):
    assert (epsilon, delta, clip, n, batch, epochs, rounds) == (4, 1e-5, 1, 16, 8, 1, 2)
    return {"steps_per_epoch": 2, "total_steps": 4, "accounting_population": n}
''')
        (package / "client_app.py").write_text('''from . import dp_harness
app = object()
def _safe_fallback_reply(*args, **kwargs):
    return {"unchanged_fallback": True}
def _dp_fit(model, X, y, pcfg, pins, n_staged, cfg, master, noise_multiplier, **kwargs):
    assert master == b"untouched-secret" and noise_multiplier == 9.
    result = dp_harness.make_private_dpsgd(**kwargs)
    return {"original_result": True}
''')
        if wrong_steps:
            path = package / "dp_harness.py"
            path.write_text(path.read_text().replace("[[9., .5, 2]]", "[[9., .5, 1]]"))
        original_hash = package_hash(package)
        manifest = self.root / "manifest"
        manifest.mkdir()
        (manifest / "manifest.json").write_text('{"dp-track":"neural"}')
        (manifest / "pinned_packages.json").write_text(json.dumps({
            "dsflower_runner": "0" * 64 if corrupt else original_hash}))
        script = '''import importlib.util, json, pathlib, sys
import segmentation_public_observer as observer
assert "dsflower_runner" not in sys.modules
config = observer.load_config(pathlib.Path(sys.argv[1]))
item = observer.PublicObserver(config)
sys.meta_path.insert(0, item)
assert "dsflower_runner" not in sys.modules
spec = importlib.util.spec_from_file_location("sitecustomize", sys.argv[2])
guard = importlib.util.module_from_spec(spec)
sys.modules["sitecustomize"] = guard
spec.loader.exec_module(guard)
from dsflower_runner import client_app
assert "dsflower_runner" in guard._verified_packages
assert client_app._dp_fit._segmentation_public_observer
class Tensor:
    def __len__(self): return 16
    def tobytes(self): return b"public-fixture-tensor"
result = client_app._dp_fit(None, Tensor(), Tensor(),
    {"epsilon":4, "delta":1e-5, "clipping_norm":1},
    {"batch_size":8,"local_epochs":1,"num_rounds":2,"round_index":1}, 18,
    {"loss-name":"segmentation_bce_dice"}, b"untouched-secret", 9., secure_rng=True)
assert result == {"original_result": True}
assert guard._hash_package(str(pathlib.Path(sys.argv[1])/"dsflower_runner")) == sys.argv[3]
print("VERIFIED_OBSERVER_PASS")
'''
        env = dict(os.environ, DSFLOWER_MANIFEST_DIR=str(manifest),
                   PYTHONPATH=os.pathsep.join((str(TOOLS / "benchmark_hooks"), str(self.root))))
        env.pop("F_SEG_PUBLIC_BENCHMARK", None)
        result = subprocess.run([sys.executable, "-c", script, str(self.root), str(HOOK), original_hash],
                                env=env, capture_output=True, text=True)
        self.assertEqual(package_hash(package), original_hash)
        return result

    def test_deferred_guarded_observer_captures_actual_steps_and_preserves_return(self):
        result = self.run_guarded_import()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = list(Path(self.config["capture_dir"]).glob("accountant-*.json"))
        self.assertEqual(len(records), 1)
        data = json.loads(records[0].read_text())
        self.assertEqual(data["accountant_history"], [[9., .5, 2]])
        self.assertEqual(data["observed_round_steps"], 2)
        self.assertEqual(data["mechanism"]["total_steps"], 4)
        self.assertEqual(data["source_rows"], 18)
        self.assertEqual(data["features_sha256"], hashlib.sha256(b"public-fixture-tensor").hexdigest())
        self.assertNotIn("untouched-secret", records[0].read_text())

    def test_wrong_accountant_steps_cannot_create_capture(self):
        result = self.run_guarded_import(wrong_steps=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("observed accountant steps do not match", result.stderr)
        self.assertFalse(list(Path(self.config["capture_dir"]).glob("accountant-*.json")))

    def test_mandatory_real_byte_hash_failure_still_exits_before_observer(self):
        result = self.run_guarded_import(corrupt=True)
        self.assertEqual(result.returncode, 99, result.stdout + result.stderr)
        self.assertIn("code hash mismatch", result.stderr)
        self.assertFalse(list(Path(self.config["capture_dir"]).glob("accountant-*.json")))

    def test_installer_refuses_non_campaign_environment(self):
        result = subprocess.run([sys.executable, str(TOOLS / "install_public_observer.py")],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("/workspace/segmentation/venv/bin/python only", result.stderr)


if __name__ == "__main__":
    unittest.main()
