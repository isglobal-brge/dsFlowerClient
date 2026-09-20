import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from test_assemble_evidence import make_cell, write_json
from resume_postprocessing import verify_completed_federation


class PostprocessingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.addCleanup(patch.stopall)
        patch("assemble_evidence.independent_accounting", return_value={}).start()
        make_cell(self.root)
        (self.root / "channel-b.json").unlink()
        shutil.rmtree(self.root / "twins")
        write_json(self.root / "execution-status.json", {"status": "failed", "phase": "federation", "exit_code": 1})
        write_json(self.root / "artifact/generated-run/metadata.json", {"status": "success", "n_clients": 3})
        (self.root / "public-probabilities.csv").write_text("0.5\n")
        patch("assemble_evidence.DEFAULT_PROVENANCE", self.root / "provenance").start()

    def test_accepts_only_completed_federation_for_postprocessing(self):
        key, previous, artifact = verify_completed_federation(self.root, 16)
        self.assertEqual(key, ("breast", "full", 8, 20260919))
        self.assertEqual(previous["status"], "failed")
        self.assertTrue(artifact.is_file())

    def test_rejects_wrong_arm(self):
        with self.assertRaises(ValueError):
            verify_completed_federation(self.root, 64)

    def test_rejects_changed_model(self):
        (self.root / "artifact/generated-run/model.pt").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "mismatched"):
            verify_completed_federation(self.root, 16)

    def test_rejects_unclean_federation(self):
        path = self.root / "federation-status.json"
        value = json.loads(path.read_text()); value["cleanup_ok"] = False
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "cleanup"):
            verify_completed_federation(self.root, 16)

    def test_rejects_any_prior_postprocessing_attempt(self):
        (self.root / "twins").mkdir()
        with self.assertRaisesRegex(ValueError, "already attempted"):
            verify_completed_federation(self.root, 16)

    def test_twin_setup_retry_preserves_completed_channel(self):
        write_json(self.root / "execution-status.json", {"status": "failed", "phase": "twins", "exit_code": 1})
        write_json(self.root / "channel-b.json", {"status": "executed"})
        key, previous, artifact = verify_completed_federation(self.root, 16, retry_twins=True)
        self.assertEqual(previous["phase"], "twins")
        with self.assertRaisesRegex(ValueError, "requested phase"):
            verify_completed_federation(self.root, 16)
        (self.root / "twins").mkdir()
        with self.assertRaisesRegex(ValueError, "already attempted"):
            verify_completed_federation(self.root, 16, retry_twins=True)

    def test_twin_retry_rejects_running_cell_or_missing_channel(self):
        for status, phase in (("running", "twins"), ("executed", "twins"), ("failed", "federation")):
            write_json(self.root / "execution-status.json", {"status": status, "phase": phase})
            with self.assertRaisesRegex(ValueError, "requested phase"):
                verify_completed_federation(self.root, 16, retry_twins=True)
        write_json(self.root / "execution-status.json", {"status": "failed", "phase": "twins"})
        with self.assertRaisesRegex(ValueError, "completed channel B"):
            verify_completed_federation(self.root, 16, retry_twins=True)
