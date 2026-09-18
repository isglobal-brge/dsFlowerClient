import base64
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "inst/flower_app"))
from dsflower_runner import segmentation, task
from central_twins import validate_twin_pins, private_key, source_row_counts


def fixture_config():
    return {"task-type": "segmentation", "loss-name": "segmentation_bce_dice",
        "data-kind": "image", "backbone": segmentation.BACKBONE,
        "vision-extractor-profile": segmentation.PROFILE,
        "num-features": segmentation.FEATURE_DIM, "image-size": 128, "num-classes": 2,
        "segmentation-selection": segmentation.SELECTION,
        "segmentation-preprocessing": segmentation.PREPROCESSING,
        "segmentation-checkpoint-sha256": segmentation.CHECKPOINT_SHA256,
        "segmentation-output-shape": "1,128,128", "segmentation-alpha": .5,
        "segmentation-smooth": 1., "mask-vocabulary": "0,255", "dp-unit": "patient",
        "model-spec-b64": base64.b64encode(json.dumps(segmentation.decoder_spec()).encode()).decode(),
        "batch-size": 16, "local-epochs": 2, "num-server-rounds": 5,
        "learning-rate": .01, "optimizer-name": "sgd", "scheduler-name": "none"}


class TwinPinTests(unittest.TestCase):
    def setUp(self):
        self.cfg = fixture_config()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "manifest.json").write_text(json.dumps(self.cfg))
            self.pins = task.load_run_pins(SimpleNamespace(node_config={"manifest-dir": temporary}))

    def test_primary_and_bce_share_identical_feature_semantics(self):
        validate_twin_pins(self.cfg, self.cfg, self.pins)
        bce = dict(self.cfg, **{"segmentation-alpha": 1.})
        validate_twin_pins(bce, self.cfg, self.pins)

    def test_batch64_shares_features_but_requires_matching_effective_pins(self):
        cfg = dict(self.cfg, **{"batch-size": 64})
        pins = dict(self.pins, batch_size=64)
        validate_twin_pins(cfg, self.cfg, pins, 64)
        with self.assertRaises(ValueError):
            validate_twin_pins(cfg, self.cfg, self.pins, 64)
        with self.assertRaises(ValueError):
            validate_twin_pins(cfg, self.cfg, pins, 16)

    def test_source_census_counts_all_images_and_duplicate_mask_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.csv"
            path.write_text("subject_id,image_id,mask_path\na,i1,m1\na,i1,m2\na,i2,m3\nb,i3,m4\nc,i4,m5\n")
            self.assertEqual(source_row_counts(path, [["a"], ["c", "b"]]), [3, 2])
            with self.assertRaisesRegex(ValueError, "absent from cached source rows"):
                source_row_counts(path, [["missing"]])

    def test_rejects_captured_or_cached_optimizer_and_scheduler_drift(self):
        for key, value in (("optimizer-momentum", .1), ("weight-decay", .1),
                           ("l1-penalty", .1), ("learning-rate", .02),
                           ("scheduler-name", "step"), ("optimizer-nesterov", True)):
            bad = dict(self.cfg, **{key: value})
            for cfg, manifest in ((bad, self.cfg), (self.cfg, bad)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_twin_pins(cfg, manifest, self.pins)
        pins = copy.deepcopy(self.pins)
        pins["optimizer"]["weight_decay"] = .1
        with self.assertRaisesRegex(ValueError, "effective twin optimizer"):
            validate_twin_pins(self.cfg, self.cfg, pins)

    def test_rejects_altered_decoder_or_mask_semantics(self):
        bad = dict(self.cfg, **{"mask-vocabulary": "0,1"})
        with self.assertRaisesRegex(ValueError, "feature semantics"):
            validate_twin_pins(bad, self.cfg, self.pins)
        spec = segmentation.decoder_spec()
        spec["layers"][1]["out_channels"] = 64
        bad = dict(self.cfg, **{"model-spec-b64": base64.b64encode(json.dumps(spec).encode()).decode()})
        with self.assertRaisesRegex(ValueError, "pinned convolutional decoder"):
            validate_twin_pins(bad, self.cfg, self.pins)


class BenchmarkKeyTests(unittest.TestCase):
    def test_key_reuses_same_cell_and_stays_outside_artifact_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("central_twins.BENCHMARK_KEY_ROOT", root / "keys"):
                first = private_key(root / "cell")
                self.assertEqual(first, private_key(root / "cell"))
                self.assertNotEqual(first, private_key(root / "different-cell"))
                self.assertEqual(len(first), 32)
                self.assertFalse((root / "cell" / "benchmark-secret").exists())

    def test_public_modes_and_symlinked_key_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keys = root / "keys"
            with patch("central_twins.BENCHMARK_KEY_ROOT", keys):
                private_key(root / "cell")
                key = next(keys.glob("*/benchmark-secret"))
                key.chmod(0o644)
                with self.assertRaisesRegex(ValueError, "0600"):
                    private_key(root / "cell")
                key.chmod(0o600)
                target = root / "other-key"
                key.rename(target)
                key.symlink_to(target)
                with self.assertRaises(OSError):
                    private_key(root / "cell")
                key.unlink()
                target.rename(key)
                keys.chmod(0o755)
                with self.assertRaisesRegex(ValueError, "0700"):
                    private_key(root / "cell")
                keys.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
