import copy
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from dsflower_runner import params, segmentation
from feature_smoke import config
from prepare_synthetic import prepare
import verify_synthetic as verifier


class FixtureEncoder(torch.nn.Module):
    def forward(self, images):
        return torch.nn.functional.adaptive_avg_pool2d(
            images.mean(1, keepdim=True), (16, 16)).repeat(1, 128, 1, 1)


class SyntheticVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        prepare(self.prepared)
        self.split = json.loads((self.prepared / "split-20260919.json").read_text())
        torch.set_num_threads(1)
        self.encoder = FixtureEncoder()
        self.X, self.y, self.sites = verifier.fixture_tensors(
            self.prepared, self.split, self.encoder, "cpu")
        mechanism = dict(adjacency="replace_one", noise_multiplier=10., clipping_norm=1.,
                         accounting_population=16, steps_per_epoch=2, sample_rate=.5,
                         expected_batch_size=8, total_epochs=2, total_steps=4)
        self.records = [dict(public_fixture_only=True, source_rows=18,
            round=round_index, mechanism=copy.deepcopy(mechanism),
            accountant_type="PRVAccountant", observed_round_steps=2,
            accountant_history=[[10., .5, 2]], features_sha256=site["features_sha256"],
            targets_sha256=site["targets_sha256"]) for site in self.sites for round_index in (1, 2)]

    def test_independent_fixture_retains_invalid_union_and_empty_subjects(self):
        self.assertEqual(self.X.shape, (51, 32768))
        self.assertEqual(self.y[:, 1, 0, 0].sum(), 48)
        for start in (0, 16, 32):
            self.assertFalse(self.y[start, 0].any())
            self.assertFalse(self.X[start + 1].any())
            self.assertFalse(self.y[start + 1].any())
            self.assertTrue(self.y[start + 2, 0, 10:20, 10:20].all())

    def test_lowercase_empty_markers_fail_instead_of_totalizing_all_subjects(self):
        path = self.prepared / "samples.csv"
        path.write_text(path.read_text().replace("TRUE", "true"))
        with self.assertRaisesRegex(ValueError, "empty declaration"):
            verifier.fixture_tensors(self.prepared, self.split, self.encoder, "cpu")

    def test_missing_duplicate_captures_or_wrong_geometry_fail(self):
        changes = [lambda r: r.pop(), lambda r: r.__setitem__(-1, copy.deepcopy(r[0])),
                   lambda r: r[0].__setitem__("source_rows", 16),
                   lambda r: r[0]["mechanism"].__setitem__("accounting_population", 15),
                   lambda r: r[0]["mechanism"].__setitem__("sample_rate", 8/18),
                   lambda r: r[0].__setitem__("observed_round_steps", 1),
                   lambda r: r[0].__setitem__("targets_sha256", "0" * 64)]
        for change in changes:
            with self.subTest(change=change):
                records = copy.deepcopy(self.records)
                change(records)
                with self.assertRaises(ValueError):
                    verifier.verify_captures(records, self.sites, 4)

    def test_accounting_composes_both_rounds_and_rejects_insufficient_noise(self):
        got = verifier.verify_captures(self.records, self.sites, 4)
        self.assertEqual(len(got), 3)
        self.assertTrue(all(r["independent_accounting"]["total_steps"] == 4 for r in got))
        with self.assertRaisesRegex(ValueError, "exceeds budget"):
            verifier.independent_accounting({"noise_multiplier": 1.}, 1)

    def make_run(self, decoder="current"):
        import shutil
        if (self.root / "run").exists():
            shutil.rmtree(self.root / "run")
        run = self.root / "run"
        capture, artifact = run / "public-capture", run / "artifact/saved-synthetic-model"
        self.artifact = artifact
        capture.mkdir(parents=True)
        artifact.mkdir(parents=True)
        cfg = dict(config(), **{"batch-size": 8, "local-epochs": 1, "num-server-rounds": 2})
        cfg["model-spec-b64"] = base64.b64encode(json.dumps(segmentation.decoder_spec(decoder)).encode()).decode()
        model = params.load_user_model(cfg, segmentation.FEATURE_DIM, "segmentation_bce_dice")
        for parameter in model.parameters():
            parameter.data.zero_()
        initial = params.get_torch_params(model)
        np.savez(capture / "public-initial-arrays.npz", **{str(i): a for i, a in enumerate(initial)})
        (capture / "public-initial.json").write_text(json.dumps(dict(
            seed=self.split["seed"], config=cfg, tensor_sha256=[verifier.tensor_hash(a) for a in initial])))
        # Unit-test synthetic records do not leave this temporary directory and
        # are never archived as real federation/gate evidence.
        for i, record in enumerate(self.records):
            (capture / f"accountant-test-{i}.json").write_text(json.dumps(record))
        for parameter in model.parameters():
            parameter.data.add_(.001)
        torch.save(model.state_dict(), artifact / "model.pt")
        with torch.no_grad():
            probability = model(torch.from_numpy(self.X[48:])).sigmoid().numpy().reshape(3, -1)
        np.savetxt(run / "public-probabilities.csv", probability, delimiter=",")
        (artifact / "metadata.json").write_text(json.dumps(dict(status="success", available=True,
            n_clients=3, model="pytorch_resnet18_segmentation", privacy="server-enforced-dp")))
        (artifact / "history.json").write_text(json.dumps([
            dict(round=r, n_failures=0, available=True) for r in (1, 2)]))
        (run / "effective-split.json").write_text(json.dumps(self.split))
        (run / "federation-status.json").write_text(json.dumps(dict(
            status="predicted_pending_public_metric_summary", cleanup_ok=True,
            synthetic=True, dataset="synthetic", seed=self.split["seed"], epsilon=4,
            output_dir=str(artifact),
            model_sha256=verifier.sha256(artifact / "model.pt"),
            split_sha256=verifier.sha256(self.prepared / "split-20260919.json"))))
        return run

    def test_complete_verifier_checks_artifact_prediction_and_cleanup(self):
        run = self.make_run()
        with mock.patch.object(segmentation, "prepare_encoder", return_value=(self.encoder, "cpu")), \
                mock.patch.object(verifier, "independent_accounting", return_value={}):
            result = verifier.verify(self.prepared, run)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["decoder_parameter_tensors"], 6)
            self.assertFalse(result["scoring_performed"])
            probability = run / "public-probabilities.csv"
            probability.write_text("0,0\n")
            with self.assertRaisesRegex(ValueError, "predictions differ"):
                verifier.verify(self.prepared, run)
        status_path = run / "federation-status.json"
        status = json.loads(status_path.read_text())
        status["cleanup_ok"] = False
        status_path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, "complete cleanly"):
            verifier.verify(self.prepared, run)

    def test_failed_history_and_extra_state_cannot_be_accepted(self):
        run = self.make_run()
        history_path = self.artifact / "history.json"
        history = json.loads(history_path.read_text())
        history[1]["n_failures"] = 1
        history_path.write_text(json.dumps(history))
        with mock.patch.object(segmentation, "prepare_encoder", return_value=(self.encoder, "cpu")), \
                mock.patch.object(verifier, "independent_accounting", return_value={}):
            with self.assertRaisesRegex(ValueError, "availability history"):
                verifier.verify(self.prepared, run)
            history[1]["n_failures"] = 0
            history_path.write_text(json.dumps(history))
            model_path = self.artifact / "model.pt"
            state = torch.load(model_path, weights_only=True)
            state["encoder.running_mean"] = torch.zeros(128)
            torch.save(state, model_path)
            status_path = run / "federation-status.json"
            status = json.loads(status_path.read_text())
            status["model_sha256"] = verifier.sha256(model_path)
            status_path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, "finite trained decoder parameters"):
                verifier.verify(self.prepared, run)

    def test_artifact_directory_must_belong_to_the_actual_run(self):
        run = self.make_run()
        status_path = run / "federation-status.json"
        status = json.loads(status_path.read_text())
        status["output_dir"] = str(self.root)
        status_path.write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, "outside this run"):
            verifier.verify(self.prepared, run)

    def test_v4_synthetic_verifier_requires_requested_decoder_and_exact_state(self):
        for decoder, tensors in (("narrow", 6), ("pointwise", 2)):
            run = self.make_run(decoder)
            with mock.patch.object(segmentation, "prepare_encoder", return_value=(self.encoder, "cpu")), \
                 mock.patch.object(verifier, "independent_accounting", return_value={}):
                result = verifier.verify(self.prepared, run, decoder)
                self.assertEqual(result["decoder_parameter_tensors"], tensors)
                with self.assertRaisesRegex(ValueError, "decoder differs"):
                    verifier.verify(self.prepared, run, "current")


if __name__ == "__main__":
    unittest.main()
