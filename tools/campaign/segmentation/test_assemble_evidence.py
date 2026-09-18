import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import assemble_evidence as evidence
from segmentation_metrics import envelopes


def mechanism(n):
    steps = math.ceil(n / 16)
    return {"adjacency": "replace_one", "clipping_norm": 1.,
            "accounting_population": n, "steps_per_epoch": steps,
            "sample_rate": 1 / steps, "expected_batch_size": max(1, n // steps),
            "total_epochs": 10, "total_steps": 10 * steps, "noise_multiplier": 2.}


def captures(populations=(69, 68, 68)):
    result = []
    for site, n in enumerate(populations):
        m = mechanism(n)
        for round_index in range(1, 6):
            result.append({"public_fixture_only": True, "round": round_index,
                           "source_rows": n + site + 1,
                           "features_sha256": str(site) * 64, "targets_sha256": str(site + 3) * 64,
                           "mechanism": copy.deepcopy(m), "accountant_type": "PRVAccountant",
                           "accountant_history": [[2., m["sample_rate"], m["steps_per_epoch"] * 2]],
                           "observed_round_steps": m["steps_per_epoch"] * 2,
                           "peak_cuda_bytes": 123})
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_cell(root):
    seed = evidence.SEEDS[0]
    train = [f"s{i}" for i in range(205)]
    test = [f"t{i}" for i in range(51)]
    split = {"seed": seed, "variant": "full", "train": train, "test": test,
             "sites": [train[:69], train[69:137], train[137:]]}
    write_json(root / "effective-split.json", split)
    split_hash = evidence.sha256(root / "effective-split.json")
    artifact_dir = root / "artifact" / "generated-run"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.pt").write_bytes(b"synthetic-test-artifact")
    artifact_hash = evidence.sha256(artifact_dir / "model.pt")
    write_json(root / "federation-status.json", {"status": "predicted_pending_public_metric_summary",
        "cleanup_ok": True, "dataset": "breast", "variant": "full", "epsilon": 8,
        "seed": seed, "model_sha256": artifact_hash, "elapsed_s": 1., "output_dir": str(artifact_dir)})
    metric = {"all": {"dice": .2, "iou": .1}, "foreground_positive": {"dice": .2, "iou": .1},
              "empty_reference": None}
    trivial = {"strongest_dice": .3}
    write_json(root / "channel-b.json", {"status": "executed", "seed": seed,
        "split_sha256": split_hash, "artifact_sha256": artifact_hash,
        "metrics": metric, "trivial": trivial})
    config = {"batch-size": 16, "local-epochs": 2, "num-server-rounds": 5,
              "learning-rate": .01, "optimizer-name": "sgd", "scheduler-name": "none",
              "segmentation-alpha": .5}
    capture = root / "public-capture"
    capture.mkdir()
    initial = np.zeros(2, dtype=np.float32)
    initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    np.savez(capture / "public-initial-arrays.npz", **{"0": initial})
    write_json(capture / "public-initial.json", {"seed": seed, "config": config,
                                               "tensor_sha256": [initial_hash]})
    for i, row in enumerate(captures()):
        write_json(capture / f"accountant-{i}.json", row)
    twins = {"status": "executed", "seed": seed, "epsilon": 8, "split_sha256": split_hash,
             "n_train": 205, "n_per_site": [69, 68, 68], "elapsed_s": 2.,
             "peak_cuda_bytes": 456, "trivial": trivial}
    (root / "twins").mkdir()
    for branch in ("pooled_dp", "pooled_nonprivate", "federated_nonprivate"):
        (root / "twins" / (branch + ".npz")).write_bytes(branch.encode())
        twins[branch] = {"metrics": metric, "initial_tensor_sha256": initial_hash,
            "artifact_sha256": evidence.sha256(root / "twins" / (branch + ".npz"))}
    twins["pooled_dp"]["mechanism"] = mechanism(205)
    twins["federated_accounting"] = evidence.validate_captures(captures(), [69, 68, 68], 8)
    write_json(root / "twins" / "twins.json", twins)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.accounting = patch.object(evidence, "independent_accounting",
            return_value={"accountant": "synthetic-unit-test", "epsilon_replace_one": 7.99})
        self.accounting.start()
        self.addCleanup(self.accounting.stop)

    def test_complete_capture_geometry_and_independent_accounting(self):
        got = evidence.validate_captures(captures(), [69, 68, 68], 8)
        self.assertEqual(len(got), 3)
        self.assertTrue(all(m["observed_round_steps"] == [10] * 5 for m in got))
        self.assertTrue(all("independent_accounting" in m for m in got))
        self.assertEqual([m["source_rows"] for m in got], [70, 70, 71])

    def test_source_row_census_is_strict_stable_and_matches_public_fixture(self):
        rows = captures()
        expected = {(r["features_sha256"], r["targets_sha256"]): r["source_rows"] for r in rows}
        evidence.validate_captures(rows, [69, 68, 68], 8, expected_source_rows=expected)
        for value in (None, True, 68, 70., 71):
            changed = copy.deepcopy(rows)
            changed[0]["source_rows"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "source row census"):
                evidence.validate_captures(changed, [69, 68, 68], 8)
        expected[next(iter(expected))] += 1
        with self.assertRaisesRegex(ValueError, "cached public source rows"):
            evidence.validate_captures(rows, [69, 68, 68], 8, expected_source_rows=expected)

    def test_reject_duplicate_round_even_with_fifteen_captures(self):
        rows = captures()
        rows[4] = copy.deepcopy(rows[3])
        with self.assertRaisesRegex(ValueError, "exactly one capture"):
            evidence.validate_captures(rows, [69, 68, 68], 8)

    def test_reject_wrong_effective_tensors_and_accounting_population(self):
        hashes = {(r["features_sha256"], r["targets_sha256"]): r["mechanism"]["accounting_population"]
                  for r in captures()}
        self.assertEqual(len(evidence.validate_captures(captures(), [69, 68, 68], 8, hashes)), 3)
        hashes[next(iter(hashes))] = 68
        with self.assertRaisesRegex(ValueError, "accounting N"):
            evidence.validate_captures(captures(), [69, 68, 68], 8, hashes)

    def test_reject_wrong_observed_history_and_horizon(self):
        for change in ("noise", "steps", "horizon"):
            rows = captures()
            if change == "noise":
                rows[0]["accountant_history"][0][0] = .1
            elif change == "steps":
                rows[0]["accountant_history"][0][2] -= 1
            else:
                for row in rows[:5]:
                    row["mechanism"]["total_steps"] = 10
            with self.subTest(change=change), self.assertRaises(ValueError):
                evidence.validate_captures(rows, [69, 68, 68], 8)

    def test_load_cell_preserves_failed_utility_and_verifies_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_cell(root)
            row = evidence.load_replicate(root, "breast", "full", 8, evidence.SEEDS[0])
            self.assertEqual(row["federated_dp"]["all"]["dice"], .2)
            self.assertEqual(row["elapsed_s"], 3.)
            evidence.released_artifact(root).write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "artifact checksum"):
                evidence.load_replicate(root, "breast", "full", 8, evidence.SEEDS[0])

    def test_artifact_resolution_uses_recorded_child_and_rejects_other_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_cell(root)
            actual = (root / "artifact" / "generated-run" / "model.pt").resolve()
            self.assertEqual(evidence.released_artifact(root), actual)
            self.assertEqual(evidence.released_artifact(root, {"output_dir": "artifact/generated-run"}), actual)
            with self.assertRaisesRegex(ValueError, "outside this campaign cell"):
                evidence.released_artifact(root, {"output_dir": str(root.parent / "other-cell")})
            with self.assertRaisesRegex(ValueError, "did not record"):
                evidence.released_artifact(root, {})

    def test_load_rejects_mismatched_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_cell(root)
            twins_path = root / "twins" / "twins.json"
            twins = evidence.read_json(twins_path)
            twins["pooled_nonprivate"]["initial_tensor_sha256"] = "0" * 64
            write_json(twins_path, twins)
            with self.assertRaisesRegex(ValueError, "initialization"):
                evidence.load_replicate(root, "breast", "full", 8, evidence.SEEDS[0])

    def test_plan_contains_all_thirty_three_cells_without_duplicates(self):
        cells = evidence.planned_cells()
        self.assertEqual(len(cells), 33)
        self.assertEqual(len(set(cells)), 33)
        self.assertEqual(sum(c[1] == "full" for c in cells), 18)

    def test_missing_runs_stay_unexecuted_and_failed_runs_are_retained(self):
        archive = Path(__file__).resolve().parents[3] / "inst/extdata/campaign/segmentation"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "breast-eps8" / "federation-status.json",
                {"status": "failed", "dataset": "breast", "variant": "full", "epsilon": 8,
                 "seed": evidence.SEEDS[0], "error": "synthetic deliberate failure"})
            result = evidence.assemble(root, archive / "provenance", {}, archive / "protocol.md")
            self.assertEqual(result["campaign-status.json"]["status"], "failed")
            cells = result["campaign-status.json"]["cells"]
            self.assertEqual(sum(c["status"] == "failed" for c in cells), 1)
            self.assertEqual(sum(c["status"] == "not_executed" for c in cells), 32)
            self.assertNotIn("replicates", result["breast-evidence.json"])
            self.assertNotIn("envelopes", result["breast-evidence.json"])

    def test_launcher_failure_before_federation_status_is_retained(self):
        archive = Path(__file__).resolve().parents[3] / "inst/extdata/campaign/segmentation"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "busbra-full-eps1" / "execution-status.json",
                {"status": "failed", "dataset": "busbra", "variant": "full", "epsilon": 1,
                 "seed": evidence.SEEDS[0], "phase": "federation", "exit_code": 2})
            result = evidence.assemble(root, archive / "provenance", {}, archive / "protocol.md")
            failed = [c for c in result["campaign-status.json"]["cells"] if c["status"] == "failed"]
            self.assertEqual(len(failed), 1)
            self.assertIn("federation (exit code 2)", failed[0]["reason"])

    def test_small_matrix_rejects_duplicate_seed_and_missing_middle_epsilon(self):
        rows = [{"epsilon": e, "seed": s, "n_train": 192, "n_per_site": [64] * 3,
                 "central_dice": .6, "federated_dice": .2, "trivial_dice": .3}
                for e in evidence.EPSILONS for s in evidence.SEEDS]
        for small in (rows + [rows[0]], [r for r in rows if r["epsilon"] != 4]):
            with self.assertRaisesRegex(ValueError, "complete matched seed matrix"):
                envelopes(rows, small)

    def test_small_near_central_flags_use_the_small_population(self):
        rows = [{"epsilon": e, "seed": s, "n_train": 852, "n_per_site": [284] * 3,
                 "central_dice": .6, "federated_dice": .6, "trivial_dice": .3}
                for e in evidence.EPSILONS for s in evidence.SEEDS]
        small = [dict(r, n_train=192, n_per_site=[64] * 3) for r in rows]
        result = envelopes(rows, small)
        self.assertFalse(any(r["flag"] for r in result["near_central"] if r["epsilon"] == 8))
        self.assertTrue(all(r["flag"] for r in result["small_n_noise_trend"]["near_central"]))


class IndependentAccountingTests(unittest.TestCase):
    def test_no_positive_budget_tolerance(self):
        evidence.independent_accounting.cache_clear()
        with patch("opacus.accountants.PRVAccountant") as cls:
            cls.return_value.get_epsilon.return_value = .501
            with self.assertRaisesRegex(ValueError, "exceeds budget"):
                evidence.independent_accounting(2., .2, 50, 1)
            cls.return_value.get_epsilon.return_value = .499
            result = evidence.independent_accounting(2., .2, 50, 1)
            self.assertEqual(result["epsilon_replace_one"], .998)
            self.assertLess(result["delta_replace_one"], 1e-5)
        evidence.independent_accounting.cache_clear()


if __name__ == "__main__":
    unittest.main()
