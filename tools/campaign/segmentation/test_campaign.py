import unittest
import json
import hashlib
from pathlib import Path
import numpy as np
from prepare_public_data import split_subjects
from segmentation_metrics import metrics, subject_scores, trivial_masks, envelopes


class CampaignTests(unittest.TestCase):
    def test_exact_empty_and_overlap_conventions(self):
        reference = np.array([[[[0, 0]]], [[[1, 0]]], [[[1, 0]]], [[[1, 1]]]])
        prediction = np.array([[[[0, 0]]], [[[0, 0]]], [[[0, 1]]], [[[1, 0]]]])
        dice, iou, _ = subject_scores(prediction, reference)
        np.testing.assert_allclose(dice, [1, 0, 0, 2/3])
        np.testing.assert_allclose(iou, [1, 0, 0, .5])

    def test_strata_do_not_hide_failed_foreground(self):
        ref = np.zeros((3, 1, 4, 4)); ref[2, 0, 0, 0] = 1
        got = metrics(np.zeros_like(ref), ref)
        self.assertEqual(got['all']['dice'], 2/3)
        self.assertEqual(got['foreground_positive']['dice'], 0)
        self.assertEqual(got['empty_reference']['dice'], 1)
        self.assertIsNone(metrics(np.zeros_like(ref), np.zeros_like(ref))['foreground_positive'])

    def test_threshold_and_input_domain(self):
        self.assertEqual(metrics(np.full((1, 1, 1, 1), .5), np.ones((1, 1, 1, 1)))['all']['dice'], 1)
        with self.assertRaises(ValueError):
            metrics(np.full((1, 1, 1, 1), np.nan), np.zeros((1, 1, 1, 1)))

    def test_fixed_trivial_shapes(self):
        ref = np.zeros((2, 1, 128, 128)); ref[:, :, 32:96, 32:96] = 1
        got = trivial_masks(ref)
        self.assertEqual(got['scores']['square_64']['all']['dice'], 1)
        self.assertEqual(got['strongest_dice'], 1)

    def test_subject_split_is_order_invariant_and_disjoint(self):
        subjects = [f's{i}' for i in range(1064)]
        labels = {s: str(i % 2) for i, s in enumerate(subjects)}
        a = split_subjects(subjects, 20260919, labels)
        self.assertEqual(a, split_subjects(subjects[::-1], 20260919, labels))
        self.assertEqual(len(a['test']), 212)
        self.assertEqual(len(a['train']), 852)
        self.assertFalse(set(a['train']) & set(a['test']))
        self.assertEqual(set(sum(a['sites'], [])), set(a['train']))
        self.assertEqual(sum(map(len, a['sites'])), len(a['train']))
        self.assertTrue(set(a['small_train']) <= set(a['train']))
        self.assertEqual(len(a['small_train']), 192)

    def test_envelopes_flag_worsening_and_failed_floor(self):
        rows = [{'epsilon': e, 'seed': s, 'n_train': 192, 'n_per_site': [64]*3,
                 'central_dice': .6, 'federated_dice': .55 if e == 1 else .4,
                 'trivial_dice': .35} for e in (1, 4, 8) for s in (1, 2, 3)]
        got = envelopes(rows, rows)
        self.assertTrue(got['epsilon_envelope'][0]['flag'])
        self.assertFalse(got['utility_floor']['pass'])
        self.assertFalse(got['small_n_noise_trend']['flag'])

    def test_archived_evidence_has_protocol_and_explicit_execution_status(self):
        root = Path(__file__).resolve().parents[3] / "inst" / "extdata" / "campaign" / "segmentation"
        document = json.loads((root / "campaign-status.json").read_text())
        self.assertIn(document["status"], ("not_executed", "failed", "executed"))
        self.assertNotIn("scores", document)
        self.assertNotIn("replicates", document)
        self.assertEqual(document["protocol_sha256"], hashlib.sha256((root / "protocol.md").read_bytes()).hexdigest())
        self.assertEqual({c["epsilon"] for c in document["cells"]}, {1, 4, 8})
        if document["schema"] == "dsflower-segmentation-campaign-summary-v1":
            from assemble_evidence import planned_cells
            keys = [(c["dataset"], c["variant"], c["epsilon"], c["seed"]) for c in document["cells"]]
            self.assertEqual(set(keys), set(planned_cells()))
            self.assertEqual(len(keys), len(set(keys)))
            if document["status"] == "executed":
                self.assertTrue(all(c["status"] == "executed" for c in document["cells"]))
        else:
            self.assertEqual(document["status"], "not_executed")
            self.assertTrue(all(len(c["seeds"]) >= 3 for c in document["cells"]))

    def test_missing_replicates_never_synthesize_scores(self):
        with self.assertRaises(ValueError):
            envelopes([])


if __name__ == '__main__':
    unittest.main()
