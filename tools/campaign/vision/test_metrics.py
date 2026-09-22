"""Known rankings, probability calibration, and ties; no cohort data."""
import unittest
import numpy as np
from score import metrics


class MetricTests(unittest.TestCase):
    def test_perfect_and_reversed_ranking(self):
        self.assertEqual(metrics([0, 1, 0, 1], [.1, .8, .2, .9])["auc"], 1.)
        self.assertEqual(metrics([0, 1], [.8, .2])["auc"], 0.)

    def test_constant_forecast_and_threshold(self):
        values = metrics([0, 0, 0, 1], [.25]*4)
        self.assertEqual(values["auc"], .5)
        self.assertEqual(values["accuracy"], .75)
        self.assertEqual(values["brier"], .1875)
        self.assertAlmostEqual(values["log_loss"], -(3*np.log(.75)+np.log(.25))/4)

    def test_partial_tie(self):
        self.assertEqual(metrics([0, 0, 1, 1], [.1, .5, .5, .9])["auc"], .875)


if __name__ == "__main__":
    unittest.main()
