import unittest
import numpy as np
from metrics import concordance, score, mean_ci, envelopes


class MetricsTests(unittest.TestCase):
    def test_ties_and_censoring(self):
        self.assertEqual(concordance([1,2,3], [1,0,1], [3,2,1]), 1)
        self.assertEqual(concordance([1,2,3], [1,0,1], [1,1,1]), .5)
        self.assertEqual(concordance([1,1,3], [1,1,0], [3,2,1]), 1)
        self.assertIsNone(concordance([1,1], [1,1], [2,1]))
        self.assertIsNone(concordance([1,2], [0,1], [2,1]))

    def test_original_time_density(self):
        cfg = dict(t_min=1,horizon=20,time_scale=2,distribution='weibull',dispersion=1)
        result = score(np.zeros((2,1)), {'time':[2,4],'event':[1,0]}, cfg)
        self.assertAlmostEqual(result['heldout_nll'], (1+np.log(2)+2)/2)

    def test_hazard_interval_end(self):
        cfg = dict(t_min=1,horizon=4,edges=[0,2,4])
        result = score(np.zeros((3,2)), {'time':[1,2,3],'event':[0,0,1]}, cfg)
        self.assertAlmostEqual(result['heldout_nll'], np.log(2)/2)

    def test_envelope_reversed_sign(self):
        def cell(seed,u):
            return dict(seed=seed,n_train=100,minimum_site_n=33,
                        central={'c_index':.7,'heldout_nll':2.},
                        central_dp={'c_index':.6,'heldout_nll':2.5},
                        federated_dp={'c_index':u,'heldout_nll':3.},
                        null={'c_index':.5,'heldout_nll':4.})
        result=envelopes({1:[cell(i,.65) for i in range(3)],4:[cell(i,.60) for i in range(3)],8:[cell(i,.59) for i in range(3)]}, primary=True,small_n=True)
        self.assertTrue(result['epsilon_envelope'][0]['flag'])
        self.assertFalse(result['utility_floor']['pass'])
        self.assertFalse(result['small_n_trend']['flag'])
        self.assertAlmostEqual(mean_ci([.4,.5,.6])['mean'], .5)


if __name__ == '__main__':
    unittest.main()
