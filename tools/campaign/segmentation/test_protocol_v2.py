import math
import unittest
from unittest.mock import patch
from assemble_evidence import validate_captures
from test_assemble_evidence import captures

class Batch64AccountingTests(unittest.TestCase):
    def test_complete_logical_batch_horizon_and_wrong_arm_rejection(self):
        rows = captures()
        for row in rows:
            m = row["mechanism"]
            n = m["accounting_population"]
            steps = math.ceil(n / 64)
            m.update(steps_per_epoch=steps, sample_rate=1/steps,
                     expected_batch_size=n//steps, total_steps=10*steps)
            row["accountant_history"] = [[2., 1/steps, 2*steps]]
            row["observed_round_steps"] = 2*steps
        with patch("assemble_evidence.independent_accounting", return_value={}):
            self.assertEqual(len(validate_captures(rows, [69,68,68], 8, batch_size=64)), 3)
            with self.assertRaisesRegex(ValueError, "subject schedule"):
                validate_captures(rows, [69,68,68], 8, batch_size=16)
