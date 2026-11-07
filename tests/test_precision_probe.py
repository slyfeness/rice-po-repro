import unittest
from ricepo.precision_probe import probability_diagnostics, matched_behavior_gate


class PrecisionProbeTests(unittest.TestCase):
    def test_behavior_shift_and_clipping_are_both_reported(self):
        d=probability_diagnostics([-1.,-1.],[-1.,0.])
        self.assertEqual(d['clip_fraction'],.5)
        self.assertFalse(d['historical_fp32_gate'])
        self.assertFalse(matched_behavior_gate(d,dict(p99_abs_log_error_max=.10,
            clip_fraction_max=.01,mean_abs_log_error_max=.02)))

    def test_unaligned_native_records_rejected(self):
        with self.assertRaises(ValueError):probability_diagnostics([-1.],[])
