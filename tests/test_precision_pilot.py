import math
import unittest
from ricepo.precision_pilot import mismatch_statistics, risk_screen


class PrecisionPilotTests(unittest.TestCase):
    def test_ess_is_stable_and_not_ratio_mean(self):
        result = mismatch_statistics([-2., -2.], [-2., -2.])
        self.assertEqual(result['ess_fraction'], 1.)
        self.assertEqual(result['outside_clip_fraction'], 0.)
        self.assertIsNone(result['effective_clip_fraction'])
        extreme = mismatch_statistics([-1002., -2.], [-2., -2.])
        self.assertAlmostEqual(extreme['ess_fraction'], .5)

    def test_effective_clipping_depends_on_advantage_sign(self):
        delta = [math.log(1.3), math.log(.7), math.log(1.3), math.log(.7), 0.]
        q = [-2.]*5
        result = mismatch_statistics(q, [a+b for a,b in zip(q,delta)], [1., -1., -1., 1., 0.])
        self.assertEqual(result['outside_clip_fraction'], .8)
        self.assertEqual(result['effective_clip_fraction'], .4)
        self.assertEqual(result['zero_advantage_fraction'], .2)

    def test_independent_risk_thresholds_do_not_erase_legacy_fail(self):
        q = [-2.]*1000
        p = [-1.9]*1000
        report = mismatch_statistics(q,p)
        self.assertTrue(risk_screen(report))
        self.assertFalse(report['historical_consistency_pass'])
        self.assertFalse(report['matched_consistency_pass'])
        report['outside_clip_fraction'] = .010001
        self.assertFalse(risk_screen(report))

    def test_nonfinite_and_unaligned_rejected(self):
        for q,p in [([],[]),([-1.],[]),([-1.],[float('nan')])]:
            with self.assertRaises(ValueError): mismatch_statistics(q,p)

    def test_tf32_remains_an_explicit_unqualified_pilot_mode(self):
        from ricepo.vllm_two_stage import vllm_numeric_options,vllm_runtime_options,VLLMEngine
        from unittest.mock import patch
        self.assertFalse(vllm_numeric_options('tf32_pilot')['ppo_compatible'])
        self.assertEqual(vllm_runtime_options(numeric_mode='tf32_pilot')['attention_backend'],'TRITON_ATTN')
        with patch.dict('os.environ',{'TRITON_F32_DEFAULT':'ieee'}):
            with self.assertRaisesRegex(ValueError,'TF32 pilot'):
                VLLMEngine('never-loaded',object(),'p',numeric_mode='tf32_pilot')
