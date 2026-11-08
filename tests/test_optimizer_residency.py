import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ricepo.vllm_lifecycle import PhaseSeparatedGenerator


class OptimizerResidency(unittest.TestCase):
    def generator(self):
        g=object.__new__(PhaseSeparatedGenerator);g.worker=None;g.defer_optimizer_residency=True
        g.device=torch.device('cpu');g.actor=SimpleNamespace(to=lambda d:None);g.reference=SimpleNamespace(to=lambda d:None)
        state={'exp_avg':torch.tensor([.125,-.5]),'step':torch.tensor(3.)}
        g.optimizer=SimpleNamespace(state={'p':state});g.moves=[(state,'exp_avg',torch.device('cpu'))]
        return g,state

    def test_moments_remain_deferred_across_entropy_and_branch_transitions(self):
        g,state=self.generator();before=state['exp_avg'].clone()
        with patch('torch.cuda.synchronize'),patch('torch.cuda.empty_cache'):
            g.prepare_scoring();self.assertEqual(len(g.moves),1)
            g._offload();self.assertEqual(len(g.moves),1)
        g.prepare_optimizer_step();self.assertEqual(g.moves,[])
        torch.testing.assert_close(state['exp_avg'],before,rtol=0,atol=0)
        self.assertEqual(float(state['step']),3.)

    def test_optimizer_residency_cannot_return_while_sampler_is_alive(self):
        g,state=self.generator();g.worker=object()
        with self.assertRaisesRegex(RuntimeError,'sampl'):
            g.prepare_optimizer_step()
