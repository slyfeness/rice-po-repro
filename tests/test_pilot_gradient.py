"""CPU tensor fixtures for the new observer, never real training evidence."""
import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch unavailable; not a pass')
class PilotGradientTests(unittest.TestCase):
    def test_same_ieee_path_and_real_step_are_separate(self):
        from ricepo.pilot_gradient import prepare
        from ricepo.training import UpdateItem
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.fill_(.1)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-6)
        item=UpdateItem('toy','toy-policy',(1,2),(False,True),(0.,1.),(0.,-1.))
        def forward(m,i):
            v=m.weight.reshape(1,1)
            return torch.cat([v*0,-1.+v],1),torch.cat([v*0,v.square()],1)
        recipe=dict(ppo_epochs=1,optimizer_minibatch_size='full_batch',loss_reduction='sequence_mean',clip_epsilon=.2,actor_kl_coefficient=.001)
        pilot=dict(candidate='T',gradient_screen=dict(T_cosine_min=.999999,T_relative_l2_max=.000001))
        observer,report=prepare(model,opt,[item],candidate_forward=forward,ieee_forward=forward,device='cpu',recipe=recipe,pilot=pilot)
        from ricepo.objective import ppo_loss
        current,kl=forward(model,item)
        loss=ppo_loss(current,torch.tensor([item.old_log_probs]),torch.tensor([item.advantages]),torch.tensor([item.generated_mask]),
            clip_epsilon=.2,kl_coefficient=.001,reference_kl=kl,reduction='sum')
        loss.backward();observer('before_clip')
        self.assertEqual(report['relative_gradient_l2'],0.)
        self.assertAlmostEqual(report['gradient_cosine'],1.)
        self.assertNotIn('optimizer_step_executed',report)
        opt.step();observer('after_step')
        self.assertTrue(report['optimizer_step_executed'])
        self.assertTrue(report['reward_gradient_nonzero'])
        self.assertGreater(report['changed_parameter_elements'],0)
