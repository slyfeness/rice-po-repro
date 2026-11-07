import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from ricepo.checkpoint_io import verified_local_copy,state_digest

class CheckpointIOTests(unittest.TestCase):
    def test_verified_copy_and_failed_verify_never_publish(self):
        with TemporaryDirectory() as d:
            root=Path(d);dst=root/'shared';dst.mkdir()
            verified_local_copy(dst,lambda p:(p/'weights').write_bytes(b'actual bytes'),
                lambda p:self.assertEqual((p/'weights').read_bytes(),b'actual bytes'),temporary_root=root)
            self.assertEqual((dst/'weights').read_bytes(),b'actual bytes')
            self.assertTrue((dst/'io-verification.json').exists())
            bad=root/'bad';bad.mkdir()
            def reject(p):raise IOError('injected bad local payload')
            with self.assertRaises(IOError):verified_local_copy(bad,lambda p:(p/'weights').write_bytes(b'bad'),reject,temporary_root=root)
            self.assertFalse((bad/'io-verification.json').exists())
            self.assertTrue((bad/'io-failure.json').exists())

    def test_eval_checkpointing_matches_direct_loss_and_gradients(self):
        import torch,copy
        from ricepo.modeling import enable_eval_layer_checkpointing
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=torch.nn.Module()
                self.model.layers=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.SiLU(),torch.nn.Linear(8,4)) for _ in range(2)])
            def forward(self,x):
                for layer in self.model.layers:x=layer(x)
                return x
        direct=Model().eval();wrapped=copy.deepcopy(direct);stats=enable_eval_layer_checkpointing(wrapped)
        x=torch.ones(2,4)
        a=direct(x).square().sum();b=wrapped(x).square().sum();a.backward();b.backward()
        torch.testing.assert_close(a,b)
        for p,q in zip(direct.parameters(),wrapped.parameters()):torch.testing.assert_close(p.grad,q.grad)
        self.assertGreater(stats['layer_executions'],stats['checkpoint_forwards'])
        self.assertFalse(wrapped.training)
        state={'optimizer':[torch.tensor([1.,2.])],'rng':torch.tensor([1,2],dtype=torch.uint8)}
        self.assertEqual(state_digest(state),state_digest(copy.deepcopy(state)))

    def test_local_credit_gradient_changes_update_direction_without_baseline_step(self):
        import torch
        from ricepo.credit_gradient import prepare_comparison
        from ricepo.training import UpdateItem,ppo_plan_update
        from test_failure_zero import runner,rollout
        t=rollout(runner());mask=t.generated_mask
        model=torch.nn.Linear(1,1,bias=False).eval()
        with torch.no_grad():model.weight.fill_(.5)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
        old=tuple(-.25 if m else 0. for m in mask)
        item=UpdateItem(t.trajectory_id,'frozen',t.token_ids,mask,tuple(1. if m else 0. for m in mask),old)
        def forward(model,item):
            v=-(model.weight.reshape(())-1).square()
            values=torch.stack([v if m else v*0 for m in item.generated_mask])[None,:]
            return values,values*0
        recipe=dict(ppo_epochs=1,optimizer_minibatch_size=1,microbatch_size=1,gradient_accumulation=1,
            loss_reduction='sequence_mean',clip_epsilon=.2,actor_kl_coefficient=.001,max_grad_norm=1.,normalization_epsilon=1e-8)
        observer,report=prepare_comparison(model,opt,[t],[item],forward=forward,device='cpu',recipe=recipe)
        result=ppo_plan_update(model,opt,[item],policy_version='frozen',forward=forward,device='cpu',recipe=recipe,step_observer=observer)
        self.assertEqual(result['optimizer_steps'],1)
        self.assertEqual(report['baseline_optimizer_steps'],0)
        self.assertEqual(report['grpo_gradient_norm'],0)
        self.assertGreater(report['gradient_difference_norm'],0)
        self.assertTrue(report['first_order_descent'])

if __name__=='__main__':unittest.main()
