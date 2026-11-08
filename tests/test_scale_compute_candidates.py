import unittest
from types import SimpleNamespace
import torch
from ricepo.training import UpdateItem, ppo_plan_update
from ricepo.fused_probability import FusedOriginalProbabilityGuard
from ricepo.backends import TransformersEntropyProvider

class Candidates(unittest.TestCase):
 def test_fused_reject_clears_gradients_before_step(self):
  m=torch.nn.Linear(1,1,bias=False); opt=torch.optim.AdamW(m.parameters(),lr=1e-6)
  items=[UpdateItem(str(i),'p',(1,2),(False,True),(0.,1.),(0.,-100.)) for i in range(2)]
  before=m.weight.detach().clone(); reports=[]
  guard=FusedOriginalProbabilityGuard(outside_max=.01,branch_gate=lambda:None,save_report=reports.append)
  def forward(model,item):
   p=torch.cat([torch.zeros(1),model.weight.flatten()])[None,:];return p,p*0
  recipe=dict(ppo_epochs=1,microbatch_size=1,optimizer_minibatch_size='full_batch',gradient_accumulation='full_batch',loss_reduction='sequence_mean',clip_epsilon=.2,actor_kl_coefficient=.001,max_grad_norm=1.)
  with self.assertRaises(ValueError):ppo_plan_update(m,opt,items,policy_version='p',forward=forward,device='cpu',recipe=recipe,probability_guard=guard)
  self.assertTrue(torch.equal(before,m.weight));self.assertFalse(opt.state)
  self.assertTrue(all(p.grad is None for p in m.parameters()));self.assertEqual(len(guard.seen),2)
 def test_entropy_chunk_boundaries(self):
  from transformers import Qwen2Config,Qwen2ForCausalLM
  torch.manual_seed(7)
  m=Qwen2ForCausalLM(Qwen2Config(vocab_size=37,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)).eval()
  t=SimpleNamespace(policy_version='p',token_ids=tuple(range(1,13)),generated_mask=(False,)+(True,)*11,steps=(SimpleNamespace(summary_span=SimpleNamespace(start=3,end=9)),))
  kw=dict(model=m,policy_version_getter=lambda:'p',device='cpu',chunk_size=4,max_context_tokens=32,model_call_kwargs={})
  a=TransformersEntropyProvider(**kw).token_entropies(t);b=TransformersEntropyProvider(**kw,projection_mode='summary_positions_v1').token_entropies(t)
  torch.testing.assert_close(torch.tensor(a[3:9]),torch.tensor(b[3:9]),atol=1e-6,rtol=1e-6)
  self.assertEqual(len(a),len(b))
 def test_guard_preserves_q(self):
  item=UpdateItem('i','p',(1,2),(False,True),(0.,1.),(0.,-.5));saved=[]
  g=FusedOriginalProbabilityGuard(outside_max=.01,branch_gate=lambda:None,save_report=saved.append);g.begin([item],'p');g.observe(item,torch.tensor([[0.,-.5]],requires_grad=True));g.finish()
  self.assertEqual(item.old_log_probs,(0.,-.5));self.assertEqual(g.report['tokens'],1)
