import unittest
from types import SimpleNamespace
import torch
from ricepo.training import UpdateItem, ppo_plan_update
from ricepo.fused_probability import FusedOriginalProbabilityGuard
from ricepo.backends import TransformersEntropyProvider

class Candidates(unittest.TestCase):
 def test_fused_reject_clears_gradients_before_step(self):
  m=torch.nn.Linear(1,1,bias=False); opt=torch.optim.AdamW(m.parameters(),lr=1e-6)
  items=[UpdateItem(str(i),'p',(1,2),(False,True),(0.,1.),(0.,-2.)) for i in range(2)]
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

 def test_selective_budget_uses_causal_horizon_and_action_count(self):
  from unittest.mock import patch
  from transformers import Qwen2Config,Qwen2ForCausalLM
  from ricepo.modeling import make_forward
  m=Qwen2ForCausalLM(Qwen2Config(vocab_size=13,hidden_size=8,intermediate_size=12,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)).eval()
  r=Qwen2ForCausalLM(m.config).eval().requires_grad_(False)
  opts=dict(reference_model=r,device='cpu',logit_chunk_size=2,max_context_tokens=16,policy_call_kwargs={},reference_call_kwargs={},projection_mode='action_positions_v1')
  f=make_forward(**opts,saved_tensor_policy='cpu_above_budget_v1',device_residency_budget={'horizon':2,'action_tokens':1})
  with patch('torch.autograd.graph.save_on_cpu',wraps=torch.autograd.graph.save_on_cpu) as spy:
   for mask,expected in [((False,False,True,False),0),((False,True,True,False),1),((False,False,False,True),2)]:
    item=UpdateItem('x','p',(1,2,3,4),mask,(0.,0.,1.,1.),(0.,0.,-.5,-.5))
    p,k=f(m,item);(p.sum()+k.sum()).backward();self.assertEqual(spy.call_count,expected);m.zero_grad(set_to_none=True)
  self.assertTrue(all(p.grad is None for p in r.parameters()))
 def test_branch_failure_discards_fused_gradients(self):
  m=torch.nn.Linear(1,1,bias=False);m.weight.data.fill_(-.5);opt=torch.optim.AdamW(m.parameters(),lr=1e-6)
  item=UpdateItem('i','p',(1,2),(False,True),(0.,1.),(0.,-.5))
  def branch():raise RuntimeError('branch audit failed')
  guard=FusedOriginalProbabilityGuard(outside_max=.01,branch_gate=branch,save_report=lambda r:None)
  def forward(model,item):
   p=torch.cat([torch.zeros(1),model.weight.flatten()])[None,:];return p,p*0
  recipe=dict(ppo_epochs=1,microbatch_size=1,optimizer_minibatch_size='full_batch',gradient_accumulation='full_batch',loss_reduction='sequence_mean',clip_epsilon=.2,actor_kl_coefficient=.001,max_grad_norm=1.)
  with self.assertRaisesRegex(RuntimeError,'branch audit'):ppo_plan_update(m,opt,[item],policy_version='p',forward=forward,device='cpu',recipe=recipe,probability_guard=guard)
  self.assertFalse(opt.state);self.assertTrue(all(p.grad is None for p in m.parameters()));self.assertEqual(float(m.weight),-.5)
