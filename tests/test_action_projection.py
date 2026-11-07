import unittest,copy
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from ricepo.modeling import make_forward,enable_eval_layer_checkpointing
from ricepo.training import UpdateItem
from ricepo.objective import ppo_loss,token_entropy_from_logits

class ActionHeadTests(unittest.TestCase):
 def test_native_actions_loss_entropy_and_all_gradients(self):
  torch.manual_seed(37)
  cfg=Qwen2Config(vocab_size=29,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.)
  a=Qwen2ForCausalLM(cfg).double().eval();b=copy.deepcopy(a);ref=copy.deepcopy(a)
  with torch.no_grad():ref.lm_head.weight.add_(torch.linspace(-.1,.1,29)[:,None])
  ref.eval().requires_grad_(False)
  ids=(1,3,5,7,2,4,8,2,11,10);mask=(False,False,True,False,True,False,True,True,False,False)
  item=UpdateItem('fixed','p',ids,mask,(0.,0.,1.,0.,-1.,0.,.4,-.2,0.,0.),(0.,0.,-3.1,0.,-3.1,0.,-3.1,-3.1,0.,0.))
  opts=dict(reference_model=ref,device='cpu',logit_chunk_size=2,max_context_tokens=64,policy_call_kwargs={},reference_call_kwargs={})
  calls=[];outputs=[]
  handle=b.lm_head.register_forward_hook(lambda mod,args,out:(calls.append(tuple(args[0].shape)),outputs.append(out.detach())) and None)
  full,kl=make_forward(**opts)(a,item)
  small,k2=make_forward(**opts,projection_mode='action_positions_v1')(b,item)
  torch.testing.assert_close(full,small,rtol=1e-7,atol=1e-9);torch.testing.assert_close(kl,k2,rtol=1e-7,atol=1e-9)
  self.assertEqual(calls,[(1,4,16)]);self.assertEqual(outputs[0].shape[-1],29)
  direct=a(input_ids=torch.tensor([ids[:7]]),use_cache=False).logits[:,[1,3,5,6],:]
  torch.testing.assert_close(token_entropy_from_logits(direct,chunk_size=2),token_entropy_from_logits(outputs[0],chunk_size=2),rtol=1e-7,atol=1e-9)
  def loss(p,k):return ppo_loss(p,torch.tensor([item.old_log_probs],dtype=p.dtype),torch.tensor([item.advantages],dtype=p.dtype),torch.tensor([mask]),clip_epsilon=.2,kl_coefficient=.001,reference_kl=k,reduction='sequence_mean')
  left,right=loss(full,kl),loss(small,k2);torch.testing.assert_close(left,right)
  left.backward();right.backward()
  for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
   self.assertEqual(n,m);torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-8)
  self.assertGreater(float(b.model.embed_tokens.weight.grad[7].abs().sum()),0.)
  self.assertTrue(all(p.grad is None for p in ref.parameters()));handle.remove()
  self.assertEqual(small[0,[0,1,3,5,8,9]].tolist(),[0.]*6)
  stats=enable_eval_layer_checkpointing(b);b.zero_grad(set_to_none=True)
  p,k=make_forward(**opts,projection_mode='action_positions_v1')(b,item);loss(p,k).backward()
  self.assertGreater(stats['layer_executions'],stats['checkpoint_forwards']);self.assertFalse(b.training)
 def test_projection_rejects_unknown_and_wrapper_without_bypass(self):
  cfg=Qwen2Config(vocab_size=17,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
  a=Qwen2ForCausalLM(cfg).eval();r=copy.deepcopy(a).requires_grad_(False)
  opts=dict(reference_model=r,device='cpu',logit_chunk_size=2,max_context_tokens=32,policy_call_kwargs={},reference_call_kwargs={})
  with self.assertRaises(ValueError):make_forward(**opts,projection_mode='sampled_target_only')
  wrapped=torch.nn.DataParallel(a)
  item=UpdateItem('i','p',(1,2,3),(False,True,True),(0.,1.,-1.),(0.,-2.,-2.))
  with self.assertRaisesRegex(ValueError,'bare Qwen2'):make_forward(**opts,projection_mode='action_positions_v1')(wrapped,item)
 def test_HF_precision_scope_restores_on_exception(self):
  from ricepo.hf_compute import hf_matmul_scope
  torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
  with self.assertRaisesRegex(RuntimeError,'sentinel'):
   with hf_matmul_scope('high'):
    self.assertEqual(torch.get_float32_matmul_precision(),'high');self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
    raise RuntimeError('sentinel')
  self.assertEqual(torch.get_float32_matmul_precision(),'highest');self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

if __name__=='__main__':unittest.main()
