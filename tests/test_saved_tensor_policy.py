"""Exact activation residency only; full head, KL and contextual gradients."""
import copy,unittest
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from ricepo.modeling import make_forward,enable_eval_layer_checkpointing
from ricepo.training import UpdateItem
from ricepo.objective import ppo_loss


class SavedTensorPolicy(unittest.TestCase):
    def test_full_sequence_loss_and_every_gradient_with_recomputation(self):
        torch.manual_seed(43)
        cfg=Qwen2Config(vocab_size=29,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.)
        a=Qwen2ForCausalLM(cfg).double().eval();b=copy.deepcopy(a);ref=copy.deepcopy(a)
        with torch.no_grad():ref.lm_head.weight.add_(torch.linspace(-.1,.1,29)[:,None])
        ref.requires_grad_(False);enable_eval_layer_checkpointing(a);stats=enable_eval_layer_checkpointing(b)
        ids=(1,3,5,7,2,4,8,2,11,10);mask=(False,False,True,False,True,False,True,True,False,False)
        item=UpdateItem('fixed','p',ids,mask,(0.,0.,1.,0.,-1.,0.,.4,-.2,0.,0.),(0.,0.,-3.1,0.,-3.1,0.,-3.1,-3.1,0.,0.))
        opts=dict(reference_model=ref,device='cpu',logit_chunk_size=2,max_context_tokens=64,
            policy_call_kwargs={},reference_call_kwargs={},projection_mode='full_sequence')
        def loss(p,k):return ppo_loss(p,torch.tensor([item.old_log_probs],dtype=p.dtype),
            torch.tensor([item.advantages],dtype=p.dtype),torch.tensor([mask]),clip_epsilon=.2,
            kl_coefficient=.001,reference_kl=k,reduction='sequence_mean')
        p,k=make_forward(**opts)(a,item)
        q,j=make_forward(**opts,saved_tensor_policy='cpu_exact_v1')(b,item)
        torch.testing.assert_close(p,q,rtol=0,atol=0);torch.testing.assert_close(k,j,rtol=0,atol=0)
        self.assertGreater(float(k.sum().detach()),0)
        x,y=loss(p,k),loss(q,j);torch.testing.assert_close(x,y,rtol=0,atol=0)
        x.backward();y.backward()
        for (name,l),(other,r) in zip(a.named_parameters(),b.named_parameters(),strict=True):
            self.assertEqual(name,other);torch.testing.assert_close(l.grad,r.grad,rtol=1e-10,atol=1e-12)
        self.assertGreater(float(b.model.embed_tokens.weight.grad[7].abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in ref.parameters()))
        self.assertGreater(stats['layer_executions'],stats['checkpoint_forwards'])
        self.assertFalse(b.training)

    def test_unknown_residency_is_rejected_before_forward(self):
        with self.assertRaisesRegex(ValueError,'saved.tensor'):
            make_forward(reference_model=object(),device='cpu',logit_chunk_size=2,max_context_tokens=64,
                policy_call_kwargs={},reference_call_kwargs={},saved_tensor_policy='detach_context')
