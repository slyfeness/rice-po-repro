"""Three saved-native items, two independent offline optimizer steps for parity."""
import argparse,json,os,gc,time,hashlib
from pathlib import Path

def main(a):
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
 import torch
 from transformers import AutoModelForCausalLM
 from ricepo.modeling import make_forward,sampled_old_log_probs,enable_eval_layer_checkpointing
 from ricepo.training import UpdateItem,ppo_plan_update
 from ricepo.fused_probability import FusedOriginalProbabilityGuard
 from ricepo.precision_pilot import mismatch_statistics,risk_screen
 from ricepo.policy_identity import identify
 from scripts.hf_compute_benchmark import gradient_comparison
 from scripts.scale_compute_probe import write
 checks=json.loads(a.paired.read_text());assert checks['status']=='PASS' and all(r['passed'] for r in checks['rows'] if r['experiment']=='A')
 plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 actor=AutoModelForCausalLM.from_pretrained(plan['actor'],dtype=torch.float32,local_files_only=True).eval().to('cuda')
 ref=AutoModelForCausalLM.from_pretrained(plan['actor'],dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to('cuda')
 binding=json.loads((Path(plan['source_run'])/'binding.json').read_text());policy,evidence=identify(actor,binding['behavior'])
 items=[UpdateItem(**{k:tuple(v) if isinstance(v,list) else v for k,v in plan['items'][label].items()}) for label in ('short','median','largest')]
 assert all(i.policy_version==policy for i in items)
 recipe=json.loads(Path('docs/reviews/biology-scale-training-v1/RESOLVED_CONFIGS/training_grpo.json').read_text())['resolved']['values']
 recipe.update(ppo_epochs=1,microbatch_size=1,optimizer_minibatch_size='full_batch',gradient_accumulation='full_batch')
 initial={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()};enable_eval_layer_checkpointing(actor)
 forward=make_forward(reference_model=ref,device='cuda',logit_chunk_size=plan['logit_chunk_size'],max_context_tokens=32768,policy_call_kwargs={},reference_call_kwargs={},saved_tensor_policy='cpu_exact_v1')
 result={'scope':'offline three fixed saved trajectories; no online training or generation','status':'IN_PROGRESS','rows':[]};baseline_grad=None;baseline_post=None
 try:
  for fused in (False,True):
   actor.load_state_dict(initial);actor.zero_grad(set_to_none=True);torch.manual_seed(plan['root_seed']);torch.cuda.manual_seed_all(plan['root_seed']);gc.collect();torch.cuda.empty_cache()
   optparams=recipe['optimizer_hyperparameters'];opt=torch.optim.AdamW(actor.parameters(),lr=recipe['actor_learning_rate'],betas=tuple(optparams['betas']),eps=optparams['eps'],weight_decay=optparams['weight_decay'])
   reports=[];guard=None;grad_result=None;torch.cuda.synchronize();s=time.perf_counter()
   if not fused:
    q=[];p=[];adv=[]
    for item in items:
     scores=sampled_old_log_probs(actor,item.token_ids,item.generated_mask,device='cuda',logit_chunk_size=plan['logit_chunk_size'],max_context_tokens=32768,policy_call_kwargs={})[0].cpu().tolist()
     for x,y,z,m in zip(item.old_log_probs,scores,item.advantages,item.generated_mask,strict=True):
      if m:q.append(x);p.append(y);adv.append(z)
    reports.append(mismatch_statistics(q,p,adv));assert risk_screen(reports[0])
   else:
    # Saved fixtures are GRPO originals: branch group is explicitly empty here.
    guard=FusedOriginalProbabilityGuard(outside_max=.01,branch_gate=lambda:None,save_report=reports.append)
   def observe(stage):
    nonlocal baseline_grad,grad_result
    if stage=='before_clip':
     if not fused:baseline_grad={k:None if v.grad is None else v.grad.detach().cpu().clone() for k,v in actor.named_parameters()}
     else:grad_result=gradient_comparison(torch,actor,baseline_grad)
   diag=ppo_plan_update(actor,opt,items,policy_version=policy,forward=forward,device='cuda',recipe=recipe,probability_guard=guard,step_observer=observe)
   torch.cuda.synchronize();elapsed=time.perf_counter()-s
   post={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
   changed=any(not torch.equal(post[k],initial[k]) for k in post)
   if not fused:baseline_post=post
   else:
    assert all(torch.equal(post[k],baseline_post[k]) for k in post),'post-update parameters differ'
    assert grad_result['global_relative_L2']<=.001 and grad_result['global_cosine']>=.99999
    assert reports[0]==result['rows'][0]['risk']
   assert changed and all(int(v['step'])==1 for v in opt.state.values())
   assert not any(v.grad is not None for v in ref.parameters())
   result['rows'].append(dict(fused=fused,diagnostics=diag,risk=reports[0],parameter_changed=changed,Adam_steps=[1],wall_including_gradient_diagnostic_seconds=elapsed,gradient_global_L2=None if grad_result is None else grad_result['global_relative_L2'],gradient_cosine=None if grad_result is None else grad_result['global_cosine']))
   del opt;gc.collect();torch.cuda.empty_cache();write(a.output/'RESULT.json',result)
  result.update(status='PASS',post_update_all_parameters_bitwise_equal=True,branches='not applicable: saved GRPO-original fixtures; online RICE branch integration not qualified by this test')
 except BaseException as e:result.update(status='FAIL',error_type=type(e).__name__,error=str(e));raise
 finally:write(a.output/'RESULT.json',result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--paired',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
