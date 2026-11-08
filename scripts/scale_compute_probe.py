"""Explicit Slurm-only saved-native probes. No generation/retrieval/scale training."""
import argparse,json,os,time,gc,math,threading
from pathlib import Path

def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False))
def main(a):
 if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
 from ricepo.artifacts import file_sha256
 if a.mode=='prepare':
  from ricepo.completed_credit_recovery import decode_trajectory
  root=Path(a.root); source=root/'runs/biology-scale-v2-training-ricepo/iteration-000000'
  raw=json.loads((source/'trajectories.json').read_text()); assert file_sha256(source/'trajectories.json')==json.loads((source/'rollout_complete.receipt.json').read_text())['artifacts']['trajectories.json']
  groups={}
  for x in raw:groups.setdefault(x['query_id'],[]).append(x)
  selected=[x for k in sorted(groups)[:8] for x in groups[k]];assert len(selected)==32
  base=json.loads((root/'reports/biology-scale-oom-prep-176722/PRIVATE_PLAN.json').read_text())
  base.update(trajectories=selected,entropy_chunk_size=64,entropy_abs_max=1e-5,speed_min=1.10,margin_bytes=8*1024**3)
  base['selection']='first eight sorted query identities, all four original samples; no reward/format filtering'
  base['source_trajectories_sha256']=file_sha256(source/'trajectories.json')
  write(a.output/'PLAN.json',base);return
 import torch
 from transformers import AutoModelForCausalLM
 from ricepo.modeling import make_forward,sampled_old_log_probs,enable_eval_layer_checkpointing
 from ricepo.objective import ppo_loss
 from ricepo.training import UpdateItem
 from ricepo.precision_pilot import mismatch_statistics,risk_screen
 from scripts.hf_compute_benchmark import gradient_comparison
 plan=json.loads(a.plan.read_text());
 if a.mode=='residency':
  paired=json.loads(a.paired.read_text());assert paired['status']=='PASS' and all(r['passed'] for r in paired['rows'] if r['experiment']=='C1')
 a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(6);torch.manual_seed(plan['root_seed']);torch.cuda.manual_seed_all(plan['root_seed'])
 torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 start=time.perf_counter();actor=AutoModelForCausalLM.from_pretrained(plan['actor'],dtype=torch.float32,local_files_only=True).eval().to('cuda')
 from ricepo.policy_identity import identify
 binding=json.loads((Path(plan['source_run'])/'binding.json').read_text())
 policy_id,evidence=identify(actor,binding['behavior'])
 if policy_id!=next(iter(plan['items'].values()))['policy_version']:raise ValueError('frozen actor identity mismatch')
 nvml={'status':'NOT_MEASURED','peak_used_bytes':None}; stop_monitor=threading.Event()
 try:
  import pynvml
  pynvml.nvmlInit();uid=str(torch.cuda.get_device_properties(0).uuid);handle=pynvml.nvmlDeviceGetHandleByUUID(uid if uid.startswith('GPU-') else 'GPU-'+uid)
  def poll():
   while not stop_monitor.is_set():
    used=pynvml.nvmlDeviceGetMemoryInfo(handle).used
    nvml.update(status='MEASURED',peak_used_bytes=max(used,nvml['peak_used_bytes'] or 0));stop_monitor.wait(.2)
  threading.Thread(target=poll,daemon=True).start()
 except Exception as exc:nvml['reason']=type(exc).__name__
 result={'status':'IN_PROGRESS' ,'plan_sha256':file_sha256(a.plan),'new_generation':0,'new_retrieval':0,'rows':[],'NVML':nvml,'initialization_seconds':time.perf_counter()-start}
 try:
  if a.mode=='entropy':
   from ricepo.backends import TransformersEntropyProvider
   from ricepo.completed_credit_recovery import decode_trajectory
   from ricepo.credit import select_anchors
   ts=[decode_trajectory(x,5) for x in plan['trajectories']];policy=ts[0].policy_version
   if any(t.policy_version!=policy_id for t in ts):raise ValueError('entropy behavior identity mismatch')
   opts=dict(model=actor,policy_version_getter=lambda:policy,device='cuda',chunk_size=plan['entropy_chunk_size'],max_context_tokens=32768,model_call_kwargs={})
   providers=[TransformersEntropyProvider(**opts),TransformersEntropyProvider(**opts,projection_mode='summary_positions_v1')]
   def run(provider,trajectories):
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();cpu=time.process_time();s=time.perf_counter()
    scores={t.trajectory_id:provider.token_entropies(t) for t in trajectories};torch.cuda.synchronize()
    return scores,dict(seconds=time.perf_counter()-s,cpu_seconds=time.process_time()-cpu,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),native_tokens=sum(len(t.token_ids) for t in trajectories),summary_tokens=sum(x.summary_span.end-x.summary_span.start for t in trajectories for x in t.steps))
   def compare(ts,x,y):
    error=max([abs(x[t.trajectory_id][i]-y[t.trajectory_id][i]) for t in ts for s in t.steps for i in range(s.summary_span.start,s.summary_span.end)] or [0.])
    groups={}
    for t in ts:groups.setdefault(t.query_id,[]).append(t)
    same=True
    for group in groups.values():
     key=lambda z:[(q.trajectory_id,q.depth) for q in z]
     for budget in (4,sum(len(t.steps) for t in group)):
      same &= key(select_anchors(group,{t.trajectory_id:x[t.trajectory_id] for t in group},budget))==key(select_anchors(group,{t.trajectory_id:y[t.trajectory_id] for t in group},budget))
    return dict(entropy_max_abs=error,all_rankings_and_top4_equal=same,passed=error<=plan['entropy_abs_max'] and same)
   for count in (8,32):
    pairs=[]
    for repeat in range(2):
     x,b=run(providers[0],ts[:count]);y,c=run(providers[1],ts[:count]);checks=compare(ts[:count],x,y)
     row=dict(experiment='B',count=count,repeat=repeat,baseline=b,candidate=c,checks=checks,speedup=b['seconds']/c['seconds']);result['rows'].append(row);write(a.output/'RESULT.json',result)
     if not checks['passed']:raise ValueError('entropy/rank gate failed; no expansion')
   result['status']='PASS';return
  reference=AutoModelForCausalLM.from_pretrained(plan['actor'],dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to('cuda');enable_eval_layer_checkpointing(actor)
  items=[UpdateItem(**{k:tuple(v) if isinstance(v,list) else v for k,v in plan['items'][label].items()}) for label in ('short','median','largest')]
  opts=dict(reference_model=reference,device='cuda',logit_chunk_size=plan['logit_chunk_size'],max_context_tokens=32768,policy_call_kwargs={},reference_call_kwargs={})
  from scripts.hf_compute_benchmark import Timers
  def calculate(item,projection,offload,separate):
   timers=Timers(torch)
   actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();nvml['peak_used_bytes']=None;start=time.perf_counter();old_seconds=0;teacher=None
   if separate:
    s=time.perf_counter();teacher=sampled_old_log_probs(actor,item.token_ids,item.generated_mask,device='cuda',logit_chunk_size=plan['logit_chunk_size'],max_context_tokens=32768,policy_call_kwargs={},projection_mode=projection).detach().cpu();torch.cuda.synchronize();old_seconds=time.perf_counter()-s
   p,k=make_forward(**opts,projection_mode=projection,saved_tensor_policy=offload,measure=timers)(actor,item)
   old=torch.tensor([item.old_log_probs],device='cuda');adv=torch.tensor([item.advantages],device='cuda');mask=torch.tensor([item.generated_mask],device='cuda')
   lp=p.detach().cpu();kl=k.detach().cpu();score=lp if teacher is None else teacher
   risk=mismatch_statistics(old[mask].cpu().tolist(),score[0,mask[0].cpu()].tolist(),adv[mask].cpu().tolist())
   loss=ppo_loss(p,old,adv,mask,clip_epsilon=.2,kl_coefficient=.001,reference_kl=k,reduction='sequence_mean');lossval=float(loss.detach());torch.cuda.synchronize();back_start=time.perf_counter();loss.backward();torch.cuda.synchronize();back_seconds=time.perf_counter()-back_start
   row=dict(seconds=time.perf_counter()-start,backward_wall_seconds=back_seconds,cuda_stage_seconds=timers.totals(),old_seconds=old_seconds,loss=lossval,risk=risk,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),nvml_peak_bytes=nvml['peak_used_bytes'],finite=all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in actor.parameters()))
   if not risk_screen(risk):actor.zero_grad(set_to_none=True);raise ValueError('saved original risk failed')
   return row,lp,kl
  for label,item in zip(('short','median','largest'),items):
   for repeat in range(2):
    b,lp,kl=calculate(item,'action_positions_v1' if a.mode=='residency' else 'full_sequence','cpu_exact_v1',True);grad={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in actor.named_parameters()}
    for name,projection,separate in ([('C2','action_positions_v1',True)] if a.mode=='residency' else [('A','full_sequence',False),('C1','action_positions_v1',True)]):
     c,l,k=calculate(item,projection,'device' if a.mode=='residency' else 'cpu_exact_v1',separate);g=gradient_comparison(torch,actor,grad)
     checks=dict(logprob_max=float((lp-l).abs().max()),KL_max=float((kl-k).abs().max()),loss_scaled=abs(b['loss']-c['loss'])/max(1.,abs(b['loss'])),gradient_L2=g['global_relative_L2'],gradient_cosine=g['global_cosine'],risk_same=b['risk']==c['risk'],finite=c['finite'])
     passed=checks['logprob_max']<=1e-4 and checks['KL_max']<=1e-6 and checks['loss_scaled']<=1e-5 and g['global_relative_L2']<=.001 and g['global_cosine']>=.99999 and c['finite']
     if a.mode=='residency':
      checks['8GiB_headroom']=c['nvml_peak_bytes'] is not None and torch.cuda.get_device_properties(0).total_memory-c['nvml_peak_bytes']>=plan['margin_bytes'];passed=passed and checks['8GiB_headroom']
     row=dict(native_tokens=len(item.token_ids),action_tokens=sum(item.generated_mask),horizon=max(i for i,m in enumerate(item.generated_mask) if m),experiment=name,fixture=label,repeat=repeat,baseline=b,candidate=c,checks=checks,passed=passed,speedup=b['seconds']/c['seconds']);result['rows'].append(row);write(a.output/'RESULT.json',result)
     if not passed:raise ValueError(name+' numerical failure; stop expansion')
    del grad
  result['status']='PASS';result['C2']='NOT_RUN: select threshold only after paired C1 measurements';result['offline_optimizer_step']='NOT_RUN_PENDING_FORWARD_GRADIENT_RESULTS'
 except BaseException as exc:
  result.update(status='FAIL',error_type=type(exc).__name__,error=str(exc));raise
 finally:
  stop_monitor.set();write(a.output/'RESULT.json',result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','entropy','compute','residency'],required=True);p.add_argument('--root');p.add_argument('--paired',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
