"""Offline measurements only; public output excludes native histories and paths."""
import hashlib, json, math, os, random, time, subprocess, threading, csv
from pathlib import Path
from contextlib import nullcontext


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def horizon(item):
    mask=item['generated_mask'] if isinstance(item,dict) else item.generated_mask
    return max(i for i,m in enumerate(mask) if m)


def sub64_indices(items):
    # Eight equal-count horizon strata, eight identities by SHA in each.
    ordered=sorted(range(len(items)),key=lambda i:(horizon(items[i]),digest(items[i]['trajectory_id'])))
    if len(items)!=512:raise ValueError('expected committed512')
    selected=[]
    for start in range(0,512,64):
        group=ordered[start:start+64]
        take=sorted(group,key=lambda i:digest(['Sub64-v1',items[i]['trajectory_id']]))[:8]
        if start==448 and ordered[-1] not in take:take[-1]=ordered[-1]
        selected.extend(take)
    return selected


def partition(items, world):
    shards=[[] for _ in range(world)];loads=[0]*world
    for i in sorted(range(len(items)),key=lambda i:(-horizon(items[i]),i)):
        rank=min(range(world),key=lambda r:(loads[r],r));shards[rank].append(i);loads[rank]+=horizon(items[i])
    return shards


def splits(count):
    result=[]
    for seed in range(202611080,202611085):
        order=list(range(count));random.Random(seed).shuffle(order)
        result.append(set(order[:count//2]))
    return result


class Monitor:
    def __init__(self,torch,device,label):
        self.rows=[];self.stop=threading.Event();self.label=label;self.start=time.perf_counter()
        uid=str(torch.cuda.get_device_properties(device).uuid)
        self.uid=uid if uid.startswith('GPU-') else 'GPU-'+uid
        self.thread=threading.Thread(target=self.poll,daemon=True);self.thread.start()
    def poll(self):
        while not self.stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','-i',self.uid,'--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True,timeout=4)
                u,m,p=map(float,raw.strip().split(','))
                self.rows.append(dict(label=self.label,seconds=time.perf_counter()-self.start,utilization_percent=u,memory_MiB=m,power_W=p))
            except Exception as e:
                self.rows.append(dict(label=self.label,seconds=time.perf_counter()-self.start,error=type(e).__name__))
            self.stop.wait(1)
    def finish(self):
        self.stop.set();self.thread.join(timeout=6)
        vals=[x for x in self.rows if 'error' not in x]
        summary={}
        for k in ['utilization_percent','memory_MiB','power_W']:
            a=[x[k] for x in vals];summary[k]={'mean':sum(a)/len(a),'max':max(a)} if a else None
        return summary


def numeric(q,p,kl,adv):
    import torch
    q,p,kl,adv=[torch.as_tensor(x,dtype=torch.float64) for x in (q,p,kl,adv)]
    d=p-q;err=d.abs();r=d.exp();w=(d-d.max()).exp()
    return dict(tokens=len(q),logprob_mean_abs=float(err.mean()),logprob_p99_abs=float(torch.quantile(err,.99)),logprob_max_abs=float(err.max()),ratio_min=float(r.min()),ratio_max=float(r.max()),ratio_outside_fraction=float(((r<.8)|(r>1.2)).double().mean()),effective_clip_fraction=float((((r>1.2)&(adv>0))|((r<.8)&(adv<0))).double().mean()),extreme_logratio_fraction=float((err>1).double().mean()),ESS_fraction=float(w.sum().square()/(len(w)*w.square().sum())),KL_sum=float(kl.sum()),KL_token_mean=float(kl.mean()))


def load_models(plan,device,precision='ieee',diver=False):
    import torch
    from transformers import AutoModelForCausalLM,AutoModel
    from ricepo.policy_identity import identify
    from ricepo.modeling import enable_eval_layer_checkpointing
    torch.set_num_threads(6);torch.manual_seed(20261108);torch.cuda.manual_seed_all(20261108)
    torch.set_float32_matmul_precision('high' if precision=='tf32' else 'highest')
    torch.backends.cuda.matmul.allow_tf32=precision=='tf32';torch.backends.cudnn.allow_tf32=False
    s=time.perf_counter()
    actor=AutoModelForCausalLM.from_pretrained(plan['actor'],dtype=torch.float32,local_files_only=True).eval().to(device)
    identity,evidence=identify(actor,plan['behavior'])
    if identity!=plan['policy_version']:raise ValueError('P1 native behavior identity differs')
    reference=AutoModelForCausalLM.from_pretrained(plan['reference'],dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
    _,ref_evidence=identify(reference,plan['behavior'])
    if ref_evidence['state_dict_sha256']!=plan['reference_state_sha256']:raise ValueError('P0 reference identity differs')
    enable_eval_layer_checkpointing(actor)
    idle=None
    if diver:
        idle=AutoModel.from_pretrained(plan['diver'],torch_dtype=getattr(torch,plan['diver_dtype']),local_files_only=True).eval().requires_grad_(False).to(device)
    torch.cuda.synchronize(device)
    return actor,reference,idle,dict(seconds=time.perf_counter()-s,actor_state_sha256=evidence['state_dict_sha256'],reference_state_sha256=ref_evidence['state_dict_sha256'],matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,actor_dtype=str(next(actor.parameters()).dtype),attention=actor.config._attn_implementation,diver_idle=diver)


def forward_factory(reference,device,chunk):
    from ricepo.modeling import make_forward
    return make_forward(reference_model=reference,device=device,logit_chunk_size=chunk,max_context_tokens=32768,policy_call_kwargs={},reference_call_kwargs={},projection_mode='action_positions_v1',saved_tensor_policy='cpu_above_budget_v1',device_residency_budget={'horizon':5535,'action_tokens':1889})


def item_backward(actor,forward,item,device,denominator,precision):
    import torch
    from ricepo.objective import ppo_loss
    with torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext():
        current,kl=forward(actor,item)
        old=torch.tensor([item.old_log_probs],device=device,dtype=torch.float32)
        adv=torch.tensor([item.advantages],device=device,dtype=torch.float32)
        mask=torch.tensor([item.generated_mask],device=device,dtype=torch.bool)
        loss=ppo_loss(current,old,adv,mask,clip_epsilon=.2,kl_coefficient=.001,reference_kl=kl,reduction='sum')
        (loss/denominator).backward()
    return float(loss.detach())/denominator,{k:v.detach()[mask].cpu() for k,v in dict(q=old,p=current,k=kl,a=adv).items()}


def tensor_statistics(reference,candidate):
    # Chunked CPU float64 statistics over ALL parameters; no gradient thresholds.
    import torch
    aa=bb=dd=ab=0.;maximum=0.;counts=[0]*7;n=0
    bounds=[0.,1e-8,1e-7,1e-6,1e-5,1e-4]
    for name,x in reference.items():
        y=candidate[name]
        if (x is None)!=(y is None):raise ValueError('parameter/gradient presence mismatch')
        if x is None:continue
        for a,b in zip(x.reshape(-1).split(1000000),y.reshape(-1).split(1000000)):
            a,b=a.double(),b.double();d=b-a;e=d.abs()
            aa+=float(a.square().sum());bb+=float(b.square().sum());dd+=float(d.square().sum());ab+=float((a*b).sum());maximum=max(maximum,float(e.max()));n+=e.numel()
            counts[0]+=int((e==0).sum())
            for j in range(1,len(bounds)):counts[j]+=int(((e>bounds[j-1])&(e<=bounds[j])).sum())
            counts[-1]+=int((e>bounds[-1]).sum())
    return dict(reference_norm=math.sqrt(aa),candidate_norm=math.sqrt(bb),difference_L2=math.sqrt(dd),relative_L2=math.sqrt(dd/aa) if aa else None,cosine=ab/math.sqrt(aa*bb) if aa*bb else None,max_abs=maximum,elements=n,absolute_difference_histogram={'upper_bounds':bounds+['inf'],'counts':counts})
