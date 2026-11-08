"""IEEE/TF32/BF16 offline gradient measurements, with fixed half-split sums."""
import argparse,json,os,time,gc,math
from pathlib import Path
from scripts.gpu_probe_common import write,Monitor,numeric,load_models,forward_factory,item_backward,tensor_statistics,splits

def main(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    import torch
    from ricepo.training import UpdateItem,_validate_items
    torch.cuda.set_device(0);device=torch.device('cuda',0)
    plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=False)
    indices=plan['sub64_indices'][:a.count]
    items=[UpdateItem(**{k:tuple(v) if isinstance(v,list) else v for k,v in plan['items'][i].items()}) for i in indices]
    _validate_items(items,plan['policy_version'])
    actor,reference,idle,init=load_models(plan,device)
    forward=forward_factory(reference,device,plan['logit_chunk_size'])
    result=dict(state='RUNNING',count=a.count,metadata=plan['public_metadata'],initialization=init,rows=[],thresholds=None,gradient_normalization='sum, denominator1, no clipping and no optimizer.step')
    base_grad=None;base_stats=None;differences={}
    try:
        for precision in ('ieee','tf32','bf16'):
            torch.set_float32_matmul_precision('high' if precision=='tf32' else 'highest');torch.backends.cuda.matmul.allow_tf32=precision=='tf32'
            for repeat in range(2):
                actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                label=f'{precision}-Sub{a.count}-repeat{repeat}';mon=Monitor(torch,device,label);start=time.perf_counter();stats=[];loss=0.
                for i,item in enumerate(items):
                    l,s=item_backward(actor,forward,item,device,1,precision);loss+=l;stats.append(s)
                    if (i+1)%8==0:write(a.output/'progress.json',dict(label=label,completed=i+1,planned=len(items),elapsed=time.perf_counter()-start))
                full={k:torch.cat([s[k] for s in stats]) for k in ('q','p','k','a')}
                risk=numeric(full['q'],full['p'],full['k'],full['a']);torch.cuda.synchronize();wall=time.perf_counter()-start;util=mon.finish()
                gradient={k:None if p.grad is None else p.grad.detach().cpu().clone() for k,p in actor.named_parameters()}
                row=dict(precision=precision,repeat=repeat,HF_wall_seconds=wall,loss_sum=loss,risk=risk,allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),GPU_util=util,OOM=False,matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,parameter_dtype=str(next(actor.parameters()).dtype),autocast_dtype='bfloat16' if precision=='bf16' else None)
                if base_grad is None:base_grad=gradient;base_stats=full
                row['gradient_vs_IEEE']=tensor_statistics(base_grad,gradient)
                delta=(full['p'].double()-base_stats['p'].double()).abs();kd=(full['k'].double()-base_stats['k'].double()).abs()
                row['same_token_vs_IEEE']=dict(logprob_mean_abs=float(delta.mean()),logprob_p99_abs=float(torch.quantile(delta,.99)),logprob_max_abs=float(delta.max()),KL_mean_abs=float(kd.mean()),KL_max_abs=float(kd.max()))
                differences.setdefault(precision,[]).append(row['gradient_vs_IEEE']['difference_L2'])
                result['rows'].append(row);write(a.output/'RESULT.json',result)
                with (a.output/'GPU_UTIL.jsonl').open('a') as f:
                    for r in mon.rows:f.write(json.dumps(r)+'\n')
                del gradient,stats,full;actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        # Linearity: one extra IEEE per-item backward contributes to all five
        # fixed half partitions. No ten redundant GPU passes, no retained graphs.
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        halves=splits(a.count);buffers=[]
        for h in halves:buffers.append({name:torch.zeros(p.shape,dtype=torch.float64,device='cpu') for name,p in actor.named_parameters() if p.requires_grad})
        label=f'half-split-noise-Sub{a.count}';mon=Monitor(torch,device,label);torch.cuda.synchronize();start=time.perf_counter();gpu_wall=0.
        for i,item in enumerate(items):
            actor.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();item_backward(actor,forward,item,device,1,'ieee');torch.cuda.synchronize();gpu_wall+=time.perf_counter()-t
            for name,p in actor.named_parameters():
                if p.grad is None:continue
                g=p.grad.detach().cpu().double()
                for half,b in zip(halves,buffers):b[name].add_(g,alpha=1 if i in half else -1)
            if (i+1)%8==0:write(a.output/'progress.json',dict(label=label,completed=i+1,planned=len(items),elapsed=time.perf_counter()-start))
        noise=[]
        for seed,b in zip(range(202611080,202611085),buffers):
            norm=math.sqrt(sum(float(t.square().sum()) for t in b.values()))
            noise.append(dict(seed=seed,half_a_count=a.count//2,half_c_count=a.count//2,half_difference_L2=norm,precision_difference_over_half_difference={k:[v/norm if norm else None for v in vals] for k,vals in differences.items()}))
        result['half_split_noise']=dict(rows=noise,wall_seconds=time.perf_counter()-start,forward_backward_seconds=gpu_wall,method='same per-trajectory IEEE gradient used for all five fixed signed half-sums; CPUfloat64 accumulation; numerically not five separate FP32 accumulation orders',GPU_util=mon.finish())
        with (a.output/'GPU_UTIL.jsonl').open('a') as f:
            for r in mon.rows:f.write(json.dumps(r)+'\n')
        result.update(state='COMPLETED',optimizer_steps=0,reference_has_gradient=any(p.grad is not None for p in reference.parameters()))
    except BaseException as e:
        result.update(state='ERROR',error_type=type(e).__name__,OOM=isinstance(e,torch.OutOfMemoryError));raise
    finally:write(a.output/'RESULT.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,choices=[16,64],required=True);main(p.parse_args())
