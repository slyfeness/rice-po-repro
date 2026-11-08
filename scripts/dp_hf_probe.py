"""Offline manual SUM data-parallel measurement; never resumes training."""
import argparse,json,os,time,gc,datetime
from pathlib import Path
from scripts.gpu_probe_common import write,partition,Monitor,numeric,load_models,forward_factory,item_backward,tensor_statistics,digest

def main(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    import torch
    import torch.distributed as dist
    from ricepo.training import UpdateItem,_validate_items
    from ricepo.policy_identity import identify
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(hours=2));world=dist.get_world_size()
    if world!=2:raise ValueError('exactly two ranks required')
    plan=json.loads(a.plan.read_text());a.output.mkdir(parents=True,exist_ok=True)
    actor,reference,idle,init=load_models(plan,device,diver=rank==1)
    initial={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    forward=forward_factory(reference,device,plan['logit_chunk_size'])
    recipe=plan['recipe'];recipe=recipe.get('resolved',{}).get('values',recipe.get('values',recipe))
    result=dict(state='RUNNING',metadata=plan['public_metadata'],initialization=init,rank=rank,rows=[],optimizer='fresh AdamW at P1; not resumed Adam',thresholds=None)
    base_grad=base_post=None
    try:
        phases=[(64,rep,kind) for rep in range(2) for kind in ('single','dual')]
        if a.dual512_only:phases=[(512,0,'dual')]
        elif a.full512:phases += [(512,0,kind) for kind in ('single','dual')]
        for count,rep,kind in phases:
            indices=plan['sub64_indices'] if count==64 else list(range(512))
            items=[UpdateItem(**{k:tuple(v) if isinstance(v,list) else v for k,v in plan['items'][i].items()}) for i in indices]
            _validate_items(items,plan['policy_version'])
            shards=partition(items,2) if kind=='dual' else [list(range(len(items))),[]]
            assert sorted(shards[0]+shards[1])==list(range(count))
            actor.load_state_dict(initial);actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
            torch.manual_seed(20261108);torch.cuda.manual_seed_all(20261108)
            label=f'{kind}-Sub{count}-repeat{rep}';mon=Monitor(torch,device,label)
            dist.barrier();torch.cuda.synchronize(device);start=time.perf_counter();rows=[];loss_sum=0.
            for j,i in enumerate(shards[rank]):
                loss,stats=item_backward(actor,forward,items[i],device,count,'ieee');loss_sum+=loss;rows.append((i,stats))
                if (j+1)%8==0:write(a.output/f'progress-rank{rank}.json',dict(label=label,completed=j+1,planned=len(shards[rank]),elapsed=time.perf_counter()-start))
            torch.cuda.synchronize(device);local_seconds=time.perf_counter()-start
            dist.barrier();reduce_start=time.perf_counter()
            if kind=='dual':
                flags=[p.grad is not None for p in actor.parameters()];allflags=[None]*2;dist.all_gather_object(allflags,flags)
                if allflags[0]!=allflags[1]:raise ValueError('gradient presence differs across ranks')
                for p in actor.parameters():
                    if p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device);reduce_seconds=time.perf_counter()-reduce_start
            gathered=[None]*2;dist.all_gather_object(gathered,dict(rows=rows,loss=loss_sum,local_seconds=local_seconds,reduce_seconds=reduce_seconds))
            ordered=sorted([r for g in gathered for r in g['rows']],key=lambda r:r[0])
            if [i for i,_ in ordered]!=list(range(count)):raise ValueError('missing or repeated sequences')
            stats={k:torch.cat([r[k] for _,r in ordered]) for k in ('q','p','k','a')}
            risk=numeric(stats['q'],stats['p'],stats['k'],stats['a'])
            dist.barrier();torch.cuda.synchronize(device);wall=time.perf_counter()-start;util=mon.finish()
            row=dict(label=label,count=count,repeat=rep,kind=kind,HF_wall_seconds=wall,rank_forward_backward_seconds=[x['local_seconds'] for x in gathered],rank_gradient_sum_seconds=[x['reduce_seconds'] for x in gathered],normalized_loss=sum(x['loss'] for x in gathered),risk=risk,allocated_peak_bytes=torch.cuda.max_memory_allocated(device),reserved_peak_bytes=torch.cuda.max_memory_reserved(device),GPU_util=util,OOM=False,optimizer_steps=0)
            with (a.output/f'GPU_UTIL-rank{rank}.jsonl').open('a') as f:
                for r in mon.rows:f.write(json.dumps(dict(rank=rank,**r))+'\n')
            # Diagnostics / step excluded from HF wall; no graphs retained.
            if count==64:
                if rank==0:
                    gradient={k:None if p.grad is None else p.grad.detach().cpu().clone() for k,p in actor.named_parameters()}
                    if kind=='single':base_grad=gradient
                    else:row['gradient_vs_single']=tensor_statistics(base_grad,gradient)
                if kind=='dual' or rank==0:
                    norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),recipe['max_grad_norm'],error_if_nonfinite=True)
                    hp=recipe['optimizer_hyperparameters'];opt=torch.optim.AdamW(actor.parameters(),lr=recipe['actor_learning_rate'],betas=tuple(hp['betas']),eps=hp['eps'],weight_decay=hp['weight_decay'])
                    t=time.perf_counter();opt.step();torch.cuda.synchronize(device);row.update(optimizer_steps=1,gradient_norm=float(norm),optimizer_step_seconds=time.perf_counter()-t)
                    post={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
                    row['parameter_changed']=any(not torch.equal(v,initial[k]) for k,v in post.items())
                    if rank==0:
                        if kind=='single':base_post=post
                        else:row['post_step_vs_single']=tensor_statistics(base_post,post)
                    row['post_step_state_sha256']=identify(actor,plan['behavior'])[1]['state_dict_sha256']
                    del opt,post
                hashes=[None]*2;dist.all_gather_object(hashes,row.get('post_step_state_sha256'))
                row['rank_weight_sha256']=hashes;row['rank_weight_hash_equal']=hashes[0]==hashes[1] if kind=='dual' else None
            row['reference_has_gradient']=any(p.grad is not None for p in reference.parameters())
            result['rows'].append(row);write(a.output/f'RESULT-rank{rank}.json',result)
            del rows,gathered,ordered,stats;actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();dist.barrier()
        result['state']='COMPLETED'
    except BaseException as e:
        result.update(state='ERROR',error_type=type(e).__name__,OOM=isinstance(e,torch.OutOfMemoryError));raise
    finally:
        write(a.output/f'RESULT-rank{rank}.json',result)
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--full512',action='store_true');p.add_argument('--dual512-only',action='store_true');main(p.parse_args())
