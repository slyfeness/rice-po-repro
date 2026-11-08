"""Bounded saved-native IEEE/full-sequence memory repair probe, zero steps."""
import argparse,gc,json,os,time,math
from pathlib import Path


def main(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicit Slurm required')
    import torch
    from transformers import AutoModelForCausalLM
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.asset_identity import local_model_identity
    from ricepo.policy_identity import identify
    from ricepo.modeling import make_forward,enable_eval_layer_checkpointing
    from ricepo.objective import ppo_loss
    from ricepo.training import UpdateItem
    from scripts.hf_compute_benchmark import gradient_comparison
    plan=json.loads(args.plan.read_text());source=Path(plan['source_run'])
    if file_sha256(source/'binding.json')!=plan['source_binding_sha256']:raise ValueError('source binding changed')
    binding=json.loads((source/'binding.json').read_text());old=json.loads((source/'model-sources.json').read_text())
    if local_model_identity(plan['actor'])!=old['actor_model_path']:raise ValueError('actual initial model assets changed')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(6);torch.manual_seed(plan['root_seed']);torch.cuda.manual_seed_all(plan['root_seed'])
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    actor=AutoModelForCausalLM.from_pretrained(plan['actor'],dtype=torch.float32,local_files_only=True,trust_remote_code=False).eval().to('cuda:0')
    ref=AutoModelForCausalLM.from_pretrained(plan['actor'],dtype=torch.float32,local_files_only=True,trust_remote_code=False).eval().requires_grad_(False).to('cuda:0')
    policy,evidence=identify(actor,binding['behavior'])
    if policy!=next(iter(plan['items'].values()))['policy_version']:raise ValueError('actual P0 weights differ from saved behavior')
    stats=enable_eval_layer_checkpointing(actor)
    opts=dict(reference_model=ref,device='cuda:0',logit_chunk_size=plan['logit_chunk_size'],max_context_tokens=32768,
        policy_call_kwargs={},reference_call_kwargs={},projection_mode='full_sequence')
    reserve=[]
    if args.role=='worst':
        # Model the deferred-residency candidate: moments stay on CPU during
        # backward, and return only after the graph has been consumed. These
        # are capacity reserves, not real optimizer states or learning steps.
        reserve=[torch.zeros_like(p,device='cpu') for p in actor.parameters() for _ in range(2)]
    result=dict(role=args.role,status='IN_PROGRESS',new_optimizer_steps=0,precision='IEEE FP32',projection='full_sequence',
        model_state_sha256=evidence['state_dict_sha256'],plan_sha256=file_sha256(args.plan),fixtures=[],
        allocator_environment=os.environ.get('PYTORCH_ALLOC_CONF'),
        allocator_settings=torch.cuda.memory._snapshot().get('allocator_settings','NOT_EXPOSED'),
        reserve_bytes=sum(p.numel()*p.element_size() for p in reserve),attention_implementation=actor.config._attn_implementation)
    def calculate(label,mode):
        data=plan['items'][label];item=UpdateItem(**{k:tuple(v) if isinstance(v,list) else v for k,v in data.items()})
        actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.monotonic();print('FIXTURE_BEGIN',args.role,label,mode,len(item.token_ids),flush=True)
        p,k=make_forward(**opts,saved_tensor_policy=mode)(actor,item)
        loss=ppo_loss(p,torch.tensor([item.old_log_probs],device='cuda:0'),torch.tensor([item.advantages],device='cuda:0'),
            torch.tensor([item.generated_mask],device='cuda:0'),clip_epsilon=.2,kl_coefficient=.001,reference_kl=k,reduction='sequence_mean')
        values=dict(lp=p.detach().cpu(),kl=k.detach().cpu(),loss=float(loss.detach()))
        loss.backward();torch.cuda.synchronize()
        values.update(seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),finite=all(x.grad is None or bool(torch.isfinite(x.grad).all()) for x in actor.parameters()),
            reference_has_gradient=any(p.grad is not None for p in ref.parameters()))
        del p,k,loss;print('FIXTURE_BACKWARD_COMPLETE',args.role,label,mode,values['seconds'],values['peak_allocated_bytes'],flush=True)
        return values
    try:
        if args.role=='compare':
            for label in ([args.fixture] if args.fixture else ('short','median')):
                a=calculate(label,'device');baseline={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in actor.named_parameters()}
                b=calculate(label,'cpu_exact_v1');g=gradient_comparison(torch,actor,baseline)
                maxerr=max((r.get('max_abs_error',0.) for r in g['parameters']),default=0.)
                gates=plan['gates'];check=dict(lp_exact=torch.equal(a['lp'],b['lp']),KL_exact=torch.equal(a['kl'],b['kl']),loss_exact=a['loss']==b['loss'],
                    finite=b['finite'],reference_frozen=not b['reference_has_gradient'],
                    gradient_L2=g['global_relative_L2'] is not None and g['global_relative_L2']<=gates['gradient_global_relative_L2_max'],
                    gradient_cosine=g['global_cosine']>=gates['gradient_global_cosine_min'],gradient_max_abs=maxerr<=gates['gradient_max_abs_max'])
                row=dict(fixture=label,checks=check,gradient_global_L2=g['global_relative_L2'],gradient_global_cosine=g['global_cosine'],
                    gradient_max_abs=maxerr,baseline={k:v for k,v in a.items() if k not in ('lp','kl')},candidate={k:v for k,v in b.items() if k not in ('lp','kl')})
                result['fixtures'].append(row);atomic_json(args.output/('fixture-'+label+'.json'),row)
                if not all(check.values()):raise ValueError('predeclared exact residency comparison failed')
                del baseline,a,b,g
        else:
            b=calculate('largest','cpu_exact_v1')
            # The old resident-moment probe is preserved as a failure; this
            # candidate explicitly changes residency, never precision/values.
            reserve=[p.to('cuda:0') for p in reserve];torch.cuda.synchronize()
            result['moments_restored_after_backward']=True
            result['allocated_after_moment_restoration']=torch.cuda.memory_allocated()
            result['fixtures'].append(dict(fixture='largest_with_deferred_FP32_Adam_moment_capacity_reserves',
                **{k:v for k,v in b.items() if k not in ('lp','kl')}))
            if not b['finite'] or b['reference_has_gradient']:raise ValueError('worst history gradient failed')
        result.update(status='PASS',actual_layer_recomputation=stats['layer_executions']>stats['checkpoint_forwards'])
    except BaseException as e:
        result.update(status='FAIL',error_type=type(e).__name__,message=str(e));raise
    finally:atomic_json(args.output/'RESULT.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=['compare','worst'],required=True);p.add_argument('--fixture',choices=['short','median']);main(p.parse_args())
