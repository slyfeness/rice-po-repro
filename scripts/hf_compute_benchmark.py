"""Explicit offline native-trajectory HF benchmark. Never generates or retrieves.

Requires a private frozen plan and explicit Slurm allocation. Baseline, candidate
and winner-step processes are separate, so precision flags cannot leak between
workers. Raw histories/gradients/checkpoints stay in the caller's private output.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import traceback


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.rename(path)


class Timers:
    def __init__(self, torch):
        self.torch, self.events = torch, []

    @contextmanager
    def __call__(self, name):
        a, b = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        try:
            yield
        finally:
            b.record()
            self.events.append((name, a, b))

    def totals(self):
        self.torch.cuda.synchronize()
        result = {}
        for name, a, b in self.events:
            result[name] = result.get(name, 0.) + a.elapsed_time(b) / 1000
        return result


def gradient_comparison(torch, model, baseline):
    aa = bb = dd = ab = 0.
    rows = []
    finite = True
    for name, p in model.named_parameters():
        a = baseline[name]
        if (a is None) != (p.grad is None):
            raise ValueError('gradient presence changed: ' + name)
        if a is None:
            rows.append(dict(parameter=name, both_absent=True))
            continue
        b = p.grad.detach().cpu()
        finite &= bool(torch.isfinite(b).all())
        x, y = a.double(), b.double()
        d = y - x
        na, nb, nd, dot = (float(v) for v in (x.square().sum(), y.square().sum(), d.square().sum(), (x*y).sum()))
        aa += na; bb += nb; dd += nd; ab += dot
        rows.append(dict(parameter=name, baseline_norm=math.sqrt(na), candidate_norm=math.sqrt(nb),
                         relative_L2=math.sqrt(nd/na) if na else (0. if nb == 0 else None),
                         cosine=dot/math.sqrt(na*nb) if na*nb else (1. if na == nb else 0.),
                         max_abs_error=float(d.abs().max())))
        del x, y, d, b
    norm = math.sqrt(aa)
    for r in rows:
        r['significant'] = r.get('baseline_norm', 0.) >= 1e-6 * norm
    return dict(finite=finite, global_cosine=ab/math.sqrt(aa*bb) if aa*bb else (1. if aa == bb else 0.),
                global_relative_L2=math.sqrt(dd/aa) if aa else (0. if bb == 0 else None),
                baseline_norm=norm, candidate_norm=math.sqrt(bb), parameters=rows)


def compare(torch, rows, loss, baseline, grads, gates):
    current = torch.cat([r['lp'] for r in rows]); old = torch.cat([r['lp'] for r in baseline['rows']])
    kl = torch.cat([r['kl'] for r in rows]); ref = torch.cat([r['kl'] for r in baseline['rows']])
    e, k = (current-old).abs().double(), (kl-ref).abs().double()
    ratio = (current-old).double().exp()
    metrics = dict(logprob_mean_abs=float(e.mean()), logprob_p99_abs=float(torch.quantile(e,.99)),
                   logprob_max_abs=float(e.max()), KL_mean_abs=float(k.mean()), KL_max_abs=float(k.max()),
                   ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                   ratio_outside_clip_fraction=float(((ratio<.8)|(ratio>1.2)).double().mean()),
                   loss=float(loss), baseline_loss=float(baseline['loss']),
                   loss_scaled_error=abs(loss-baseline['loss'])/max(1.,abs(baseline['loss'])))
    checks = {
        'logprob_abs_max': metrics['logprob_max_abs'] <= gates['logprob_abs_max'],
        'KL_abs_max': metrics['KL_max_abs'] <= gates['KL_abs_max'],
        'loss': metrics['loss_scaled_error'] <= gates['loss_scaled_error_max'],
        'gradient_finite': grads['finite'],
        'gradient_cosine': grads['global_cosine'] >= gates['gradient_cosine_min'],
        'gradient_relative_L2': grads['global_relative_L2'] is not None and grads['global_relative_L2'] <= gates['gradient_relative_L2_max'],
        'parameter_cosine': all(r['cosine'] >= gates['significant_parameter_cosine_min'] for r in grads['parameters'] if r['significant']),
        'parameter_relative_L2': all(r['relative_L2'] is not None and r['relative_L2'] <= gates['significant_parameter_relative_L2_max'] for r in grads['parameters'] if r['significant']),
    }
    for key, metric in [('logprob_mean_max','logprob_mean_abs'),('logprob_p99_max','logprob_p99_abs'),('KL_mean_abs_max','KL_mean_abs')]:
        if key in gates: checks[key] = metrics[metric] <= gates[key]
    return dict(metrics=metrics, checks=checks, passed=all(checks.values()), gradients=grads)


def run(args):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm allocation required')
    import torch
    from transformers import AutoModelForCausalLM
    from ricepo.hf_compute import hf_matmul_scope
    from ricepo.modeling import make_forward, sampled_old_log_probs, enable_eval_layer_checkpointing
    from ricepo.training import UpdateItem, ppo_plan_update
    from ricepo.objective import ppo_loss
    plan = json.loads(args.plan.read_text())
    out = args.output / args.scheme
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(6); torch.manual_seed(plan['root_seed']); torch.cuda.manual_seed_all(plan['root_seed'])
    props = torch.cuda.get_device_properties(0)
    if 'A800' not in props.name: raise RuntimeError('paired A800 measurement required')
    precision = 'high' if args.scheme in ('S1','S3') else 'highest'
    projection = 'action_positions_v1' if args.scheme in ('S2','S3') else 'full_sequence'
    recipe = plan['recipe']
    items = [UpdateItem(**{k:tuple(v) if isinstance(v,list) else v for k,v in x.items()}) for x in plan['items']]
    started = time.perf_counter()
    actor = AutoModelForCausalLM.from_pretrained(plan['actor'], dtype=torch.float32, local_files_only=True,
                                                attn_implementation='sdpa').eval().to('cuda')
    reference = AutoModelForCausalLM.from_pretrained(plan['reference'], dtype=torch.float32, local_files_only=True,
                                                    attn_implementation='sdpa').eval().requires_grad_(False).to('cuda')
    torch.cuda.synchronize(); init_seconds = time.perf_counter()-started
    if plan['activation_checkpointing'] == 'eval_layers_v1': enable_eval_layer_checkpointing(actor)
    elif plan['activation_checkpointing'] != 'none': raise ValueError('unknown checkpointing mode')
    assert all(p.dtype == torch.float32 for p in actor.parameters())
    device = 'cuda:0'
    nvml = dict(status='NOT_MEASURED', peak_used_bytes=None)
    stop = threading.Event()
    try:
        import pynvml
        pynvml.nvmlInit()
        uuid = str(props.uuid)
        handle = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith('GPU-') else 'GPU-'+uuid)
        def sample():
            while not stop.is_set():
                used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
                nvml.update(status='MEASURED',peak_used_bytes=max(used,nvml['peak_used_bytes'] or 0))
                stop.wait(.2)
        threading.Thread(target=sample,daemon=True).start()
    except Exception as exc: nvml['reason'] = type(exc).__name__
    opts = dict(reference_model=reference,device=device,logit_chunk_size=plan['logit_chunk_size'],
                max_context_tokens=32768,policy_call_kwargs={},reference_call_kwargs={},projection_mode=projection)

    def batch(selected, old_pass=True):
        actor.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        timers = Timers(torch); begin = time.perf_counter(); old_seconds=0.
        if old_pass:
            torch.cuda.synchronize(); begin_old=time.perf_counter()
            for item in selected:
                old_scores = sampled_old_log_probs(actor,item.token_ids,item.generated_mask,device=device,
                    logit_chunk_size=plan['logit_chunk_size'],max_context_tokens=32768,policy_call_kwargs={},
                    projection_mode=projection)
                if not bool(torch.isfinite(old_scores).all()): raise FloatingPointError('old scoring nonfinite')
                del old_scores
            torch.cuda.synchronize(); old_seconds=time.perf_counter()-begin_old
        torch.cuda.synchronize(); begin_ppo=time.perf_counter(); rows=[]; total_loss=0.
        forward = make_forward(**opts,measure=timers)
        denominator={'sequence_mean':len(selected),'token_mean':sum(sum(x.generated_mask) for x in selected),'sum':1}[recipe['loss_reduction']]
        for item in selected:
            p,k = forward(actor,item)
            m=torch.tensor([item.generated_mask],dtype=torch.bool,device=device)
            old=torch.tensor([item.old_log_probs],dtype=p.dtype,device=device)
            adv=torch.tensor([item.advantages],dtype=p.dtype,device=device)
            with timers('PPO_objective'):
                loss=ppo_loss(p,old,adv,m,clip_epsilon=recipe['clip_epsilon'],
                    kl_coefficient=recipe['actor_kl_coefficient'],reference_kl=k,reduction='sum')/denominator
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
            with timers('backward'):loss.backward()
            rows.append(dict(lp=p[m].detach().cpu(),kl=k[m].detach().cpu()))
            total_loss+=float(loss.detach());del p,k,loss,m,old,adv
        torch.cuda.synchronize(); ppo_seconds=time.perf_counter()-begin_ppo
        stages=timers.totals()
        if any(p.grad is not None for p in reference.parameters()):raise ValueError('reference acquired gradients')
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters()):raise FloatingPointError('nonfinite gradient')
        timing=dict(old_probability_seconds=old_seconds,PPO_accumulation_seconds=ppo_seconds,
                    HF_total_seconds=old_seconds+ppo_seconds,stage_cuda_seconds=stages,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        return dict(rows=rows,loss=total_loss,timing=timing)

    result=dict(scheme=args.scheme,precision=precision,projection=projection,hardware=props.name,
                initialization_seconds=init_seconds,torch_version=torch.__version__,
                native_tokens=sum(len(i.token_ids) for i in items),action_tokens=sum(sum(i.generated_mask) for i in items),
                source_plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                checkpointing=plan['activation_checkpointing'],model_mode='eval',optimizer_steps=0)
    with hf_matmul_scope(precision):
        result['actual_precision_flags']=dict(precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        if args.step:
            # A separate fresh offline replay step; no generation and no claim of on-policy training.
            before={n:p.detach().cpu().clone() for n,p in actor.named_parameters()}
            opt=recipe['optimizer_hyperparameters']
            optimizer=torch.optim.AdamW(actor.parameters(),lr=recipe['actor_learning_rate'],betas=tuple(opt['betas']),
                                        eps=opt['eps'],weight_decay=opt['weight_decay'])
            torch.cuda.synchronize(); ts=time.perf_counter()
            result['update']=ppo_plan_update(actor,optimizer,items,policy_version=items[0].policy_version,
                                            forward=make_forward(**opts),device=device,recipe=recipe)
            torch.cuda.synchronize();result['optimizer_step_seconds']=time.perf_counter()-ts
            changed=0;maximum=0.
            for n,p in actor.named_parameters():
                d=(p.detach().cpu()-before[n]).abs();changed+=int((d!=0).sum());maximum=max(maximum,float(d.max()))
            result.update(optimizer_steps=result['update']['optimizer_steps'],changed_parameter_elements=changed,max_parameter_delta=maximum,
                          optimizer_state_dtypes=sorted({str(v.dtype) for s in optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor)}))
            del before,optimizer;actor.zero_grad(set_to_none=True)
            target=out/'actor.partial';actor.save_pretrained(target,safe_serialization=True)
            restored=AutoModelForCausalLM.from_pretrained(target,dtype=torch.float32,local_files_only=True,attn_implementation='sdpa').eval()
            equal=all(torch.equal(p.detach().cpu(),q.detach()) for p,q in zip(actor.parameters(),restored.parameters()))
            result['actor_reload_all_parameters_equal']=equal
            if not equal:raise ValueError('saved actor differs after reload')
            write(target/'OFFLINE_DIAGNOSTIC_COMPLETE.json',dict(kind='actor_only_not_training_resume',all_parameters_equal=True))
            target.rename(out/'actor');result['status']='STEP_COMPLETE';write(out/'RESULT.json',result);return
        # Profile real saved sequence under the actual requested precision; excluded from timings.
        if args.scheme in ('S0','S1'):
            actor.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                item=items[0];x=torch.tensor([item.token_ids[:512]],device=device)
                actor(input_ids=x,use_cache=False).logits.sum().backward();torch.cuda.synchronize()
            prof.export_chrome_trace(str(out/'KERNEL_TRACE_PRIVATE.json'))
            trace=json.loads((out/'KERNEL_TRACE_PRIVATE.json').read_text())
            kernels=sorted({e['name'] for e in trace['traceEvents'] if e.get('cat')=='kernel'})
            result['GEMM_kernels']=[k for k in kernels if any(t in k.lower() for t in ('gemm','1688','tf32'))]
            result['kernel_tensor_float_evidence']=any('tf32' in k.lower() or 's1688' in k.lower() for k in kernels)
            actor.zero_grad(set_to_none=True);del prof,x
            if args.scheme=='S1' and not result['kernel_tensor_float_evidence']:
                result['status']='NO_EFFECT_UNPROVEN_KERNEL';result['correctness_passed']=False
                write(out/'RESULT.json',result);return
        # Small real GPU check precedes expansion to the frozen six-sequence cohort.
        small=batch(items[:2],old_pass=False)
        if args.scheme=='S0':
            torch.save({n:None if p.grad is None else p.grad.detach().cpu() for n,p in actor.named_parameters()},out/'SMALL_GRADS.pt')
            torch.save(small,out/'SMALL.pt');result['small_passed']=True
        else:
            base=args.output/'S0';g=torch.load(base/'SMALL_GRADS.pt',map_location='cpu',weights_only=True)
            grads=gradient_comparison(torch,actor,g);del g
            gates=plan['projection_gates'] if args.scheme=='S2' else plan['precision_gates']
            comparison=compare(torch,small['rows'],small['loss'],torch.load(base/'SMALL.pt',weights_only=True),grads,gates)
            result['small_comparison']=comparison;result['small_passed']=comparison['passed']
            if not comparison['passed']:
                result['status']='REJECTED_SMALL';result['NVML']=nvml;write(out/'RESULT.json',result);return
        del small;actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        batch(items) # exactly one warmup, never an optimizer step
        repeats=[]
        for rep in range(2):
            row=batch(items);repeats.append(row['timing']);print('REPEAT_COMPLETE',args.scheme,rep,row['timing'],flush=True)
        result['repeats']=repeats
        if args.scheme=='S0':
            torch.save({n:None if p.grad is None else p.grad.detach().cpu() for n,p in actor.named_parameters()},out/'FULL_GRADS.pt')
            torch.save(row,out/'FULL.pt');result['correctness_passed']=True
        else:
            base=args.output/'S0';g=torch.load(base/'FULL_GRADS.pt',map_location='cpu',weights_only=True)
            grads=gradient_comparison(torch,actor,g);del g
            gates=plan['projection_gates'] if args.scheme=='S2' else plan['precision_gates']
            result['comparison']=compare(torch,row['rows'],row['loss'],torch.load(base/'FULL.pt',weights_only=True),grads,gates)
            result['correctness_passed']=result['comparison']['passed']
        result.update(status='BENCHMARK_COMPLETE',NVML=nvml,checkpoint_statistics=getattr(actor,'_rice_checkpoint_statistics',None))
        write(out/'RESULT.json',result)
    stop.set()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scheme',choices=['S0','S1','S2','S3'],required=True);parser.add_argument('--step',action='store_true')
    args=parser.parse_args()
    try:run(args)
    except Exception as exc:
        path=args.output/args.scheme;path.mkdir(parents=True,exist_ok=True)
        write(path/'FAILURE_PRIVATE.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
