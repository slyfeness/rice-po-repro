"""One saved-native GPU1 longest-history forward/backward with resident Diver."""
import argparse,json,os,time
from pathlib import Path


def main(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
    import torch
    from transformers import AutoModel,AutoModelForCausalLM
    from ricepo.modeling import make_forward,enable_eval_layer_checkpointing
    from ricepo.training import UpdateItem
    from ricepo.policy_identity import identify
    from scripts.gpu_probe_common import write,horizon,Monitor,item_backward
    plan=json.loads(a.plan.read_text());assets=json.loads(a.assets.read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda:1');torch.cuda.set_device(device)
    torch.set_num_threads(6);torch.manual_seed(20261108);torch.cuda.manual_seed_all(20261108)
    torch.set_float32_matmul_precision('high');torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=False
    item=UpdateItem(**{k:tuple(v) if isinstance(v,list) else v for k,v in plan['items']['largest'].items()})
    if horizon(item)!=16947:raise ValueError('saved preflight identity is not requested horizon16947')
    result=dict(status='RUNNING',horizon=horizon(item),action_tokens=sum(item.generated_mask),optimizer_steps=0,
        precision='FP32 weights, TF32 GEMM',projection='action_positions_v1',Diver_resident=True)
    monitor=None
    try:
        diver=AutoModel.from_pretrained(assets['diver'],torch_dtype=getattr(torch,assets['diver_dtype']),local_files_only=True).eval().requires_grad_(False).to(device)
        actor=AutoModelForCausalLM.from_pretrained(plan['actor'],dtype=torch.float32,local_files_only=True).eval().to(device)
        reference=AutoModelForCausalLM.from_pretrained(plan['actor'],dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
        binding=json.loads((Path(plan['source_run'])/'binding.json').read_text())
        policy,evidence=identify(actor,binding['behavior'])
        if policy!=item.policy_version:raise ValueError('saved-native P0 identity mismatch')
        result['actor_state_sha256']=evidence['state_dict_sha256']
        enable_eval_layer_checkpointing(actor)
        forward=make_forward(reference_model=reference,device=device,logit_chunk_size=plan['logit_chunk_size'],
            max_context_tokens=32768,policy_call_kwargs={},reference_call_kwargs={},projection_mode='action_positions_v1',
            saved_tensor_policy='cpu_above_budget_v1',device_residency_budget={'horizon':5535,'action_tokens':1889})
        torch.cuda.reset_peak_memory_stats(device);monitor=Monitor(torch,device,'GPU1-longest-Diver')
        torch.cuda.synchronize(device);tick=time.perf_counter()
        loss,stats=item_backward(actor,forward,item,device,512,'tf32')
        torch.cuda.synchronize(device)
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in actor.parameters() if p.grad is not None]))
        if not torch.isfinite(norm):raise FloatingPointError('nonfinite preflight gradient')
        result.update(status='COMPLETED',HF_seconds=time.perf_counter()-tick,loss=loss,gradient_norm=float(norm),
            allocated_peak_bytes=torch.cuda.max_memory_allocated(device),reserved_peak_bytes=torch.cuda.max_memory_reserved(device),
            matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,OOM=False)
    except BaseException as exc:
        result.update(status='ERROR',error_type=type(exc).__name__,OOM=isinstance(exc,torch.OutOfMemoryError))
        raise
    finally:
        if monitor is not None:result['GPU_util']=monitor.finish()
        write(a.output/'RESULT.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
