# Public driver: explicit Slurm execution only; no downloads.
import os,json,time,math,gc,hashlib
from pathlib import Path

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT'])
P=R/'reports/gemm-independent-plan-175841/PRIVATE_VALIDATION_PLAN.json'

def main():
    from ricepo.environment import configure_caches
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.asset_identity import local_model_identity
    from ricepo.vllm_two_stage import VLLMEngine
    from ricepo.precision_pilot import mismatch_statistics,risk_screen
    from ricepo.audit import attempt,fingerprint
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import torch
    O=R/'reports'/f'gemm-independent-numeric-{os.environ["SLURM_JOB_ID"]}';O.mkdir();configure_caches(R/'cache'/O.name)
    plan=json.loads(P.read_text());m=json.loads((R/'benchmarks/tf32-swap-175674/MANIFEST.json').read_text());model=m['models']['initial']
    # Speed is a hard precondition, not merely a report field.
    a=json.loads((R/'benchmarks/tf32-swap-175674/S2-confirm-175705/cold/RESULT.json').read_text())
    b=json.loads((R/'benchmarks/tf32-gemm-B-175839/cold/RESULT.json').read_text())
    speedup=a['wall_seconds']/b['wall_seconds']
    if speedup<1.5:raise ValueError('cold speed gate not passed')
    identity=local_model_identity(model);tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    os.environ['VLLM_FLOAT32_MATMUL_PRECISION']='high';engine=None;saved=[]
    try:
        with attempt(O):
            engine=VLLMEngine(model,tok,identity['payload_sha256'],numeric_mode='tf32_pilot',runtime_options=m['candidates']['S2'],
                worker_extension_cls='ricepo.gemm_diagnostic.GemmInspection')
            state=engine.llm.collective_rpc('gemm_state')
            if not state[0]['matmul_allow_tf32'] or state[0]['torch_float32_matmul_precision']!='high':raise ValueError('wrong B worker')
            atomic_json(O/'BEHAVIOR.json',dict(worker_state=state,model_identity=identity,private_plan_sha256=file_sha256(P),speedup=speedup))
            for stage in ('think','summary'):
                rows=[r for r in plan['requests'] if r['stage']==stage]
                req=[dict(prefix=tuple(r['prefix_token_ids']),seed=r['seed'],budget=r['remaining_token_budget']) for r in rows]
                result=engine.sample(req,stage,engine.version)
                for r,o in zip(rows,result,strict=True):
                    atomic_json(O/(r['request_id']+'-native.json'),dict(request=r,completion=o,model_payload_sha256=identity['payload_sha256']))
                    if fingerprint(r['prefix_token_ids'])!=r['prefix_sha256']:raise ValueError('changed prefix')
                    if len(o['token_ids'])!=len(o['sampled_log_probs']) or any(not math.isfinite(x) for x in o['sampled_log_probs']):raise ValueError('invalid behavior LP')
                    saved.append((r,o))
    finally:
        if engine is not None:engine.llm.llm_engine.engine_core.shutdown()
    del engine;gc.collect();torch.cuda.empty_cache()
    os.environ['VLLM_FLOAT32_MATMUL_PRECISION']='highest'
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1':raise ValueError('unexpected global cuBLAS TF32 override')
    teacher=AutoModelForCausalLM.from_pretrained(model,dtype=torch.float32,local_files_only=True,trust_remote_code=False).eval().to('cuda:0')
    teacher.requires_grad_(False)
    if any(p.dtype!=torch.float32 for p in teacher.parameters()):raise ValueError('teacher weights not FP32')
    from ricepo.modeling import sampled_old_log_probs
    allq=[];allp=[];records=[];tick=time.monotonic()
    for r,o in saved:
        prefix=r['prefix_token_ids'];ids=o['token_ids'];mask=[False]*len(prefix)+[True]*len(ids)
        with torch.no_grad():
            full=sampled_old_log_probs(teacher,prefix+ids,mask,device='cuda:0',logit_chunk_size=64,
                max_context_tokens=32768,policy_call_kwargs={})[0].cpu().tolist()
        lp=[x for x,a in zip(full,mask,strict=True) if a];q=o['sampled_log_probs']
        diag=mismatch_statistics(q,lp);allq+=q;allp+=lp
        records.append(dict(request_id=r['request_id'],stage=r['stage'],prefix_tokens=len(prefix),finish_reason=o['finish_reason'],**diag))
        atomic_json(O/(r['request_id']+'-scored.json'),dict(hf_log_probs=lp,action_mask=mask,diagnostics=records[-1]))
    overall=mismatch_statistics(allq,allp);passed=risk_screen(overall)
    atomic_json(O/'HELDOUT_RESULT.json',dict(candidate='B',status='RISK_SCREEN_PASS' if passed else 'FAIL',
        risk_screen_pass=passed,private_plan_sha256=file_sha256(P),requests=len(records),overall=overall,request_diagnostics=records,
        teacher_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,teacher_precision=torch.get_float32_matmul_precision(),
        scoring_seconds=time.monotonic()-tick,optimizer_steps=0,exact_backend_equivalence=False))
    print('NUMERICAL_RESULT',json.dumps(dict(pass_risk=passed,overall=overall)),flush=True)

if __name__=='__main__':main()
