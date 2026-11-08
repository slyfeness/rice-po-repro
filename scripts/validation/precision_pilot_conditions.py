"""Explicit frozen-condition generation/scoring; not complete on-policy episodes.

Use separate processes for generation and scoring inside an authorized Slurm
allocation. Inputs/outputs contain private native histories and must stay private.
"""
import argparse
import json
import os
from pathlib import Path
import time

from ricepo.artifacts import atomic_json, file_sha256
from ricepo.audit import fingerprint
from ricepo.asset_identity import local_model_identity
from ricepo.precision_pilot import mismatch_statistics, risk_screen


def main(args):
    plan = json.loads(Path(args.plan).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    from ricepo.environment import configure_caches
    configure_caches(Path(args.cache))
    expected = plan['initial_model_payload_sha256']
    identity = local_model_identity(args.model)
    if identity['payload_sha256'] != expected:
        raise ValueError('initial model/tokenizer payload differs from frozen plan')
    if any(fingerprint(r['prefix_token_ids']) != r['prefix_sha256'] for r in plan['requests']):
        raise ValueError('native condition identity mismatch')
    if {r['prefix_sha256'] for r in plan['requests']} & set(plan['calibrated_prefixes']):
        raise ValueError('held-out condition overlaps calibration')
    if args.phase == 'generate':
        from transformers import AutoTokenizer
        from ricepo.vllm_two_stage import VLLMEngine
        from ricepo.audit import attempt
        if (args.candidate=='T' and os.environ.get('TRITON_F32_DEFAULT')!='tf32'):
            raise ValueError('TF32 must be selected before imports')
        tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
        numeric = 'tf32_pilot' if args.candidate=='T' else 'bf16_fast'
        runtime = dict(attention_backend='TRITON_ATTN' if args.candidate=='T' else 'FLASH_ATTN',
                       gpu_memory_utilization=.80 if args.candidate=='T' else .90)
        tick = time.monotonic()
        with attempt(out):
            engine = VLLMEngine(args.model,tokenizer,identity['payload_sha256'],numeric_mode=numeric,runtime_options=runtime)
            init_seconds = time.monotonic()-tick
            try:
                started = time.monotonic()
                count = 0
                for stage in ('think','summary'):
                    rows = [r for r in plan['requests'] if r['stage']==stage]
                    requests = [dict(prefix=tuple(r['prefix_token_ids']),seed=r['seed'],budget=r['remaining_token_budget']) for r in rows]
                    outputs = engine.sample(requests,stage,identity['payload_sha256'])
                    for r,o in zip(rows,outputs,strict=True):
                        if len(o['token_ids']) != len(o['sampled_log_probs']) or not o['token_ids']:
                            raise ValueError('incomplete sampled probability evidence')
                        atomic_json(out/(r['request_id']+'.json'),dict(request=r,completion=o))
                        count += len(o['token_ids'])
                seconds=time.monotonic()-started
                atomic_json(out/'GENERATION.json',dict(candidate=args.candidate,model_identity=identity,
                    private_plan_sha256=file_sha256(args.plan),requests=len(plan['requests']),generated_tokens=count,
                    initialization_seconds=init_seconds,formal_seconds=seconds,tokens_per_second=count/seconds,
                    runtime=engine.runtime_options,numeric_mode=numeric,
                    prefix_cache_hit_ratio=None,prefill_seconds=None,decode_seconds=None,
                    timing_limitation='no per-phase engine counter exported; unknown is not inferred from token ratios'))
            finally: engine.llm.llm_engine.engine_core.shutdown()
        return

    import torch
    from transformers import AutoModelForCausalLM
    from ricepo.modeling import sampled_old_log_probs
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    generation=json.loads((out/'GENERATION.json').read_text())
    if generation['model_identity'] != identity or generation['private_plan_sha256'] != file_sha256(args.plan):
        raise ValueError('generation/teacher identity or plan mismatch')
    started=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,
            local_files_only=True,trust_remote_code=False).eval().to('cuda:0')
    assert all(p.dtype==torch.float32 for p in model.parameters())
    model.requires_grad_(False)
    allq=[];allp=[];records=[]
    for r in plan['requests']:
        saved=json.loads((out/(r['request_id']+'.json')).read_text())
        if saved['request'] != r: raise ValueError('saved condition changed')
        c=saved['completion'];a=c['token_ids'];prefix=r['prefix_token_ids'];mask=[False]*len(prefix)+[True]*len(a)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.candidate=='A'):
            full=sampled_old_log_probs(model,prefix+a,mask,device='cuda:0',logit_chunk_size=64,
                max_context_tokens=32768,policy_call_kwargs={})[0].cpu().tolist()
        lp=[v for v,m in zip(full,mask,strict=True) if m]
        q=c['sampled_log_probs']
        diag=mismatch_statistics(q,lp)
        records.append(dict(request_id=r['request_id'],stage=r['stage'],prefix_tokens=len(prefix),
            context_bin='short' if len(prefix)<2048 else 'medium' if len(prefix)<=8192 else 'long',**diag))
        allq+=q;allp+=lp
        atomic_json(out/(r['request_id']+'-scored.json'),dict(request_sha256=file_sha256(out/(r['request_id']+'.json')),
            hf_log_probs=lp,action_mask=mask,diagnostics=records[-1]))
    overall=mismatch_statistics(allq,allp)
    atomic_json(out/'HELDOUT_RESULT.json',dict(candidate=args.candidate,overall=overall,
        request_diagnostics=records,risk_screen_pass=risk_screen(overall),
        status='CONDITIONAL_SCREEN_PASS' if risk_screen(overall) else 'REJECTED',
        private_plan_sha256=file_sha256(args.plan),scoring_seconds=time.monotonic()-started,
        no_optimizer_step=True,scope='native conditional histories, not complete on-policy trajectories'))
    print(json.dumps(dict(candidate=args.candidate,overall=overall,risk_screen_pass=risk_screen(overall))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['generate','score'])
    parser.add_argument('--candidate',choices=['T','A'],required=True)
    for key in ('plan','model','output','cache'):parser.add_argument('--'+key,required=True)
    main(parser.parse_args())
