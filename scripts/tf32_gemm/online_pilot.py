# Public driver: explicit Slurm execution only; no downloads.
"""New initial-model two-query RICE experiment; no development or big batch."""
import json,os,time
from pathlib import Path

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
    import torch
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.config import load_config,resolve
    from ricepo.hf_runtime import run
    R=Path(os.environ['RICE_PROJECT_ROOT'])
    S=Path(os.environ['RICE_SOURCE_ROOT'])
    candidate='B'
    if candidate!='B':raise ValueError('only B')
    plan_path=R/'reports/gemm-independent-plan-175841/PRIVATE_VALIDATION_PLAN.json'
    heldout=R/'reports/gemm-independent-numeric-175846/HELDOUT_RESULT.json'
    frozen=json.loads(plan_path.read_text());screen=json.loads(heldout.read_text())
    if not screen['risk_screen_pass'] or screen['candidate']!=candidate:
        raise ValueError('held-out gate did not pass; do not run PPO')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tag=f'gemm-online-B-{os.environ["SLURM_JOB_ID"]}'
    deployment=json.loads((R/'runs/two-stage-online-ricepo-174610/deployment.json').read_text())
    deployment.update(output_root=str(R/'runs'/tag),cache_root=str(R/'cache'/tag),
        actor_device='cuda:0',retriever_device='cuda:1',actor_precision='float32',
        prompt_path=str(S/'prompts/ricepo_prefilled_think_smoke.txt'),sampling_backend='vllm_mixed_precision_pilot_v1',
        vllm_runtime=dict(max_num_seqs=144,max_num_batched_tokens=32768,
            gpu_memory_utilization=.90,enforce_eager=False,
            enable_prefix_caching=True,enable_chunked_prefill=True,
            attention_backend='TRITON_ATTN'),
        vllm_rpc_root=str(Path(os.environ['RICE_RPC_ROOT'])/f'p{os.environ["SLURM_JOB_ID"]}'),
        checkpoint_staging_root=str(R/'tmp'/tag),compare_grpo_credit_gradients=False,
        stop_after_checkpoint_diagnostic=True,retrieval_ranking_impl='array_minmax_v1',
        retrieval_query_batch_size=1,branch_probability_audit='all',
        precision_pilot=dict(candidate=candidate,validation_plan_path=str(plan_path),
            validation_plan_sha256=file_sha256(plan_path),heldout_result_path=str(heldout),speed_gate_evidence_path=str(R/'reports/gemm-independent-numeric-175846/BEHAVIOR.json')))
    for key in ('parent_run_root','parent_source_root','continuation_schedule_path','native_replay_sources',
                'native_action_replay_sources','initial_evaluation_only','frozen_schedule_path'):
        deployment.pop(key,None)
    Path(deployment['checkpoint_staging_root']).mkdir(parents=True,exist_ok=False)
    config=load_config(S/'configs/two_stage_ricepo.json')
    config.setdefault('runtime_bindings',{}).update(total_updates=1,train_split_manifest=deployment['train_manifest_path'],
        development_split_manifest=deployment['development_manifest_path'])
    values=dict(queries_per_update=2,training_seeds=[frozen['training_seed']],
        query_composition=frozen['conditions']['query_composition'],development_every_updates=4,
        checkpoint_every_updates=2,checkpoint_selection='last',eval_samples_per_query=4)
    for key,value in values.items():
        config['deviations'][key]=dict(value=value,reason='Explicit TF32 GEMM B one-update diagnostic; frozen independent plan; no paper-scale or backend-equivalence claim.')
    config['execution']['enabled']=True
    resolved=resolve(config,'train')
    if resolved['issues']:raise ValueError(resolved['issues'])
    atomic_json(R/'reports'/f'{tag}-config.json',dict(config=config,deployment=deployment,recipe=resolved,
        scope='exactly one two-query B update; no development or larger training'))
    start=time.monotonic()
    try:
        result=run(config,deployment)
        atomic_json(R/'reports'/f'{tag}-result.json',result)
        print('PILOT_COMPLETE',json.dumps(result),flush=True)
    finally:
        atomic_json(R/'reports'/f'{tag}-resources.json',dict(elapsed_seconds=time.monotonic()-start,
            peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
            peak_reserved_bytes=[torch.cuda.max_memory_reserved(i) for i in range(2)]))

if __name__=='__main__':main()
