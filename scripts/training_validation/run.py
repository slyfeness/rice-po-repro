"""Four-query paired RICE run or one checkpoint-to-checkpoint continuation."""
import os,json,time,sys,threading,subprocess
from pathlib import Path
R=Path(os.environ['RICE_PROJECT_ROOT'])

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicit Slurm allocation required')
    import torch
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.config import load_config,resolve
    from ricepo.hf_runtime import run
    from ricepo.state import recovery_checkpoint
    from ricepo.audit import fingerprint
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1':raise ValueError('unexpected TF32 override')
    source=Path(os.environ['RICE_SOURCE_ROOT']);resume=sys.argv[1]=='resume'
    if resume:
        root=R/'runs'/sys.argv[2];config=json.loads((root/'config.json').read_text());deployment=json.loads((root/'deployment.json').read_text())
        if deployment['precision_pilot']['candidate']!='G_B':raise ValueError('only B continuation authorized')
        cp,saved=recovery_checkpoint(root,fingerprint(json.loads((root/'binding.json').read_text())))
        if saved['iteration']>=2:raise ValueError('three-update budget already complete')
        if not (cp.parent/'training-validation.json').is_file():raise ValueError('previous post-update gate incomplete')
        if not json.loads((cp.parent/'pilot-post-update-risk.json').read_text())['pass_risk_screen']:raise ValueError('previous update gate failed')
        tag=root.name;update_index=saved['iteration']+1
    else:
        group=sys.argv[1]
        if group not in ('A','B'):raise ValueError('explicit A/B required')
        tag=f'gemm-training-{group}-{os.environ["SLURM_JOB_ID"]}';root=R/'runs'/tag;update_index=0
        plan=R/'reports/gemm-training-plan-176255/PRIVATE_PLAN.json';frozen=json.loads(plan.read_text())
        deployment=json.loads((R/'runs/gemm-online-B-175853/deployment.json').read_text())
        deployment.update(output_root=str(root),cache_root=str(R/'cache'/tag),checkpoint_staging_root=str(R/'tmp'/tag),
            prompt_path=str(source/'prompts/ricepo_prefilled_think_smoke.txt'),
            vllm_rpc_root=str(Path(os.environ['RICE_RPC_ROOT'])/f'e{os.environ["SLURM_JOB_ID"]}'),
            branch_probability_audit='stratified_five_percent_v1',
            precision_pilot=dict(candidate='G_'+group,validation_plan_path=str(plan),validation_plan_sha256=file_sha256(plan),
                qualification_result_path=str(R/'reports/gemm-independent-numeric-175846/HELDOUT_RESULT.json')))
        Path(deployment['checkpoint_staging_root']).mkdir(parents=True,exist_ok=False)
        config=load_config(source/'configs/two_stage_ricepo.json')
        config.setdefault('runtime_bindings',{}).update(total_updates=1 if group=='A' else 3,
            train_split_manifest=deployment['train_manifest_path'],development_split_manifest=deployment['development_manifest_path'])
        for k,v in dict(queries_per_update=4,training_seeds=[frozen['training_seed']],query_composition=frozen['conditions']['query_composition'],
                checkpoint_every_updates=2,development_every_updates=4,checkpoint_selection='last',eval_samples_per_query=4).items():
            config['deviations'][k]=dict(value=v,reason='Frozen four-query paired GEMM/three-update bounded validation; checkpoint handoffs; no development evaluation.')
        config['execution']['enabled']=True
        recipe=resolve(config,'train')
        if recipe['issues']:raise ValueError(recipe['issues'])
        atomic_json(R/'reports'/f'{tag}-config.json',dict(config=config,deployment=deployment,recipe=recipe))
    stop=threading.Event();samples=[];uuids=[]
    for i in range(2):
        key=str(torch.cuda.get_device_properties(i).uuid);uuids.append(key if key.startswith('GPU-') else 'GPU-'+key)
    def sample():
        while not stop.is_set():
            try:
                values=subprocess.check_output(['nvidia-smi','-i',','.join(uuids),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3)
                samples.append([list(map(float,r.split(','))) for r in values.strip().splitlines()])
            except Exception:pass
            stop.wait(.5)
    thread=threading.Thread(target=sample,daemon=True);thread.start();started=time.monotonic()
    try:
        result=run(config,deployment,resume=resume)
        atomic_json(R/'reports'/f'{tag}-u{update_index+1}-result.json',result)
        print('COMMITTED_VALIDATION',tag,update_index+1,result['status'],flush=True)
    finally:
        elapsed=time.monotonic()-started;stop.set();thread.join(4)
        atomic_json(R/'reports'/f'{tag}-u{update_index+1}-resources.json',dict(elapsed_seconds=elapsed,update_index=update_index,
            parent_peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
            sampled_device_peak_mib=[max((s[i][0] for s in samples if len(s)>i),default=None) for i in range(2)],
            hf_precision=torch.get_float32_matmul_precision(),hf_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            scope='run() wall including initialization or resume, update, checkpoint and new-weight probe',samples=len(samples)))
if __name__=='__main__':main()
