"""One continuous scale training process; private paths are supplied explicitly."""
import argparse,json,os,signal,time,threading
from pathlib import Path


def main(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    import torch
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.audit import fingerprint
    from ricepo.config import load_config,resolve
    from ricepo.hf_runtime import run
    from ricepo.scale_training import request_stop,run_tag
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.device_count()!=2:raise ValueError('two Slurm-assigned GPUs required')
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1':raise ValueError('TF32 override in HF parent')
    signal.signal(signal.SIGUSR1,lambda *_:request_stop())
    R=Path(os.environ['RICE_PROJECT_ROOT']);source=Path(os.environ['RICE_SOURCE_ROOT'])
    plan_path=Path(os.environ['RICE_SCALE_PLAN']);frozen=json.loads(plan_path.read_text())
    tag=run_tag(args.stage,args.method)
    root=R/'runs'/tag
    if args.resume:
        from ricepo.scale_training import preserve_unpersisted_tail
        from ricepo.state import writer
        # Only a verified persistent state survives; prior computed tails remain archived.
        binding=fingerprint(json.loads((root/'binding.json').read_text()))
        with writer(root,resume=True):preserve_unpersisted_tail(root,binding)
        config=json.loads((root/'config.json').read_text());dep=json.loads((root/'deployment.json').read_text())
    else:
        dep=json.loads((plan_path.parent/'PRIVATE_BASE_DEPLOYMENT.json').read_text())
        for key in ('parent_run_root','parent_source_root','continuation_schedule_path','native_replay_sources',
                    'native_action_replay_sources','frozen_schedule_path','initial_evaluation_only','hf_matmul_precision'):
            dep.pop(key,None)
        dep.update(output_root=str(root),cache_root=str(R/'cache'/tag),checkpoint_staging_root=str(R/'tmp'/tag),
            actor_device='cuda:0',retriever_device='cuda:1',actor_precision='float32',hf_projection_mode='full_sequence',
            prompt_path=str(source/'prompts/ricepo_prefilled_think_smoke.txt'),
            vllm_rpc_root=str(R.parent.parent/'i'/f'bs{os.environ["SLURM_JOB_ID"]}'),
            sampling_backend='vllm_mixed_precision_pilot_v1',checkpoint_io='local_verified_v1',
            compare_grpo_credit_gradients=False,stop_after_checkpoint_diagnostic=False,
            branch_probability_audit='stratified_five_percent_v1',initial_evaluation_only=args.stage=='initial',
            vllm_runtime=dict(max_num_seqs=144,max_num_batched_tokens=32768,gpu_memory_utilization=.90,
                enforce_eager=False,enable_prefix_caching=True,enable_chunked_prefill=True,attention_backend='TRITON_ATTN'),
            precision_pilot=dict(candidate='SCALE',stage=args.stage,validation_plan_path=str(plan_path),
                validation_plan_sha256=file_sha256(plan_path),qualification_result_path=frozen['prior_qualification_path']))
        if os.environ.get('RICE_MEMORY_QUALIFICATION'):
            if args.stage!='training':raise ValueError('memory recovery belongs to training only')
            qualification=Path(os.environ['RICE_MEMORY_QUALIFICATION'])
            dep['hf_memory_repair']=dict(mode='cpu_saved_expandable_deferred_adam_v1',
                qualification_result_path=str(qualification),qualification_result_sha256=file_sha256(qualification))
            dep['completed_credit_recovery']=dict(run=os.environ['RICE_CREDIT_RECOVERY_RUN'],source=os.environ['RICE_CREDIT_RECOVERY_SOURCE'])
        Path(dep['checkpoint_staging_root']).mkdir(parents=True,exist_ok=True)
        stage='capacity' if args.stage=='capacity' else 'training'
        config=load_config(source/f'configs/biology_scale_v1_{stage}_{args.method}.json')
        config['runtime_bindings'].update(train_split_manifest=dep['train_manifest_path'],development_split_manifest=dep['development_manifest_path'])
        config['execution']['enabled']=True
    resolved=resolve(config,'train')
    if resolved['issues']:raise ValueError(resolved['issues'])
    if resolved['values']['training_seeds']!=[frozen['training_seed']]:raise ValueError('seed schedule differs')
    torch.manual_seed(frozen['training_seed']);torch.cuda.manual_seed_all(frozen['training_seed'])
    evidence=R/'reports'/f'{tag}-launch-{os.environ["SLURM_JOB_ID"]}.json'
    atomic_json(evidence,dict(config=config,deployment=dep,resolved=resolved,source_commit=os.environ['RICE_SOURCE_COMMIT'],
        frozen_sampling_plan_source_commit=frozen['execution_source_commit'],
        scale_sampling_response_timeout_seconds=43200,initialization_response_timeout_seconds=1800,
        plan_sha256=file_sha256(plan_path),resume=args.resume,slurm_job_id=os.environ['SLURM_JOB_ID']))
    stop=threading.Event();memory=[0,0]
    try:
        import pynvml
        pynvml.nvmlInit();handles=[]
        for i in range(2):
            uuid=str(torch.cuda.get_device_properties(i).uuid)
            handles.append(pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith('GPU-') else 'GPU-'+uuid))
        def sample():
            while not stop.is_set():
                for i,h in enumerate(handles):memory[i]=max(memory[i],pynvml.nvmlDeviceGetMemoryInfo(h).used)
                stop.wait(1)
        threading.Thread(target=sample,daemon=True).start()
    except Exception:memory=None
    start=time.monotonic()
    try:
        result=run(config,dep,resume=args.resume)
        atomic_json(R/'reports'/f'{tag}-return-{os.environ["SLURM_JOB_ID"]}.json',result)
        print('SCALE_RUNTIME_RETURN',tag,json.dumps(result),flush=True)
        if result.get('status')=='stopped_at_persistent_checkpoint':return 75
        if args.stage=='capacity' and result.get('status')!='capacity_update_complete':raise ValueError('capacity did not pass')
        if args.stage=='training' and result.get('completed_updates')!=10:raise ValueError('ten updates not completed')
        return 0
    finally:
        stop.set()
        atomic_json(R/'reports'/f'{tag}-resources-{os.environ["SLURM_JOB_ID"]}.json',dict(elapsed_seconds=time.monotonic()-start,
            parent_peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],NVML_peak_bytes=memory,
            scope='actual run wall incl initialization/checkpoint/evaluation; not queue',HF_precision=torch.get_float32_matmul_precision()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['capacity','initial','training']);p.add_argument('method',choices=['ricepo','grpo']);p.add_argument('--resume',action='store_true')
    raise SystemExit(main(p.parse_args()))
