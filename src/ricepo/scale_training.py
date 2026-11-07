"""Independent, user-authorized Biology scale run; legacy pilot gates unchanged."""
from pathlib import Path
from uuid import uuid4
from .artifacts import atomic_json, file_sha256, require_cluster_storage
from .state import read_json, verify_checkpoint, development_score

KIND='biology_scale_training_v1'
_stop=False


def run_tag(stage,method):
    import os,re
    tag=os.environ.get('RICE_SCALE_RUN_TAG')
    if tag is not None:
        if stage!='training' or re.fullmatch(r'biology-scale-v[2-9][0-9]*-training-'+re.escape(method),tag) is None:
            raise ValueError('explicit recovery run tag differs from stage/method or escapes the run namespace')
        return tag
    return 'biology-scale-v1-initial' if stage=='initial' else f'biology-scale-v1-{stage}-{method}'


def request_stop():
    global _stop
    _stop=True


def stop_at_checkpoint(durations, completed, total):
    """Use measured updates, never a four-query extrapolation, for job handoff."""
    import os,time
    if completed>=total:return False
    if _stop:return True
    end=os.environ.get('SLURM_JOB_END_TIME')
    if not end or not durations:return False
    budget=1.3*max(durations[-2:])*min(2,total-completed)+900
    return float(end)-time.time()<budget


def validate_contract(stage, frozen, recipe):
    if stage not in ('capacity','initial','training') or frozen.get('kind')!=KIND:
        raise ValueError('unknown independent scale stage')
    count,updates=(32,1) if stage=='capacity' else (128,10)
    expected=dict(queries_per_update=count,total_updates=updates,training_seeds=[frozen['training_seed']],
        trajectories_per_query=4,rollout_depth=5,actor_learning_rate=1e-6,actor_kl_coefficient=.001,
        clip_epsilon=.2,ppo_epochs=1,microbatch_size=1,loss_reduction='sequence_mean',
        query_composition={'mode':'original_and_summary','separator':'\n'},temperature=1.,top_p=1.,top_k=0,
        max_context_tokens=32768,max_new_tokens_per_step=4096,checkpoint_every_updates=2,development_every_updates=4,
        checkpoint_selection='best_development_macro',eval_samples_per_query=4,
        native_span_parser='two_stage_env_summary_v1',format_failure_policy='invalid_action_terminal_zero_v1')
    if recipe.get('method')=='ricepo':
        expected.update(anchors_per_query=4,total_branches_including_original=6,local_advantage_normalization='center_only',
                        influence_threshold=.05,residual_threshold=.03)
    if (recipe.get('method') not in ('ricepo','grpo')
            or frozen.get('data_counts')!={'train':128,'development':32,'test':103}
            or any(recipe.get(k)!=v for k,v in expected.items())
            or any(recipe.get(k) not in ('full_batch',count*4) for k in ('optimizer_minibatch_size','gradient_accumulation'))):
        raise ValueError('scale recipe differs from the frozen independent experiment')


def settings(spec, deployment, recipe):
    path=require_cluster_storage(spec['validation_plan_path'])
    if file_sha256(path)!=spec['validation_plan_sha256']:raise ValueError('scale plan changed')
    frozen=read_json(path);stage=spec['stage'];validate_contract(stage,frozen,recipe)
    prior_path=require_cluster_storage(spec['qualification_result_path']);prior=read_json(prior_path)
    if (file_sha256(prior_path)!=frozen['prior_qualification_sha256'] or prior.get('candidate')!='B'
            or not prior.get('risk_screen_pass')):
        raise ValueError('scale TF32 GEMM qualification identity differs')
    if (deployment.get('hf_projection_mode','full_sequence')!='full_sequence'
            or deployment.get('actor_precision')!='float32' or deployment.get('parent_run_root')
            or deployment.get('compare_grpo_credit_gradients')
            or deployment.get('sampling_backend')!='vllm_mixed_precision_pilot_v1'
            or deployment.get('branch_probability_audit')!='stratified_five_percent_v1'
            or deployment.get('stop_after_checkpoint_diagnostic',False)):
        raise ValueError('scale requires fresh FP32 HF/full head and independent continuous runtime')
    runtime=deployment['vllm_runtime']
    if any(runtime.get(k)!=v for k,v in dict(max_num_seqs=144,max_num_batched_tokens=32768,
            gpu_memory_utilization=.90,enforce_eager=False,enable_prefix_caching=True,
            enable_chunked_prefill=True,attention_backend='TRITON_ATTN').items()):
        raise ValueError('frozen scale vLLM runtime differs')
    batches=frozen['capacity_batches'] if stage=='capacity' else frozen['training_batches']
    if len(batches)!=recipe['total_updates'] or any(len(b)!=recipe['queries_per_update'] or len(set(b))!=len(b) for b in batches):
        raise ValueError('incomplete scale query schedule')
    return {**frozen,**spec,'query_batches':batches,'numeric_mode':'tf32_gemm_pilot'}


def risk_limit(index):
    if type(index) is not int or not 0<=index<10:raise ValueError('scale update index outside authorized ten')
    return .01 if index==0 else .02


def cadence(completed,total):
    if type(completed) is not int or not 1<=completed<=total:raise ValueError('invalid completed position')
    dev=completed==total or completed%4==0
    return dict(development=dev,checkpoint=dev or completed%2==0)


def publish_initial_state(root,payload):
    """Durable empty Adam/RNG state; Actor stays in the verified initial asset."""
    import os,torch
    root=Path(root);staging=root/('.pending-initial-'+uuid4().hex);staging.mkdir()
    with (staging/'optimizer-and-rng.pt').open('wb') as fd:
        torch.save(payload,fd);fd.flush();os.fsync(fd.fileno())
    atomic_json(staging/'complete.json',dict(iteration=-1,policy_version=payload['policy_version'],
        binding_sha256=payload['binding'],files={'optimizer-and-rng.pt':file_sha256(staging/'optimizer-and-rng.pt')}))
    fd=os.open(staging,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    target=root/'initial-state'
    if target.exists():raise FileExistsError(target)
    staging.rename(target)
    fd=os.open(root,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def verify_initial_state(root,binding):
    path=Path(root)/'initial-state';record=read_json(path/'complete.json')
    if (record.get('iteration')!=-1 or record.get('binding_sha256')!=binding
            or record.get('files')!={'optimizer-and-rng.pt':file_sha256(path/'optimizer-and-rng.pt')}):
        raise ValueError('Initial optimizer/RNG identity changed')
    return record


def preserve_unpersisted_tail(root,binding):
    """Retain lost-work evidence before replay from the latest verified durable step.

    Caller must own the campaign/run writer lock. Runtime then verifies the
    retained checkpoint and step receipts normally; no unverified state is used.
    Actual recomputation is distinct from logical learning-curve update counts.
    """
    root=Path(root);dirs=sorted(root.glob('iteration-[0-9]*'));latest=None
    for d in reversed(dirs):
        if (d/'checkpoint/complete.json').is_file():
            unpublished=not any((d/(stage+'.receipt.json')).exists() for stage in ('development_complete','iteration_complete'))
            record=verify_checkpoint(d/'checkpoint',binding,allow_missing_stage=unpublished)
            latest=(d,record);break
    if latest is None:latest=(None,verify_initial_state(root,binding))
    d,record=latest;tail=dirs if d is None else [p for p in dirs if p.name>d.name];computed=[]
    if tail:
        archive=root/'recovery-archives'/uuid4().hex;archive.mkdir(parents=True)
        for path in tail:
            if (path/'optimizer_step_committed.receipt.json').is_file():computed.append(int(path.name[10:])+1)
            path.rename(archive/path.name)
        atomic_json(archive/'recovery.json',dict(persisted_updates=record['iteration']+1,
            discarded_computed_updates=computed,retained_directories=[p.name for p in tail],
            unpersisted_update_positions=[int(p.name[10:])+1 for p in tail],
            incomplete_optimizer_execution='UNKNOWN when no commit receipt; never count as zero or a persisted learning update',
            policy='recompute from verified durable weights/Adam/RNG and original frozen seeds; do not count as additional logical updates'))
    return dict(persisted_updates=record['iteration']+1,discarded_computed_updates=computed)


def recomputation_generations(root):
    generations={}
    for path in Path(root).glob('recovery-archives/*/recovery.json'):
        for name in read_json(path)['retained_directories']:
            index=int(name[10:]);generations[index]=generations.get(index,0)+1
    return generations


def bind_initial(root, baseline_root, binding, initial_policy, actor_path):
    """Copy only a verified common Initial score into this run's selection set."""
    root=Path(root);baseline=Path(baseline_root);old=read_json(baseline/'binding.json')
    for key in ('manifest_sha256','model_sha256','retriever_sha256','prompt_sha256','behavior'):
        if old[key]!=binding[key]:raise ValueError('common Initial differs: '+key)
    result=read_json(baseline/'initial-evaluation-complete.json')
    if result['policy_version']!=initial_policy or result.get('samples_per_query')!=4:
        raise ValueError('Initial model or evaluation sample schedule differs')
    report=baseline/'initial-development/summary.json'
    score=development_score(root,report,initial_policy)
    summary=read_json(report)
    if summary.get('samples_per_query')!=4 or summary.get('evaluated_sample_total')!=128:
        raise ValueError('common Initial must contain32 queries times4 samples')
    choice=dict(iteration=-1,checkpoint=str(actor_path),policy_version=initial_policy,
        development_macro=score,kind='frozen_initial_source',report_path=str(report),report_sha256=file_sha256(report))
    target=root/'initial-selection.json'
    if target.exists():
        if read_json(target)!=choice:raise ValueError('Initial selection changed')
    else:atomic_json(target,choice)
    return choice
