"""Finish only a committed capacity checkpoint's sampling gate; never train.

Called by the independent restore process, using its verified Actor/Adam state.
The failed training attempt and its binding remain immutable. New evidence has
its own source identity and attempt directory. No retrieval or reward is run.
"""
import os,time
from pathlib import Path


def checked_prefix(saved):
    if len(saved)!=128 or len({x['trajectory_id'] for x in saved})!=128:
        raise ValueError('capacity originals are not a complete unique group')
    groups={}
    for x in saved:
        key=x['query_id'];groups[key]=groups.get(key,0)+1
    if len(groups)!=32 or set(groups.values())!={4}:raise ValueError('capacity query groups differ')
    first=saved[0]
    prefix=first['steps'][0]['prefix_token_ids'] if first['steps'] else first['failure']['prefix_token_ids']
    if not prefix or any(type(t) is not int for t in prefix):raise ValueError('missing native prefix')
    return tuple(prefix)


def complete(*,run,cp,record,binding,plan,actor,optimizer,restore_evidence):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.audit import attempt,fingerprint,stable_seed
    from ricepo.state import writer,read_json,complete_stage
    from ricepo.policy_identity import identify
    from ricepo.modeling import sampled_old_log_probs
    from ricepo.asset_identity import local_model_identity
    from ricepo.precision_pilot import mismatch_statistics,risk_screen
    from ricepo.vllm_lifecycle import PhaseSeparatedGenerator
    start=time.monotonic();R=Path(os.environ['RICE_PROJECT_ROOT'])
    if record['iteration']!=0 or plan['total_updates']!=1 or plan['queries_per_update']!=32:
        raise ValueError('repair is restricted to one committed capacity step')
    with writer(run,resume=True),attempt(run,recovery=True) as evidence:
        if (run/'capacity-complete.json').exists():raise ValueError('capacity already complete; do not repeat probe')
        directory=cp.parent;dep=read_json(run/'deployment.json')
        receipt=read_json(directory/'rollout_complete.receipt.json')
        if receipt['artifacts']['trajectories.json']!=file_sha256(directory/'trajectories.json'):
            raise ValueError('saved originals changed')
        prefix=checked_prefix(read_json(directory/'trajectories.json'))
        if local_model_identity(dep['actor_model_path'])!=read_json(run/'model-sources.json')['actor_model_path']:
            raise ValueError('initial Reference assets changed')
        tokenizer=AutoTokenizer.from_pretrained(cp/'actor',local_files_only=True,trust_remote_code=False)
        reference=AutoModelForCausalLM.from_pretrained(dep['actor_model_path'],dtype=torch.float32,
            local_files_only=True,trust_remote_code=False).eval().requires_grad_(False).to('cuda:0')
        generator=PhaseSeparatedGenerator(actor=actor,reference=reference,optimizer=optimizer,tokenizer=tokenizer,
            version=lambda:record['policy_version'],behavior=binding['behavior'],device=torch.device('cuda:0'),
            root=evidence/'vllm',rpc_root=R.parent.parent/'i'/f'cr{os.environ["SLURM_JOB_ID"]}',
            max_new_tokens=4096,max_context_tokens=32768,vllm_runtime=dep['vllm_runtime'],
            numeric_mode='tf32_gemm_pilot',reuse_verified_assets=True,response_timeout_seconds=43200)
        try:
            generated=generator.generate(prefix,seed=stable_seed(plan['training_seeds'][0],'post_update_probe',0))
            atomic_json(evidence/'post-update-sampling-probe.json',dict(policy_version=record['policy_version'],generation=generated))
            generator.prepare_scoring()
            ids=prefix+generated.token_ids;mask=(False,)*len(prefix)+generated.action_mask
            with torch.no_grad():
                teacher=sampled_old_log_probs(actor,ids,mask,device=torch.device('cuda:0'),
                    logit_chunk_size=dep['logit_chunk_size'],max_context_tokens=32768,policy_call_kwargs={},projection_mode='full_sequence')[0].cpu().tolist()
            q=[x for x,b in zip(generated.sampled_log_probs,generated.action_mask,strict=True) if b]
            p=[x for x,b in zip(teacher,mask,strict=True) if b]
            stats=mismatch_statistics(q,p);passed=risk_screen(stats,outside_max=.01)
            atomic_json(evidence/'post-update-risk.json',dict(statistics=stats,pass_risk_screen=passed,outside_limit=.01))
            if not passed:raise ValueError('capacity recovery post-update risk failed')
            if identify(actor,binding['behavior'])[0]!=record['policy_version']:raise ValueError('restored Actor changed')
            steps=sorted({int(s['step'].item()) for s in optimizer.state.values() if 'step' in s})
            if steps!=[1] or any(p.grad is not None for p in actor.parameters()):raise ValueError('repair changed optimizer position or accumulated gradients')
            result=dict(status='capacity_update_complete',method=plan['method'],completed_updates=1,
                optimizer_steps=1,queries=32,original_trajectories=128,policy_version=record['policy_version'],
                checkpoint='iteration-000000/checkpoint',optimizer_step_replayed=False,
                recovery=dict(evidence=str(evidence),source_commit=os.environ['RICE_SOURCE_COMMIT'],
                    original_binding_sha256=fingerprint(binding),independent_restore_sha256=file_sha256(restore_evidence),
                    checkpoint_complete_sha256=file_sha256(cp/'complete.json'),new_optimizer_steps=0,
                    elapsed_seconds=time.monotonic()-start,scope='post-checkpoint sampling gate only; original failed attempt preserved'))
            complete_stage(directory,'development_complete',record['policy_version'],{'skipped':'independent capacity batch, no development'})
            complete_stage(directory,'iteration_complete',record['policy_version'],{'capacity_complete':True})
            atomic_json(evidence/'completion.json',result);atomic_json(run/'capacity-complete.json',result)
        finally:generator.close()
