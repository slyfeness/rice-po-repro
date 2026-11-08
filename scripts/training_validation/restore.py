"""Read-only independent restore of the one committed B update; zero steps."""
import os,json,time
from pathlib import Path

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicit Slurm allocation required')
    import torch
    from transformers import AutoModelForCausalLM
    from ricepo.audit import fingerprint
    from ricepo.state import recovery_checkpoint,verify_checkpoint
    from ricepo.policy_identity import identify
    from ricepo.continuation import validate_state_position
    from ricepo.checkpoint_io import state_digest,rng_probe
    from ricepo.artifacts import atomic_json,file_sha256
    from ricepo.environment import configure_caches
    R=Path(os.environ['RICE_PROJECT_ROOT']);run=Path(os.environ['RICE_RESTORE_RUN'])
    out=R/'reports'/f'gemm-training-restore-{os.environ["SLURM_JOB_ID"]}.json'
    configure_caches(R/'cache'/out.stem)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();binding=json.loads((run/'binding.json').read_text());bound=fingerprint(binding)
    if os.environ.get('RICE_RESTORE_ITERATION'):
        index=int(os.environ['RICE_RESTORE_ITERATION']);cp=run/f'iteration-{index:06d}'/'checkpoint'
        record=verify_checkpoint(cp,bound)
        step=json.loads((cp.parent/'optimizer_step_committed.receipt.json').read_text())
        if record['iteration']!=index or step['policy_version']!=record['policy_version'] or step['artifacts']['update.json']!=file_sha256(cp.parent/'update.json'):raise ValueError('committed step differs')
    else:cp,record=recovery_checkpoint(run,bound)
    plan=json.loads((run/'recipe.json').read_text())['values']
    state=torch.load(cp/'optimizer-and-rng.pt',map_location='cpu',weights_only=True)
    validate_state_position(state,binding=bound,iteration=record['iteration'],seed=plan['training_seeds'][0],policy_version=record['policy_version'])
    if state.get('global_update_index')!=record['iteration']:raise ValueError('unexpected update position')
    actor,info=AutoModelForCausalLM.from_pretrained(cp/'actor',dtype=torch.float32,local_files_only=True,
        trust_remote_code=False,output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):raise ValueError('partial weights')
    actor=actor.eval().to('cuda:0')
    version,identity=identify(actor,binding['behavior'])
    if version!=record['policy_version']:raise ValueError('restored policy identity mismatch')
    h=plan['optimizer_hyperparameters']
    optimizer=torch.optim.AdamW(actor.parameters(),lr=plan['actor_learning_rate'],betas=tuple(h['betas']),eps=h['eps'],weight_decay=h['weight_decay'])
    optimizer.load_state_dict(state['optimizer'])
    digest=state_digest(optimizer.state_dict())
    if digest!=state_digest(state['optimizer']):raise ValueError('restored optimizer differs')
    steps=sorted({int(x['step'].item()) for x in optimizer.state.values() if 'step' in x})
    if steps!=[record['iteration']+1]:raise ValueError('optimizer position differs from committed update')
    torch.random.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'],'cuda:0')
    if not torch.equal(torch.random.get_rng_state(),state['cpu_rng']) or not torch.equal(torch.cuda.get_rng_state('cuda:0'),state['cuda_rng']):raise ValueError('live RNG state differs after restoration')
    probe=rng_probe(state,'cuda:0')
    if probe!=json.loads((cp/'rng-probe.json').read_text()):raise ValueError('RNG restoration mismatch')
    atomic_json(out,dict(status='PASS',scope='independent read-only checkpoint restore; zero optimizer steps',
        training_run_alias=run.name,completed_updates=record['iteration']+1,policy_version=version,actor_state_dict_sha256=identity['state_dict_sha256'],
        binding_sha256=bound,checkpoint_complete_sha256=file_sha256(cp/'complete.json'),
        optimizer_state_sha256=digest,restored_optimizer_steps=steps,rng_probe=probe,
        newly_executed_optimizer_steps=0,checkpoint_modified=False,elapsed_seconds=time.monotonic()-start,
        hf_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,hf_precision=torch.get_float32_matmul_precision()))
    print('INDEPENDENT_RESTORE_PASS',out.name,flush=True)
if __name__=='__main__':main()
