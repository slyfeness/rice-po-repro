"""Explicit Slurm-only model/optimizer/RNG recovery check; never trains."""
import argparse
import hashlib


def verify(run_root, output, device, iteration=0):
    from ricepo.artifacts import require_slurm,require_cluster_storage,atomic_json
    from ricepo.state import read_json,verify_checkpoint
    from ricepo.audit import fingerprint
    from ricepo.hf_runtime import _source_hashes
    require_slurm()
    root=require_cluster_storage(run_root);out=require_cluster_storage(output)
    if out.exists():raise FileExistsError(out)
    if read_json(root/'sources.json')!=_source_hashes():
        raise ValueError('use the exact immutable package source recorded by this run')
    if type(iteration) is not int or iteration<0:raise ValueError('invalid local checkpoint iteration')
    binding=read_json(root/'binding.json');cp=root/f'iteration-{iteration:06d}'/'checkpoint'
    receipt=verify_checkpoint(cp,fingerprint(binding),allow_missing_stage=True)
    import torch
    from transformers import AutoModelForCausalLM
    from ricepo.policy_identity import identify
    from ricepo.checkpoint_io import state_digest
    model,info=AutoModelForCausalLM.from_pretrained(str(cp/'actor'),torch_dtype=torch.float32,
        local_files_only=True,trust_remote_code=False,output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
        raise ValueError('incomplete restored model')
    model.to(device).eval();version,identity=identify(model,binding['behavior'])
    if version!=receipt['policy_version']:raise ValueError('restored actual policy differs')
    saved=torch.load(cp/'optimizer-and-rng.pt',map_location='cpu',weights_only=True)
    global_index=binding.get('continuation',{}).get('global_start',0)+iteration
    if (saved['policy_version']!=version or saved['binding']!=fingerprint(binding)
            or saved['iteration']!=iteration or receipt['iteration']!=iteration
            or saved.get('global_update_index',iteration)!=global_index):
        raise ValueError('optimizer binding differs')
    recipe=read_json(root/'recipe.json')['values'];hp=recipe['optimizer_hyperparameters']
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['actor_learning_rate'],
        betas=tuple(hp['betas']),eps=hp['eps'],weight_decay=hp['weight_decay'])
    expected=state_digest(saved['optimizer']);optimizer.load_state_dict(saved['optimizer'])
    if state_digest(optimizer.state_dict())!=expected:raise ValueError('optimizer logical payload differs')
    steps=sorted({float(s['step']) for s in optimizer.state.values()})
    mini=recipe['optimizer_minibatch_size']
    sequences=recipe['queries_per_update']*recipe['trajectories_per_query']
    mini=sequences if mini=='full_batch' else mini
    expected_steps=(global_index+1)*recipe['ppo_epochs']*(sequences//mini)
    if steps!=[float(expected_steps)]:raise ValueError('optimizer step position differs from frozen update plan')
    torch.random.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],device)
    probes=dict(cpu_draw_sha256=hashlib.sha256(torch.rand(32).numpy().tobytes()).hexdigest(),
        cuda_draw_sha256=hashlib.sha256(torch.rand(32,device=device).cpu().numpy().tobytes()).hexdigest())
    if probes!=read_json(cp/'rng-probe.json'):raise ValueError('restored random draws differ')
    report=dict(policy_version=version,state_dict_sha256=identity['state_dict_sha256'],
        optimizer_steps=steps,optimizer_restored=True,optimizer_logical_sha256=expected,
        cpu_cuda_rng_restored=True,rng_draw_hashes=probes,actual_actor_reloaded=True,
        checkpoint_files_verified=True,new_optimizer_steps=0,separate_process=True,
        local_iteration=iteration,global_completed_updates=global_index+1,
        peak_allocated=torch.cuda.max_memory_allocated(device))
    atomic_json(out,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--iteration',type=int,default=0)
    args=parser.parse_args()
    print(verify(args.run_root,args.output,args.device,args.iteration),flush=True)
