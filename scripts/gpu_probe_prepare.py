"""Read-only extraction of committed GRPO update2; no reward recomputation."""
import argparse,json,os
from pathlib import Path
from scripts.gpu_probe_common import write,digest,horizon,sub64_indices

def main(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    from ricepo.artifacts import file_sha256
    from ricepo.training import UpdateItem,_validate_items
    from ricepo.completed_credit_recovery import decode_trajectory
    from ricepo.contracts import trajectory_behavior_log_probs
    root=a.run;it=root/'iteration-000001';items=json.loads((it/'old-policy-update-items.json').read_text())
    if len(items)!=512:raise ValueError('expected512 originals')
    version=items[0]['policy_version'];_validate_items([UpdateItem(**x) for x in items],version)
    tfile=it/'trajectories.json';receipt=json.loads((it/'rollout_complete.receipt.json').read_text())
    if file_sha256(tfile)!=receipt['artifacts']['trajectories.json']:raise ValueError('trajectory receipt differs')
    ts={x['trajectory_id']:decode_trajectory(x,5) for x in json.loads(tfile.read_text())}
    for x in items:
        t=ts[x['trajectory_id']]
        if tuple(x['token_ids'])!=t.token_ids or tuple(x['generated_mask'])!=t.generated_mask or tuple(x['old_log_probs'])!=trajectory_behavior_log_probs(t):raise ValueError('native items/q differ from recorded generation')
    if not (it/'optimizer_step_committed.receipt.json').is_file() or not(it/'checkpoint/complete.json').is_file():raise ValueError('not committed update2')
    binding=json.loads((root/'binding.json').read_text());dep=json.loads((root/'deployment.json').read_text())
    actor=list(root.glob('attempts/*/vllm/snapshot-'+version+'/actor'))
    if len(actor)!=1:raise ValueError('P1 snapshot must be unique')
    ref=binding['common_initial'];refid=json.loads((root/'behavior-identities'/(ref['policy_version']+'.json')).read_text())
    indices=sub64_indices(items)
    plan=dict(items=items,sub64_indices=indices,actor=str(actor[0]),reference=ref['actor_path'],reference_state_sha256=refid['state_dict_sha256'],policy_version=version,behavior=binding['behavior'],diver=dep['retriever_model_path'],diver_dtype=dep['dtype'],logit_chunk_size=dep['logit_chunk_size'],recipe=json.loads((root/'recipe.json').read_text()))
    meta=dict(source_items_sha256=file_sha256(it/'old-policy-update-items.json'),source_trajectories_sha256=file_sha256(tfile),checkpoint2_receipt_sha256=file_sha256(it/'checkpoint/complete.json'),sub64_sha256=digest([items[i] for i in indices]),sub16_sha256=digest([items[i] for i in indices[:16]]),selected_identity_sha256=[digest(items[i]['trajectory_id']) for i in indices],horizons=[horizon(items[i]) for i in indices],action_counts=[sum(items[i]['generated_mask']) for i in indices],longest_horizon=max(map(horizon,items)),selection='8 horizon quantile strata;8 identity-SHA sorted per stratum;force global maximum replacing eighth in final stratum;Sub16 first16 in this order',optimizer='fresh AdamW at P1 for offline comparison; not a resumed P1 Adam state')
    plan['public_metadata']=meta
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'PLAN.json',plan);write(a.output/'METADATA.json',meta)
    print(json.dumps(dict(items=len(items),selected=len(indices),longest=meta['longest_horizon'],sub64_sha256=meta['sub64_sha256'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
