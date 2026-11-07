"""Read-only audit of complete vLLM native diagnostic; no model or generation."""
import json,os,math,hashlib
from pathlib import Path
from ricepo.contracts import generation_from_record,generation_action_mask
r=Path(os.environ['RICE_PROJECT']);run=r/'runs/vllm-native-rollout-175004';d=run/'vllm'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
originals=read(d/'originals.json');groups=[read(p) for p in sorted(d.glob('branches-*.json'))];adv=read(d/'advantages.json')
traces=originals+[t for g in groups for t in g['trajectories']]
actions={};failures={};records=[]
for t in traces:
 mask=[False]*len(t['token_ids'])
 steps=list(t['steps'])+([t['failure']] if t['failure'] else [])
 for s in steps:
  g=generation_from_record(s['generation']);gm=list(generation_action_mask(g));p=s['prefix_token_ids'];start=len(p);end=start+len(g.token_ids)
  assert t['token_ids'][:start]==p and t['token_ids'][start:end]==list(g.token_ids)
  assert sum(gm)<=4096
  mask[start:end]=gm
  # Reused branch history is a single original generation, not new output cost.
  key=hashlib.sha256(json.dumps([p,g.token_ids],separators=(',',':')).encode()).hexdigest()
  actions[key]=dict(tokens=sum(gm),environment_tokens=len(gm)-sum(gm),reason=g.finish_reason)
 assert mask==t['generated_mask']
 if t['failure']:
  assert t['final_reward']==0 and t['failure_terminal_reward']==0
  cat=t['failure']['generation']['category'];failures[cat]=failures.get(cat,0)+1
 else:assert len(t['steps'])==5 and t['final_reward']==t['steps'][-1]['local_reward']
byid={t['trajectory_id']:t for t in originals}
for item in groups:
 group=item['group'];a=group['anchor'];out=group['outcomes'];t=byid[a['trajectory_id']]
 step=next(s for s in t['steps'] if s['depth']==a['depth'])
 assert len(out)==6 and sum(o['is_original'] for o in out)==1 and len(item['trajectories'])==5
 assert all(o['prefix_token_ids']==step['prefix_token_ids'] for o in out)
 assert all(b['token_ids'][:len(step['prefix_token_ids'])]==step['prefix_token_ids'] for b in item['trajectories'])
 local=[o['local_reward'] for o in out];final=[o['final_reward'] for o in out]
 mean=sum(local)/6;var=sum((x-mean)**2 for x in local)/6;res=[f-l for f,l in zip(final,local)];rm=sum(res)/6;rv=sum((x-rm)**2 for x in res)/6
 record=next(c for c in adv['records'] if c['anchor']==a)
 assert math.isclose(var,record['influence_variance'],abs_tol=1e-12) and math.isclose(rv,record['residual_variance'],abs_tol=1e-12)
 assert record['propagate']==(var>=.05 and rv<=.03)
 original=next(o for o in out if o['is_original']);assert math.isclose(record['summary_advantage'],original['local_reward']-mean,abs_tol=1e-12)
 i=adv['trajectory_ids'].index(a['trajectory_id']);span=a['summary_span'];assert all(x==record['summary_advantage'] for x in adv['advantages'][i][span['start']:span['end']])
 records.append(dict(depth=a['depth'],entropy=a['entropy'],local_rewards=local,final_rewards=final,
  residuals=res,influence_variance=var,residual_variance=rv,propagate=record['propagate'],summary_advantage=record['summary_advantage']))
for row,mask in zip(adv['advantages'],adv['generated_masks']):assert all(x==0 for x,m in zip(row,mask) if not m)
result=dict(source='vllm-native-rollout-175004',source_summary_sha256=sha(run/'summary.json'),
  originals_sha256=sha(d/'originals.json'),advantages_sha256=sha(d/'advantages.json'),
  branch_files=[dict(ordinal=i,sha256=sha(p)) for i,p in enumerate(sorted(d.glob('branches-*.json')))],
  native_masks_verified=True,frozen_prefixes_verified=True,original_count_once_verified=True,credit_moments_recomputed=True,
  action_budget=4096,max_observed_action_tokens=max(x['tokens'] for x in actions.values()),unique_action_records=len(actions),
  observed_budget_exhaustion=sum(x['tokens']==4096 for x in actions.values()),failure_categories=failures,
  normal_originals=sum(t['failure'] is None for t in originals),normal_branches=sum(t['failure'] is None for g in groups for t in g['trajectories']),records=records)
p=r/'reports'/('vllm-native-audit-'+os.environ['SLURM_JOB_ID']+'.json');p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
