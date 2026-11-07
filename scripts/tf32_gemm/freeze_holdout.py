# Public driver: explicit Slurm execution only; no downloads.
import json,os,hashlib,time
from pathlib import Path
from ricepo.audit import fingerprint,stable_seed
from ricepo.artifacts import atomic_json,file_sha256
from ricepo.hf_runtime import _manifest

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT']);M=R/'benchmarks/tf32-swap-175674/MANIFEST.json'
O=R/'reports'/f'gemm-independent-plan-{os.environ["SLURM_JOB_ID"]}';O.mkdir()
read=lambda p:json.loads(p.read_text());excluded=set();index=[]
def collect(value):
    if isinstance(value,dict):
        if isinstance(value.get('prefix_sha256'),str):excluded.add(value['prefix_sha256'])
        if isinstance(value.get('prefix_token_ids'),list):excluded.add(fingerprint(value['prefix_token_ids']))
        for k,v in value.items():
            if k=='prefixes' and isinstance(v,list):
                for p in v:excluded.add(fingerprint(p))
            elif k=='calibrated_prefixes':excluded.update(v)
            else:collect(v)
    elif isinstance(value,list):
        for v in value:collect(v)
sources=[R/'reports/bf16-matched-175507/REQUEST_SELECTION.json',R/'runs/vllm-fp32-probe-174858/selection.json',
    R/'reports/precision-plan-175586/PRIVATE_VALIDATION_PLAN.json']
for base in ('s1-probability-175518','explicit-bf16-175560','explicit-bf16-175561'):
    sources+=list((R/'reports'/base).glob('*.json'))
for p in sources:
    collect(read(p));index.append(dict(alias='historical/'+p.parent.name+'/'+p.name,sha256=file_sha256(p)))
# Include all actual Think/Summary condition histories returned in the prior T pilot.
for p in (R/'runs/precision-pilot-T-175590').glob('attempts/*/events/*.json'):
    with p.open() as f:head=f.read(4096)
    if '"kind": "raw_completion"' in head:collect(read(p))
for p in (R/'runs/precision-pilot-T-175590').glob('attempts/*/vllm/worker-*/attempts/*/events/*.json'):
    with p.open() as f:head=f.read(4096)
    if 'vllm_engine_raw_response' in head:
        e=read(p)
        for r in e.get('requests',[]):excluded.add(fingerprint(r['prefix']))
m=read(M);selected=[]
for stage in ('think','summary'):
    pool={r['prefix_sha256']:r for r in m['requests'] if r['stage']==stage and r['prefix_sha256'] not in excluded}
    rows=sorted(pool.values(),key=lambda r:fingerprint(['gemm-independent-v1',r['prefix_sha256'],stage]))
    chosen=[]
    for r in [min(rows,key=lambda r:len(r['prefix_token_ids'])),max(rows,key=lambda r:len(r['prefix_token_ids']))]+rows:
        if r not in chosen:chosen.append(r)
        if len(chosen)==8:break
    for r in chosen:
        key=fingerprint(['gemm-independent-v1',r['prefix_sha256'],stage]);selected.append({**r,'request_id':key,'seed':stable_seed(20261107,'gemm-independent-v1',key)})
assert len(selected)==16 and len({r['prefix_sha256'] for r in selected})==16
assert not ({r['prefix_sha256'] for r in selected}&excluded)
old=read(R/'reports/precision-plan-175586/PRIVATE_VALIDATION_PLAN.json')
dep=read(R/'runs/precision-pilot-T-175590/deployment.json')
_,train,_=_manifest(dep['train_manifest_path'],set(dep['domains']),expected_sha256=old['train_manifest_sha256'])
oldq={q for batch in old['query_batches'] for q in batch}
queries=sorted(set(train)-oldq,key=lambda q:fingerprint(['gemm-B-training-v1',q]))[:2]
plan=dict(kind='tf32_gemm_B_independent_v1',candidate='B',frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    requests=selected,excluded_prefix_count=len(excluded),excluded_prefixes=sorted(excluded),exclusion_sources=index,
    selection='8/stage; shortest+longest then identity-hash rank, new stable seeds; no format/reward/output-length filtering',
    query_batches=[queries],training_seed=20261107,train_manifest_sha256=old['train_manifest_sha256'],
    conditions=old['conditions'],maximum_updates=1,worker_matmul_precision='high',hf_matmul_precision='highest',
    speed_gate=dict(primary='same1152 cold makespan',minimum_speedup=1.5),
    screen=dict(outside_clip_fraction_max=.01,extreme_tail_fraction_max=.0005),
    prior_precision_calibration_reused=False,maximum_generation_budget='saved remaining shared4096 action budget',
    no_long_training=True,no_full_backend_equivalence_claim=True)
atomic_json(O/'PRIVATE_VALIDATION_PLAN.json',plan)
public={k:v for k,v in plan.items() if k not in ('requests','excluded_prefixes','query_batches')}
public.update(private_plan_sha256=file_sha256(O/'PRIVATE_VALIDATION_PLAN.json'),requests=[
    {k:r[k] for k in ('request_id','prefix_sha256','stage','seed','remaining_token_budget')}|dict(prefix_tokens=len(r['prefix_token_ids'])) for r in selected],
    query_schedule_sha256=fingerprint([queries]),execution={'enabled':False})
atomic_json(O/'VALIDATION_PLAN.json',public)
print('INDEPENDENT_PLAN',json.dumps(public),flush=True)
