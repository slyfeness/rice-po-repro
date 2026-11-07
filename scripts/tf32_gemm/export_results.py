# Public aggregate export; no native inputs or logs are released.
"""Sanitized aggregate export, inside a CPU Slurm job; no native output release."""
import os,json,csv,shutil,collections,hashlib
from pathlib import Path
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT'])
O=R/'reports'/f'gemm-public-export-{os.environ["SLURM_JOB_ID"]}';O.mkdir()
read=lambda p:json.loads(p.read_text());index=[]
def tracked(p):
    index.append(dict(alias=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size));return read(p)
def scrub(v):
    if isinstance(v,dict):return {k:('sha256:'+hashlib.sha256(str(x).encode()).hexdigest() if k in ('query','query_id') else scrub(x)) for k,x in v.items()}
    if isinstance(v,list):return [scrub(x) for x in v]
    return v
def dump(n,v):(O/n).write_text(json.dumps(scrub(v),indent=2,allow_nan=False)+'\n')
rows=[];aggregate={}
from ricepo.inference_timing import distribution
bases={'A':R/'benchmarks/tf32-swap-175674/S2-confirm-175705','B':R/'benchmarks/tf32-gemm-B-175839','C':R/'benchmarks/tf32-gemm-C-175840'}
for group,base in bases.items():
    init=tracked(base/'INITIALIZATION.json') if (base/'INITIALIZATION.json').exists() else {}
    aggregate[group]={'initialization':{k:v for k,v in init.items() if k in ('initialization_seconds','runtime','manifest_sha256')},'batches':{}}
    for cache in ('cold','hot'):
        p=base/cache/'RESULT.json';v=tracked(p);times=[]
        for stage in ('think','summary'):times+=tracked(base/cache/(stage+'-timing.json'))
        counts=collections.Counter(t['finish_reason'] for t in times)
        input_tokens=sum(s['input_tokens'] for s in v['stages'])
        cached=sum(s['cached_tokens'] for s in v['stages']) if all(isinstance(s['cached_tokens'],(int,float)) for s in v['stages']) else None
        d={k:distribution(t[k] for t in times) for k in ('ttft','queue','prefill','decode','tpot','core_completion')}
        peak=max((s['observation'].get('observed_device_peak_mib',0) for s in v['stages'] if isinstance(s['observation'],dict)),default=None)
        preempt=sum(s['engine_metrics_delta'].get('vllm:num_preemptions',0) for s in v['stages'])
        row=dict(group=group,cache=cache,requests=v['requests'],generated_tokens=v['generated_tokens'],wall_seconds=v['wall_seconds'],tokens_per_second=v['tokens_per_second'],input_tokens=input_tokens,cached_tokens=cached,uncached_tokens=input_tokens-cached if cached is not None else None,cache_hit_fraction=cached/input_tokens if cached is not None else None,observed_device_peak_mib=peak,preemptions=preempt,oom=0,initialization_seconds=init.get('initialization_seconds','UNKNOWN'),timing_basis='host batch makespan; per-request latency intervals overlap; initialization excluded')
        for k,x in d.items():
            for metric in ('mean','p50','p99','max'):row[k+'_'+metric]=x.get(metric,'UNKNOWN') if isinstance(x,dict) else 'UNKNOWN'
        rows.append(row);aggregate[group]['batches'][cache]=dict(summary=row,latencies=d,finish_reasons=dict(counts),output_length=distribution(t['generated_tokens'] for t in times),stages=v['stages'])
with (O/'GEMM_COMPARISON.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
dump('BENCHMARK_DETAILS.json',aggregate)
for src,name in [(R/'reports/tf32-gemm-kernels-175838/KERNEL_COMPARISON.json','KERNEL_COMPARISON.json'),(R/'reports/tf32-gemm-cpu-175837/CACHE_ANALYSIS.json','CACHE_ANALYSIS.json'),(R/'reports/gemm-independent-plan-175841/VALIDATION_PLAN.json','VALIDATION_PLAN.json'),(R/'reports/gemm-cost-exclusive-175854/AMDAHL_ESTIMATES.json','AMDAHL_ESTIMATES.json')]:dump(name,tracked(src))
src=R/'reports/gemm-cost-exclusive-175854/TRAINING_COST_BREAKDOWN.csv';index.append(dict(alias=str(src.relative_to(R)),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),bytes=src.stat().st_size));shutil.copyfile(src,O/src.name)
dump('NUMERICAL_VALIDATION.json',tracked(R/'reports/gemm-independent-numeric-175846/HELDOUT_RESULT.json'))
run=R/'runs/gemm-online-B-175853';it=run/'iteration-000000';online={}
for name in ('update.json','credit-diagnostics.json','pilot-gradient-comparison.json','pilot-probability-risk.json','pilot-post-update-risk.json','reload-verification.json','rollout_complete.receipt.json','branches_complete.receipt.json','credit_complete.receipt.json','optimizer_step_committed.receipt.json','checkpoint_complete.receipt.json'):
    p=it/name
    if p.exists():online[name]=tracked(p)
for name in ('trajectories.json','counterfactual-trajectories.json','advantages.json','anchors.json'):
    p=it/name
    if p.exists():
        v=tracked(p);online[name+'-shape']=dict(type=type(v).__name__,length=len(v),keys=list(v)[:12] if isinstance(v,dict) else None,first_keys=list(v[0]) if isinstance(v,list) and v and isinstance(v[0],dict) else None)
        if name=='trajectories.json':
            online['originals']=[{k:t.get(k) for k in ('trajectory_id','final_reward','last_valid_ndcg','failure_terminal_reward','termination','initial_ndcg')}|dict(legal_steps=len(t['steps']),local_rewards=[s.get('local_reward') for s in t['steps']]) for t in v]
groups=read(it/'counterfactual-trajectories.json')
actual=[t for g in groups for t in g['trajectories']]
online['branch_group_sizes']=[len(g['trajectories']) for g in groups]
online['actual_new_branch_count']=len(actual)
online['branch_failure_count']=sum(t.get('failure') is not None for t in actual)
online['unique_new_branch_identities']=len({t['trajectory_id'] for t in actual})
for name in ('diagnostic-complete.json',):
    p=run/name
    if p.exists():online[name]=tracked(p)
p=R/'reports/gemm-online-B-175853-resources.json'
if p.exists():online['resources']=tracked(p)
# No raw trajectories/prefixes/token IDs, actor paths, hostnames or logs are exported.
from ricepo.audit import cost_report
dump('ONLINE_COST_AGGREGATES.json',cost_report(run))
dump('ONLINE_EVIDENCE.json',online)
for p in (R/'reports').glob('gemm-restore-*.json'):dump('INDEPENDENT_RESTORE.json',tracked(p))
for p in sorted(set((R/'control').glob('26*-gemm*'))|set((R/'control').glob('27*-gemm*'))):
    if p.is_file():index.append(dict(alias='executed-control/'+p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
for p in (R/'runs/tf32-gemm-source-v2/src/ricepo').glob('*.py'):
    index.append(dict(alias='executed-source-v2/src/ricepo/'+p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
import importlib.metadata,sys
versions={}
for pkg in ('torch','vllm','transformers','numpy','pytrec-eval-terrier','gensim','pyserini'):
    try:versions[pkg]=importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:versions[pkg]='NOT_AVAILABLE'
vd=importlib.metadata.distribution('vllm');installed=[]
for rel in ('vllm/v1/worker/gpu_worker.py','vllm/model_executor/layers/linear.py','vllm/model_executor/layers/utils.py','vllm/envs.py'):
    path=Path(vd.locate_file(rel))
    if path.exists():installed.append(dict(alias='installed/'+rel,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
dump('ENVIRONMENT.json',dict(versions=versions,python='.'.join(map(str,sys.version_info[:3])),installed_source=installed,packages_modified=False,dependencies_installed=False))
dump('EVIDENCE_INDEX.json',index)
print('EXPORT_COMPLETE',O.name,flush=True)
