# Public driver: explicit Slurm execution only; no downloads.
"""Saved evidence only: operation timing inventory and native-prefix cache bounds."""
import json,os,time,hashlib,struct,collections
from pathlib import Path

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT'])
O=R/'reports'/f'tf32-gemm-cpu-{os.environ["SLURM_JOB_ID"]}';O.mkdir()
def dump(name,value):(O/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
runs=['scale32-ricepo-175312','scale32-grpo-175313','optimization4-reference-ricepo-175521',
      'optimization4-reference-grpo-175523','optimization4-candidate-ricepo-175524','optimization4-candidate-grpo-175525']
for name in runs:
    run=R/'runs'/name;phases=collections.Counter();intervals=[];incomplete=collections.Counter()
    for path in run.glob('attempts/*/events/*.json'):
        e=json.loads(path.read_text());phase=e.get('phase')
        if phase and e.get('kind') in ('completed','failed'):
            phases[phase]+=1
            intervals.append({k:e.get(k) for k in ('kind','utc','phase','operation_id','elapsed_seconds','iteration','split','request_id')})
        elif phase and e.get('kind')=='started':incomplete[phase]+=1
    dump(name+'-operations.json',dict(run=name,phases=dict(phases),started=dict(incomplete),intervals=intervals))
    small={}
    for p in run.glob('*.json'):
        if p.stat().st_size<2*1024*1024:
            small[p.name]=json.loads(p.read_text())
    dump(name+'-metadata-private.json',small)
    print('TIMING_INVENTORY',name,dict(phases),flush=True)
from ricepo.inference_scheduling import partition_requests
M=R/'benchmarks/tf32-swap-175674/MANIFEST.json';m=json.loads(M.read_text());rows=m['requests']
parts=partition_requests(rows,4);assignment={rid:i for i,part in enumerate(parts) for rid in part}
groups={g:i for i,g in enumerate(sorted({r['group_key'] for r in rows}))}
cache=[]
for block in (16,32):
    nodes={};total=0;input_tokens=sum(len(r['prefix_token_ids']) for r in rows)
    for r in rows:
        ids=r['prefix_token_ids'];h=hashlib.sha256();g=1<<groups[r['group_key']];w=1<<assignment[r['request_id']]
        for start in range(0,len(ids)-block+1,block):
            h.update(struct.pack('<'+str(block)+'I',*ids[start:start+block]));key=h.digest();total+=block
            count,gm,wm=nodes.get(key,(0,0,0));nodes[key]=(count+1,gm|g,wm|w)
    unique=len(nodes)*block;replicated=sum(wm.bit_count() for _,_,wm in nodes.values())*block
    cross=sum(1 for _,gm,_ in nodes.values() if gm.bit_count()>1)*block
    lost=replicated-unique
    cache.append(dict(block_size_assumption=block,input_tokens=input_tokens,full_block_occurrence_tokens=total,
        unique_global_prefix_tokens=unique,sum_unique_per_worker_prefix_tokens=replicated,
        unique_cross_group_shared_prefix_tokens=cross,extra_replicated_prefix_tokens=lost,
        ideal_cold_single_hit_fraction=(total-unique)/input_tokens,
        ideal_cold_four_hit_fraction=(total-replicated)/input_tokens,
        ideal_hit_loss_percentage_points=100*lost/input_tokens,
        basis='ESTIMATED unbounded sequential input-prefix reuse; not actual concurrent/evicting KV cache simulation'))
dump('CACHE_ANALYSIS.json',dict(manifest_sha256=hashlib.sha256(M.read_bytes()).hexdigest(),requests=len(rows),groups=len(groups),
    partition_counts=list(map(len,parts)),group_never_split=all(len({assignment[r['request_id']] for r in rows if r['group_key']==g})==1 for g in groups),
    estimates=cache,original_call='partition_requests unchanged; input group IDs preserved'))
print('CACHE_ANALYSIS',json.dumps(cache),flush=True)
