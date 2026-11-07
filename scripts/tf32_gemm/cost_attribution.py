# Public driver: explicit Slurm execution only; no downloads.
"""Exclusive interval union from saved terminal-operation timestamps; no models."""
import json,os,csv,collections,datetime,hashlib
from pathlib import Path

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT']);I=R/'reports/tf32-gemm-cpu-175837'
O=R/'reports'/f'gemm-cost-exclusive-{os.environ["SLURM_JOB_ID"]}';O.mkdir()
read=lambda p:json.loads(p.read_text())
def category(phase):
    if phase in ('original_generation_batch','branch_generation_batch'):return 'GENERATION_HOST_WRAPPER'
    if phase.startswith('retrieval'):return 'RETRIEVAL'
    if phase in ('old_log_probs','old_probability_forward','update_forward','optimizer_minibatch_step','update'):return 'HF_SCORING_PPO'
    if phase in ('entropy','credit'):return 'ENTROPY_CREDIT_COORDINATION'
    if phase=='behavior_weight_identity':return 'WEIGHT_IDENTITY'
    if phase=='asset_binding':return 'ASSET_VALIDATION'
    return 'UNKNOWN_INSTRUMENTED_OVERHEAD'
def flatten(intervals):
    events=[];counts=collections.defaultdict(float);active={};evalgen=0.
    for i,r in enumerate(intervals):
        events.extend([(r['start'],1,i),(r['end'],-1,i)])
    previous=None
    for t,kind,i in sorted(events):
        if previous is not None and t>previous and active:
            rows=list(active.values());phases={r['phase'] for r in rows};dt=t-previous
            is_eval=bool(phases&{'development','development_evaluation','evaluation_query','initial_development_evaluation'})
            if is_eval:
                key='EVALUATION'
                if phases&{'original_generation','original_generation_batch','branch_generation_batch'}:evalgen+=dt
            elif 'post_update_sampling_probe' in phases:key='CHECKPOINT_POST_UPDATE_PROBE'
            elif 'checkpoint_write' in phases:key='CHECKPOINT_WRITE_VERIFY'
            elif 'update' in phases:key='HF_SCORING_PPO'
            else:
                leaves=[r for r in rows if not any(s is not r and s['start']>=r['start'] and s['end']<=r['end']
                    and (s['start']>r['start'] or s['end']<r['end']) for s in rows)]
                cats={category(r['phase']) for r in leaves}
                key=cats.pop() if len(cats)==1 else 'UNKNOWN_CONCURRENT_OVERLAP'
            counts[key]+=dt
        if kind==1:active[i]=intervals[i]
        else:active.pop(i,None)
        previous=t
    return dict(counts),evalgen
# Mathematical accounting checks, not a training test.
t,_=flatten([dict(start=0.,end=10.,phase='credit'),dict(start=2.,end=7.,phase='branch_generation_batch')])
assert t=={'ENTROPY_CREDIT_COORDINATION':5.,'GENERATION_HOST_WRAPPER':5.}
t,_=flatten([dict(start=0.,end=5.,phase='entropy'),dict(start=3.,end=7.,phase='retrieval_followup')])
assert abs(sum(t.values())-7)<1e-9 and t['UNKNOWN_CONCURRENT_OVERLAP']==2.
rows=[];details=[];evidence=[]
for file in sorted(I.glob('*-operations.json')):
    raw=read(file);run=R/'runs'/raw['run'];intervals=[]
    for r in raw['intervals']:
        end=datetime.datetime.fromisoformat(r['utc']).timestamp();elapsed=r['elapsed_seconds']
        if elapsed is None or elapsed<0:continue
        intervals.append(dict(start=end-elapsed,end=end,phase=r['phase'],split=r.get('split')))
    counts,evalgen=flatten(intervals);wholefile=R/'reports'/(run.name+'-resources.json')
    whole=read(wholefile)['elapsed_seconds'];initial=read(run/'iteration-000000/rollout_complete.receipt.json')['policy_version']
    loads=[]
    for p in run.glob('attempts/*/vllm/worker-*-ready.json'):
        v=read(p);loads.append(dict(train=v['policy']==initial,seconds=v['load_seconds']))
    load_train=sum(v['seconds'] for v in loads if v['train']);load_other=sum(v['seconds'] for v in loads if not v['train'])
    rawgen=counts.get('GENERATION_HOST_WRAPPER',0.)
    # Lifecycle timer is nested in the first generation call, confirmed in executed source.
    if load_train<=rawgen+1e-3:
        counts['GENERATION_HOST_WRAPPER']=max(0.,rawgen-load_train)
        counts['SAMPLER_OFFLOAD_SNAPSHOT_ENGINE_START']=load_train
        genstatus='MEASURED host call excluding measured lifecycle; includes transport/format/evidence I/O'
    else:
        genstatus='UNKNOWN lifecycle/operation scope mismatch; raw wrapper retained';load_train=None
    covered=sum(counts.values());counts['UNKNOWN_UNINSTRUMENTED']=whole-covered
    if covered>whole+.5:raise ValueError('operation envelope exceeds run wall; cannot invent exclusive accounting')
    # Eval remains a disjoint top-level stage. Its generation subset is ONLY a separate Amdahl input.
    eval_only_gen=max(0.,evalgen-load_other) if load_other<=evalgen else None
    generation=counts.get('GENERATION_HOST_WRAPPER',0.)
    known_total_gen=generation+(eval_only_gen or 0.)
    for cat,seconds in sorted(counts.items()):
        rows.append(dict(run=run.name,queries=32 if run.name.startswith('scale32') else 4,
            category=cat,exclusive_seconds=seconds,run_wall_seconds=whole,fraction_of_run=seconds/whole,
            status='UNKNOWN cause; measured residual duration' if cat.startswith('UNKNOWN') else 'MEASURED',
            generation_definition=genstatus if cat=='GENERATION_HOST_WRAPPER' else '',
            denominator='existing driver elapsed_seconds; includes initialization/checkpoint/eval or diagnostic scope'))
    details.append(dict(run=run.name,run_wall_seconds=whole,generation_train_host_seconds=generation,
        generation_train_fraction=generation/whole,generation_pure_GPU_seconds='UNKNOWN',
        lifecycle_train_seconds=load_train,lifecycle_inside_eval_or_probe_seconds=load_other,
        evaluation_total_seconds=counts.get('EVALUATION',0.),evaluation_generation_subset_seconds=eval_only_gen,
        component_rows_are_disjoint=True,exclusive_seconds_sum=sum(counts.values()),
        saving_if_only_training_generation_1_5x=generation/3/whole,
        saving_if_train_and_known_eval_generation_1_5x=known_total_gen/3/whole,
        whole_speedup_if_training_generation_1_5x=whole/(whole-generation+generation/1.5),
        whole_speedup_if_train_and_known_eval_generation_1_5x=whole/(whole-known_total_gen+known_total_gen/1.5),
        estimate_basis='ESTIMATED Amdahl, other stages fixed; transport/format inside generation may not accelerate; no new training measurement'))
    for p in (file,wholefile):evidence.append(dict(alias=run.name+'/'+p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
with (O/'TRAINING_COST_BREAKDOWN.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(O/'AMDAHL_ESTIMATES.json').write_text(json.dumps(dict(runs=details,evidence_index=evidence,interval_accounting_checks=2),indent=2)+'\n')
print(json.dumps(details),flush=True)
