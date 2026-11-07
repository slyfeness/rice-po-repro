"""Slurm CPU export of bounded, anonymized scale learning curves.

Never loads a model or changes a run. Large native arrays are streamed once
per completed update; existing aggregate caches bind the immutable receipts.
Only the caller's explicit run names and job IDs are queried.
"""
import argparse,csv,hashlib,json,math,os,re,subprocess
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def array_items(path,expected):
    """Stream a top-level JSON array and verify its original receipt SHA."""
    decoder=json.JSONDecoder();h=hashlib.sha256();buffer='';eof=False
    with Path(path).open(encoding='utf-8',newline='') as f:
        def more():
            nonlocal buffer,eof
            text=f.read(1024*1024)
            if not text:eof=True
            else:h.update(text.encode('utf-8'));buffer+=text
        more();buffer=buffer.lstrip()
        if not buffer.startswith('['):raise ValueError('native artifact is not an array')
        buffer=buffer[1:]
        while True:
            buffer=buffer.lstrip()
            if buffer.startswith(','):buffer=buffer[1:].lstrip()
            if buffer.startswith(']'):
                buffer=buffer[1:]
                while not eof:more()
                if buffer.strip():raise ValueError('extra JSON after native array')
                break
            try:item,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:raise
                more();continue
            yield item;buffer=buffer[end:]
        if h.hexdigest()!=expected:raise ValueError('native artifact differs from its completed receipt')


def generation_work(t,minimum_depth=1):
    generated=inputs=tools=0
    actions=[s for s in t['steps'] if s['depth']>=minimum_depth]
    tools=len(actions)
    failure=t.get('failure')
    if failure and failure['depth']>=minimum_depth:actions.append(failure)
    for action in actions:
        g=action['generation'];prefix=len(action['prefix_token_ids'])
        if g['protocol']!='two_stage_env_summary_v1':raise ValueError('unexpected generation protocol')
        for stage in g['stages']:
            span=stage['token_span'];generated+=span['end']-span['start'];inputs+=prefix+span['start']
    return generated,inputs,tools


def update_stats(directory,cache):
    up=directory/'update.json';credit=directory/'credit-diagnostics.json'
    key=digest(up);target=cache/(directory.parent.name+'-'+directory.name+'.json')
    metadata={p.name:[p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns]
              for p in directory.iterdir() if p.name in ('trajectories.json','counterfactual-trajectories.json',
              'rollout_complete.receipt.json','branches_complete.receipt.json','credit-diagnostics.json','update.json')}
    if target.exists():
        saved=read(target)
        if saved['update_sha256']!=key or saved['immutable_metadata']!=metadata:raise ValueError('previous completed update changed')
        return saved
    update=read(up);recipe=read(directory.parent/'recipe.json')['values'];old=read(directory/'rollout_complete.receipt.json')
    counts=Counter();grouped=defaultdict(list);identities=set();last=[];normal=[]
    for t in array_items(directory/'trajectories.json',old['artifacts']['trajectories.json']):
        if t['trajectory_id'] in identities or t['policy_version']!=old['policy_version']:raise ValueError('original identity changed')
        identities.add(t['trajectory_id']);grouped[t['query_id']].append(t['final_reward'])
        counts['originals']+=1;counts['legal_originals']+=t['failure'] is None
        g,i,n=generation_work(t);counts['generated_tokens']+=g;counts['input_tokens_submitted']+=i;counts['followup_calls']+=n
        last.append(t['last_valid_ndcg'])
        if t['failure'] is None:
            if len(t['steps'])!=5:raise ValueError('normal original did not reach five rounds')
            normal.append(t['final_reward'])
    if len(grouped)!=recipe['queries_per_update'] or any(len(v)!=4 for v in grouped.values()):raise ValueError('incomplete original query group')
    branch=directory/'counterfactual-trajectories.json';branch_ids=set()
    if branch.exists():
        receipt=read(directory/'branches_complete.receipt.json')
        for collection in array_items(branch,receipt['artifacts']['counterfactual-trajectories.json']):
            depth=collection['group']['anchor']['depth'];rows=collection['trajectories']
            if len(rows)!=5:raise ValueError('incomplete counterfactual group')
            for t in rows:
                if t['trajectory_id'] in branch_ids:raise ValueError('duplicate branch ID')
                branch_ids.add(t['trajectory_id']);g,i,n=generation_work(t,depth)
                counts['generated_tokens']+=g;counts['input_tokens_submitted']+=i;counts['followup_calls']+=n
    counts['new_branches']=len(branch_ids);counts['retrieval_calls']=len(grouped)+counts['followup_calls']
    recovery=directory/'completed-credit-recovery.json'
    if recovery.exists():
        counts['reused_generation_tokens']=counts['generated_tokens']
        counts['new_generation_tokens_in_this_attempt']=0
    else:counts['new_generation_tokens_in_this_attempt']=counts['generated_tokens']
    report=read(credit) if credit.exists() else {};anchors=report.get('anchors',[])
    signal=sum(x!=math.fsum(v)/4 for v in grouped.values() for x in v)
    risk_path=directory/'pilot-probability-risk.json';risk=read(risk_path)['originals'] if risk_path.exists() else None
    result=dict(update_sha256=key,immutable_metadata=metadata,optimizer=update['optimizer'],old_policy=old['policy_version'],new_policy=update['new_policy_version'],
        **counts,task_reward_mean=math.fsum(x for v in grouped.values() for x in v)/(len(grouped)*4),
        trajectory_nonzero_advantage_fraction=signal/(len(grouped)*4),last_valid_mean=math.fsum(last)/len(last),
        normal_final_ndcg_mean=math.fsum(normal)/len(normal) if normal else None,
        selected_anchors=report.get('selected_anchor_count',0),estimable_anchors=report.get('estimable_anchor_count',0),
        nonzero_local_anchors=report.get('nonzero_local_credit_count',0),
        summary_replaced_tokens=sum(a['summary_tokens'] for a in anchors),
        summary_changed_tokens=sum(a['summary_tokens'] for a in anchors if a['summary_after']!=a['trajectory_advantage']),
        think_terminal=sum(a['gates']['joint'] and a['terminal_anchor'] for a in anchors),
        think_nonterminal=sum(a['gates']['joint'] and not a['terminal_anchor'] for a in anchors),
        influence_variances=[a['influence_variance'] for a in anchors],residual_variances=[a['residual_variance'] for a in anchors],
        original_probability=risk)
    write(target,result);return result


def event_times(root,cache):
    """Main process only; worker child spans are not added to host wall again."""
    target=cache/(root.name+'-events.json')
    state=read(target) if target.exists() else dict(seen=[],totals={},started={},ended=[])
    seen=set(state['seen']);ended=set(state['ended']);started=state['started'];totals=state['totals']
    for path in root.glob('attempts/*/events/*.json'):
        name=str(path.relative_to(root))
        if name in seen:continue
        with path.open() as f:head=f.read(8192)
        match=re.search(r'"kind"\s*:\s*"([^\"]+)"',head)
        kind=match.group(1) if match else None
        if kind not in ('started','completed','failed'):
            seen.add(name);continue
        e=read(path);seen.add(name);oid=e['operation_id']
        if kind=='started':started[oid]=dict(phase=e['phase'],utc=e['utc'],iteration=e.get('iteration'));continue
        ended.add(oid);iteration=str(e.get('iteration','initialization'));phase=e['phase']
        if e.get('evaluation_scope'):category='evaluation'
        elif phase in ('model_lifecycle','behavior_weight_identity'):category='model_lifecycle'
        elif phase=='checkpoint_write':category='checkpoint'
        elif phase.startswith('retrieval'):category='retrieval'
        elif 'generation' in phase or phase=='post_update_sampling_probe':category='generation'
        elif phase in ('entropy','credit','cross_query_branch_wave'):category='entropy_credit'
        elif phase=='update' or phase.startswith(('old_','update_','optimizer_','final_weight_')):category='HF_scoring_PPO'
        else:category='unattributed'
        row=totals.setdefault(iteration,{})
        row[category]=row.get(category,0.)+e.get('exclusive_elapsed_seconds',0.)
    state.update(seen=sorted(seen),ended=sorted(ended),started=started,totals=totals);write(target,state)
    active=[v for k,v in started.items() if k not in ended]
    active.sort(key=lambda v:v['utc'])
    return totals,active[-1] if active else None


def evaluations(root,method):
    results=[]
    paths=[(-1,root/'initial-development')] if method=='Initial' and (root/'initial-evaluation-complete.json').is_file() else []
    for marker in sorted(root.glob('iteration-*/development_complete.receipt.json')):
        r=read(marker)['artifacts']
        if not r.get('skipped'):
            if digest(marker.parent/r['relative_report'])!=r['sha256']:raise ValueError('completed evaluation changed')
            paths.append((int(marker.parent.name[10:]),marker.parent/Path(r['relative_report']).parent))
    for index,path in paths:
        if not (path/'summary.json').exists():continue
        summary=read(path/'summary.json')
        if summary['manifest_sha256']!=read(root/'binding.json')['manifest_sha256']['development']:raise ValueError('evaluation manifest changed')
        if method=='Initial' and summary!=read(root/'initial-evaluation-complete.json')['summary']:raise ValueError('Initial summary differs from completion')
        rows=read(path/'results-by-query.json');groups=defaultdict(list)
        for row in rows:
            alias=hashlib.sha256(json.dumps([row['domain'],row['query_id']],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            groups[alias].append({k:row.get(k) for k in ('sample_index','failed','failure_adjusted_task_score','ndcg_at_10','last_valid_ndcg','initial_ndcg_at_10','generation_seeds')})
        if len(groups)!=32 or any(sorted(r['sample_index'] for r in rows)!=[0,1,2,3] for rows in groups.values()):raise ValueError('evaluation membership/samples incomplete')
        results.append(dict(method=method,update=index+1,summary=summary,
            queries=[dict(query_alias=k,samples=sorted(v,key=lambda r:r['sample_index']),task_score_mean=math.fsum(r['failure_adjusted_task_score'] for r in v)/4) for k,v in sorted(groups.items())]))
    return results


def main(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('CPU Slurm export required')
    R=Path(os.environ['RICE_PROJECT_ROOT']);args.output.mkdir(parents=True,exist_ok=False);cache=R/'reports/biology-scale-v1-export-cache';cache.mkdir(exist_ok=True)
    public=args.output/'public';public.mkdir();curves=[];status=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),sampling_plan_source_commit='b0ee0bb0f789f803770566572ef8fd1499f14550',runs={},raw_assets_public=False)
    evals=evaluations(R/'runs/biology-scale-v1-initial','Initial')
    registry=R/'reports/biology-scale-active-training.json'
    active_runs=read(registry) if registry.exists() else {}
    for stage in ('capacity','training'):
        for method in ('ricepo','grpo'):
            alias=active_runs.get(method,f'biology-scale-v1-training-{method}') if stage=='training' else f'biology-scale-v1-capacity-{method}'
            if not re.fullmatch(r'biology-scale-v[1-9][0-9]*-'+stage+'-'+method,alias):raise ValueError('invalid active run alias')
            root=R/'runs'/alias;info=dict(stage=stage,method=method,computed_updates=0,persisted_updates=0,status='NOT_STARTED')
            status['runs'][stage+'_'+method]=info
            if not root.exists():continue
            launches=[read(p) for p in (R/'reports').glob(root.name+'-launch-*.json')]
            info['execution_source_commits']=sorted({r['source_commit'] for r in launches})
            info['actual_run_alias']=alias
            info['job_ids']=sorted({r['slurm_job_id'] for r in launches})
            recovery=root/'iteration-000000/completed-credit-recovery.json'
            if recovery.exists():info['completed_credit_recovery']=read(recovery)
            if (root/'capacity-complete.json').exists():
                recovery=read(root/'capacity-complete.json').get('recovery')
                if recovery:info['capacity_recovery']={k:v for k,v in recovery.items() if k not in ('evidence',)}
            times,active=event_times(root,cache);info.update(status='IN_PROGRESS',active=active,phase_seconds=times)
            for d in sorted(root.glob('iteration-[0-9]*')):
                if (d/'optimizer_step_committed.receipt.json').exists():
                    info['computed_updates']=int(d.name[10:])+1
                    info['latest_committed_update']=read(d/'update.json')
                if (d/'checkpoint/complete.json').exists() and (d/'checkpoint_complete.receipt.json').exists():info['persisted_updates']=int(d.name[10:])+1
                if not (d/'iteration_complete.receipt.json').exists():continue
                stats=update_stats(d,cache);progress=read(d/'scale-progress.json') if (d/'scale-progress.json').exists() else {}
                if stage=='capacity':info['capacity_result']={k:v for k,v in stats.items() if k!='immutable_metadata'}
                else:
                    row=dict(method=method,update=int(d.name[10:])+1,persisted=(d/'checkpoint/complete.json').exists(),
                        **{k:v for k,v in stats.items() if k not in ('optimizer','immutable_metadata','influence_variances','residual_variances','original_probability')})
                    for k in ('policy_loss','reference_kl','grad_norm_before_clip','optimizer_steps','actual_state_dict_changed','outside_clip_fraction','effective_clip_fraction','ratio_min','ratio_max','ratio_mean','positive_advantages','negative_advantages','zero_advantages','zero_gradient_all_steps','parameter_probe_changed'):
                        row[k]=stats['optimizer'].get(k)
                    for name in ('influence','residual'):
                        values=sorted(stats[name+'_variances'])
                        for label,f in [('min',lambda v:v[0]),('max',lambda v:v[-1]),('mean',lambda v:math.fsum(v)/len(v)),('p50',lambda v:v[max(0,math.ceil(.5*len(v))-1)]),('p90',lambda v:v[max(0,math.ceil(.9*len(v))-1)])]:
                            row[name+'_variance_'+label]=f(values) if values else None
                    risk=stats['original_probability']
                    if risk:row.update(original_ratio_outside=risk['outside_clip_fraction'],original_extreme_tail=risk['extreme_tail_fraction'],original_ESS=risk['ess_fraction'],original_log_error_mean=risk['absolute_error']['mean'],original_log_error_p99=risk['absolute_error']['p99'],original_log_error_max=risk['absolute_error']['max'],original_legacy_consistency_pass=risk['historical_consistency_pass'])
                    row.update(iteration_seconds=progress.get('iteration_seconds'),**{k+'_seconds':v for k,v in times.get(str(int(d.name[10:])),{}).items()});curves.append(row)
            marker=root/f'{stage}-restore-complete.json'
            if marker.exists():
                receipt=read(marker)
                if digest(receipt['evidence'])!=receipt['sha256']:raise ValueError('restore evidence changed')
                info['independent_restore']=read(receipt['evidence'])
                info['status']='COMPLETE' if stage=='capacity' or (root/'completed.json').exists() else 'RESTORE_COMPLETE'
            elif stage=='training' and (root/'completed.json').exists():info['status']='FINAL_RESTORE_PENDING'
            failures=list(root.glob('attempts/*/failure.json'))
            if failures:info['failure_categories']=[read(p).get('category') for p in failures]
            if (root/'completed.json').exists():
                selected=read(root/'completed.json')['selected_checkpoint'];info['selected']={k:v for k,v in selected.items() if k in ('kind','iteration','policy_version','development_macro')}
            evals.extend(evaluations(root,method) if stage=='training' else [])
    jobs=[s for s in args.jobs.split(',') if s]
    for p in (R/'reports').glob('biology-scale-handoff-*.json'):jobs.append(read(p)['successor_job'])
    jobs=sorted(set(jobs));records=subprocess.check_output(['sacct','-j',','.join(jobs),'--parsable2','--noheader','--format=JobID,State,ElapsedRaw,Submit,Start,End,AllocTRES'],text=True).strip().splitlines()
    # Slurm appends the cancelling operator's UID to State. Publish the state
    # itself only; the operator/account identifier is not audit evidence.
    status['slurm_records']=[];allocations=[]
    for line in records:
        fields=line.split('|');fields[1]=fields[1].split(' by ',1)[0]
        status['slurm_records'].append('|'.join(fields))
        if not fields[0].isdigit():continue  # .batch/srun are not extra allocations
        gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',fields[6])
        allocations.append(dict(job_id=fields[0],state=fields[1],allocated_gpus=int(gpu[1]) if gpu else 0,
            elapsed_seconds=int(fields[2]),GPU_hours=int(fields[2])*(int(gpu[1]) if gpu else 0)/3600))
    states={r['job_id']:r['state'] for r in allocations}
    for info in status['runs'].values():
        if info['status'] not in ('IN_PROGRESS','RESTORE_COMPLETE'):continue
        observed=[states[j] for j in info.get('job_ids',[]) if j in states]
        if any(s=='RUNNING' for s in observed):info['status']='RUNNING'
        elif any(s=='PENDING' for s in observed):info['status']='PENDING'
        elif observed:info['status']='FAILED' if any(s.startswith(('FAILED','OUT_OF_MEMORY','TIMEOUT')) for s in observed) else 'STOPPED'
    status['GPU_allocations']=allocations;status['measured_GPU_hours_at_snapshot']=math.fsum(r['GPU_hours'] for r in allocations)
    status['sampling_note']='training generation tokens exclude reused branch history and environment markers; submitted input counts include full repeated native context, not measured uncached prefill FLOPs'
    status['timing_note']='main-process exclusive spans only; evaluation includes its nested generation/tools. Uninstrumented verification/serialization remains unattributed. In-flight spans excluded from completed-phase totals.'
    write(public/'RUN_STATUS.json',status);write(public/'EVALUATION.json',dict(expected_queries=32,samples_per_query=4,evaluations=evals))
    columns=list(dict.fromkeys(k for r in curves for k in r)) or ['method','update','persisted','iteration_seconds']
    with (public/'TRAINING_CURVES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(curves)
    print('SCALE_EXPORT_COMPLETE',args.output.name,{k:(v['computed_updates'],v['persisted_updates'],v['status']) for k,v in status['runs'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--jobs',required=True);main(p.parse_args())
