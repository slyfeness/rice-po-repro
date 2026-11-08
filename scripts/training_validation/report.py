"""CPU-only aggregate training validation report from immutable native evidence."""
import os,json,csv,datetime,collections,math
from pathlib import Path
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('explicit Slurm allocation required')
R=Path(os.environ['RICE_PROJECT_ROOT']);read=lambda p:json.loads(p.read_text())
O=R/'reports'/f'gemm-training-export-{os.environ["SLURM_JOB_ID"]}';O.mkdir()
def dump(name,data):(O/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def classify(phase):
    if phase=='engine_generation':return 'generation_engine'
    if phase=='sampler_load':return 'sampler_offload_snapshot_engine_start'
    if phase=='post_update_sampling_probe':return 'post_update_probe_other'
    if phase=='checkpoint_write':return 'checkpoint_write_verify'
    if phase in ('update','old_log_probs','old_probability_forward','update_forward','optimizer_minibatch_step'):return 'hf_scoring_ppo'
    if phase.startswith('retrieval'):return 'retrieval'
    if phase in ('credit','entropy'):return 'entropy_credit'
    if phase=='asset_binding':return 'asset_validation'
    if phase=='behavior_weight_identity':return 'weight_identity'
    return 'other_instrumented'

def exclusive(intervals):
    events=[];active={};sums=collections.defaultdict(float)
    for n,row in enumerate(intervals):events.extend([(row['start'],1,n),(row['end'],-1,n)])
    old=None
    for t,k,n in sorted(events):
        if old is not None and t>old and active:
            rows=list(active.values());leaves=[r for r in rows if not any(s is not r and s['start']>=r['start'] and s['end']<=r['end'] and (s['start']>r['start'] or s['end']<r['end']) for s in rows)]
            cats={classify(r['phase']) for r in leaves};cat=next(iter(cats)) if len(cats)==1 else 'unknown_overlap'
            sums[cat]+=t-old
        if k==1:active[n]=intervals[n]
        else:active.pop(n,None)
        old=t
    return dict(sums)

paired=[];continuous=[];details={}
for tag in ('gemm-training-A-176262','gemm-training-B-176263'):
    root=R/'runs'/tag
    for it in sorted(root.glob('iteration-*')):
        idx=int(it.name.split('-')[-1]);completed=(it/'training-validation.json').exists()
        resources_path=R/'reports'/f'{tag}-u{idx+1}-resources.json'
        if not resources_path.exists():continue
        resources=read(resources_path);update=read(it/'update.json') if (it/'update.json').exists() else None
        result=read(it/'training-validation.json') if completed else {}
        risk=read(it/'pilot-probability-risk.json') if (it/'pilot-probability-risk.json').exists() else None
        credit=read(it/'credit-diagnostics.json') if (it/'credit-diagnostics.json').exists() else None
        originals=read(it/'trajectories.json') if (it/'trajectories.json').exists() else []
        attempts=list(root.glob('attempts/*'));attempt=None
        for a in attempts:
            ls=a/'loaded-state.json'
            if idx==0 and not ls.exists():
                has_update=False
                for ep in a.glob('events/*.json'):
                    with ep.open() as handle:head=handle.read(2048)
                    if '"kind": "completed"' in head and '"phase": "update"' in head:
                        has_update=True;break
                if not has_update:continue
                candidate=a
            elif idx>0 and ls.exists() and read(ls)['local_iteration']==idx-1:candidate=a
            else:continue
            attempt=candidate
        if attempt is None:raise ValueError('missing unique attempt')
        intervals=[];generation=[];worker_states=[];retry=collections.Counter()
        for p in attempt.glob('events/*.json'):
            e=read(p);phase=e.get('phase');kind=e.get('kind');elapsed=e.get('elapsed_seconds')
            if kind in ('completed','failed') and phase and isinstance(elapsed,(int,float)):
                end=datetime.datetime.fromisoformat(e['utc']).timestamp();intervals.append(dict(start=end-elapsed,end=end,phase=phase))
            if kind=='vllm_policy_ready':
                end=datetime.datetime.fromisoformat(e['utc']).timestamp();intervals.append(dict(start=end-e['load_seconds'],end=end,phase='sampler_load'))
            if phase and kind=='failed':retry[phase]+=1
        for p in attempt.glob('vllm/worker-*-ready.json'):
            w=read(p);worker_states.append(dict(policy=w['policy'],numeric_mode=w['worker_ready']['numeric_mode'],cache_policy=w['cache_policy'],gemm_state=[{k:v for k,v in s.items() if k!='pid'} for s in w['worker_ready'].get('gemm_state',[])]))
        policy=read(it/'rollout_complete.receipt.json')['policy_version'] if originals else None
        for p in attempt.glob('vllm/worker-*/attempts/*/events/*.json'):
            e=read(p)
            if e['kind']=='vllm_native_generation_timing':
                end=datetime.datetime.fromisoformat(e['utc']).timestamp();intervals.append(dict(start=end-e['elapsed_seconds'],end=end,phase='engine_generation'))
                generation.append(e)
        snapshots=[]
        for p in attempt.glob('vllm/snapshot-*/complete.json'):
            snap=read(p);snapshots.append(dict(policy=snap['policy'],state_dict_sha256=snap['state_dict_sha256'],payload_sha256=snap['assets']['payload_sha256']))
        train_gen=[e for e in generation if e['policy_version']==policy]
        train_seconds=sum(e['elapsed_seconds'] for e in train_gen);gen_tokens=sum(e['generated_tokens'] for e in train_gen)
        costs=exclusive(intervals);covered=sum(costs.values());whole=resources['elapsed_seconds']
        if covered>whole+1:raise ValueError('overlapping/incorrect total accounting')
        costs['unknown_uninstrumented']=whole-covered
        retrievals=sum(1 for p in attempt.glob('events/*.json') if (lambda e:e.get('kind')=='completed' and e.get('phase') in ('retrieval_D0','retrieval_followup'))(read(p)))
        optimizer=update['optimizer'] if update else {};gradient=optimizer.get('pilot_gradient',{})
        base=dict(group='A' if '-A-' in tag else 'B',update=idx+1,status='COMPLETE' if completed else 'INCOMPLETE',queries=4,
            full_run_seconds=whole,engine_training_seconds=train_seconds,actual_training_generated_tokens=gen_tokens,
            training_engine_tokens_per_second=gen_tokens/train_seconds if train_seconds else None,
            submitted_training_input_tokens=sum(e['input_tokens'] for e in train_gen),training_engine_calls=len(train_gen),
            training_stage_requests=sum(e['requests'] for e in train_gen),retrieval_calls=retrievals,
            hf_scoring_ppo_exclusive_seconds=costs.get('hf_scoring_ppo',0),entropy_credit_exclusive_seconds=costs.get('entropy_credit',0),
            sampler_offload_snapshot_engine_start_exclusive_seconds=costs.get('sampler_offload_snapshot_engine_start',0),
            checkpoint_write_verify_exclusive_seconds=costs.get('checkpoint_write_verify',0),retrieval_exclusive_seconds=costs.get('retrieval',0),
            initial_hf_weight_load_exclusive_seconds='UNKNOWN',device0_sampled_peak_mib=resources['sampled_device_peak_mib'][0],device1_sampled_peak_mib=resources['sampled_device_peak_mib'][1],
            original_count=len(originals),original_failures=sum(t['failure'] is not None for t in originals),mean_training_task_utility=sum(t['final_reward'] for t in originals)/len(originals) if originals else None,mean_last_valid_ndcg=sum(t['last_valid_ndcg'] for t in originals)/len(originals) if originals else None,legal_five_round_originals=sum(t['failure'] is None and len(t['steps'])==5 for t in originals),
            selected_anchors=credit.get('selected_anchor_count') if credit else None,estimable_anchors=credit.get('estimable_anchor_count') if credit else None,
            nonzero_local_credit=credit.get('nonzero_local_credit_count') if credit else None,reasoning_propagations=credit.get('gate_pass_count') if credit else None,
            summary_replaced_tokens=sum(a['summary_tokens'] for a in credit['anchors']) if credit else None,summary_changed_tokens=sum(a['summary_tokens'] for a in credit['anchors'] if a['summary_after']!=a['trajectory_advantage']) if credit else None,reasoning_changed_tokens=sum(a['reasoning_tokens'] for a in credit['anchors'] if a['propagated'] and a['reasoning_after']!=a['trajectory_advantage']) if credit else None,propagated_terminal_anchor_count=sum(a['propagated'] and a['terminal_anchor'] for a in credit['anchors']) if credit else None,optimizer_steps=optimizer.get('optimizer_steps'),errors=sum(retry.values()))
        if idx==0:paired.append(base)
        post=read(it/'pilot-post-update-risk.json') if (it/'pilot-post-update-risk.json').exists() else None
        branch_records=read(it/'counterfactual-trajectories.json') if (it/'counterfactual-trajectories.json').exists() else []
        base['new_branch_failures']=sum(t['failure'] is not None for g in branch_records for t in g['trajectories'])
        original_stats=risk['originals'] if risk else {};combined=risk['all_actual_tokens'] if risk else {}
        row={**base,'policy_loss':optimizer.get('policy_loss'),'reference_kl':optimizer.get('reference_kl'),'grad_norm':optimizer.get('grad_norm_before_clip'),
             'parameter_delta_l2':gradient.get('parameter_delta_l2'),'changed_parameter_elements':gradient.get('changed_parameter_elements'),
             'nonzero_advantage_positions':optimizer.get('positive_advantages',0)+optimizer.get('negative_advantages',0),
             'effective_clip_fraction':optimizer.get('effective_clip_fraction'),'original_ratio_outside_fraction':original_stats.get('outside_clip_fraction'),
             'original_log_error_mean':original_stats.get('absolute_error',{}).get('mean'),'original_log_error_max':original_stats.get('absolute_error',{}).get('max'),'original_log_error_p99':original_stats.get('absolute_error',{}).get('p99'),'original_ratio_p99':original_stats.get('ratio',{}).get('p99'),
             'original_ess_fraction':original_stats.get('ess_fraction'),'original_extreme_fraction':original_stats.get('extreme_tail_fraction'),
             'checked_branch_count':risk.get('checked_branch_count') if risk else None,'branch_count':risk.get('branch_count') if risk else None,
             'combined_ratio_outside_fraction':combined.get('outside_clip_fraction'),'combined_extreme_fraction':combined.get('extreme_tail_fraction'),
             'combined_log_error_mean':combined.get('absolute_error',{}).get('mean'),'combined_log_error_p99':combined.get('absolute_error',{}).get('p99'),'combined_log_error_max':combined.get('absolute_error',{}).get('max'),'combined_ess_fraction':combined.get('ess_fraction'),'historical_consistency_pass':combined.get('historical_consistency_pass'),'post_update_risk_pass':post.get('pass_risk_screen') if post else None,
             'policy_version':update.get('new_policy_version') if update else None,'checkpoint_complete_sha256':result.get('checkpoint_complete_sha256')}
        if base['group']=='B':continuous.append(row)
        reward_groups=collections.defaultdict(list)
        for t in originals:reward_groups[__import__('hashlib').sha256(t['query_id'].encode()).hexdigest()].append(t['final_reward'])
        loaded=read(attempt/'loaded-state.json') if (attempt/'loaded-state.json').exists() else None
        if loaded:
            previous=it.parent/f'iteration-{idx-1:06d}'/'checkpoint'
            savedprobe=read(previous/'rng-probe.json')
            if loaded['rng_probe']!=savedprobe or loaded['optimizer_steps']!=[idx]:raise ValueError('resume RNG or optimizer position mismatch')
        details[f'{base["group"]}-{idx+1}']=dict(costs_exclusive=costs,row=row,post_update=post,online_risk=risk,reward_groups=dict(reward_groups),
            credit_anchors=[{k:v for k,v in a.items() if k not in ('query','trajectory','outcomes')}|dict(outcomes=[{k:v for k,v in b.items() if k!='id'} for b in a['outcomes']]) for a in credit['anchors']] if credit else None,hf_precision=optimizer.get('hf_precision'),gradient=gradient,worker_states=worker_states,loaded_state=loaded,
            checkpoint_complete=bool((it/'checkpoint_complete.receipt.json').exists()),reload=read(it/'reload-verification.json') if (it/'reload-verification.json').exists() else None,
            immutable_snapshots=snapshots,asset_binding={k:read(root/'binding.json')[k] for k in ('dataset_receipt_sha256','manifest_sha256','model_sha256','retriever_sha256','prompt_sha256')},checkpoint_sha256=result.get('checkpoint_complete_sha256'),query_schedule_fingerprint=__import__('hashlib').sha256((it/'query-batch.json').read_bytes()).hexdigest())
for name,rows in [('PAIRED_E2E.csv',paired),('CONTINUOUS_UPDATES.csv',continuous)]:
    if rows:
        with (O/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
dump('AGGREGATES.json',details)
for p in (R/'reports').glob('gemm-training-restore-*.json'):dump(p.name,read(p))
print('TRAINING_EXPORT_COMPLETE',O.name,flush=True)
