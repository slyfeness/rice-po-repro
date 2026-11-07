"""Post-hoc engineering risk screens; NEVER backend-equivalence certificates.

Historical thresholds remain reported. Pilot thresholds were revised after
observing historical results and must be frozen before held-out generation.
"""
import math


def _summary(values):
    ordered = sorted(values)
    def quantile(q): return ordered[max(0, math.ceil(q*len(ordered))-1)]
    return dict(mean=math.fsum(ordered)/len(ordered), min=ordered[0],
                p50=quantile(.5), p90=quantile(.9), p99=quantile(.99), max=ordered[-1])


def mismatch_statistics(sampled, current, advantages=None):
    if not sampled or len(sampled) != len(current):
        raise ValueError('aligned nonempty native probability records required')
    delta = [float(p)-float(q) for q,p in zip(sampled,current,strict=True)]
    if any(not math.isfinite(float(x)) for x in (*sampled,*current)):
        raise ValueError('nonfinite probabilities')
    absolute = [abs(x) for x in delta]
    # Stable even for grossly incompatible policies. Never exponentiate raw
    # large delta to calculate ESS; overflowed ratios remain a reported tail.
    maximum = max(delta)
    weights = [math.exp(x-maximum) for x in delta]
    ess = math.fsum(weights)**2/(len(weights)*math.fsum(w*w for w in weights))
    ratios = [math.exp(x) if x < 709. else float('inf') for x in delta]
    outside = sum(x < math.log(.8) or x > math.log(1.2) for x in delta)/len(delta)
    result = dict(tokens=len(delta), signed_delta=_summary(delta), absolute_error=_summary(absolute),
        ratio=_summary(ratios), outside_clip_fraction=outside,
        extreme_tail_fraction=sum(abs(x)>1 for x in delta)/len(delta), ess_fraction=ess,
        ratio_mean_in_diagnostic_band=.995 <= math.fsum(ratios)/len(ratios) <= 1.005,
        historical_consistency_pass=(math.fsum(absolute)/len(delta)<=.02 and _summary(absolute)['p99']<=.06 and max(absolute)<=.12),
        matched_consistency_pass=(math.fsum(absolute)/len(delta)<=.02 and _summary(absolute)['p99']<=.10 and outside<=.01),
        effective_clip_fraction=None,zero_advantage_fraction=None)
    if advantages is not None:
        if len(advantages)!=len(delta) or any(not math.isfinite(float(x)) for x in advantages):
            raise ValueError('aligned finite actual advantages required')
        result['effective_clip_fraction']=sum((a>0 and d>math.log(1.2)) or (a<0 and d<math.log(.8))
            for a,d in zip(advantages,delta,strict=True))/len(delta)
        result['zero_advantage_fraction']=sum(a==0 for a in advantages)/len(delta)
    return result


def risk_screen(report, *, outside_max=.01, extreme_max=.0005):
    """Finite evidence and exact identity/alignment are separate mandatory gates."""
    return (report['outside_clip_fraction']<=outside_max
            and report['extreme_tail_fraction']<=extreme_max)


def pilot_settings(deployment, recipe):
    """Explicit opt-in, separate provenance and bounded new experiment only."""
    import json
    from pathlib import Path
    from .artifacts import file_sha256, require_cluster_storage
    spec=deployment.get('precision_pilot')
    if spec is None:return None
    if deployment.get('sampling_backend')!='vllm_mixed_precision_pilot_v1':
        raise ValueError('pilot must use its independent backend identity')
    if spec.get('candidate') not in ('T','A'):
        raise ValueError('only predeclared T/A candidates are supported')
    p=require_cluster_storage(spec['validation_plan_path'])
    if file_sha256(p)!=spec['validation_plan_sha256']:
        raise ValueError('frozen pilot plan fingerprint differs')
    frozen=json.loads(Path(p).read_text())
    check=json.loads(require_cluster_storage(spec['heldout_result_path']).read_text())
    if (check['private_plan_sha256']!=spec['validation_plan_sha256'] or
            check['candidate']!=spec['candidate'] or not check['risk_screen_pass']):
        raise ValueError('independent held-out risk screen not passed')
    if (recipe['method']!='ricepo' or recipe['queries_per_update']!=2 or recipe['total_updates']!=2
            or recipe['trajectories_per_query']!=4 or recipe['anchors_per_query']!=4
            or recipe['total_branches_including_original']!=6 or recipe['rollout_depth']!=5
            or recipe['training_seeds']!=[frozen['training_seed']]
            or recipe['query_composition']!=frozen['conditions']['query_composition']
            or recipe['local_advantage_normalization']!='center_only'
            or recipe['ppo_epochs']!=1 or recipe['clip_epsilon']!=.2
            or recipe['actor_learning_rate']!=1e-6 or recipe['actor_kl_coefficient']!=.001
            or recipe['influence_threshold']!=.05 or recipe['residual_threshold']!=.03
            or deployment.get('parent_run_root') or not deployment.get('stop_after_checkpoint_diagnostic')):
        raise ValueError('pilot exceeds the frozen bounded two-query protocol')
    return {**frozen,**spec,'numeric_mode':'tf32_pilot' if spec['candidate']=='T' else 'bf16_fast'}
