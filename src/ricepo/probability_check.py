"""Scalar native-action probability diagnostics, never vocabulary dumps."""
import math

def compare_log_probs(sampled,teacher,mask):
    if not len(sampled)==len(teacher)==len(mask):raise ValueError('unaligned probability evidence')
    delta=[float(t)-float(s) for s,t,m in zip(sampled,teacher,mask,strict=True) if m]
    if not delta or any(not math.isfinite(d) for d in delta):raise ValueError('missing/nonfinite action probabilities')
    errors=sorted(abs(d) for d in delta)
    mean=math.fsum(errors)/len(errors);p99=errors[max(0,math.ceil(.99*len(errors))-1)];maximum=errors[-1]
    return dict(action_tokens=len(errors),mean_abs_error=mean,p99_abs_error=p99,max_abs_error=maximum,
        ratio_min=math.exp(min(delta)),ratio_max=math.exp(max(delta)),
        ratio_mean=math.fsum(math.exp(d) for d in delta)/len(delta),
        within_bf16_tolerance=mean<=.02 and p99<=.06 and maximum<=.12,
        tolerances=dict(mean=.02,p99=.06,max=.12),units='natural log probability')
