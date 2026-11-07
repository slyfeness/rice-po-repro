"""Scalar acceptance diagnostics for an explicitly separate BF16 training probe."""
import math


def probability_diagnostics(sampled, current, *, clip_epsilon=.2):
    if len(sampled) != len(current) or not sampled:
        raise ValueError("aligned nonempty native probability records required")
    delta = [float(c)-float(s) for s,c in zip(sampled,current,strict=True)]
    if any(not math.isfinite(x) for x in delta):
        raise ValueError("nonfinite native probability record")
    ordered = sorted(abs(x) for x in delta)
    ratio = [math.exp(x) for x in delta]
    return dict(tokens=len(delta),mean_abs_log_error=math.fsum(ordered)/len(ordered),
        p99_abs_log_error=ordered[max(0,math.ceil(.99*len(ordered))-1)],
        max_abs_log_error=ordered[-1],ratio_min=min(ratio),ratio_max=max(ratio),
        ratio_mean=math.fsum(ratio)/len(ratio),
        clip_fraction=sum(x<1-clip_epsilon or x>1+clip_epsilon for x in ratio)/len(ratio),
        historical_fp32_gate=(math.fsum(ordered)/len(ordered)<=.02 and
            ordered[max(0,math.ceil(.99*len(ordered))-1)]<=.06 and ordered[-1]<=.12))


def matched_behavior_gate(diagnostics, criteria):
    """Thresholds must be bound in the probe receipt before model execution."""
    return (diagnostics['p99_abs_log_error'] <= criteria['p99_abs_log_error_max']
            and diagnostics['clip_fraction'] <= criteria['clip_fraction_max']
            and diagnostics['mean_abs_log_error'] <= criteria['mean_abs_log_error_max'])
