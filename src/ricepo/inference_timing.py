"""Native vLLM request intervals; no token-ratio timing approximations."""
import math

NOT_MEASURED='NOT_MEASURED'


def request_timing(metrics, tokens):
    if metrics is None:
        return {key:NOT_MEASURED for key in ('ttft','queue','prefill','decode','tpot','core_completion')}
    queued=metrics.get('queued_ts',0);scheduled=metrics.get('scheduled_ts',0)
    first=metrics.get('first_token_ts',0);last=metrics.get('last_token_ts',0)
    def interval(start,end):
        return end-start if start>0 and end>=start else NOT_MEASURED
    decode=interval(first,last)
    return dict(ttft=metrics.get('first_token_latency',NOT_MEASURED),queue=interval(queued,scheduled),
        prefill=interval(scheduled,first),decode=decode,
        tpot=decode/(tokens-1) if isinstance(decode,(int,float)) and tokens>1 else NOT_MEASURED,
        core_completion=interval(queued,last))


def distribution(values):
    values=sorted(float(v) for v in values if isinstance(v,(int,float)) and math.isfinite(v))
    if not values:return NOT_MEASURED
    return dict(n=len(values),mean=math.fsum(values)/len(values),p50=values[math.ceil(.5*len(values))-1],
        p90=values[math.ceil(.9*len(values))-1],p99=values[math.ceil(.99*len(values))-1],max=values[-1])
