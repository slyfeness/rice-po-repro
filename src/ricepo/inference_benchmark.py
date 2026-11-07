"""Validation for a fixed, token-native RICE inference benchmark manifest.

Importing this module does not load a model or issue a generation request.
The original manifest validation remains unchanged. Explicit benchmark helpers
below observe an already-created engine only when called by an opt-in driver.
"""

from collections import Counter


MAX_CONTEXT_TOKENS = 32768
MAX_ACTION_TOKENS = 4096
_STOP_MARKERS = {"think": "</think>", "summary": "</summary>"}
_SOURCE_KINDS = frozenset({"original", "branch"})


def context_bin(prefix_tokens):
    """Classify an unchanged native prefix for fixed-load coverage reporting."""
    if prefix_tokens <= 8192:
        return "short"
    if prefix_tokens <= 16384:
        return "medium"
    return "long"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_request_manifest(rows, *, expected_count=128):
    """Reject an incomplete, altered, or non-representative fixed request pool.

    Every selected row retains the original native prefix, explicit seed,
    original remaining action budget and stage-specific endpoint.  A benchmark
    cannot silently discard a Think/Summary, original/branch, or length class.
    """
    _require(type(expected_count) is int and expected_count > 0, "expected_count must be positive")
    _require(isinstance(rows, list) and len(rows) == expected_count,
             f"expected exactly {expected_count} requests")
    identifiers = set()
    stages = Counter()
    sources = Counter()
    bins = Counter()
    for row in rows:
        _require(isinstance(row, dict), "request row must be an object")
        request_id = row.get("request_id")
        _require(isinstance(request_id, str) and request_id, "request_id must be nonempty")
        _require(request_id not in identifiers, "duplicate request_id")
        identifiers.add(request_id)
        stage = row.get("stage")
        _require(stage in _STOP_MARKERS, "unsupported request stage")
        _require(row.get("stop_marker") == _STOP_MARKERS[stage], "stage stop marker differs from native protocol")
        source = row.get("source_kind")
        _require(source in _SOURCE_KINDS, "unsupported request source_kind")
        prefix = row.get("prefix_token_ids")
        _require(isinstance(prefix, list) and prefix and all(type(token) is int and token >= 0 for token in prefix),
                 "native prefix_token_ids required")
        budget = row.get("remaining_token_budget")
        _require(type(budget) is int and 0 < budget <= MAX_ACTION_TOKENS,
                 "invalid remaining_token_budget")
        _require(len(prefix) + budget <= MAX_CONTEXT_TOKENS,
                 "request would silently shorten its saved token budget")
        seed = row.get("seed")
        _require(type(seed) is int and 0 <= seed < 2**63, "explicit nonnegative seed required")
        stages[stage] += 1
        sources[source] += 1
        bins[context_bin(len(prefix))] += 1
    for stage in _STOP_MARKERS:
        _require(stages[stage] > 0, f"missing {stage} request")
    for source in _SOURCE_KINDS:
        _require(sources[source] > 0, f"missing {source} request")
    for name in ("short", "medium", "long"):
        _require(bins[name] > 0, f"missing {name} context request")
    return {
        "request_count": len(rows),
        "stages": dict(sorted(stages.items())),
        "source_kinds": dict(sorted(sources.items())),
        "context_bins": dict(sorted(bins.items())),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_action_tokens": MAX_ACTION_TOKENS,
    }


def stage_batches(rows):
    """Keep each native request under its saved stage endpoint."""
    batches = {"think": [], "summary": []}
    for row in rows:
        stage = row["stage"]
        if stage not in batches:
            raise ValueError("unsupported request stage")
        batches[stage].append(row)
    return {stage: tuple(batch) for stage, batch in batches.items()}


def response_path(directory, request_id):
    """Keep prewarm and formal completions in distinct immutable paths."""
    if not isinstance(directory, str) or not directory or "/" in request_id or not request_id:
        raise ValueError("safe response directory and request_id required")
    return f"{directory}/{request_id}.json"


"""Opt-in inference observations. No retrieval, credit, optimizer or downloads."""
from dataclasses import asdict
import math
import os
from pathlib import Path
import subprocess
import threading
import time

from .artifacts import atomic_json
from .inference_timing import request_timing,distribution,NOT_MEASURED


class Observations:
    """Sample native engine gauges and assigned-device NVML; peaks are sampled."""
    def __init__(self,engine=None):
        import torch
        self.engine=engine;self.samples=[];self.errors=[];self.stop=threading.Event()
        self.device_uuid=str(torch.cuda.get_device_properties(0).uuid)
        if not self.device_uuid.startswith('GPU-'):self.device_uuid='GPU-'+self.device_uuid
        self.thread=threading.Thread(target=self._poll,daemon=True)
    def snapshot(self):
        if self.engine is None:return []
        return [asdict(m) for m in self.engine.llm.get_metrics()]
    def _poll(self):
        while not self.stop.is_set():
            row=dict(monotonic=time.monotonic())
            try:
                rows=self.snapshot()
                row['gauges']={m['name']:m['value'] for m in rows if 'value' in m and
                    any(k in m['name'] for k in ('num_requests_running','num_requests_waiting','kv_cache_usage','num_preemptions'))}
            except Exception as e:self.errors.append(type(e).__name__)
            try:
                text=subprocess.check_output(['nvidia-smi','-i',self.device_uuid,
                    '--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=3)
                used,total,util=map(float,text.strip().split(','))
                row.update(device_used_mib=used,device_total_mib=total,gpu_util_percent=util)
            except Exception as e:self.errors.append('nvml:'+type(e).__name__)
            self.samples.append(row);self.stop.wait(.2)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(5)
    def interval(self,start,end):
        rows=[r for r in self.samples if start<=r['monotonic']<=end]
        names={n for r in rows for n in r.get('gauges',{})}
        return dict(samples=len(rows),gauge_peaks={n:max(r['gauges'].get(n,0) for r in rows) for n in names},
            observed_device_peak_mib=max((r['device_used_mib'] for r in rows if 'device_used_mib' in r),default=NOT_MEASURED),
            basis='0.2s native-gauge/NVML samples; sampled peaks, not an absolute maximum',errors=sorted(set(self.errors)))


def metric_deltas(before,after):
    def key(r):return (r['name'],tuple(sorted(r['labels'].items())))
    prior={key(r):r for r in before};result={};seen=set()
    for r in after:
        if r['name'] in seen:
            # A source-labelled counter cannot be reduced by overwriting the
            # previous label. Preserve uncertainty instead of reporting zero.
            result[r['name']]=NOT_MEASURED
            continue
        seen.add(r['name'])
        b=prior.get(key(r),{})
        if 'count' in r:
            count=r['count']-b.get('count',0);total=r['sum']-b.get('sum',0)
            result[r['name']]=dict(count=count,sum=total,mean=total/count if count else None)
        elif 'value' in r and any(x in r['name'] for x in ('prefix_cache','preempt','generation_tokens','prompt_tokens')):
            result[r['name']]=r['value']-b.get('value',0)
    return result


def run_conditions(engine,rows,output,*,cap=512,observations=None):
    """All preset requests, including EOS/invalid outputs, must be retained."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    results=[];start=time.monotonic()
    for stage in ('think','summary'):
        selected=[r for r in rows if r['stage']==stage]
        if not selected:continue
        requests=[dict(prefix=tuple(r['prefix_token_ids']),seed=r['seed'],budget=min(cap,r['remaining_token_budget'])) for r in selected]
        before=observations.snapshot() if observations else []
        tick=time.monotonic();returned=engine.sample(requests,stage,engine.version);end=time.monotonic()
        # Preserve paid native records before numerical/identity validation.
        atomic_json(output/(stage+'-native.json'),dict(request_ids=[r['request_id'] for r in selected],outputs=returned))
        if len(returned)!=len(selected):raise ValueError('omitted benchmark requests')
        timings=[];cached=[]
        for r,o in zip(selected,returned,strict=True):
            ids=o['token_ids'];lp=o['sampled_log_probs']
            if not ids or len(ids)!=len(lp) or any(not math.isfinite(v) for v in lp):
                raise ValueError('incomplete/nonfinite actual sampling evidence')
            timings.append(dict(request_id=r['request_id'],stage=stage,prefix_tokens=len(r['prefix_token_ids']),
                generated_tokens=len(ids),finish_reason=o['finish_reason'],action_mask_identity='all returned stage tokens are actions',
                **request_timing(o.get('request_metrics'),len(ids))))
            cached.append(o.get('num_cached_tokens'))
        after=observations.snapshot() if observations else []
        token_count=sum(len(o['token_ids']) for o in returned)
        row=dict(stage=stage,requests=len(selected),generated_tokens=token_count,wall_seconds=end-tick,
            tokens_per_second=token_count/(end-tick),input_tokens=sum(len(r['prefix_token_ids']) for r in selected),
            cached_tokens=sum(cached) if all(v is not None for v in cached) else NOT_MEASURED,
            request_timing={k:distribution(t[k] for t in timings) for k in ('ttft','queue','prefill','decode','tpot','core_completion')},
            output_length=distribution(len(o['token_ids']) for o in returned),
            engine_metrics_delta=metric_deltas(before,after),
            observation=observations.interval(tick,end) if observations else NOT_MEASURED,
            timing_basis='native per-request intervals overlap across requests, not exclusive GPU-kernel time')
        atomic_json(output/(stage+'-timing.json'),timings);results.append(row)
    wall=time.monotonic()-start
    summary=dict(requests=sum(r['requests'] for r in results),generated_tokens=sum(r['generated_tokens'] for r in results),
        wall_seconds=wall,tokens_per_second=sum(r['generated_tokens'] for r in results)/wall,stages=results)
    atomic_json(output/'RESULT.json',summary)
    return summary
