"""Validation for a fixed, token-native RICE inference benchmark manifest.

This module does not load a model or issue a generation request.  The private
Slurm driver extracts eligible requests from immutable saved action evidence,
then uses this validation gate before any candidate engine is started.
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
