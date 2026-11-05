"""Append-only attempt evidence. All model calls take place outside this module."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import hashlib
import json

from .artifacts import atomic_json

_attempt = ContextVar("ricepo_attempt", default=None)
_recovery = ContextVar("ricepo_recovery", default=False)
_request = ContextVar("ricepo_request", default=None)


class GenerationTruncated(RuntimeError):
    code = "generation_length"


def stable_seed(root_seed, stream, *identity):
    payload = json.dumps([root_seed, stream, *identity], ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def failure_kind(exc, phase):
    code = getattr(exc, "code", None)
    if code:
        return code
    message = str(exc).lower()
    if isinstance(exc, FloatingPointError) or "nonfinite" in message or "non-finite" in message:
        return "nonfinite_score"
    if "context" in message and ("exceed" in message or "truncat" in message or "limit" in message):
        return "context_limit"
    if phase.startswith("retrieval"):
        return "retrieval_failure"
    if phase == "asset_binding" or "receipt mismatch" in message:
        return "asset_mismatch"
    if "generat" in phase or type(exc).__module__.startswith("torch"):
        return "model_or_device_error"
    return "stage_failure"


def event(kind, **details):
    root = _attempt.get()
    if root is None:
        return
    atomic_json(root / "events" / f"{uuid4().hex}.json", {
        "kind": kind, "utc": datetime.now(timezone.utc).isoformat(),
        **(_request.get() or {}), **details})


@contextmanager
def attempt(run, *, recovery=False):
    root = Path(run) / "attempts" / uuid4().hex
    root.mkdir(parents=True)
    token = _attempt.set(root)
    recovery_token = _recovery.set(recovery)
    try:
        event("attempt_started", recovery=recovery)
        yield root
    except BaseException as exc:
        # Evidence already on disk survives process termination; hard kills
        # leave the started operation without a synthetic completion event.
        atomic_json(root / "failure-cost.json", cost_report(run))
        atomic_json(root / "failure.json", {
            "type": type(exc).__name__, "reason": str(exc),
            "category": getattr(exc,"audit_category",failure_kind(exc, "")),
            "operation": getattr(exc,"audit_operation",None),
            "completed_artifacts": sorted(str(p.relative_to(run)) for p in Path(run).rglob("*.json")
                                          if "events" not in p.parts and not p.name.startswith(".")),
            "event_directory": "events", "recovery": recovery})
        raise
    finally:
        _attempt.reset(token)
        _recovery.reset(recovery_token)


@contextmanager
def operation(phase, **identity):
    started = perf_counter()
    operation_id = uuid4().hex
    fields = {**(_request.get() or {}), **identity, "phase": phase, "operation_id": operation_id}
    if phase in {"branch_generation","original_generation"} and _attempt.get() is not None:
        fields.update(_register_request(fields))
    token = _request.set(fields)
    try:
        event("started")
        yield
    except BaseException as exc:
        category = failure_kind(exc, phase)
        if not hasattr(exc,"audit_category"):
            exc.audit_category = category
            exc.audit_operation = fields
        event("failed", category=category, exception=type(exc).__name__, message=str(exc),
              elapsed_seconds=perf_counter() - started)
        raise
    else:
        event("completed", elapsed_seconds=perf_counter() - started)
    finally:
        _request.reset(token)


def raw_completion(ids, reason, prefix, seed, version, preview=None):
    event("raw_completion", token_ids=tuple(ids), finish_reason=reason, seed=seed,
          policy_version=version, prefix_token_ids=tuple(prefix), prefix_sha256=fingerprint(tuple(prefix)),
          input_tokens=len(prefix), generated_tokens=len(ids), decoded_preview=preview)


def saved_item(kind, identity, item):
    root = _attempt.get()
    if root:
        file = root / "items" / kind / f"{fingerprint(identity)}.json"
        atomic_json(file, item)
        event("item_saved", item_kind=kind, item_id=identity, path=str(file.relative_to(root)))


def cost_report(run):
    """Inclusive wall time per category. Nested categories must not be summed."""
    groups = {}
    for path in Path(run).glob("attempts/*/events/*.json"):
        item = json.loads(path.read_text(encoding="utf8"))
        phase = item.get("phase")
        if not phase:
            continue
        # Preserve the enclosing split: dev/test generations are separate from training.
        key = (item.get("split") or "train") + "/" + phase
        row = groups.setdefault(key, dict(started=0, completed=0, failed=0, elapsed_seconds=0.0,
                                           input_tokens=0, submitted_input_tokens=0, scored_input_positions=0,
                                           generated_tokens=0, raw_completions=0, recovery_replays=0))
        kind = item["kind"]
        if kind in {"started", "completed", "failed"}:
            row[kind] += 1
            row["elapsed_seconds"] += item.get("elapsed_seconds", 0)
        if kind == "started":
            row["submitted_input_tokens"] += item.get("input_tokens",0)
            row["scored_input_positions"] += item.get("input_tokens",0)*item.get("model_passes",0)
            row["recovery_replays"] += int(item.get("recovery_replay",False))
        if kind == "raw_completion":
            row["input_tokens"] += item["input_tokens"]
            row["generated_tokens"] += item["generated_tokens"]
            row["raw_completions"] += 1
    return {"categories": groups, "time_basis": "inclusive host wall seconds",
            "tokens_basis": "returned native completion IDs; failed calls with no returned IDs have unknown cost"}


def logical_request_id(version, query, trajectory, depth, seed):
    parent = _request.get() or {}
    return fingerprint([version, query, trajectory, depth, seed, parent.get("split"), parent.get("iteration")])


def _register_request(fields):
    attempt_root = _attempt.get()
    key = fields["request_id"]
    target = attempt_root.parent.parent/"requests"/key/"first.json"
    identity = {k:fields[k] for k in ("seed","query_id","trajectory_id","depth","policy_version","prefix_sha256")}
    entry = {"attempt":attempt_root.name, "identity":identity}
    try:
        atomic_json(target,entry)
    except FileExistsError:
        prior = json.loads(target.read_text(encoding="utf8"))
        if not _recovery.get() or prior["identity"] != identity or prior["attempt"] == attempt_root.name:
            raise ValueError("duplicate or inconsistent generation request")
        return {"recovery_replay":True,"first_attempt":prior["attempt"],"independent_sample":False}
    return {"recovery_replay":False,"first_attempt":attempt_root.name,"independent_sample":True}
