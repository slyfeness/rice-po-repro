"""Atomic stage receipts and append-only, versioned experiment artifacts."""

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(type(value).__name__)


def atomic_json(path: str | Path, value: object) -> str:
    """Publish a new JSON artifact without replacing any existing result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    payload = (json.dumps(value, default=_json_default, ensure_ascii=False,
                          sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if another process won.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_receipt(root: Path, stage: str, *, policy_version: str, artifacts: dict) -> None:
    if stage not in {"rollout_complete", "entropy", "branches_complete", "credit_complete",
                     "optimizer_step_committed", "checkpoint_complete", "development_complete",
                     "iteration_complete", "evaluation"}:
        raise ValueError("unknown workflow stage")
    atomic_json(root / f"{stage}.receipt.json", {
        "schema_version": 1, "stage": stage, "policy_version": policy_version,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts,
    })


def require_cluster_storage(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    # This guard is for cluster execution, not local source authoring.
    if not resolved.is_relative_to(Path("/file_storage01/home")):
        raise ValueError("cluster data/output/cache must be under /file_storage01/home")
    return resolved


def require_slurm() -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        raise RuntimeError("cluster compute requires an authorized Slurm allocation")
    return job_id
