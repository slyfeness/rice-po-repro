"""Single writer, immutable state transitions and checkpoint-only recovery on POSIX."""
from contextlib import contextmanager
from pathlib import Path
import os
import json
from uuid import uuid4

from .artifacts import atomic_json, file_sha256, stage_receipt
from .audit import fingerprint, operation
from .asset_identity import local_model_identity


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


@contextmanager
def writer(root, *, resume):
    """Requires coherent cross-node advisory flock and same-filesystem atomic rename."""
    import fcntl
    root = Path(root)
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(root)
    else:
        root.mkdir(parents=True, exist_ok=False)
    with (root / ".writer.lock").open("ab") as fd:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another writer owns this run") from exc
        try:
            yield root
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def bind_run(root, data, *, resume):
    if resume:
        if read_json(root / "binding.json") != data:
            raise ValueError("run binding differs: code/config/split/model/retrieval or environment")
    else:
        atomic_json(root / "binding.json", data)
    return fingerprint(data)


def publish_checkpoint(root, *, iteration, policy_version, binding_sha256, write):
    final = root / "checkpoint"
    if final.exists():
        raise FileExistsError(final)
    staging = root / (".pending-checkpoint-" + uuid4().hex)
    staging.mkdir()
    # Interrupted staging directories are retained for diagnosis and never selectable.
    with operation("checkpoint_write", iteration=iteration):
        write(staging)
        if not (staging / "actor" / "config.json").is_file() or not (staging / "optimizer-and-rng.pt").is_file():
            raise ValueError("incomplete checkpoint payload")
        actor_identity = local_model_identity(staging / "actor")
        files = {str(p.relative_to(staging)): file_sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        atomic_json(staging / "complete.json", dict(iteration=iteration, policy_version=policy_version,
                   binding_sha256=binding_sha256, actor_payload_sha256=actor_identity["payload_sha256"], files=files))
        # Crash durability relies on same-filesystem directory rename and fsync.
        for item in staging.rglob("*"):
            if item.is_file():
                with item.open("rb") as handle:
                    os.fsync(handle.fileno())
        for directory in [*sorted((p for p in staging.rglob("*") if p.is_dir()),reverse=True),staging]:
            fd = os.open(directory,os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        staging.rename(final)
        fd = os.open(root,os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return final


def verify_checkpoint(path, binding_sha256, *, allow_missing_stage=False):
    path = Path(path)
    if path.name != "checkpoint" or path.is_symlink() or not path.is_dir():
        raise ValueError("only a published checkpoint directory can be loaded")
    value = read_json(path / "complete.json")
    expected = {str(p.relative_to(path)):file_sha256(p) for p in path.rglob("*")
                if p.is_file() and p != path / "complete.json"}
    if value["binding_sha256"] != binding_sha256 or value["files"] != expected:
        raise ValueError("checkpoint contents or run binding changed")
    if "optimizer-and-rng.pt" not in expected or "actor/config.json" not in expected:
        raise ValueError("missing optimizer or model")
    if local_model_identity(path / "actor")["payload_sha256"] != value.get("actor_payload_sha256"):
        raise ValueError("checkpoint actor identity differs")
    stage_path = path.parent / "checkpoint_complete.receipt.json"
    if stage_path.exists():
        stage = read_json(stage_path)
        if (stage.get("stage") != "checkpoint_complete" or stage["policy_version"] != value["policy_version"]
                or stage["artifacts"] != {"checkpoint/complete.json":file_sha256(path/"complete.json")}):
            raise ValueError("checkpoint differs from the externally committed stage receipt")
    elif not allow_missing_stage:
        raise ValueError("missing checkpoint completion stage receipt")
    return value


def complete_stage(root, name, version, artifacts):
    target = root / f"{name}.receipt.json"
    if target.exists():
        record = read_json(target)
        if record["policy_version"] != version or record["artifacts"] != artifacts:
            raise ValueError("existing stage receipt differs")
    else:
        stage_receipt(root, name, policy_version=version, artifacts=artifacts)


def recovery_checkpoint(root, binding_sha256):
    """Never rewind over even a possibly committed optimizer step."""
    dirs = sorted(root.glob("iteration-[0-9]*"))
    if not dirs or any(d.name != f"iteration-{i:06d}" for i,d in enumerate(dirs)):
        raise ValueError("recovery requires a contiguous history ending in a complete checkpoint")
    final = dirs[-1]
    if not (final / "optimizer_step_committed.receipt.json").is_file():
        raise ValueError("ambiguous partial iteration: no automatic replay")
    cp = final / "checkpoint"
    unpublished = not any((final / (stage + ".receipt.json")).exists()
                          for stage in ("development_complete","iteration_complete"))
    saved = verify_checkpoint(cp, binding_sha256, allow_missing_stage=unpublished)
    if saved["iteration"] != len(dirs)-1:
        raise ValueError("checkpoint iteration mismatch")
    step = read_json(final / "optimizer_step_committed.receipt.json")
    if step["policy_version"] != saved["policy_version"] or step["artifacts"].get("update.json") != file_sha256(final / "update.json"):
        raise ValueError("checkpoint and committed optimizer identity differ")
    return cp, saved


def development_score(root, path, version):
    value = read_json(path)
    digest = read_json(root/"binding.json")["manifest_sha256"]["development"]
    if (value.get("scope") != "complete" or value.get("split") != "development"
            or value.get("policy_version") != version or value.get("manifest_sha256") != digest
            or not value.get("expected_total") or value.get("evaluated_total") != value["expected_total"]
            or type(value.get("macro")) not in {int,float} or not 0 <= value["macro"] <= 1):
        raise ValueError("development report differs from selected policy, split or complete membership")
    return value["macro"]


def selected_so_far(root, rule):
    if rule not in {"last","best_development_macro"}:
        raise ValueError("unsupported checkpoint selection rule")
    choices = []
    initial = root / 'initial-selection.json'
    if initial.exists():
        choice=read_json(initial)
        report=Path(choice['report_path'])
        frozen=read_json(root/'binding.json').get('common_initial',{})
        if (choice.get('kind')!='frozen_initial_source' or choice.get('iteration')!=-1
                or frozen.get('report_sha256')!=choice['report_sha256']
                or frozen.get('policy_version')!=choice['policy_version'] or frozen.get('actor_path')!=choice['checkpoint']
                or file_sha256(report)!=choice['report_sha256']
                or development_score(root,report,choice['policy_version'])!=choice['development_macro']):
            raise ValueError('Initial selection identity or score changed')
        choices.append(choice)
    for path in sorted(root.glob("iteration-*/development_complete.receipt.json")):
        value = read_json(path)
        ref = value["artifacts"]
        if ref.get("skipped"):
            continue
        report = path.parent / ref["relative_report"]
        if file_sha256(report) != ref["sha256"]:
            raise ValueError("development report identity changed")
        score = development_score(root,report,value["policy_version"])
        choices.append(dict(policy_version=value["policy_version"], checkpoint=str(path.parent/"checkpoint"),
                            development_macro=score, iteration=int(path.parent.name[10:])))
    if not choices:
        return None
    return choices[-1] if rule == "last" else max(choices, key=lambda c:c["development_macro"])


def finish_development(directory, version, *, due, output, evaluate):
    """Retry only development in a fresh attempt; a completed report is immutable."""
    marker = directory/"development_complete.receipt.json"
    if marker.exists():
        record = read_json(marker)
        info = record["artifacts"]
        if record["policy_version"] != version or ("skipped" in info) == due:
            raise ValueError("development receipt disagrees with checkpoint/cadence")
        if due and file_sha256(directory/info["relative_report"]) != info["sha256"]:
            raise ValueError("completed development report changed")
        return
    if not due:
        complete_stage(directory,"development_complete",version,{"skipped":"cadence"})
        return
    evaluate(output)
    report = output/"summary.json"
    development_score(directory.parent,report,version)
    complete_stage(directory,"development_complete",version,
                   {"relative_report":str(report.relative_to(directory)),"sha256":file_sha256(report)})
