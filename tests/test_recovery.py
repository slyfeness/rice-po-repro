"""POSIX checkpoint/recovery fixtures. Source only in this delivery."""
import json
import os
from pathlib import Path
import pytest

from ricepo.artifacts import atomic_json, file_sha256, stage_receipt
from ricepo.state import (bind_run, complete_stage, finish_development, publish_checkpoint,
                          recovery_checkpoint, selected_so_far, verify_checkpoint, writer)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="cluster filesystem protocol uses POSIX")


def payload(staging):
    actor = staging/"actor"
    actor.mkdir()
    for name, data in (("config.json","{}"),("tokenizer.json","{}"),("model.safetensors","fixture")):
        (actor/name).write_text(data,"utf8")
    (staging/"optimizer-and-rng.pt").write_bytes(b"fixture")


def setup_run(root):
    root.mkdir()
    bound = bind_run(root,{"recipe":"frozen","manifest_sha256":{"development":"abc"}},resume=False)
    directory = root/"iteration-000000"
    directory.mkdir()
    h = atomic_json(directory/"update.json",{"optimizer_steps":1,"new_policy_version":"p1"})
    stage_receipt(directory,"optimizer_step_committed",policy_version="p1",artifacts={"update.json":h})
    return directory,bound


def test_pending_checkpoint_cannot_be_selected_or_replayed(tmp_path):
    directory,bound = setup_run(tmp_path/"run")
    def interrupt(staging):
        payload(staging)
        raise RuntimeError("power loss")
    with pytest.raises(RuntimeError):
        publish_checkpoint(directory,iteration=0,policy_version="p1",binding_sha256=bound,write=interrupt)
    assert not (directory/"checkpoint").exists()
    pending, = directory.glob(".pending-checkpoint-*")
    with pytest.raises(ValueError):
        verify_checkpoint(pending,bound)
    with pytest.raises(ValueError):
        recovery_checkpoint(directory.parent,bound)


def test_development_retry_and_repeated_recovery_never_repeat_step(tmp_path):
    directory,bound = setup_run(tmp_path/"run")
    cp = publish_checkpoint(directory,iteration=0,policy_version="p1",binding_sha256=bound,write=payload)
    complete_stage(directory,"checkpoint_complete","p1",{"checkpoint/complete.json":file_sha256(cp/"complete.json")})
    step_hash = file_sha256(directory/"optimizer_step_committed.receipt.json")
    def fail(out):
        atomic_json(out/"one-query.json",{"query":"q"})
        raise RuntimeError("injected development failure")
    with pytest.raises(RuntimeError):
        finish_development(directory,"p1",due=True,output=directory/"dev-1",evaluate=fail)
    assert not (directory/"development_complete.receipt.json").exists()
    restored, metadata = recovery_checkpoint(directory.parent,bound)
    assert restored == cp and metadata["policy_version"]=="p1"
    def complete(out):
        atomic_json(out/"summary.json",{"macro":0.5,"scope":"complete","split":"development","policy_version":"p1",
                    "manifest_sha256":"abc","expected_total":1,"evaluated_total":1})
    finish_development(directory,"p1",due=True,output=directory/"dev-2",evaluate=complete)
    finish_development(directory,"p1",due=True,output=directory/"dev-3",evaluate=fail)
    assert (directory/"dev-1"/"one-query.json").exists() and not (directory/"dev-3").exists()
    assert selected_so_far(directory.parent,"best_development_macro")["checkpoint"] == str(cp)
    assert file_sha256(directory/"optimizer_step_committed.receipt.json")==step_hash
    with pytest.raises(ValueError,match="binding"):
        recovery_checkpoint(directory.parent,"different")
    (cp/"optimizer-and-rng.pt").write_bytes(b"changed")
    with pytest.raises(ValueError,match="contents"):
        recovery_checkpoint(directory.parent,bound)


def test_lock_and_binding_refuse_competing_writer_or_changed_inputs(tmp_path):
    root = tmp_path/"run"
    with writer(root,resume=False):
        bind_run(root,{"index":"A"},resume=False)
        with pytest.raises(RuntimeError,match="writer"), writer(root,resume=True):
            pytest.fail("unexpected second writer")
    with writer(root,resume=True), pytest.raises(ValueError,match="binding"):
        bind_run(root,{"index":"B"},resume=True)


def test_no_roll_back_over_newer_partial_iteration(tmp_path):
    directory,bound = setup_run(tmp_path/"run")
    publish_checkpoint(directory,iteration=0,policy_version="p1",binding_sha256=bound,write=payload)
    (directory.parent/"iteration-000001").mkdir()
    with pytest.raises(ValueError,match="partial"):
        recovery_checkpoint(directory.parent,bound)


def test_recomputed_inner_receipt_cannot_replace_committed_checkpoint(tmp_path):
    directory,bound = setup_run(tmp_path/"run")
    cp=publish_checkpoint(directory,iteration=0,policy_version="p1",binding_sha256=bound,write=payload)
    complete_stage(directory,"checkpoint_complete","p1",{"checkpoint/complete.json":file_sha256(cp/"complete.json")})
    inner = json.loads((cp/"complete.json").read_text())
    (cp/"optimizer-and-rng.pt").write_bytes(b"substitute")
    inner["files"]["optimizer-and-rng.pt"] = file_sha256(cp/"optimizer-and-rng.pt")
    (cp/"complete.json").write_text(json.dumps(inner))
    with pytest.raises(ValueError,match="externally committed"):
        verify_checkpoint(cp,bound)
