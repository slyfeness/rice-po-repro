"""An authorized deterministic replay gets a cost record, not a new branch ID."""
import json
import pytest
from ricepo.audit import attempt, cost_report, logical_request_id, operation, raw_completion, fingerprint


def call():
    key = logical_request_id("policy","q","test",1,2)
    with operation("original_generation",request_id=key,seed=2,query_id="q",trajectory_id="test",
                   depth=1,policy_version="policy",input_tokens=1,prefix_sha256=fingerprint([9])):
        raw_completion([8,7],"eos",[9],2,"policy")
    return key


def test_recovery_replay_identity_and_cost_include_both_calls(tmp_path):
    with attempt(tmp_path) as first, operation("development",split="development",iteration=0):
        key = call()
    with attempt(tmp_path,recovery=True) as second, operation("development",split="development",iteration=0):
        assert call()==key
    entries=[json.loads(p.read_text()) for p in (second/"events").glob("*.json")]
    record=next(e for e in entries if e["kind"]=="raw_completion")
    assert record["recovery_replay"] and not record["independent_sample"]
    assert record["first_attempt"] == first.name
    cost=cost_report(tmp_path)["categories"]["development/original_generation"]
    assert cost["generated_tokens"]==4 and cost["recovery_replays"]==1


def test_duplicate_request_outside_recovery_stops(tmp_path):
    with attempt(tmp_path):
        call()
        with pytest.raises(ValueError,match="duplicate"):
            call()
