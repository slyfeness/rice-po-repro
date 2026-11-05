"""Small complete workflow fixtures; no models, jobs, or real checkpoints."""

from dataclasses import replace
import json

import pytest

from ricepo.contracts import BranchGroup, BranchOutcome, Generation, RetrievalResult, RetrievalStep, Span, Trajectory
from ricepo.rollout import BranchRollouts
from ricepo.workflow import IterationComponents, run_iteration


def trajectory(query, identity, final):
    generation = Generation((10, 11, 12), Span(0, 1), Span(1, 2), "summary", "stop")
    retrieval = RetrievalResult("query", (), (99,), "fixed-index")
    step = RetrievalStep(1, (1,), generation, retrieval, final, Span(1, 2), Span(2, 3))
    return Trajectory(query, identity, "old", (1, 10, 11, 12, 99),
                      (False, True, True, True, False), (step,), final, "test")


class Entropies:
    policy_version = "old"

    def token_entropies(self, source):
        # The undefined first-token entropy must not enter JSON artifacts.
        return (float("nan"), 1.0, 2.0, 1.0, 1.0)


def branch(source, anchor):
    new_id = source.trajectory_id + f"-depth-{anchor.depth}-new"
    alternative = replace(source, trajectory_id=new_id)
    prefix = source.steps[0].prefix_token_ids
    group = BranchGroup(anchor, (
        BranchOutcome(source.trajectory_id, True, source.final_reward, source.final_reward, "old", prefix),
        BranchOutcome(new_id, False, source.final_reward, source.final_reward, "old", prefix),
    ))
    return BranchRollouts(group, (alternative,))


def components(*, provider=None, collect=branch):
    sources = (trajectory("q1", "q1-high", 0.8), trajectory("q2", "q2-high", 0.9),
               trajectory("q1", "q1-low", 0.2), trajectory("q2", "q2-low", 0.7))
    updates, barriers = [], []

    def update(originals, batch, version, root):
        updates.append((originals, batch, version, root))
        return {"new_policy_version": "new", "checkpoint": "verified-by-test-callback"}

    result = IterationComponents(lambda iteration, version: sources, provider or Entropies(),
                                 collect, update, lambda before, after: barriers.append((before, after)))
    return result, sources, updates, barriers


def run(value, root, method="ricepo"):
    recipe = dict(method=method, queries_per_update=2, trajectories_per_query=2, rollout_depth=1,
                  anchors_per_query=2, total_branches_including_original=2, normalization_epsilon=0.1,
                  local_advantage_normalization="center_only", influence_threshold=0.05, residual_threshold=0.03,
                  gate_mode="paper", anchor_selection="entropy")
    return run_iteration(value, iteration=0, policy_version="old", output_root=root,
                         recipe=recipe, root_seed=17)


def test_multiquery_credit_is_normalized_per_query_and_merged_in_original_order(tmp_path):
    value, sources, updates, barriers = components()
    result = run(value, tmp_path)
    assert result["new_policy_version"] == "new"
    originals, batch, version, _ = updates[0]
    assert originals == sources and version == "old"
    assert batch.trajectory_ids == tuple(source.trajectory_id for source in sources)
    assert [row[1] for row in batch.advantages] == pytest.approx([0.75, 0.5, -0.75, -0.5])
    assert all(row[2] == 0 for row in batch.advantages)
    assert all(row[0] == row[-1] == 0 for row in batch.advantages)
    assert len(batch.records) == 4
    assert barriers == [("rollout", "entropy"), ("entropy", "branches"), ("branches", "credit"),
                        ("credit", "update")]
    root = tmp_path / "iteration-000000"
    traces = json.loads((root / "counterfactual-trajectories.json").read_text(encoding="utf-8"))
    assert len(traces) == 4
    assert all(collection["trajectories"][0]["token_ids"] == [1, 10, 11, 12, 99] for collection in traces)
    assert all(collection["trajectories"][0]["steps"][0]["prefix_token_ids"] == [1] for collection in traces)
    assert "NaN" not in (root / "summary-entropies.json").read_text(encoding="utf-8")
    assert json.loads((root / "optimizer_step_committed.receipt.json").read_text(encoding="utf-8"))["policy_version"] == "new"
    with pytest.raises(FileExistsError):
        run(value, tmp_path)
    assert len(updates) == 1


def test_entropy_policy_drift_stops_before_branching_and_update(tmp_path):
    class DriftingEntropies(Entropies):
        def token_entropies(self, source):
            self.policy_version = "changed"
            return super().token_entropies(source)

    value, _, updates, _ = components(provider=DriftingEntropies())
    with pytest.raises(ValueError, match="changed policy"):
        run(value, tmp_path)
    assert updates == []
    assert not (tmp_path / "iteration-000000" / "entropy.receipt.json").exists()


@pytest.mark.parametrize("mutation", [
    lambda result: replace(result, trajectories=()),
    lambda result: replace(result, trajectories=(replace(result.trajectories[0], final_reward=0.01),)),
    lambda result: replace(result, trajectories=(replace(result.trajectories[0], token_ids=(1, 10)),)),
    lambda result: replace(result, group=replace(result.group, outcomes=(
        replace(result.group.outcomes[0], branch_id="forged-original"), result.group.outcomes[1]))),
])
def test_missing_or_inconsistent_counterfactual_evidence_stops_before_update(tmp_path, mutation):
    value, _, updates, _ = components(collect=lambda source, anchor: mutation(branch(source, anchor)))
    with pytest.raises(ValueError):
        run(value, tmp_path)
    assert updates == []
    assert not (tmp_path / "iteration-000000" / "branches_complete.receipt.json").exists()


def test_grpo_never_calls_entropy_or_branches(tmp_path):
    class Forbidden:
        def __getattribute__(self,name):
            raise AssertionError("GRPO accessed entropy")
    def forbidden(*args):
        raise AssertionError("GRPO sampled a branch")
    value, _, updates, _ = components(collect=forbidden)
    value.entropy_provider = Forbidden()
    run(value,tmp_path,method="grpo")
    assert updates[0][1].records == ()
    assert not list(tmp_path.rglob("anchors.json"))


@pytest.mark.parametrize("failure", ["original", "branch", "credit", "before_step"])
def test_fail_fast_keeps_evidence_and_never_commits_incomplete_update(tmp_path, failure, monkeypatch):
    from ricepo.audit import attempt
    from ricepo import workflow
    value, sources, updates, _ = components()
    def fail(*args, **kwargs):
        raise RuntimeError("injected")
    if failure == "original":
        def partial(*args):
            yield sources[0]
            fail()
        value.rollout_batch = partial
    elif failure == "branch":
        value.collect_branches = fail
    elif failure == "credit":
        monkeypatch.setattr(workflow, "build_advantages", fail)
    else:
        value.update = fail
    with pytest.raises(RuntimeError), attempt(tmp_path):
        run(value,tmp_path)
    assert list(tmp_path.glob("attempts/*/failure.json"))
    assert list(tmp_path.glob("attempts/*/items/original/*.json"))
    assert not list(tmp_path.rglob("optimizer_step_committed.receipt.json"))
