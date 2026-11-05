"""Offline fake-component tests; creating these tests does not run an experiment."""

from dataclasses import replace

import pytest

from ricepo.contracts import Anchor, Generation, RankedDocument, RetrievalResult, Span
from ricepo.rollout import (
    BranchRequest,
    MalformedGenerationError,
    PolicyVersionError,
    RetrievalContractError,
    RolloutLengthError,
    RolloutRules,
    RolloutRunner,
)


class SeedGenerator:
    policy_version = "old-policy-7"

    def __init__(self):
        self.calls = []

    def generate(self, prefix_token_ids, *, seed):
        self.calls.append((tuple(prefix_token_ids), seed))
        # Repeated IDs are legitimate model output, not duplicate documents.
        return Generation((seed, seed, seed + 100), Span(0, 2), Span(2, 3), str(seed), "stop")


class Composer:
    def __init__(self):
        self.calls = []

    def compose(self, original_query, summary):
        self.calls.append((original_query, summary))
        return original_query + " EXPLICIT-TEST-SEPARATOR " + summary


class Retriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, *, excluded_ids):
        self.calls.append((query, excluded_ids))
        seed = query.split()[-1]
        return RetrievalResult(
            query, (RankedDocument("doc-" + seed, 0.5),), (800, int(seed), 801), "test-index"
        )


class Scorer:
    def __init__(self):
        self.calls = []

    def score(self, query_id, ranked_ids):
        self.calls.append((query_id, tuple(ranked_ids)))
        return float(ranked_ids[0].removeprefix("doc-")) / 100


def make_runner(*, stop_depth=5, generator=None, retriever=None, rules=None, excluded_ids_by_query=None):
    return RolloutRunner(
        generator=generator or SeedGenerator(),
        retriever=retriever or Retriever(),
        reward_scorer=Scorer(),
        query_composer=Composer(),
        rules=rules or RolloutRules(frozenset({"stop"}), 10, 200),
        should_stop=lambda steps: len(steps) == stop_depth,
        final_reward=lambda steps: steps[-1].local_reward,
        excluded_ids_by_query=excluded_ids_by_query,
    )


def run(runner):
    return runner.run(
        query_id="qrels-key",
        trajectory_id="original",
        original_query="original query",
        initial_prefix_token_ids=(41, 42),
        seeds=(1, 2, 3, 4, 5),
        old_policy_version="old-policy-7",
    )


def anchor_for(trajectory, depth):
    step = trajectory.steps[depth - 1]
    return Anchor(trajectory.query_id, trajectory.trajectory_id, depth, 1.2,
                  step.reasoning_span, step.summary_span)


def test_rollout_preserves_native_ids_absolute_spans_and_observation_masks():
    runner = make_runner(stop_depth=2)
    trajectory = run(runner)
    assert trajectory.token_ids == (41, 42, 1, 1, 101, 800, 1, 801, 2, 2, 102, 800, 2, 801)
    assert trajectory.generated_mask == (False, False, True, True, True, False, False, False,
                                          True, True, True, False, False, False)
    assert trajectory.steps[0].reasoning_span == Span(2, 4)
    assert trajectory.steps[1].summary_span == Span(10, 11)
    assert trajectory.steps[1].prefix_token_ids == trajectory.token_ids[:8]
    assert runner.generator.calls[1][0] == trajectory.token_ids[:8]
    assert runner.query_composer.calls == [("original query", "1"), ("original query", "2")]
    assert runner.retriever.calls[1][1] == frozenset()
    assert runner.reward_scorer.calls == [("qrels-key", ("doc-1",)), ("qrels-key", ("doc-2",))]
    assert trajectory.policy_version == "old-policy-7"
    assert trajectory.final_reward == 0.02


def test_five_step_limit_applies_even_if_caller_never_stops():
    runner = make_runner(stop_depth=99)
    trajectory = run(runner)
    assert [step.depth for step in trajectory.steps] == [1, 2, 3, 4, 5]
    assert len(runner.generator.calls) == 5
    assert trajectory.termination == "depth_limit"


def test_branch_uses_saved_prefix_original_outcome_and_independent_continuations():
    runner = make_runner()
    original = run(runner)
    anchor = anchor_for(original, 2)
    # A new stop preference cannot shorten counterfactuals below the full horizon.
    runner.should_stop = lambda steps: True
    result = runner.branch(
        original_query="original query", original=original, anchor=anchor,
        requests=(BranchRequest("new-a", (11, 12, 13, 14)), BranchRequest("new-b", (21, 22, 23, 24))),
    )
    assert len(result.group.outcomes) == 3
    assert result.group.outcomes[0].is_original
    assert result.group.outcomes[0].local_reward == original.steps[1].local_reward
    assert result.group.outcomes[0].final_reward == original.final_reward
    saved_prefix = original.steps[1].prefix_token_ids
    assert runner.generator.calls[5] == (saved_prefix, 11)
    assert runner.generator.calls[9] == (saved_prefix, 21)
    assert all(outcome.prefix_token_ids == saved_prefix for outcome in result.group.outcomes)
    assert all(len(branch.steps) == 5 for branch in result.trajectories)
    assert result.trajectories[0].steps[0] is original.steps[0]
    assert result.trajectories[1].steps[1].generation.token_ids == (21, 21, 121)
    assert runner.retriever.calls[5][1] == frozenset()
    assert runner.retriever.calls[9][1] == frozenset()
    assert result.trajectories[0].final_reward == 0.14
    assert result.trajectories[1].final_reward == 0.24
    assert original.final_reward == 0.05


def test_branch_at_last_depth_generates_one_new_step_only():
    runner = make_runner()
    original = run(runner)
    result = runner.branch(
        original_query="original query", original=original, anchor=anchor_for(original, 5),
        requests=(BranchRequest("last-step-branch", (18,)),),
    )
    assert len(runner.generator.calls) == 6
    assert result.trajectories[0].steps[:4] == original.steps[:4]
    assert result.group.outcomes[1].local_reward == result.group.outcomes[1].final_reward == 0.18


def test_branch_rejects_duplicate_seed_streams_and_foreign_anchor():
    runner = make_runner()
    original = run(runner)
    anchor = anchor_for(original, 5)
    with pytest.raises(ValueError, match="distinct per-step seeds"):
        runner.branch(original_query="original query", original=original, anchor=anchor,
                      requests=(BranchRequest("a", (9,)), BranchRequest("b", (9,))))
    with pytest.raises(ValueError, match="does not identify"):
        runner.branch(original_query="original query", original=original,
                      anchor=replace(anchor, query_id="different-qrels"),
                      requests=(BranchRequest("a", (9,)),))


def test_branch_rejects_truncated_saved_prefix_instead_of_losing_tool_history():
    runner = make_runner()
    original = run(runner)
    shortened = replace(original.steps[1], prefix_token_ids=original.token_ids[:2])
    corrupted = replace(original, steps=(original.steps[0], shortened) + original.steps[2:])
    with pytest.raises(ValueError, match="complete saved history"):
        runner.branch(original_query="original query", original=corrupted,
                      anchor=anchor_for(corrupted, 2), requests=(BranchRequest("a", (9, 10, 11, 12)),))


def test_credit_branch_rejects_early_stopped_original_instead_of_mixing_horizons():
    runner = make_runner(stop_depth=3)
    original = run(runner)
    with pytest.raises(ValueError, match="full 5-step horizon"):
        runner.branch(original_query="original query", original=original,
                      anchor=anchor_for(original, 2), requests=(BranchRequest("a", (9, 10, 11, 12)),))
    assert len(runner.generator.calls) == 3


def test_policy_version_must_match_before_and_after_generation():
    runner = make_runner()
    runner.generator.policy_version = "new-policy"
    with pytest.raises(PolicyVersionError):
        run(runner)
    assert runner.generator.calls == []

    class ChangingGenerator(SeedGenerator):
        def generate(self, prefix_token_ids, *, seed):
            result = super().generate(prefix_token_ids, seed=seed)
            self.policy_version = "changed-in-flight"
            return result

    with pytest.raises(PolicyVersionError):
        run(make_runner(generator=ChangingGenerator()))


@pytest.mark.parametrize("replacement", [
    {"token_ids": (False, 2, 3)},
    {"token_ids": ()},
    {"reasoning_span": Span(0, 3)},
    {"summary_span": Span(2, 4)},
    {"summary_text": ""},
    {"summary_span": Span(2, 2)},
    {"reasoning_span": Span(False, 2)},
    {"finish_reason": "length"},
])
def test_malformed_generation_raises_instead_of_inventing_failure_reward(replacement):
    class BrokenGenerator(SeedGenerator):
        def generate(self, prefix_token_ids, *, seed):
            return replace(super().generate(prefix_token_ids, seed=seed), **replacement)

    runner = make_runner(generator=BrokenGenerator())
    with pytest.raises(MalformedGenerationError):
        run(runner)
    assert runner.reward_scorer.calls == []


def test_explicit_context_limit_is_not_silently_truncated():
    runner = make_runner(rules=RolloutRules(frozenset({"stop"}), 10, 4))
    with pytest.raises(RolloutLengthError):
        run(runner)
    assert runner.retriever.calls == []


def test_retriever_cannot_return_duplicate_ids_in_one_ranked_response():
    class BrokenRetriever(Retriever):
        def retrieve(self, query, *, excluded_ids):
            result = super().retrieve(query, excluded_ids=excluded_ids)
            doc = result.documents[0]
            return replace(result, documents=(doc, doc))

    with pytest.raises(RetrievalContractError):
        run(make_runner(retriever=BrokenRetriever()))


def test_fixed_dataset_exclusions_allow_documents_retrieved_in_previous_steps():
    exclusions = {"qrels-key": frozenset({"dataset-excluded"})}
    runner = make_runner(excluded_ids_by_query=exclusions)
    # Mutation of the caller's mapping cannot change the runner's query exclusions.
    exclusions["qrels-key"] = frozenset({"doc-1"})
    original = runner.run(
        query_id="qrels-key", trajectory_id="original", original_query="original query",
        initial_prefix_token_ids=(41, 42), seeds=(1, 1, 1, 1, 1), old_policy_version="old-policy-7",
    )
    assert all(step.retrieval.documents[0].document_id == "doc-1" for step in original.steps)
    assert all(excluded == frozenset({"dataset-excluded"}) for _, excluded in runner.retriever.calls)
    branched = runner.branch(
        original_query="original query", original=original, anchor=anchor_for(original, 5),
        requests=(BranchRequest("new", (1,)),),
    )
    assert branched.trajectories[0].steps[-1].retrieval.documents[0].document_id == "doc-1"
    assert runner.retriever.calls[-1][1] == frozenset({"dataset-excluded"})


def test_retriever_cannot_return_fixed_dataset_exclusions():
    runner = make_runner(excluded_ids_by_query={"qrels-key": frozenset({"doc-1"})})
    with pytest.raises(RetrievalContractError, match="excluded document"):
        run(runner)
