"""Hand-computable fixtures for paper equations (1)--(9); no model required."""

import math
from dataclasses import replace

import pytest

from ricepo.contracts import (
    Anchor,
    BranchGroup,
    BranchOutcome,
    Generation,
    RetrievalResult,
    RetrievalStep,
    Span,
    Trajectory,
)
from ricepo.credit import build_advantages, compute_credit, select_anchors


def trajectory(identity="a", final=0.8, locals_=(1.0,)):
    tokens = (1,)
    mask = (False,)
    steps = []
    for depth, local in enumerate(locals_, start=1):
        prefix = tokens
        generation = Generation((10, 11, 12, 13, 14), Span(0, 2), Span(2, 4), "query", "stop")
        retrieval = RetrievalResult("query", (), (99,), "retriever-v1")
        offset = len(prefix)
        steps.append(
            RetrievalStep(
                depth,
                prefix,
                generation,
                retrieval,
                local,
                Span(offset, offset + 2),
                Span(offset + 2, offset + 4),
            )
        )
        tokens += generation.token_ids + retrieval.observation_token_ids
        mask += (True,) * 5 + (False,)
    return Trajectory("q", identity, "policy-v1", tokens, mask, tuple(steps), final, "max_depth")


def group_for(source, alternatives=((0.0, 0.0),)):
    step = source.steps[0]
    anchor = Anchor("q", source.trajectory_id, step.depth, 1.0, step.reasoning_span, step.summary_span)
    outcomes = [
        BranchOutcome(source.trajectory_id, True, step.local_reward, source.final_reward, source.policy_version, step.prefix_token_ids)
    ]
    outcomes.extend(
        BranchOutcome(f"new-{index}", False, local, final, source.policy_version, step.prefix_token_ids)
        for index, (local, final) in enumerate(alternatives)
    )
    return BranchGroup(anchor, tuple(outcomes))


def credit(group, **kwargs):
    return compute_credit(
        group,
        expected_branch_count=len(group.outcomes),
        local_normalization="center_only",
        epsilon=0.1,
        **kwargs,
    )


def test_anchor_selection_pools_all_steps_across_one_query():
    sources = [trajectory("a", locals_=(0.1, 0.2, 0.3)), trajectory("b", locals_=(0.4, 0.5, 0.6))]
    scores = {}
    for source, step_scores in zip(sources, ((1, 9, 3), (8, 7, 2))):
        values = [float("nan")] * len(source.token_ids)
        for step, score in zip(source.steps, step_scores):
            values[step.summary_span.start : step.summary_span.end] = (score - 0.5, score + 0.5)
        scores[source.trajectory_id] = values
    anchors = select_anchors(sources, scores)
    assert [(a.trajectory_id, a.depth, a.entropy) for a in anchors] == [
        ("a", 2, 9), ("b", 1, 8), ("b", 2, 7), ("a", 3, 3)
    ]
    assert select_anchors(sources, scores, top_k=0) == ()
    assert select_anchors([], {}) == ()


def test_summary_entropy_must_be_finite_and_native_token_aligned():
    source = trajectory()
    with pytest.raises(ValueError, match="align"):
        select_anchors([source], {"a": [1.0]})
    values = [1.0] * len(source.token_ids)
    values[source.steps[0].summary_span.start] = float("nan")
    with pytest.raises(ValueError, match="entropy"):
        select_anchors([source], {"a": values})


def test_branch_statistics_use_population_denominator_and_final_minus_local():
    source = trajectory(final=0.1, locals_=(0.0,))
    group = group_for(source, alternatives=((0.6, 0.7), (0.9, 1.0)))
    record = credit(group)
    assert record.local_mean == pytest.approx(0.5)
    assert record.influence_variance == pytest.approx(0.14)
    assert record.residual_mean == pytest.approx(0.1)
    assert record.residual_variance == pytest.approx(0.0)
    assert record.summary_advantage == pytest.approx(-0.5)
    assert record.propagate
    standardized = compute_credit(
        group, expected_branch_count=3, local_normalization="center_std", epsilon=0.1
    )
    assert standardized.summary_advantage == pytest.approx(-0.5 / (math.sqrt(0.14) + 0.1))


@pytest.mark.parametrize(
    "local, final, alternative, expected",
    [
        (0.0, 0.0, (1.0, 1.0), True),
        (0.4, 0.4, (0.6, 0.6), False),
        (0.0, 1.0, (1.0, 0.0), False),
        (0.4, 1.0, (0.6, 0.0), False),
    ],
)
def test_gate_requires_both_influence_and_residual_stability(local, final, alternative, expected):
    assert credit(group_for(trajectory(final=final, locals_=(local,)), (alternative,))).propagate is expected


def test_gate_comparisons_include_exact_threshold_boundaries():
    group = group_for(trajectory(final=0.0, locals_=(0.0,)), ((1.0, 1.0),))
    assert credit(group, variance_threshold=0.25, residual_threshold=0.0).propagate


@pytest.mark.parametrize("other_final, propagate", [(0.0, True), (1.0, False)])
def test_selected_spans_replace_baseline_and_tool_tokens_stay_masked(other_final, propagate):
    selected = trajectory("high", final=0.8, locals_=(1.0,))
    unselected = trajectory("low", final=0.2, locals_=(0.2,))
    result = build_advantages(
        [selected, unselected],
        [group_for(selected, ((0.0, other_final),))],
        expected_branch_count=2,
        local_normalization="center_only",
        epsilon=0.1,
    )
    reasoning = 0.5 if propagate else 0.75
    assert result.advantages[0] == pytest.approx((0, reasoning, reasoning, 0.5, 0.5, 0.75, 0))
    assert result.advantages[1] == pytest.approx((0, -0.75, -0.75, -0.75, -0.75, -0.75, 0))
    assert result.generated_masks == (selected.generated_mask, unselected.generated_mask)
    assert result.records[0].propagate is propagate


def test_no_anchors_and_constant_rewards_produce_finite_zero_advantages():
    sources = [trajectory("a", final=0.5), trajectory("b", final=0.5)]
    result = build_advantages(
        sources, [], expected_branch_count=2, local_normalization="center_std", epsilon=1e-8
    )
    assert result.records == ()
    assert all(value == 0 for row in result.advantages for value in row)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda g: replace(g, outcomes=g.outcomes[:1]),
        lambda g: replace(g, outcomes=(g.outcomes[0], replace(g.outcomes[1], is_original=True))),
        lambda g: replace(g, outcomes=(replace(g.outcomes[0], is_original=False), g.outcomes[1])),
        lambda g: replace(g, outcomes=(g.outcomes[0], replace(g.outcomes[1], branch_id=g.outcomes[0].branch_id))),
        lambda g: replace(g, outcomes=(g.outcomes[0], replace(g.outcomes[1], policy_version="different"))),
        lambda g: replace(g, outcomes=(g.outcomes[0], replace(g.outcomes[1], prefix_token_ids=(42,)))),
        lambda g: replace(g, outcomes=(g.outcomes[0], replace(g.outcomes[1], local_reward=50.0))),
        lambda g: replace(g, outcomes=(g.outcomes[0], replace(g.outcomes[1], final_reward=float("nan")))),
    ],
)
def test_reject_malformed_or_incomplete_branch_groups(mutation):
    with pytest.raises(ValueError):
        compute_credit(
            mutation(group_for(trajectory())),
            expected_branch_count=2,
            local_normalization="center_only",
            epsilon=0.1,
        )


def test_duplicate_sampled_rewards_are_valid_when_branch_identities_are_distinct():
    source = trajectory(final=0.5, locals_=(0.5,))
    record = credit(group_for(source, ((0.5, 0.5),)))
    assert record.influence_variance == 0
    assert record.summary_advantage == 0
    assert not record.propagate


def test_reject_cross_query_groups_duplicate_anchors_and_forged_originals():
    source = trajectory()
    with pytest.raises(ValueError, match="one query"):
        select_anchors([source, replace(trajectory("b"), query_id="q2")], {})
    group = group_for(source)
    kwargs = dict(expected_branch_count=2, local_normalization="center_only", epsilon=0.1)
    with pytest.raises(ValueError, match="duplicate anchor"):
        build_advantages([source], [group, group], **kwargs)
    forged = replace(group, outcomes=(replace(group.outcomes[0], local_reward=0.9), group.outcomes[1]))
    with pytest.raises(ValueError, match="source trajectory"):
        build_advantages([source], [forged], **kwargs)
    forged_id = replace(group, outcomes=(replace(group.outcomes[0], branch_id="foreign"), group.outcomes[1]))
    with pytest.raises(ValueError, match="source trajectory"):
        build_advantages([source], [forged_id], **kwargs)


def test_reject_tool_tokens_accidentally_included_in_generated_mask():
    source = trajectory()
    source = replace(source, generated_mask=source.generated_mask[:-1] + (True,))
    with pytest.raises(ValueError, match="tool observation"):
        build_advantages(
            [source], [], expected_branch_count=2, local_normalization="center_only", epsilon=0.1
        )


def test_reject_overlapping_or_unaligned_spans_and_duplicate_trajectory_ids():
    source = trajectory()
    bad_step = replace(source.steps[0], summary_span=Span(2, 5))
    kwargs = dict(expected_branch_count=2, local_normalization="center_only", epsilon=0.1)
    with pytest.raises(ValueError, match="shifted"):
        build_advantages([replace(source, steps=(bad_step,))], [], **kwargs)
    with pytest.raises(ValueError, match="unique"):
        build_advantages([source, source], [], **kwargs)


@pytest.mark.parametrize("gate,expected",[("paper",False),("influence_only",True),
                                         ("residual_only",False),("always",True),("never",False)])
def test_gate_ablation_keeps_anchor_and_local_credit(gate, expected):
    source = trajectory(final=0.0,locals_=(1.0,))
    record = credit(group_for(source,[(0.0,1.0)]),gate_mode=gate)
    assert record.summary_advantage == 0.5
    assert record.propagate is expected


def test_last_step_has_zero_residual_for_each_independent_outcome():
    source = trajectory(final=1.0,locals_=(1.0,))
    record = credit(group_for(source,[(0.0,0.0),(0.0,0.0)]))
    assert record.residual_mean == record.residual_variance == 0
    assert record.local_mean == pytest.approx(1/3)
