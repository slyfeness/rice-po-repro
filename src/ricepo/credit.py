"""RICE-PO credit assignment, arXiv:2605.26352v3, equations (1)--(9).

Rewards are fractional NDCG values in [0, 1], not the percentages in paper tables.
Local normalization is deliberately required: the proposition centers rewards,
while the implementation text does not specify whether to divide by local std.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .contracts import (
    AdvantageBatch,
    Anchor,
    BranchGroup,
    CreditRecord,
    LocalNormalization,
    Span,
    Trajectory,
)


def _reward(value: float) -> None:
    """Check the NDCG reward convention used in equations (4)--(5)."""
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("rewards must be finite fractional values in [0, 1]")


def _moments(values: Sequence[float]) -> tuple[float, float]:
    """Population moments, using the 1/(K+1) denominator in (4)--(5)."""
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance


def _normalization(local_normalization: LocalNormalization, epsilon: float) -> None:
    """Validate the explicit convention needed to instantiate (8)--(9)."""
    if local_normalization not in ("center_only", "center_std"):
        raise ValueError("local_normalization must be center_only or center_std")
    if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive")


def _thresholds(variance_threshold: float, residual_threshold: float) -> None:
    """Validate variance thresholds in the inclusive propagation gate (6)."""
    for threshold in (variance_threshold, residual_threshold):
        if isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("variance thresholds must be finite and nonnegative")


def _span(span: Span, size: int, *, nonempty: bool = False) -> None:
    """Check the half-open token spans in (8)."""
    if (
        type(span.start) is not int
        or type(span.end) is not int
        or not 0 <= span.start <= span.end <= size
        or (nonempty and span.start == span.end)
    ):
        raise ValueError("span must contain valid half-open native token offsets")


def _validate_trajectories(trajectories: Sequence[Trajectory]) -> None:
    """Validate one query's policy group and alignment before (1), (2), or (8)."""
    if not trajectories:
        return
    query_id = trajectories[0].query_id
    policy_version = trajectories[0].policy_version
    if not query_id or not policy_version:
        raise ValueError("query_id and policy_version must be nonempty")
    ids: set[str] = set()
    for trajectory in trajectories:
        if trajectory.query_id != query_id or trajectory.policy_version != policy_version:
            raise ValueError("trajectories must share one query and one policy version")
        if not trajectory.trajectory_id or trajectory.trajectory_id in ids:
            raise ValueError("trajectory IDs must be nonempty and unique")
        ids.add(trajectory.trajectory_id)
        _reward(trajectory.final_reward)
        size = len(trajectory.token_ids)
        if len(trajectory.generated_mask) != size:
            raise ValueError("generated mask and native token IDs must have equal lengths")
        if any(type(flag) is not bool for flag in trajectory.generated_mask):
            raise ValueError("generated_mask must contain booleans")
        if any(type(token) is not int or token < 0 for token in trajectory.token_ids):
            raise ValueError("native token IDs must be nonnegative integers")
        depths: set[int] = set()
        previous_end = 0
        expected_mask = [False] * size
        for step in trajectory.steps:
            if type(step.depth) is not int or step.depth < 0 or step.depth in depths:
                raise ValueError("step depths must be nonnegative unique integers")
            depths.add(step.depth)
            _reward(step.local_reward)
            prefix_size = len(step.prefix_token_ids)
            generation = step.generation
            end = prefix_size + len(generation.token_ids)
            if prefix_size < previous_end:
                raise ValueError("step generation and observation ranges must not overlap")
            if trajectory.token_ids[:prefix_size] != step.prefix_token_ids:
                raise ValueError("step prefix must match its native trajectory history")
            if trajectory.token_ids[prefix_size:end] != generation.token_ids:
                raise ValueError("step generation must match its native trajectory tokens")
            _span(generation.reasoning_span, len(generation.token_ids))
            _span(generation.summary_span, len(generation.token_ids), nonempty=True)
            _span(step.reasoning_span, size)
            _span(step.summary_span, size, nonempty=True)
            if generation.reasoning_span.end > generation.summary_span.start:
                raise ValueError("reasoning must precede summary without overlapping it")
            for relative, absolute in (
                (generation.reasoning_span, step.reasoning_span),
                (generation.summary_span, step.summary_span),
            ):
                if absolute != Span(prefix_size + relative.start, prefix_size + relative.end):
                    raise ValueError("step spans must be shifted native generation spans")
            if not all(trajectory.generated_mask[prefix_size:end]):
                raise ValueError("every generated token must be marked as generated")
            expected_mask[prefix_size:end] = [True] * len(generation.token_ids)
            observation = step.retrieval.observation_token_ids
            observation_end = end + len(observation)
            if trajectory.token_ids[end:observation_end] != observation:
                raise ValueError("tool observation must follow the native generation")
            if any(trajectory.generated_mask[end:observation_end]):
                raise ValueError("tool observation tokens must be excluded from the loss")
            previous_end = observation_end
        if tuple(expected_mask) != trajectory.generated_mask:
            raise ValueError("generated_mask must mark exactly the recorded native generations")


def select_anchors(
    trajectories: Sequence[Trajectory],
    entropy_by_trajectory: Mapping[str, Sequence[float]],
    top_k: int = 4,
) -> tuple[Anchor, ...]:
    """Pool all steps of one query and select top-K mean summary entropy (1)--(2).

    Each entropy array has the length of its native trajectory. Entry u must be
    full-vocabulary Shannon entropy conditioned on history before token u; it
    must not be the surprisal of the sampled token. Non-summary entries are
    unused. Ties preserve input trajectory/step order. Fewer than K candidates
    returns all candidates, and K=0 explicitly disables anchors.
    """
    _validate_trajectories(trajectories)
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    ids = {trajectory.trajectory_id for trajectory in trajectories}
    if set(entropy_by_trajectory) != ids:
        raise ValueError("entropy mapping must contain exactly the trajectory IDs")
    anchors: list[Anchor] = []
    for trajectory in trajectories:
        entropies = entropy_by_trajectory[trajectory.trajectory_id]
        if len(entropies) != len(trajectory.token_ids):
            raise ValueError("entropy entries must align with native trajectory tokens")
        for step in trajectory.steps:
            scores = entropies[step.summary_span.start : step.summary_span.end]
            if any(not math.isfinite(value) or value < 0 for value in scores):
                raise ValueError("summary entropy must be finite and nonnegative")
            anchors.append(
                Anchor(
                    trajectory.query_id,
                    trajectory.trajectory_id,
                    step.depth,
                    math.fsum(scores) / len(scores),
                    step.reasoning_span,
                    step.summary_span,
                )
            )
    return tuple(sorted(anchors, key=lambda anchor: -anchor.entropy)[:top_k])


def trajectory_advantages(trajectories, *, epsilon):
    """Group-relative final reward for original trajectories of ONE query."""
    _validate_trajectories(trajectories)
    if not trajectories or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("a complete query group and positive finite epsilon are required")
    reward_mean, reward_variance = _moments([t.final_reward for t in trajectories])
    scale = math.sqrt(reward_variance) + epsilon
    rows = [
        [
            (trajectory.final_reward - reward_mean) / scale if generated else 0.0
            for generated in trajectory.generated_mask
        ]
        for trajectory in trajectories
    ]
    return AdvantageBatch(tuple(t.trajectory_id for t in trajectories), tuple(map(tuple, rows)),
                          tuple(t.generated_mask for t in trajectories), ())


def compute_credit(
    group: BranchGroup,
    *,
    expected_branch_count: int,
    local_normalization: LocalNormalization,
    epsilon: float,
    variance_threshold: float = 0.05,
    residual_threshold: float = 0.03,
    gate_mode: str = "paper",
) -> CreditRecord:
    """Compute branch moments (4)--(5), inclusive gate (6), and local credit.

    expected_branch_count includes the original and all new branches (K+1 in
    the paper). Appendix A.1 reports five Monte Carlo branches, but whether
    that count includes the original is ambiguous; the total remains explicit.
    center_only uses the proposition's original local reward minus local mean;
    center_std divides that result by population std + epsilon. Identical
    sampled outcomes remain valid; repeated branch identities do not.
    """
    _normalization(local_normalization, epsilon)
    if type(expected_branch_count) is not int or expected_branch_count < 2:
        raise ValueError("expected_branch_count must include original and >=1 new branch")
    if len(group.outcomes) != expected_branch_count:
        raise ValueError("branch group is incomplete or has unexpected extra outcomes")
    _thresholds(variance_threshold, residual_threshold)
    anchor = group.anchor
    if not anchor.query_id or not anchor.trajectory_id:
        raise ValueError("anchor query and trajectory IDs must be nonempty")
    if type(anchor.depth) is not int or anchor.depth < 0:
        raise ValueError("anchor depth must be a nonnegative integer")
    if not math.isfinite(anchor.entropy) or anchor.entropy < 0:
        raise ValueError("anchor entropy must be finite and nonnegative")
    _span(anchor.reasoning_span, anchor.summary_span.end)
    _span(anchor.summary_span, anchor.summary_span.end, nonempty=True)
    if anchor.reasoning_span.end > anchor.summary_span.start:
        raise ValueError("anchor reasoning and summary spans must not overlap")
    originals = [outcome for outcome in group.outcomes if outcome.is_original]
    if len(originals) != 1:
        raise ValueError("a branch group must contain exactly one original outcome")
    original = originals[0]
    if original.branch_id != anchor.trajectory_id:
        raise ValueError("original branch ID must identify the anchor's source trajectory")
    if not original.policy_version:
        raise ValueError("branch policy_version must be nonempty")
    branch_ids: set[str] = set()
    for outcome in group.outcomes:
        if type(outcome.is_original) is not bool:
            raise ValueError("is_original must be a boolean")
        if not outcome.branch_id or outcome.branch_id in branch_ids:
            raise ValueError("branch IDs must be nonempty and unique within a group")
        branch_ids.add(outcome.branch_id)
        if outcome.policy_version != original.policy_version:
            raise ValueError("all branches must use the same policy version")
        if outcome.prefix_token_ids != original.prefix_token_ids:
            raise ValueError("all branches must share the exact native prefix")
        if len(outcome.prefix_token_ids) > anchor.reasoning_span.start:
            raise ValueError("branch prefix must end before the anchor reasoning begins")
        if any(type(token) is not int or token < 0 for token in outcome.prefix_token_ids):
            raise ValueError("branch prefix must contain nonnegative native token IDs")
        _reward(outcome.local_reward)
        _reward(outcome.final_reward)
    local_mean, influence_variance = _moments([b.local_reward for b in group.outcomes])
    residual_mean, residual_variance = _moments(
        [b.final_reward - b.local_reward for b in group.outcomes]
    )
    advantage = original.local_reward - local_mean
    if local_normalization == "center_std":
        advantage /= math.sqrt(influence_variance) + epsilon
    passed_local = influence_variance >= variance_threshold
    passed_residual = residual_variance <= residual_threshold
    gates = {"paper": passed_local and passed_residual, "influence_only": passed_local,
             "residual_only": passed_residual, "always": True, "never": False}
    if gate_mode not in gates:
        raise ValueError("unsupported propagation rule")
    return CreditRecord(
        anchor=group.anchor,
        local_mean=local_mean,
        influence_variance=influence_variance,
        residual_mean=residual_mean,
        residual_variance=residual_variance,
        propagate=gates[gate_mode],
        summary_advantage=advantage,
    )


def build_advantages(
    trajectories: Sequence[Trajectory],
    groups: Sequence[BranchGroup],
    *,
    expected_branch_count: int,
    local_normalization: LocalNormalization,
    epsilon: float,
    variance_threshold: float = 0.05,
    residual_threshold: float = 0.03,
    gate_mode: str = "paper",
) -> AdvantageBatch:
    """Replace selected generated-token advantages as in (8)--(9).

    Final trajectory rewards are centered within one query, then divided by
    population std + epsilon. Population std and additive epsilon are explicit
    implementation conventions: the paper says group-normalized but does not
    give that formula. Tool/prompt tokens always receive zero and a false mask.
    No branch groups gives the ordinary trajectory-level baseline.
    """
    _normalization(local_normalization, epsilon)
    _thresholds(variance_threshold, residual_threshold)
    _validate_trajectories(trajectories)
    if type(expected_branch_count) is not int or expected_branch_count < 2:
        raise ValueError("expected_branch_count must include original and >=1 new branch")
    if not trajectories:
        if groups:
            raise ValueError("branch groups require matching trajectories")
        return AdvantageBatch((), (), (), ())
    rows = [list(row) for row in trajectory_advantages(trajectories, epsilon=epsilon).advantages]
    indices = {trajectory.trajectory_id: index for index, trajectory in enumerate(trajectories)}
    seen: set[tuple[str, int]] = set()
    records: list[CreditRecord] = []
    for group in groups:
        anchor = group.anchor
        identity = (anchor.trajectory_id, anchor.depth)
        if identity in seen:
            raise ValueError("duplicate anchor group")
        seen.add(identity)
        if anchor.trajectory_id not in indices:
            raise ValueError("anchor trajectory is missing")
        index = indices[anchor.trajectory_id]
        trajectory = trajectories[index]
        if anchor.query_id != trajectory.query_id:
            raise ValueError("anchor query does not match its trajectory")
        if not math.isfinite(anchor.entropy) or anchor.entropy < 0:
            raise ValueError("anchor entropy must be finite and nonnegative")
        steps = [step for step in trajectory.steps if step.depth == anchor.depth]
        if len(steps) != 1:
            raise ValueError("anchor must identify an existing retrieval step")
        step = steps[0]
        if (anchor.reasoning_span, anchor.summary_span) != (
            step.reasoning_span,
            step.summary_span,
        ):
            raise ValueError("anchor spans must match the source retrieval step")
        record = compute_credit(
            group,
            expected_branch_count=expected_branch_count,
            local_normalization=local_normalization,
            epsilon=epsilon,
            variance_threshold=variance_threshold,
            residual_threshold=residual_threshold, gate_mode=gate_mode,
        )
        original = next(outcome for outcome in group.outcomes if outcome.is_original)
        if (
            original.branch_id != trajectory.trajectory_id
            or original.policy_version != trajectory.policy_version
            or original.prefix_token_ids != step.prefix_token_ids
            or original.local_reward != step.local_reward
            or original.final_reward != trajectory.final_reward
        ):
            raise ValueError("original branch must reproduce its source trajectory metadata")
        for token in range(anchor.summary_span.start, anchor.summary_span.end):
            rows[index][token] = record.summary_advantage
        if record.propagate:
            for token in range(anchor.reasoning_span.start, anchor.reasoning_span.end):
                rows[index][token] = record.summary_advantage
        records.append(record)
    return AdvantageBatch(
        tuple(trajectory.trajectory_id for trajectory in trajectories),
        tuple(tuple(row) for row in rows),
        tuple(trajectory.generated_mask for trajectory in trajectories),
        tuple(records),
    )
