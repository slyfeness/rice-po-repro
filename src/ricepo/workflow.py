"""Staged RICE-PO iteration with frozen policy versions and durable evidence."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Sequence

from .artifacts import atomic_json, stage_receipt
from .contracts import AdvantageBatch, Anchor, EntropyProvider, Span, Trajectory
from .credit import build_advantages, select_anchors
from .rollout import BranchRollouts


@dataclass
class IterationComponents:
    # Deployment callbacks own model lifecycle/barriers. They may not alter the
    # frozen scientific recipe or reload a newer policy during this iteration.
    rollout_batch: Callable[[int, str], Sequence[Trajectory]]
    entropy_provider: EntropyProvider
    collect_branches: Callable[[Trajectory, Anchor], BranchRollouts]
    update: Callable
    phase_barrier: Callable[[str, str], None]


def _validate_branch_trace(trace: Trajectory, source: Trajectory, anchor: Anchor) -> None:
    if trace.query_id != source.query_id or trace.policy_version != source.policy_version:
        raise ValueError("counterfactual trace must retain the source query and frozen policy")
    selected = next(index for index, step in enumerate(source.steps) if step.depth == anchor.depth)
    if len(trace.steps) != len(source.steps) or trace.steps[:selected] != source.steps[:selected]:
        raise ValueError("counterfactual trace must retain prior steps and the original final depth")
    if trace.steps[selected].prefix_token_ids != source.steps[selected].prefix_token_ids:
        raise ValueError("counterfactual trace does not start at the saved anchor prefix")
    history = trace.steps[0].prefix_token_ids
    mask = (False,) * len(history)
    for step, source_step in zip(trace.steps, source.steps, strict=True):
        if step.depth != source_step.depth or step.prefix_token_ids != history:
            raise ValueError("counterfactual trace has discontinuous native history")
        generation = step.generation
        for relative, absolute in ((generation.reasoning_span, step.reasoning_span),
                                   (generation.summary_span, step.summary_span)):
            if (not 0 <= relative.start <= relative.end <= len(generation.token_ids)
                    or absolute != Span(len(history) + relative.start, len(history) + relative.end)):
                raise ValueError("counterfactual spans do not align with native generated IDs")
        if step.retrieval.backend_version != source_step.retrieval.backend_version:
            raise ValueError("counterfactual retrieval backend changed from the fixed source backend")
        history += generation.token_ids + step.retrieval.observation_token_ids
        mask += (True,) * len(generation.token_ids) + (False,) * len(step.retrieval.observation_token_ids)
    if history != trace.token_ids or mask != trace.generated_mask:
        raise ValueError("counterfactual trace IDs or mask do not reproduce its saved steps")
    if any(type(token) is not int or token < 0 for token in history):
        raise ValueError("counterfactual trace contains invalid native token IDs")
    if any(type(flag) is not bool for flag in trace.generated_mask):
        raise ValueError("counterfactual generated mask must contain booleans")


def _validate_branch_rollouts(branch_rollouts, anchors, originals, total_branches):
    seen_ids = set(originals)
    for anchor, collection in zip(anchors, branch_rollouts, strict=True):
        if not isinstance(collection, BranchRollouts) or collection.group.anchor != anchor:
            raise ValueError("branch collection must retain the exact anchor and full native trajectories")
        outcomes = collection.group.outcomes
        traces = collection.trajectories
        if len(outcomes) != total_branches or len(traces) != total_branches - 1:
            raise ValueError("every new branch outcome requires exactly one persisted full trace")
        original_outcomes = [outcome for outcome in outcomes if outcome.is_original]
        alternatives = [outcome for outcome in outcomes if not outcome.is_original]
        if len(original_outcomes) != 1:
            raise ValueError("branch collection requires exactly one observed original")
        if len({outcome.branch_id for outcome in outcomes}) != len(outcomes):
            raise ValueError("branch outcome identities must be unique")
        trace_map = {trace.trajectory_id: trace for trace in traces}
        if len(trace_map) != len(traces) or set(trace_map) != {outcome.branch_id for outcome in alternatives}:
            raise ValueError("counterfactual trace identities do not match the branch outcomes")
        if seen_ids.intersection(trace_map):
            raise ValueError("counterfactual IDs must be globally unique and distinct from original IDs")
        seen_ids.update(trace_map)
        source = originals[anchor.trajectory_id]
        selected = next(step for step in source.steps if step.depth == anchor.depth)
        observed = original_outcomes[0]
        if (observed.branch_id != source.trajectory_id or observed.policy_version != source.policy_version
                or observed.prefix_token_ids != selected.prefix_token_ids
                or observed.local_reward != selected.local_reward or observed.final_reward != source.final_reward):
            raise ValueError("original branch outcome does not reproduce the source trajectory")
        for outcome in alternatives:
            trace = trace_map[outcome.branch_id]
            _validate_branch_trace(trace, source, anchor)
            local = next(step.local_reward for step in trace.steps if step.depth == anchor.depth)
            if (outcome.policy_version != trace.policy_version or outcome.prefix_token_ids != selected.prefix_token_ids
                    or outcome.local_reward != local or outcome.final_reward != trace.final_reward):
                raise ValueError("counterfactual outcome rewards or metadata disagree with its full trace")


def run_iteration(components: IterationComponents, *, iteration: int, policy_version: str,
                  output_root: Path, expected_trajectories_per_query: int,
                  total_branches: int, local_normalization: str, epsilon: float):
    if type(iteration) is not int or iteration < 0 or not isinstance(policy_version, str) or not policy_version:
        raise ValueError("iteration must be nonnegative and policy_version must be identified")
    if (type(expected_trajectories_per_query) is not int or expected_trajectories_per_query <= 0
            or type(total_branches) is not int or total_branches < 2):
        raise ValueError("explicit positive query group size and >=2 total branches are required")
    if (local_normalization not in {"center_only", "center_std"} or isinstance(epsilon, bool)
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("explicit local normalization and positive finite epsilon are required")
    root = Path(output_root) / f"iteration-{iteration:06d}"
    root.mkdir(parents=True, exist_ok=False)  # no silent replay of an update
    trajectories = tuple(components.rollout_batch(iteration, policy_version))
    if not trajectories or any(t.policy_version != policy_version for t in trajectories):
        raise ValueError("rollout batch is empty or has mixed policy versions")
    index = {t.trajectory_id: t for t in trajectories}
    if len(index) != len(trajectories):
        raise ValueError("trajectory IDs must be globally unique within the iteration")
    grouped = {}
    for trajectory in trajectories:
        grouped.setdefault(trajectory.query_id, []).append(trajectory)
    if any(len(group) != expected_trajectories_per_query for group in grouped.values()):
        raise ValueError("incomplete query group; cannot estimate advantages")
    digest = atomic_json(root / "trajectories.json", trajectories)
    stage_receipt(root, "rollout", policy_version=policy_version, artifacts={"trajectories.json": digest})
    components.phase_barrier("rollout", "entropy")
    if components.entropy_provider.policy_version != policy_version:
        raise ValueError("entropy provider has a different old policy")
    entropies = {}
    for trajectory in trajectories:
        if components.entropy_provider.policy_version != policy_version:
            raise ValueError("entropy provider changed policy within the iteration")
        entropies[trajectory.trajectory_id] = tuple(components.entropy_provider.token_entropies(trajectory))
        if components.entropy_provider.policy_version != policy_version:
            raise ValueError("entropy provider changed policy during teacher forcing")
    anchors = tuple(a for group in grouped.values() for a in select_anchors(
        group, {t.trajectory_id: entropies[t.trajectory_id] for t in group}, top_k=4))
    # Save all summary-token entropies, including candidates that were not selected.
    summary_entropies = [
        {"trajectory_id": t.trajectory_id, "depth": s.depth,
         "span": s.summary_span, "values": list(entropies[t.trajectory_id][s.summary_span.start:s.summary_span.end])}
        for t in trajectories for s in t.steps
    ]
    entropy_digest = atomic_json(root / "summary-entropies.json", summary_entropies)
    digest = atomic_json(root / "anchors.json", anchors)
    stage_receipt(root, "entropy", policy_version=policy_version,
                  artifacts={"anchors.json": digest, "summary-entropies.json": entropy_digest})
    components.phase_barrier("entropy", "branches")
    branch_rollouts = tuple(components.collect_branches(index[a.trajectory_id], a) for a in anchors)
    _validate_branch_rollouts(branch_rollouts, anchors, index, total_branches)
    branches = tuple(b.group for b in branch_rollouts)
    trace_digest = atomic_json(root / "counterfactual-trajectories.json", branch_rollouts)
    digest = atomic_json(root / "branches.json", branches)
    stage_receipt(root, "branches", policy_version=policy_version,
                  artifacts={"branches.json": digest, "counterfactual-trajectories.json": trace_digest})
    components.phase_barrier("branches", "credit")
    per_query = [build_advantages(
        group, [b for b in branches if b.anchor.query_id == query_id],
        local_normalization=local_normalization, epsilon=epsilon,
        expected_branch_count=total_branches) for query_id, group in grouped.items()]
    rows = {identity: (adv, mask) for part in per_query for identity, adv, mask in zip(
        part.trajectory_ids, part.advantages, part.generated_masks, strict=True)}
    batch = AdvantageBatch(tuple(t.trajectory_id for t in trajectories),
                           tuple(rows[t.trajectory_id][0] for t in trajectories),
                           tuple(rows[t.trajectory_id][1] for t in trajectories),
                           tuple(record for part in per_query for record in part.records))
    digest = atomic_json(root / "advantages.json", batch)
    stage_receipt(root, "credit", policy_version=policy_version, artifacts={"advantages.json": digest})
    components.phase_barrier("credit", "update")
    result = components.update(trajectories, batch, policy_version, root)
    # Callback returns the newly verified checkpoint identity, not a guessed path.
    if (not isinstance(result, dict) or not isinstance(result.get("new_policy_version"), str)
            or not result["new_policy_version"]):
        raise ValueError("update did not return a verified checkpoint identity")
    if result["new_policy_version"] == policy_version:
        raise ValueError("nonzero update did not produce a new policy version")
    digest = atomic_json(root / "update.json", result)
    stage_receipt(root, "update", policy_version=result["new_policy_version"], artifacts={"update.json": digest})
    components.phase_barrier("update", "rollout")
    return result
