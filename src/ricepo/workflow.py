"""Staged RICE-PO iteration with frozen policy versions and durable evidence."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .artifacts import atomic_json, stage_receipt
from .contracts import AdvantageBatch, Anchor, EntropyProvider, Span, Trajectory
from .credit import build_advantages, select_anchors, trajectory_advantages
from .audit import operation, saved_item, stable_seed
from .diagnostics import credit_report
from .rollout import BranchRollouts


@dataclass
class IterationComponents:
    # Deployment callbacks own model lifecycle/barriers. They may not alter the
    # frozen scientific recipe or reload a newer policy during this iteration.
    rollout_batch: Callable[[int, str], Sequence[Trajectory]]
    entropy_provider: EntropyProvider | None
    collect_branches: Callable[[Trajectory, Anchor], BranchRollouts] | None
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
    if trace.final_reward != trace.steps[-1].local_reward:
        raise ValueError("counterfactual terminal reward must equal final retrieval")
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


def _collect_credit(components, grouped, originals, root, policy_version, recipe, seed, iteration):
    def grouped_baseline():
        return [trajectory_advantages(g, epsilon=recipe["normalization_epsilon"]) for g in grouped.values()]
    if recipe["method"] == "grpo":
        # Do not even read the entropy provider or branch sampler in this path.
        stage_receipt(root, "branches_complete", policy_version=policy_version,
                      artifacts={"applicability": "skipped_grpo", "branch_count": 0})
        return grouped_baseline()
    entropy, anchors, summaries = {}, [], []
    provider = components.entropy_provider
    components.phase_barrier("rollout", "entropy")
    for q, group in grouped.items():
        for t in group:
            if provider.policy_version != policy_version:
                raise ValueError("entropy provider changed policy within the iteration")
            with operation("entropy", query=q, trajectory=t.trajectory_id, input_tokens=len(t.token_ids)-1, model_passes=1):
                entropy[t.trajectory_id] = tuple(provider.token_entropies(t))
            if provider.policy_version != policy_version:
                raise ValueError("entropy provider changed policy during teacher forcing")
            for step in t.steps:
                span = step.summary_span
                summaries.append(dict(trajectory_id=t.trajectory_id, depth=step.depth, span=span,
                                      values=entropy[t.trajectory_id][span.start:span.end]))
        candidates = select_anchors(group, {t.trajectory_id:entropy[t.trajectory_id] for t in group},
                                    top_k=sum(len(t.steps) for t in group))
        k = recipe["anchors_per_query"]
        if len(candidates) < k:
            raise ValueError("fewer eligible actions than configured anchor budget")
        if recipe["anchor_selection"] == "random":
            candidates = sorted(candidates, key=lambda a: stable_seed(seed, "random_anchor", iteration,
                                                                        q, a.trajectory_id, a.depth))
        elif recipe["anchor_selection"] != "entropy":
            raise ValueError("unsupported anchor selection")
        anchors.extend(candidates[:k])
    entropy_digest = atomic_json(root / "summary-entropies.json", summaries)
    anchor_digest = atomic_json(root / "anchors.json", anchors)
    stage_receipt(root, "entropy", policy_version=policy_version,
                  artifacts={"anchors.json":anchor_digest, "summary-entropies.json":entropy_digest})
    components.phase_barrier("entropy","branches")
    collections = []
    for a in anchors:
        collection = components.collect_branches(originals[a.trajectory_id], a)
        saved_item("branch_group", [iteration,a.trajectory_id,a.depth], collection)
        collections.append(collection)
    total = recipe["total_branches_including_original"]
    _validate_branch_rollouts(collections, anchors, originals, total)
    branches = [item.group for item in collections]
    digest = atomic_json(root / "counterfactual-trajectories.json", collections)
    stage_receipt(root, "branches_complete", policy_version=policy_version,
                  artifacts={"counterfactual-trajectories.json":digest})
    components.phase_barrier("branches","credit")
    parts = [build_advantages(group, [b for b in branches if b.anchor.query_id == q],
             expected_branch_count=total, epsilon=recipe["normalization_epsilon"],
             local_normalization=recipe["local_advantage_normalization"], gate_mode=recipe["gate_mode"],
             variance_threshold=recipe["influence_threshold"], residual_threshold=recipe["residual_threshold"])
             for q,group in grouped.items()]
    atomic_json(root / "credit-diagnostics.json", credit_report(grouped, branches, parts, recipe, iteration))
    return parts


def run_iteration(components, *, iteration, policy_version, output_root, recipe, root_seed):
    """Commit one optimizer call. Checkpoint publication and selection follow separately."""
    if recipe["method"] not in {"ricepo","grpo"} or type(iteration) is not int or iteration < 0:
        raise ValueError("invalid training method or iteration")
    root = Path(output_root) / f"iteration-{iteration:06d}"
    root.mkdir(parents=True, exist_ok=False)
    originals, grouped = {}, {}
    with operation("rollout_group", iteration=iteration):
        for trajectory in components.rollout_batch(iteration,policy_version):
            if trajectory.trajectory_id in originals or trajectory.policy_version != policy_version:
                raise ValueError("duplicate trajectory or mixed behavior policy")
            if len(trajectory.steps) != recipe["rollout_depth"] or trajectory.final_reward != trajectory.steps[-1].local_reward:
                raise ValueError("incomplete configured horizon or final reward differs from last retrieval")
            saved_item("original", trajectory.trajectory_id, trajectory)
            originals[trajectory.trajectory_id] = trajectory
            grouped.setdefault(trajectory.query_id,[]).append(trajectory)
        if len(grouped) != recipe["queries_per_update"] or any(
                len(g) != recipe["trajectories_per_query"] for g in grouped.values()):
            raise ValueError("incomplete original query group")
    trajectories = tuple(originals.values())
    digest = atomic_json(root / "trajectories.json", trajectories)
    stage_receipt(root, "rollout_complete", policy_version=policy_version, artifacts={"trajectories.json":digest})
    with operation("credit", iteration=iteration):
        parts = _collect_credit(components,grouped,originals,root,policy_version,recipe,root_seed,iteration)
        rows = {key:(adv,mask) for part in parts for key,adv,mask in zip(
            part.trajectory_ids,part.advantages,part.generated_masks,strict=True)}
        batch = AdvantageBatch(tuple(originals),tuple(rows[k][0] for k in originals),
                               tuple(rows[k][1] for k in originals),tuple(r for part in parts for r in part.records))
        digest = atomic_json(root / "advantages.json",batch)
    stage_receipt(root, "credit_complete", policy_version=policy_version, artifacts={"advantages.json":digest})
    components.phase_barrier("credit","update")
    with operation("update", iteration=iteration):
        result = components.update(trajectories,batch,policy_version,root)
    if not isinstance(result,dict) or not result.get("new_policy_version"):
        raise ValueError("update did not return a policy identity")
    digest = atomic_json(root / "update.json",result)
    stage_receipt(root, "optimizer_step_committed", policy_version=result["new_policy_version"],
                  artifacts={"update.json":digest})
    return result
