"""Explicit independently authorized zero-utility failure environment."""
from .contracts import Span

MODE = "invalid_action_terminal_zero_v1"


def validate_trajectory(trace, depth):
    if not trace.steps and trace.failure is None:
        raise ValueError("trajectory has neither a valid action nor failure evidence")
    history = trace.steps[0].prefix_token_ids if trace.steps else trace.failure.prefix_token_ids
    if not history:
        raise ValueError("terminal trajectory needs its native initial prefix")
    mask = (False,) * len(history)
    for expected, step in enumerate(trace.steps, 1):
        if step.depth != expected or step.prefix_token_ids != history:
            raise ValueError("terminal trajectory has discontinuous native history")
        gen = step.generation
        if not (0 <= gen.reasoning_span.start <= gen.reasoning_span.end <= gen.summary_span.start
                < gen.summary_span.end <= len(gen.token_ids)) or not gen.summary_text.strip():
            raise ValueError("invalid recorded valid action")
        if not 0 <= step.local_reward <= 1:
            raise ValueError("invalid local NDCG")
        for relative, absolute in ((gen.reasoning_span,step.reasoning_span),(gen.summary_span,step.summary_span)):
            if absolute != Span(len(history)+relative.start,len(history)+relative.end):
                raise ValueError("native body offsets changed")
        history += gen.token_ids + step.retrieval.observation_token_ids
        mask += (True,)*len(gen.token_ids)+(False,)*len(step.retrieval.observation_token_ids)
    if trace.failure:
        failure = trace.failure
        if (trace.termination != MODE or failure.depth != len(trace.steps)+1 or failure.depth>depth
                or failure.prefix_token_ids != history or failure.policy_version != trace.policy_version
                or trace.final_reward != 0 or trace.failure_terminal_reward != 0):
            raise ValueError("inconsistent invalid terminal evidence")
        if not failure.generation.token_ids or failure.seed < 0:
            raise ValueError("missing failure generation identity")
        history += failure.generation.token_ids
        mask += (True,)*len(failure.generation.token_ids)
        last = trace.steps[-1].local_reward if trace.steps else trace.initial_ndcg
        if last is None or trace.last_valid_ndcg != last or not 0 <= last <= 1:
            raise ValueError("last measured NDCG differs")
    elif len(trace.steps)!=depth or trace.final_reward!=trace.steps[-1].local_reward:
        raise ValueError("normal completion requires all scheduled depths and last NDCG")
    if history != trace.token_ids or mask != trace.generated_mask:
        raise ValueError("terminal native IDs or direct-loss mask changed")
    if any(type(x) is not int or x<0 for x in history) or any(type(x) is not bool for x in mask):
        raise ValueError("invalid native token or mask")


def validate_collections(collections, anchors, originals, count, depth):
    seen = set(originals)
    for collection, anchor in zip(collections,anchors,strict=True):
        source = originals[anchor.trajectory_id]
        validate_trajectory(source,depth)
        selected = source.steps[anchor.depth-1]
        if collection.group.anchor != anchor or len(collection.group.outcomes)!=count or len(collection.trajectories)!=count-1:
            raise ValueError("incomplete scheduled counterfactual group")
        original, *others = collection.group.outcomes
        if (not original.is_original or original.branch_id!=source.trajectory_id
                or original.local_reward!=selected.local_reward or original.final_reward!=source.final_reward
                or original.prefix_token_ids!=selected.prefix_token_ids
                or original.policy_version!=source.policy_version):
            raise ValueError("original outcome changed")
        for outcome, trace in zip(others,collection.trajectories,strict=True):
            validate_trajectory(trace,depth)
            first_prefix = (trace.steps[anchor.depth-1].prefix_token_ids if len(trace.steps)>=anchor.depth
                            else trace.failure.prefix_token_ids)
            local = trace.steps[anchor.depth-1].local_reward if len(trace.steps)>=anchor.depth else None
            if (trace.trajectory_id in seen or outcome.is_original or trace.query_id!=source.query_id
                    or trace.policy_version!=source.policy_version or outcome.policy_version!=source.policy_version
                    or first_prefix!=selected.prefix_token_ids or outcome.prefix_token_ids!=selected.prefix_token_ids
                    or trace.steps[:anchor.depth-1]!=source.steps[:anchor.depth-1]
                    or outcome.branch_id!=trace.trajectory_id or outcome.local_reward!=local
                    or outcome.final_reward!=trace.final_reward):
                raise ValueError("counterfactual identity, prefix, prior steps or outcome changed")
            seen.add(trace.trajectory_id)
            if any(s.retrieval.backend_version != selected.retrieval.backend_version for s in trace.steps):
                raise ValueError("counterfactual retrieval backend changed")
