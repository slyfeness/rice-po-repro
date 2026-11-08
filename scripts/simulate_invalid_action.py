"""Pure design simulation. Synthetic fixture scores; never called by training."""
from dataclasses import dataclass, replace
import json


@dataclass(frozen=True)
class ObservedState:
    ranked_ids: tuple[str, ...]
    ndcg: float
    failed: bool = False
    generated_actions: tuple[tuple[int, ...], ...] = ()
    actual_tool_calls: int = 0


def transition(state, native_ids, summary, retrieve):
    if state.failed:
        raise ValueError("absorbing states admit no further sampled action")
    actions = state.generated_actions + (tuple(native_ids),)
    if summary is None:
        return replace(state, failed=True, generated_actions=actions)
    if not summary.strip():
        raise ValueError("a valid action requires a nonempty exact summary")
    ranking, measured_ndcg = retrieve(summary)
    if not 0 <= measured_ndcg <= 1:
        raise ValueError("observed NDCG must be fractional")
    return ObservedState(tuple(ranking), measured_ndcg, generated_actions=actions,
                         actual_tool_calls=state.actual_tool_calls + 1)


def simulate():
    calls = []
    def synthetic_retriever(summary):
        calls.append(summary)
        return ("fixture-relevant",), 1.0
    initial = ObservedState(("fixture-d0",), .25)
    failed_first = transition(initial, (10, 11, 2), None, synthetic_retriever)
    assert failed_first.generated_actions == ((10, 11, 2),)
    assert failed_first.ndcg == .25 and failed_first.actual_tool_calls == 0 and not calls
    valid = transition(initial, (20, 21), "fixture summary", synthetic_retriever)
    failed_later = transition(valid, (30, 31, 2), None, synthetic_retriever)
    assert failed_later.ranked_ids == valid.ranked_ids and failed_later.ndcg == 1
    assert failed_later.generated_actions == ((20, 21), (30, 31, 2))
    assert failed_later.actual_tool_calls == 1 and calls == ["fixture summary"]
    try:
        transition(failed_later, (40,), "forbidden retry", synthetic_retriever)
    except ValueError:
        pass
    else:
        raise AssertionError("absorbing failure admitted a retry")
    assert len(calls) == 1
    return dict(design_only=True, synthetic_scores=True, simulated_cases=3,
                invalid_action_has_tool_call=False, native_ids_retained=True,
                observed_reward_inherited=True, absorbing_retry_rejected=True)


if __name__ == "__main__":
    print(json.dumps(simulate(), sort_keys=True))
