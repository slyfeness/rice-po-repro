"""Predeclared deterministic stratified HF checks of non-optimized branches."""
import hashlib
import math


def select_branch_checks(trajectories, *, update_index, policy="all"):
    """First update is exhaustive; later 5% is rounded up within each stratum.

    Whole trajectories are checked, including every Think/Summary stage and
    the failure action, so selecting a trajectory never drops its failed tail.
    """
    rows = tuple(trajectories)
    if policy not in {"all", "stratified_five_percent_v1"}:
        raise ValueError("unknown branch probability audit policy")
    if type(update_index) is not int or update_index < 0:
        raise ValueError("explicit nonnegative update index required")
    if len({t.trajectory_id for t in rows}) != len(rows):
        raise ValueError("duplicate branch identity")
    if policy == "all" or update_index == 0:
        return rows
    strata = {}
    for t in rows:
        length = len(t.token_ids)
        failure = getattr(t, "failure", None)
        category = getattr(getattr(failure, "generation", failure), "category", type(failure).__name__) if failure else "valid"
        stage = "summary_present" if t.steps else "no_legal_summary"
        key = ("short" if length <= 8192 else "medium" if length <= 16384 else "long", stage, category)
        strata.setdefault(key, []).append(t)
    selected = []
    for key in sorted(strata):
        ranked = sorted(strata[key], key=lambda t: hashlib.sha256(
            f"branch-audit-v1:{update_index}:{t.trajectory_id}".encode()).hexdigest())
        selected.extend(ranked[:max(1, math.ceil(.05 * len(ranked)))])
    return tuple(selected)
