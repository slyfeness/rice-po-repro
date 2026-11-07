"""Predeclared deterministic stratified HF checks of non-optimized branches."""
import hashlib
import math


def branch_check_stratum(trajectory):
    """Public audit metadata for the selector's existing strata."""
    length = len(trajectory.token_ids)
    failure = getattr(trajectory, "failure", None)
    category = getattr(getattr(failure, "generation", failure), "category", type(failure).__name__) if failure else "valid"
    return ("short" if length <= 8192 else "medium" if length <= 16384 else "long",
            "summary_present" if trajectory.steps else "no_legal_summary", category)


def branch_check_plan(trajectories, *, update_index, policy="all"):
    """Persist whole-trajectory selection and per-stratum scheduled coverage."""
    rows = tuple(trajectories)
    selected = select_branch_checks(rows, update_index=update_index, policy=policy)
    selected_ids = {t.trajectory_id for t in selected}
    strata = {}
    for t in rows:
        key = branch_check_stratum(t)
        entry = strata.setdefault(key, {"total": 0, "selected": 0})
        entry["total"] += 1
        entry["selected"] += t.trajectory_id in selected_ids
    return dict(policy=policy, update_index=update_index, total=len(rows), checked=len(selected),
                selected=[t.trajectory_id for t in selected],
                strata=[dict(length=k[0], stage=k[1], failure_category=k[2], **value)
                        for k, value in sorted(strata.items())],
                failure_action="abort before optimizer; expand to all branches before any update retry")


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
        key = branch_check_stratum(t)
        strata.setdefault(key, []).append(t)
    selected = []
    for key in sorted(strata):
        ranked = sorted(strata[key], key=lambda t: hashlib.sha256(
            f"branch-audit-v1:{update_index}:{t.trajectory_id}".encode()).hexdigest())
        selected.extend(ranked[:max(1, math.ceil(.05 * len(ranked)))])
    return tuple(selected)
