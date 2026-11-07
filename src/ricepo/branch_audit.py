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
                failure_action="numerical mismatch expands to all saved branches immediately; this optimizer update stays blocked")


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


def execute_branch_checks(trajectories, selected, *, score, save_check, save_expansion):
    """Score selected native traces; numerical failure expands and blocks PPO.

    The caller keeps one frozen model for both phases. Already saved checks are
    reused. Infrastructure exceptions stop immediately; an exception during
    expansion also publishes an incomplete expansion record before propagating.
    """
    rows, chosen = tuple(trajectories), tuple(selected)
    by_id = {t.trajectory_id:t for t in rows}
    chosen_ids = [t.trajectory_id for t in chosen]
    if (len(by_id)!=len(rows) or len(set(chosen_ids))!=len(chosen_ids)
            or not set(chosen_ids)<=by_id.keys()):
        raise ValueError('branch audit identities must be unique and selected from the full group')
    checks={}
    for t in chosen:
        check=score(t)
        save_check(t,check,False)
        checks[t.trajectory_id]=check
        if check['within_bf16_tolerance']:
            continue
        reused=list(checks)
        base=dict(trigger_trajectory=t.trajectory_id,total=len(rows),
                  reused_trajectory_ids=reused,optimizer_update_permitted=False)
        save_expansion('started',dict(base,checked=len(checks),coverage_complete=False))
        try:
            for remaining in rows:
                if remaining.trajectory_id in checks:
                    continue
                expanded=score(remaining)
                save_check(remaining,expanded,True)
                checks[remaining.trajectory_id]=expanded
        except BaseException as exc:
            save_expansion('failed',dict(base,checked=len(checks),coverage_complete=False,
                error_type=type(exc).__name__,error=str(exc),
                failed_trajectory=remaining.trajectory_id,
                completed_trajectory_ids=list(checks)))
            raise
        save_expansion('complete',dict(base,checked=len(checks),coverage_complete=True,
            reused_checks=len(reused),new_expanded_checks=len(checks)-len(reused),
            failed_trajectories=[identity for identity,c in checks.items() if not c['within_bf16_tolerance']]))
        raise ValueError('branch probability mismatch; expanded full audit completed; optimizer remains blocked')
    return checks
