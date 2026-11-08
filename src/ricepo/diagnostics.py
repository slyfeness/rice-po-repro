"""Anchor-level credit evidence. No vocabulary distributions are serialized."""
from collections import defaultdict
import math
from .credit import trajectory_advantages


def credit_report(grouped, branches, parts, recipe, iteration):
    records = {(r.anchor.trajectory_id, r.anchor.depth): r for part in parts for r in part.records}
    baselines = {}
    for group in grouped.values():
        batch = trajectory_advantages(group, epsilon=recipe["normalization_epsilon"])
        for trajectory, adv in zip(group, batch.advantages, strict=True):
            baselines[trajectory.trajectory_id] = (trajectory, adv)
    ranks, buckets, detail = defaultdict(int), defaultdict(list), []
    import json
    for branch in branches:
        a = branch.anchor
        r = records[a.trajectory_id, a.depth]
        t, row = baselines[a.trajectory_id]
        local_ok = r.influence_variance >= recipe["influence_threshold"]
        residual_ok = r.residual_variance <= recipe["residual_threshold"]
        before = row[a.summary_span.start]
        try:
            domain, _ = json.loads(t.query_id)
        except (ValueError, TypeError):
            domain = "unspecified"
        ranks[a.query_id] += 1
        item = dict(query=a.query_id, trajectory=a.trajectory_id, domain=domain, iteration=iteration,
                    depth=a.depth, entropy=a.entropy, selected_rank=ranks[a.query_id],
                    gates={"local": local_ok, "residual": residual_ok, "joint": local_ok and residual_ok},
                    failed_conditions=[k for k, ok in (("influence", local_ok), ("residual", residual_ok)) if not ok],
                    propagated=r.propagate, gate_rule=recipe["gate_mode"],
                    influence_threshold=recipe["influence_threshold"], residual_threshold=recipe["residual_threshold"],
                    influence_variance=r.influence_variance, residual_variance=r.residual_variance,
                    trajectory_advantage=before, summary_after=r.summary_advantage,
                    reasoning_after=r.summary_advantage if r.propagate else before,
                    summary_tokens=a.summary_span.end-a.summary_span.start,
                    reasoning_tokens=a.reasoning_span.end-a.reasoning_span.start,
                    terminal_anchor=a.depth == recipe["rollout_depth"],
                    constant_original_final_rewards=len({trajectory.final_reward for trajectory in grouped[a.query_id]})==1, outcomes=[
                        dict(id=o.branch_id, original=o.is_original, local=o.local_reward, final=o.final_reward,
                             residual=o.final_reward-o.local_reward) for o in branch.outcomes])
        detail.append(item)
        buckets[domain, a.depth].append(item)
    summaries = []
    for (domain, depth), rows in sorted(buckets.items()):
        n = len(rows)
        local = [v for row in rows for v in row["outcomes"]]
        mu = math.fsum(v["local"] for v in local)/len(local)
        mf = math.fsum(v["final"] for v in local)/len(local)
        sums = {k: sum(row["gates"][k] for row in rows)/n for k in ("local","residual","joint")}
        a = [row["summary_after"] for row in rows]
        summaries.append(dict(iteration=iteration, domain=domain, depth=depth, anchors=n, gate_rates=sums,
            positive=sum(v>0 for v in a)/n, negative=sum(v<0 for v in a)/n, zero=a.count(0)/n,
            mean_abs_local_advantage=math.fsum(map(abs,a))/n,
            constant_local_groups=sum(len({v["local"] for v in row["outcomes"]})==1 for row in rows)/n,
            constant_original_final_groups=sum(row["constant_original_final_rewards"] for row in rows)/n,
            constant_branch_final_groups=sum(len({v["final"] for v in row["outcomes"]})==1 for row in rows)/n,
            sign_conflicts=sum(row["trajectory_advantage"]*row["summary_after"]<0 for row in rows)/n,
            local_final_covariance=math.fsum((v["local"]-mu)*(v["final"]-mf) for v in local)/len(local),
            terminal_anchors=sum(row["terminal_anchor"] for row in rows)))
    return {"anchors":detail, "by_iteration_domain_depth":summaries}
