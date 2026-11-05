"""BRIGHT NDCG aggregation: macro across domains, not pooled queries."""

from collections import Counter, defaultdict
import math
from typing import Iterable, Mapping


def summarize(rows: Iterable[Mapping], expected_ids: Iterable[tuple[str, str]], *, scope="complete") -> dict:
    """Average only the authorized membership. Diagnostics expose their missing rows."""
    if scope not in {"complete", "partial_diagnostic"}:
        raise ValueError("scope must be complete or partial_diagnostic")
    expected = set()
    for item in expected_ids:
        if (not isinstance(item, (tuple, list)) or len(item) != 2
                or any(not isinstance(x, str) or not x.strip() for x in item)):
            raise ValueError("expected IDs must be (domain, query_id) pairs")
        if tuple(item) in expected:
            raise ValueError("duplicate expected query ID")
        expected.add(tuple(item))
    if not expected:
        raise ValueError("frozen expected query membership is empty")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row["domain"], row["query_id"])
        if identity not in expected:
            raise ValueError(f"unexpected query ID: {identity}")
        if identity in seen:
            raise ValueError(f"duplicate query result: {identity}")
        seen.add(identity)
        value = row["ndcg_at_10"]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("NDCG must be a finite fraction in [0,1]")
        groups[row["domain"]].append(value)
    missing = sorted(expected - seen)
    if missing and scope == "complete":
        raise ValueError(f"missing expected query IDs ({len(missing)}): {missing}")
    if not seen:
        raise ValueError("no query scores to summarize")
    expected_counts = dict(sorted(Counter(d for d, _ in expected).items()))
    per_domain = {domain: math.fsum(values) / len(values) for domain, values in sorted(groups.items())}
    macro = math.fsum(per_domain.values()) / len(per_domain)
    score_name = "macro" if scope == "complete" else "partial_macro"
    return {"metric": "ndcg_cut_10", "scale": "0_to_1", "scope": scope,
            "expected_total": len(expected), "evaluated_total": len(seen),
            "expected_query_counts": expected_counts,
            "query_counts": {d: len(groups[d]) for d in expected_counts},
            "missing_ids": [list(identity) for identity in missing],
            "per_domain": per_domain, score_name: macro,
            "display_percent": {score_name: 100 * macro, "per_domain": {k: 100 * v for k, v in per_domain.items()}}}


def mechanism_summary(records) -> dict:
    records = tuple(records)
    if not records:
        return {"anchors": 0, "propagation_rate": None}
    return {"anchors": len(records),
            "propagation_rate": sum(r.propagate for r in records) / len(records),
            "mean_influence_variance": math.fsum(r.influence_variance for r in records) / len(records),
            "mean_residual_variance": math.fsum(r.residual_variance for r in records) / len(records)}
