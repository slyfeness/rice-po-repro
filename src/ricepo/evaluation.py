"""BRIGHT NDCG aggregation: macro across domains, not pooled queries."""

from collections import defaultdict
import math
from typing import Iterable, Mapping


def summarize(rows: Iterable[Mapping], expected_domains) -> dict:
    groups = defaultdict(list)
    seen = set()
    expected = set(expected_domains)
    if not expected:
        raise ValueError("expected domain set must be explicit and nonempty")
    for row in rows:
        identity = (row["domain"], row["query_id"])
        if identity in seen or row["domain"] not in expected:
            raise ValueError("duplicate query result or unexpected domain")
        seen.add(identity)
        value = row["ndcg_at_10"]
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("NDCG must be a finite fraction in [0,1]")
        groups[row["domain"]].append(value)
    if set(groups) != expected:
        raise ValueError("missing domains: do not report a partial evaluation as BRIGHT macro")
    per_domain = {domain: math.fsum(values) / len(values) for domain, values in sorted(groups.items())}
    macro = math.fsum(per_domain.values()) / len(per_domain)
    return {"metric": "ndcg_cut_10", "scale": "0_to_1", "macro": macro,
            "per_domain": per_domain, "query_counts": {k: len(v) for k, v in sorted(groups.items())},
            "display_percent": {"macro": 100 * macro, "per_domain": {k: 100 * v for k, v in per_domain.items()}}}


def mechanism_summary(records) -> dict:
    records = tuple(records)
    if not records:
        return {"anchors": 0, "propagation_rate": None}
    return {"anchors": len(records),
            "propagation_rate": sum(r.propagate for r in records) / len(records),
            "mean_influence_variance": math.fsum(r.influence_variance for r in records) / len(records),
            "mean_residual_variance": math.fsum(r.residual_variance for r in records) / len(records)}
