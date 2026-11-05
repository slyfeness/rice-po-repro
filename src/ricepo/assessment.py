"""One query, one original retrieval, one frozen policy and one final NDCG@10."""

import json
from pathlib import Path

from .artifacts import atomic_json
from .audit import event, fingerprint, operation, stable_seed
from .dataset_bundle import is_digest
from .evaluation import summarize


METHODS = ("retrieval_only", "prompting", "grpo", "ricepo")


def _check_scope(split, scope, manifest_sha256):
    if split not in ("development", "test") or scope not in ("complete", "partial_diagnostic"):
        raise ValueError("unknown evaluation split or scope")
    if split == "development" and scope != "complete":
        raise ValueError("checkpoint selection requires complete development membership")
    if not is_digest(manifest_sha256):
        raise ValueError("evaluation requires the SHA-256 of the frozen split manifest")


def summarize_assessment(rows, expected_ids, *, split, manifest_sha256, scope="complete"):
    """Aggregate one identified evaluation stream against the precommitted split."""
    _check_scope(split, scope, manifest_sha256)
    stream, domains = {}, {}

    def checked_rows():
        for row in rows:
            if not isinstance(row, dict) or row.get("split") != split or row.get("manifest_sha256") != manifest_sha256:
                raise ValueError("result row differs from the requested split or frozen manifest SHA-256")
            method, policy = row.get("method"), row.get("policy_version")
            if method not in METHODS or not isinstance(policy, str) or not policy.strip():
                raise ValueError("result row must name its method and active policy")
            identity = dict(split=split, method=method, policy_version=policy, manifest_sha256=manifest_sha256,
                            behavior_sha256=row.get("behavior_sha256"), retrieval_set_sha256=row.get("retrieval_set_sha256"))
            if not is_digest(identity["behavior_sha256"]) or not is_digest(identity["retrieval_set_sha256"]):
                raise ValueError("result row must bind its sampling behavior and retrieval set")
            if stream and stream != identity:
                raise ValueError("cannot mix methods, behavior policies, manifests or retrieval sets in one score")
            stream.update(identity)
            domain, backend = row.get("domain"), (row.get("backend_fingerprint"), row.get("backend_version"))
            if (not isinstance(domain, str) or not domain or not is_digest(backend[0])
                    or not isinstance(backend[1], str) or not backend[1]):
                raise ValueError("result row must bind its domain and actual retrieval backend")
            if domains.setdefault(domain, backend) != backend:
                raise ValueError(f"mixed retrieval backend identities within domain {domain}")
            yield row

    summary = summarize(checked_rows(), expected_ids, scope=scope)
    return {**summary, **stream, "backend_fingerprints": {d: v[0] for d, v in sorted(domains.items())},
            "backend_versions": {d: v[1] for d, v in sorted(domains.items())}}


def evaluate_queries(*, examples, expected_ids, split, phase, method, root_seed, policy_version,
                     rollout_depth, retrievers, runners, renderer, reward, output, manifest_sha256,
                     backend_fingerprints, behavior_sha256=None, scope="complete"):
    """Evaluate the supplied split. Baselines and trained methods use identical folds and seeds.

    expected_ids is the frozen set of (domain, original query ID) pairs from the data
    receipt. examples is the manifest's mapping keyed by the canonical JSON identity.
    Generative methods share step seeds across checkpoints and methods, independently
    of iteration or ordering. A query failure aborts and preserves preceding evidence.
    """
    _check_scope(split, scope, manifest_sha256)
    if method not in METHODS or not isinstance(policy_version, str) or not policy_version.strip() or not phase:
        raise ValueError("evaluation requires an explicit method, phase and policy/retriever identity")
    expected = [tuple(item) for item in expected_ids]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected evaluation membership is empty or repeated")
    observed = {}
    for key, ex in examples.items():
        identity = tuple(json.loads(key))
        if (len(identity) != 2 or identity[0] != ex.domain or ex.query_id != key
                or not all(isinstance(v, str) and v for v in identity) or identity in observed):
            raise ValueError("invalid or duplicate manifest query identity")
        observed[identity] = key
    if (not observed or not set(observed) <= set(expected)
            or scope == "complete" and set(observed) != set(expected)):
        raise ValueError("evaluation membership differs from the required frozen query identities")
    if (not isinstance(backend_fingerprints, dict) or set(backend_fingerprints) != set(retrievers)
            or not {d for d, _ in observed} <= set(backend_fingerprints)
            or any(not is_digest(v) for v in backend_fingerprints.values())):
        raise ValueError("evaluation requires a content fingerprint for every bound retrieval backend")
    retrieval_set_sha256 = fingerprint(backend_fingerprints)
    if behavior_sha256 is None:
        behavior_sha256 = fingerprint(dict(policy_version=policy_version, root_seed=root_seed, rollout_depth=rollout_depth))
    if not is_digest(behavior_sha256):
        raise ValueError("invalid evaluation behavior identity")
    generate = method != "retrieval_only"
    if generate and (renderer is None or runners is None or type(rollout_depth) is not int
                     or rollout_depth < 1 or type(root_seed) is not int or not 0 <= root_seed < 2**63):
        raise ValueError("generated evaluation requires explicit runners, renderer, horizon and seed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with operation(phase, split=split, method=method, policy_version=policy_version):
        for serial, ((domain, query_id), key) in enumerate(sorted(observed.items())):
            example = examples[key]
            with operation("evaluation_query", split=split, query_id=key, method=method, phase_label=phase):
                with operation("retrieval_initial", query=example.query, excluded_ids=sorted(example.excluded_ids)):
                    initial = retrievers[domain].retrieve(example.query, excluded_ids=example.excluded_ids)
                    ranked = tuple(doc.document_id for doc in initial.documents)
                    if initial.query != example.query or example.excluded_ids.intersection(ranked):
                        raise ValueError("initial retrieval must use the original query and its fixed exclusions")
                initial_score = reward.score(key, ranked)
                seeds = ([stable_seed(root_seed, "evaluation", split, domain, query_id, step)
                          for step in range(1, rollout_depth + 1)] if generate else [])
                trace_id = f"{split}-q{serial:06d}"
                trace = {"identity": [domain, query_id], "initial": initial, "generation_seeds": seeds,
                         "method": method, "split": split, "policy_version": policy_version}
                # Save D0 before any model call, including failed first generations.
                initial_path = f"initial/{serial:06d}.json"
                artifacts = {initial_path: atomic_json(output / initial_path, trace)}
                final_score = initial_score
                if generate:
                    trajectory = runners[domain].run(
                        query_id=key, trajectory_id=trace_id, original_query=example.query,
                        initial_prefix_token_ids=renderer.initial(example.query, initial.documents),
                        seeds=seeds, old_policy_version=policy_version,initial_ndcg=initial_score)
                    zero = getattr(runners[domain],"failure_semantics","raise") == "invalid_action_terminal_zero_v1"
                    if (trajectory.policy_version != policy_version or trajectory.query_id != key
                            or not zero and len(trajectory.steps) != rollout_depth):
                        raise ValueError("evaluation trajectory differs from the frozen policy, query or horizon")
                    if zero:
                        from .failure_zero import validate_trajectory
                        validate_trajectory(trajectory,rollout_depth)
                    ranked = (tuple(doc.document_id for doc in trajectory.steps[-1].retrieval.documents)
                              if trajectory.steps else ranked)
                    final_score = trajectory.final_reward if zero else reward.score(key, ranked)
                    if not zero and trajectory.final_reward != final_score:
                        raise ValueError("last ranking's NDCG disagrees with recorded trajectory reward")
                    trace_path = f"trajectories/{serial:06d}.json"
                    artifacts[trace_path] = atomic_json(output / trace_path, trajectory)
                row = {"domain": domain, "query_id": query_id, "split": split, "method": method,
                       "policy_version": policy_version, "backend_version": initial.backend_version,
                       "manifest_sha256": manifest_sha256, "backend_fingerprint": backend_fingerprints[domain],
                       "behavior_sha256": behavior_sha256, "retrieval_set_sha256": retrieval_set_sha256,
                       "initial_ndcg_at_10": initial_score, "ndcg_at_10": final_score,
                       "ranked_ids": ranked, "generation_seeds": seeds, "artifact_sha256": artifacts}
                if generate and zero:
                    row.update(failure_semantics="invalid_action_terminal_zero_v1",
                        failed=trajectory.failure is not None,failure_adjusted_task_score=final_score,
                        last_valid_ndcg=trajectory.last_valid_ndcg,
                        failure_terminal_reward=trajectory.failure_terminal_reward,
                        ndcg_at_10=None if trajectory.failure else final_score)
                rows.append(row)
                atomic_json(output / "queries" / f"{serial:06d}.json", row)
                event("evaluation_query_scored", ndcg_at_10=row["ndcg_at_10"],
                      failure_adjusted_task_score=row.get("failure_adjusted_task_score"))
    if generate and zero:
        aggregate = summarize_assessment([{**r,"ndcg_at_10":r["failure_adjusted_task_score"]} for r in rows],
            expected,split=split,manifest_sha256=manifest_sha256,scope=scope)
        aggregate.update(metric="failure_adjusted_task_score",
            failure_semantics="invalid_action_terminal_zero_v1",
            failed_queries=sum(r["failed"] for r in rows),
            legitimate_complete_queries=sum(not r["failed"] for r in rows),
            legitimate_complete_fraction=sum(not r["failed"] for r in rows)/len(rows),
            legitimate_final_ndcg=[r["ndcg_at_10"] for r in rows if not r["failed"]],
            d0_macro=sum(sum(r["initial_ndcg_at_10"] for r in rows if r["domain"]==d)/
                sum(r["domain"]==d for r in rows) for d in {r["domain"] for r in rows})/
                len({r["domain"] for r in rows}))
    else:
        aggregate = summarize_assessment(rows, expected, split=split, manifest_sha256=manifest_sha256, scope=scope)
    atomic_json(output / "results-by-query.json", rows)
    atomic_json(output / "summary.json", aggregate)
    return aggregate
