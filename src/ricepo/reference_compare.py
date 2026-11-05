"""Explicit, offline collection and comparison for a small, user-supplied retrieval case."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from .artifacts import atomic_json, require_cluster_storage, require_slurm
from .config import load_config, require_execution, resolve
from .data import content_sha256
from .preparation import MODEL_REVISION, load_vector_bundle
from .retrieval import (DenseVectorScorer, DiverEncoder, ExplicitQueryComposer, FixedHybridRetriever,
                        LuceneBM25Scorer, PyTrecReward, minmax)

DIVER_CODE_REVISION = "cfdf09ad863540f78a655880ef133c4cb28b3676"
EXACT_LAYERS = ("rendered_inputs", "token_ids", "attention_masks", "ranked_ids")
NUMERIC_LAYERS = ("document_vectors", "query_vectors", "dense_raw", "sparse_raw", "dense_normalized",
                  "sparse_normalized", "fused", "ndcg_cut_10")
LAYERS = (*EXACT_LAYERS, *NUMERIC_LAYERS)


def _record_inputs(layers, kind, keys, texts, inputs):
    for name, values in (("rendered_inputs", texts), ("token_ids", inputs["input_ids"].tolist()),
                         ("attention_masks", inputs["attention_mask"].tolist())):
        layers[name][kind].update(zip(keys, values, strict=True))


def _check_case(case):
    if not isinstance(case, dict) or case.get("schema_version") != 1:
        raise ValueError("expected schema-v1 case")
    ids = case.get("document_ids")
    if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids) or not isinstance(case.get("corpus_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", case["corpus_sha256"])):
        raise ValueError("case requires frozen corpus SHA-256 and ordered document IDs")
    queries = case.get("queries")
    if not isinstance(queries, list) or not queries or len({q["id"] for q in queries}) != len(queries):
        raise ValueError("case requires unique queries")
    for q in queries:
        qrels = q.get("qrels")
        excluded = q.get("excluded_ids")
        if (not isinstance(q["id"], str) or not q["id"] or not isinstance(q["original_query"], str)
                or not q["original_query"].strip() or not (q["summary"] is None or isinstance(q["summary"], str))
                or not isinstance(qrels, dict) or not qrels or not set(qrels) <= set(ids)
                or any(type(rel) is not int or rel < 0 for rel in qrels.values())
                or not isinstance(excluded, list) or any(not isinstance(i, str) or not i for i in excluded)
                or len(set(excluded)) != len(excluded) or not set(excluded) <= (set(ids) | {"N/A"})):
            raise ValueError("invalid query, qrels or fixed exclusions in reference case")


def collect(config, deployment, case):
    require_execution(config,"compare-retrieval")
    require_slurm()
    _check_case(case)
    domain = case["domain"]
    # Use the same effective deviations as the runtime. This case needs only
    # the resolved retrieval fields, not training budgets or split paths.
    settings = resolve(config, "compare-retrieval")["values"]
    if (settings.get("document_encoding_profile") != "original_model_card_raw_documents"
            or settings.get("hybrid_normalization") != "minmax"
            or type(settings.get("excluded_before_normalization")) is not bool
            or type(deployment["retrieval_top_k"]) is not int or deployment["retrieval_top_k"] < 10):
        raise ValueError("explicit raw document, minmax, fixed exclusions and cutoff >= 10 required")

    spec = deployment["domains"][domain]
    encoder = DiverEncoder.from_local(deployment)
    corpus, ids, vectors, audit = load_vector_bundle(domain, spec, config["model"], encoder)
    if case["document_ids"] != ids or case["corpus_sha256"] != audit["source_sha256"]["corpus_jsonl"]:
        raise ValueError("case corpus bytes or row order differs from bound vector bundle")
    layers = {name: {} for name in LAYERS}
    for key in ("rendered_inputs", "token_ids", "attention_masks"):
        layers[key] = {"documents": {}, "queries": {}}
    size = audit["encoder_batch_size"]
    for start in range(0, len(ids), size):
        batch = ids[start:start + size]
        texts = [corpus[key] for key in batch]
        inputs = encoder.tokenize(texts, is_query=False)
        _record_inputs(layers, "documents", batch, [encoder.document_prefix + s for s in texts], inputs)
        layers["document_vectors"].update(zip(batch, vectors[start:start + size].tolist(), strict=True))

    queries = case["queries"]
    reward = PyTrecReward({q["id"]: q["qrels"] for q in queries})
    composer = None
    if any(q["summary"] is not None for q in queries):
        if not isinstance(settings.get("query_composition"), dict):
            raise ValueError("summary case requires an explicit query composition in the resolved recipe")
        composer = ExplicitQueryComposer(**settings["query_composition"])

    # Capture the exact input and vector consumed by DenseVectorScorer in its
    # normal call, rather than encoding the query a second time for the audit.
    query_observation = {}
    def encode_query(text):
        inputs = encoder.tokenize([text], is_query=True)
        vector = encoder.encode_inputs(inputs)[0]
        query_observation.update(inputs=inputs, vector=vector.tolist())
        return vector

    backend = FixedHybridRetriever(dense_score=DenseVectorScorer(ids, vectors, encode_query,
                                       chunk_size=deployment["dense_chunk_size"]),
                                   sparse_score=LuceneBM25Scorer(corpus), corpus=corpus, normalization=minmax,
                                   exclude_before_normalization=settings["excluded_before_normalization"],
                                   top_k=deployment["retrieval_top_k"], render_observation=lambda docs: (),
                                   backend_version=spec["backend_version"])
    for row in queries:
        qid = row["id"]
        query = row["original_query"] if row["summary"] is None else composer.compose(row["original_query"], row["summary"])
        trace = backend.score_trace(query, excluded_ids=frozenset(row["excluded_ids"]))
        _record_inputs(layers, "queries", [qid], [encoder.QUERY_PREFIX + query], query_observation["inputs"])
        layers["query_vectors"][qid] = query_observation["vector"]
        for key in ("dense_raw", "sparse_raw", "dense_normalized", "sparse_normalized", "fused"):
            layers[key][qid] = trace[key]
        ranked_ids = [d.document_id for d in trace["ranking"]]
        layers["ranked_ids"][qid] = ranked_ids
        layers["ndcg_cut_10"][qid] = reward.score(qid, ranked_ids)
    return {"schema_version": 1, "role": "local_observed", "case": case, "case_sha256": content_sha256(case),
            "domain": domain, "bundle": audit, "encoder": encoder.identity, "recipe": backend.recipe(),
            "composition": settings.get("query_composition"), "layers": layers}


def _layer_result(local, expected, tolerance):
    counts = {"compared": 0, "mismatches": 0, "max_abs_error": 0.0, "examples": []}
    def mismatch(path, reason):
        counts["mismatches"] += 1
        if len(counts["examples"]) < 12:
            counts["examples"].append({"path": path, "reason": reason})

    def visit(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                mismatch(path, "key sets differ")
            for key in sorted(left.keys() & right.keys()):
                visit(left[key], right[key], [*path, key])
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatch(path, "lengths differ")
            for i, (x, y) in enumerate(zip(left, right)):
                visit(x, y, [*path, i])
        else:
            counts["compared"] += 1
            if tolerance is None:
                if type(left) is not type(right) or left != right:
                    mismatch(path, "exact value differs")
                return
            if not all(type(x) in {int, float} and math.isfinite(x) for x in (left, right)):
                mismatch(path, "numeric finite values required")
                return
            error = abs(left - right)
            counts["max_abs_error"] = max(counts["max_abs_error"], error)
            if error > tolerance["atol"] + tolerance["rtol"] * abs(right):
                mismatch(path, "outside absolute + relative tolerance")
    visit(local, expected, [])
    return {"status": "mismatch" if counts["mismatches"] else "matched", **counts}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(observed_file, expected_file, expected_sha256):
    """Compare to an externally pinned expectation file; never derive a target locally."""
    require_slurm()
    observed_file = require_cluster_storage(observed_file)
    expected_file = require_cluster_storage(expected_file)
    observed_bytes = observed_file.read_bytes()
    expected_bytes = expected_file.read_bytes()
    if observed_file == expected_file or hashlib.sha256(expected_bytes).hexdigest() != expected_sha256:
        raise ValueError("independently pinned expected artifact SHA-256 required")
    local, ext = json.loads(observed_bytes), json.loads(expected_bytes)
    if (local.get("schema_version") != 1 or local.get("role") != "local_observed"
            or ext.get("schema_version") != 1 or ext.get("role") != "upstream_expected"):
        raise ValueError("local observations and independent upstream expectations are distinct inputs")
    source = ext["source"]
    if (source.get("repository") != "https://github.com/AQ-MedAI/Diver"
            or source.get("revision") != DIVER_CODE_REVISION
            or source.get("retriever_revision") != MODEL_REVISION
            or any(not isinstance(source.get(k), str) or not source[k].strip()
                   for k in ("producer", "command", "encoding_profile"))):
        raise ValueError("upstream evidence requires pinned Diver revisions and explicit producer, command and profile")
    _check_case(local["case"])
    if (content_sha256(local["case"]) != local["case_sha256"]
            or local["case_sha256"] != ext.get("case_sha256")):
        raise ValueError("local and upstream cases must have identical frozen inputs")
    if not ext.get("layers") or not set(ext["layers"]) <= set(LAYERS) or set(local["layers"]) != set(LAYERS):
        raise ValueError("invalid or empty layer set")
    document_ids = set(local["case"]["document_ids"])
    query_ids = {q["id"] for q in local["case"]["queries"]}
    for key in LAYERS:
        rows = local["layers"][key]
        if key in ("rendered_inputs", "token_ids", "attention_masks"):
            if set(rows) != {"documents", "queries"} or set(rows["documents"]) != document_ids or set(rows["queries"]) != query_ids:
                raise ValueError(f"incomplete local document/query inputs in {key}")
        elif set(rows) != (document_ids if key == "document_vectors" else query_ids):
            raise ValueError(f"incomplete local case membership in {key}")
    needed = set(NUMERIC_LAYERS) & ext["layers"].keys()
    if set(ext.get("tolerances", {})) != needed:
        raise ValueError("explicit per-layer tolerances required for every supplied numeric layer")
    results = {}
    for name in LAYERS:
        if name not in ext["layers"]:
            results[name] = {"status": "missing_upstream_evidence"}
            continue
        tolerance = ext.get("tolerances", {}).get(name)
        if name in needed and (not isinstance(tolerance, dict) or set(tolerance) != {"atol", "rtol"}
                or any(type(x) not in {int, float} or not math.isfinite(x) or x < 0 for x in tolerance.values())):
            raise ValueError(f"invalid pinned tolerance for {name}")
        results[name] = _layer_result(local["layers"][name], ext["layers"][name], tolerance)
    return {"schema_version": 1, "case_sha256": local["case_sha256"], "source": source,
            "local_artifact_sha256": hashlib.sha256(observed_bytes).hexdigest(), "upstream_artifact_sha256": expected_sha256,
            "tolerances": ext.get("tolerances", {}), "layers": results,
            "all_layers_matched": all(v["status"] == "matched" for v in results.values()),
            "scope": "comparison to supplied case outputs; producer provenance is declared by the evidence provider"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ricepo.reference_compare")
    sub = parser.add_subparsers(dest="action", required=True)
    gather = sub.add_parser("collect")
    for key in ("config", "deployment", "case", "output"):
        gather.add_argument(f"--{key}", required=True)
    check = sub.add_parser("compare")
    for key in ("observed", "expected", "expected-sha256", "output"):
        check.add_argument(f"--{key}", required=True)
    args = parser.parse_args(argv)
    require_slurm()
    output = require_cluster_storage(args.output)
    if args.action == "collect":
        result = collect(load_config(args.config), _json(args.deployment), _json(require_cluster_storage(args.case)))
    else:
        result = compare(args.observed, args.expected, args.expected_sha256)
    atomic_json(output, result)
    return 1 if args.action == "compare" and not result["all_layers_matched"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
