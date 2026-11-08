"""Recheck 48 saved retrieval requests after a CPU-to-GPU Diver transition."""
import argparse
import json
import math
import os
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    from ricepo.artifacts import atomic_json
    from ricepo.audit import fingerprint
    from ricepo.dataset_bundle import verify_bundle
    from ricepo.environment import configure_caches
    from ricepo.hf_runtime import _domain_retriever, _manifest
    from ricepo.retrieval import DiverEncoder, ExplicitQueryComposer, PyTrecReward

    root = args.project
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    parent = root / "runs" / "two-stage-online-ricepo-174610"
    reference = root / "reports" / "query-composition-174994" / "rankings"
    deployment = read(parent / "deployment.json")
    deployment.update(retriever_device="cpu", retrieval_ranking_impl="array_minmax_v1")
    configure_caches(root / "cache" / out.name)
    frozen = verify_bundle(deployment)
    _, examples, qrels = _manifest(
        frozen.paths["development"], set(deployment["domains"]),
        expected_sha256=frozen.manifest_sha256["development"],
    )
    encoder = DiverEncoder.from_local(deployment)
    import torch
    move_start = time.monotonic()
    encoder.model.to("cuda:1")
    encoder.device = "cuda:1"
    torch.cuda.synchronize(1)
    restore_seconds = time.monotonic() - move_start
    backend, _ = _domain_retriever(
        "biology", deployment["domains"]["biology"], deployment,
        read(parent / "recipe.json")["values"], read(parent / "config.json")["model"],
        encoder, None, examples, qrels,
    )
    composer = ExplicitQueryComposer("original_and_summary", "\n")
    reward = PyTrecReward(qrels)
    candidates = []
    for row in read(root / "reports" / "query-composition-174994" / "private-cases.json"):
        query_id = row["query_id"]
        example = examples[query_id]
        for mode in ("summary_only", "original_and_summary"):
            saved = row[mode]
            request = saved["request_sha256"]
            reference_path = reference / f"{request}.json"
            if reference_path.is_file():
                query = row["summary"] if mode == "summary_only" else composer.compose(example.query, row["summary"])
                candidates.append((request, query_id, query, tuple(example.excluded_ids), saved["ndcg"], reference_path))
    unique = {request: values for request, *values in candidates}
    if len(unique) < 48:
        raise ValueError("fewer than 48 saved rank-level references")
    selected = [(request, *values) for request, values in sorted(unique.items())[:48]]
    start = time.monotonic()
    observed = []
    for request, query_id, query, excluded_ids, expected_ndcg, reference_path in selected:
        expected = read(reference_path)["documents"]
        expected_ids = [doc["document_id"] for doc in expected]
        result = backend.retrieve(query, excluded_ids=set(excluded_ids))
        observed_ids = [doc.document_id for doc in result.documents]
        ndcg = reward.score(query_id, observed_ids)
        if observed_ids != expected_ids:
            raise ValueError(f"ranking mismatch for {request}")
        if not math.isclose(ndcg, expected_ndcg, abs_tol=0.0, rel_tol=0.0):
            raise ValueError(f"NDCG mismatch for {request}")
        observed.append(dict(request_sha256=request, ranking_sha256=fingerprint(observed_ids), ndcg=ndcg))
    atomic_json(out / "RESULT.json", {
        "status": "COMPLETED",
        "requests": len(observed),
        "selection": "first 48 distinct saved rank-level request hashes in lexical order",
        "selection_sha256": fingerprint([row["request_sha256"] for row in observed]),
        "ranking_identity": "exact document ID sequence",
        "ndcg_identity": "exact",
        "elapsed_seconds": time.monotonic() - start,
        "Diver_transition": {"phase": "generation", "device": "cuda:1", "seconds": restore_seconds},
        "results": observed,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
