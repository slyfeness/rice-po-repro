"""Build independent BRIGHT manifests from explicit offline source selections.

Future Slurm entry point: python -m ricepo.manifest --plan PLAN --output NEW_DIR
PLAN is a JSON object with schema_version=1, execution_enabled=true, an
explicit count_policy ("report_only" or "require_paper_counts"), reasonir_revision,
bright_revision, selection_jsonl, and domains. Each asset is {"path": absolute
local JSONL path, "sha256": expected hex digest}. Each domain provides
bright_documents_jsonl and bright_examples_jsonl. Domains selected for training
or development also provide reasonir_hq_jsonl. BRIGHT exports contain original
short documents {id, content} and examples {id, query, gold_ids, excluded_ids}.
Exported assets and the selection file must reside on cluster bulk storage.

Each selection row supplies split ("train" or "development"), domain, query_id,
and source_row_index (zero-based nonblank JSON records in reasonir_hq_jsonl).
All BRIGHT reference examples in the provided domains go into test. No RNG or
implicit sampling is used. Expected hashes identify local exports. Declared Hub
revisions alone cannot authenticate how an export was produced.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from .artifacts import atomic_json, file_sha256, require_cluster_storage, require_slurm
from .data import (
    BRIGHT_TEST_COUNTS, BRIGHT_TRAIN_COUNTS, content_sha256, join_reasonir_hq,
    project_bright_example, project_qrels, validate_split_manifests,
)

REASONIR_REVISION = "0275f825929b206d4ead23d34b4f8a50d4eddbc8"
BRIGHT_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"


def _asset(spec):
    path = Path(spec["path"])
    digest = spec["sha256"]
    if not path.is_absolute() or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("each asset requires an absolute path and expected SHA-256")
    if any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("asset SHA-256 must use lowercase hexadecimal")
    return require_cluster_storage(path), digest


def _rows(asset):
    """Hash the same byte stream used to parse rows, including blank lines."""
    path, expected = asset
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if raw.strip():
                row = json.loads(raw.decode("utf-8"))
                if not isinstance(row, dict):
                    raise ValueError(f"{path}, line {line_number}: expected JSON object")
                yield row
    if digest.hexdigest() != expected:
        raise ValueError(f"input SHA-256 mismatch: {path}")


def _join(domain, query_id, query, qrels, excluded, corpus, source):
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError(f"{domain}: expected a nonempty query ID")
    if not qrels or not set(qrels).issubset(corpus):
        raise ValueError(f"{domain}/{query_id}: absent positive or unresolved corpus ID")
    if set(qrels).intersection(excluded):
        raise ValueError(f"{domain}/{query_id}: positive document is also excluded")
    return {"domain": domain, "query_id": query_id, "query": query, "qrels": qrels,
            "excluded_ids": sorted(excluded), "source": source}


def _source(dataset, revision, asset, index, row):
    return {"dataset": dataset, "revision": revision, "export_sha256": asset[1],
            "source_row_index": index, "source_row_sha256": content_sha256(row)}


def build_manifests(plan_path: str | Path, output: str | Path) -> dict:
    """Create a new bundle only after joins, hashes, split and count checks pass."""
    require_slurm()
    plan_bytes = Path(plan_path).read_bytes()
    plan = json.loads(plan_bytes.decode("utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or plan.get("execution_enabled") is not True:
        raise ValueError("an explicitly enabled version-1 preparation plan is required")
    if (plan.get("bright_revision") != BRIGHT_REVISION
            or plan.get("reasonir_revision") != REASONIR_REVISION):
        raise ValueError("source revision differs from the pinned independent profile")
    policy = plan.get("count_policy")
    if policy not in {"report_only", "require_paper_counts"}:
        raise ValueError("set count_policy to report_only or require_paper_counts")
    target = require_cluster_storage(output)
    if target.exists():
        raise FileExistsError(target)
    domains = plan["domains"]
    if not isinstance(domains, dict) or not domains or not set(domains).issubset(BRIGHT_TEST_COUNTS):
        raise ValueError("plan requires one or more known BRIGHT domains")
    inputs = {"selection": _asset(plan["selection_jsonl"])}
    selected = {}
    seen_sources = set()
    for selection in _rows(inputs["selection"]):
        domain, split = selection["domain"], selection["split"]
        index, query_id = selection["source_row_index"], selection["query_id"]
        if split not in {"train", "development"} or domain not in domains:
            raise ValueError("invalid ReasonIR split or domain in selection")
        if type(index) is not int or index < 0:
            raise ValueError("source_row_index must be a nonnegative integer")
        key = f"{domain}/reasonir_hq"
        if key not in inputs:
            inputs[key] = _asset(domains[domain]["reasonir_hq_jsonl"])
        asset = inputs[key]
        source_id = (asset[1], index)
        if source_id in seen_sources:
            raise ValueError("a ReasonIR source row was selected more than once")
        seen_sources.add(source_id)
        selected.setdefault(asset, {})[index] = (split, domain, query_id)

    splits = {"train": [], "development": [], "test": []}
    corpora = {}
    for domain, spec in sorted(domains.items()):
        documents = _asset(spec["bright_documents_jsonl"])
        examples = _asset(spec["bright_examples_jsonl"])
        inputs[f"{domain}/documents"], inputs[f"{domain}/examples"] = documents, examples
        corpus = {}
        for row in _rows(documents):
            doc_id = row["id"]
            if not isinstance(doc_id, str) or not doc_id or doc_id in corpus:
                raise ValueError(f"{domain}: invalid or duplicate corpus document ID")
            if not isinstance(row["content"], str):
                raise ValueError(f"{domain}: BRIGHT documents require string content")
            # Only IDs are needed for joins; retain no duplicate corpus payload.
            corpus[doc_id] = ""
        if not corpus:
            raise ValueError(f"{domain}: empty document export")
        corpora[domain] = corpus
        for index, row in enumerate(_rows(examples)):
            item = project_bright_example(row, domain)
            splits["test"].append(_join(
                domain, item.query_id, item.query, project_qrels(row), item.excluded_ids,
                corpus, _source("xlangai/BRIGHT", BRIGHT_REVISION, examples, index, row)))

    for asset, requests in selected.items():
        matched = set()
        for index, row in enumerate(_rows(asset)):
            if index in requests:
                split, domain, query_id = requests[index]
                query, qrels = join_reasonir_hq(row, corpora[domain])
                splits[split].append(_join(
                    domain, query_id, query, qrels, (), corpora[domain],
                    _source("reasonir/reasonir-data", REASONIR_REVISION, asset, index, row)))
                matched.add(index)
        if matched != requests.keys():
            raise ValueError(f"ReasonIR selection indices absent from export: {sorted(requests.keys() - matched)}")
    if any(not rows for rows in splits.values()):
        raise ValueError("each of train, development, and test must contain selected rows")
    validate_split_manifests(splits)
    counts = {split: dict.fromkeys(BRIGHT_TEST_COUNTS, 0) for split in splits}
    for split, rows in splits.items():
        for row in rows:
            counts[split][row["domain"]] += 1
        rows.sort(key=lambda row: (row["domain"], row["query_id"]))
    matches = {"train": counts["train"] == BRIGHT_TRAIN_COUNTS,
               "test": counts["test"] == BRIGHT_TEST_COUNTS}
    if policy == "require_paper_counts" and not all(matches.values()):
        raise ValueError(f"domain counts differ from Table 3: {counts}")
    receipt = {"schema_version": 1, "kind": "independent_explicit_selection",
               "author_split_identity_verified": False, "count_policy": policy,
               "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
               "input_sha256": {role: asset[1] for role, asset in sorted(inputs.items())},
               "declared_revisions": {"reasonir": REASONIR_REVISION, "bright": BRIGHT_REVISION},
               "counts": counts, "paper_train_counts": BRIGHT_TRAIN_COUNTS,
               "paper_test_counts": BRIGHT_TEST_COUNTS, "paper_count_matches": matches,
               "cross_split_validation": "domain-qualified IDs and normalized query text disjoint",
               "manifest_sha256": {}}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".manifest-") as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        for split, rows in splits.items():
            path = bundle / f"{split}.jsonl"
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            receipt["manifest_sha256"][split] = file_sha256(path)
        atomic_json(bundle / "receipt.json", receipt)
        bundle.rename(target)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_manifests(args.plan, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
