"""Freeze and verify all three split identities without exposing held-out judgments."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .artifacts import file_sha256, require_cluster_storage, require_slurm
from .data import BRIGHT_REVISION, BRIGHT_TEST_COUNTS, BRIGHT_TRAIN_COUNTS, REASONIR_REVISION, audit_identities


SPLITS = ("train", "development", "test")
REVISIONS = {"bright": BRIGHT_REVISION, "reasonir": REASONIR_REVISION}
SOURCE_RULES = {
    "train_and_development": {
        "dataset": "reasonir/reasonir-data", "query_field": "query[1]", "id_field": "selection.query_id",
        "positives": "pos[*][1] joined by exact ID to same-domain BRIGHT documents.id",
        "excluded_ids": [], "negatives": "ignore synthetic neg text",
        "membership": "exact explicit ReasonIR HQ indices, no additional sampling or deduplication",
        "judgments": "one binary relevance grade per distinct joined positive ID; fail on any unresolved ID",
    },
    "test": {
        "dataset": "xlangai/BRIGHT", "query_field": "query", "id_field": "id",
        "positives": "gold_ids joined by exact ID to same-domain BRIGHT documents.id",
        "excluded_ids": "excluded_ids, missing field means []",
        "membership": "every example in every declared BRIGHT export",
        "judgments": "one binary grade per distinct gold ID; reject empty gold or conflict with excluded IDs",
    },
    "row_numbering": "zero-based nonblank JSON records",
    "ordering": ["domain", "query_id"],
    "normalized_text": "NFKC, casefold, collapse Unicode whitespace; SHA-256 of canonical JSON string",
}


def is_digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def verified_jsonl(path, expected_sha256):
    """Exhaust before using any rows; hash exactly the bytes parsed, including blank lines."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if raw.strip():
                row = json.loads(raw.decode("utf-8"))
                if not isinstance(row, dict):
                    raise ValueError(f"{path}, line {line_number}: expected JSON object")
                yield row
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")


def read_frozen_rows(path, expected_sha256):
    return list(verified_jsonl(path, expected_sha256))


def selection_provenance(plan):
    declared = plan.get("selection_provenance")
    if declared is None and plan["schema_version"] == 1:
        return {split: {"rule": "explicit selection JSONL; upstream selection history undisclosed",
                        "filters": [], "seed": None, "upstream_history": "undisclosed"}
                for split in SPLITS[:2]}
    if not isinstance(declared, dict) or set(declared) != set(SPLITS[:2]):
        raise ValueError("selection_provenance must specify train and independent development")
    for split, entry in declared.items():
        if (not isinstance(entry, dict) or not isinstance(entry.get("rule"), str) or not entry["rule"].strip()
                or not isinstance(entry.get("filters"), list)
                or any(not isinstance(f, str) or not f.strip() for f in entry["filters"])
                or "seed" not in entry or (entry["seed"] is not None and type(entry["seed"]) is not int)):
            raise ValueError(f"selection_provenance.{split} requires rule, filters and integer or null seed")
    return declared


@dataclass(frozen=True)
class FrozenBundle:
    receipt_sha256: str
    paths: dict[str, Path]
    manifest_sha256: dict[str, str]
    identities: dict[str, frozenset[tuple[str, str]]]


def _verify_membership(receipt, domains):
    inventory = receipt["membership"]
    if set(inventory) != set(SPLITS):
        raise ValueError("all three frozen memberships are required")
    identities = {}
    for split in SPLITS:
        rows = inventory[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"empty frozen {split} membership")
        counts = dict.fromkeys(BRIGHT_TEST_COUNTS, 0)
        test_indices = {domain: set() for domain in domains}
        for row in rows:
            domain = row["domain"]
            if (domain not in domains or not isinstance(row["query_id"], str) or not row["query_id"].strip()
                    or not is_digest(row["normalized_query_sha256"])):
                raise ValueError(f"invalid {split} membership record")
            counts[domain] += 1
            is_test = split == "test"
            role = f"{domain}/" + ("examples" if is_test else "reasonir_hq")
            asset = receipt["inputs"][role]
            source = row["source"]
            index = source["source_row_index"]
            if (source["dataset"] != ("xlangai/BRIGHT" if is_test else "reasonir/reasonir-data")
                    or source["revision"] != (BRIGHT_REVISION if is_test else REASONIR_REVISION)
                    or source["export_sha256"] != asset["sha256"] or source["asset_role"] != role
                    or not is_digest(source["source_row_sha256"]) or type(index) is not int
                    or not 0 <= index < asset["records"]):
                raise ValueError(f"{split}/{domain}: source evidence differs from frozen export")
            if is_test:
                test_indices[domain].add(index)
            else:
                sel_index = source["selection_row_index"]
                if (type(sel_index) is not int or not 0 <= sel_index < receipt["inputs"]["selection"]["records"]
                        or not is_digest(source["selection_row_sha256"])):
                    raise ValueError(f"{split}/{domain}: invalid explicit selection evidence")
        if counts != receipt["counts"][split]:
            raise ValueError(f"frozen {split} counts disagree with membership")
        if split == "test":
            for domain, indices in test_indices.items():
                total = receipt["inputs"][f"{domain}/examples"]["records"]
                if indices != set(range(total)) or counts[domain] != total:
                    raise ValueError(f"{domain}: test membership must retain every official example once")
        identities[split] = frozenset((r["domain"], r["query_id"]) for r in rows)
    indices = [row["source"]["selection_row_index"] for s in SPLITS[:2] for row in inventory[s]]
    if sorted(indices) != list(range(receipt["inputs"]["selection"]["records"])):
        raise ValueError("every explicit ReasonIR selection must be present exactly once")
    if audit_identities(inventory) != receipt["overlap_audit"]:
        raise ValueError("receipt overlap audit differs from the frozen identities")
    matches = {"train": receipt["counts"]["train"] == BRIGHT_TRAIN_COUNTS,
               "test": receipt["counts"]["test"] == BRIGHT_TEST_COUNTS}
    policy = receipt["count_policy"]
    if (policy not in {"require_paper_counts", "report_only"} or matches != receipt["paper_count_matches"]
            or (policy == "require_paper_counts" and not all(matches.values()))):
        raise ValueError("count policy or paper count matches differ from frozen membership")
    return identities


def verify_bundle(deployment) -> FrozenBundle:
    """Verify a pre-pinned preparation receipt. Hash test bytes without parsing its qrels."""
    require_slurm()
    receipt_path = require_cluster_storage(deployment["dataset_receipt_path"])
    pinned = deployment["dataset_receipt_sha256"]
    payload = receipt_path.read_bytes()
    if not is_digest(pinned) or hashlib.sha256(payload).hexdigest() != pinned:
        raise ValueError("dataset receipt differs from the precommitted SHA-256")
    receipt = json.loads(payload)
    if (receipt_path.name != "receipt.json" or receipt.get("schema_version") != 2
            or receipt.get("kind") != "independent_explicit_selection"
            or receipt.get("author_split_identity_verified") is not False
            or receipt.get("source_rules") != SOURCE_RULES or receipt.get("declared_revisions") != REVISIONS
            or not is_digest(receipt.get("plan_sha256"))):
        raise ValueError("expected a pinned independent version-2 preparation receipt")
    plan_bytes = (receipt_path.parent / "preparation-plan.json").read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != receipt["plan_sha256"]:
        raise ValueError("preparation plan differs from the pinned receipt")
    plan = json.loads(plan_bytes)
    if (plan.get("schema_version") not in (1, 2) or plan.get("execution_enabled") is not True
            or plan.get("bright_revision") != BRIGHT_REVISION or plan.get("reasonir_revision") != REASONIR_REVISION
            or plan.get("count_policy") != receipt["count_policy"]
            or selection_provenance(plan) != selection_provenance(receipt)):
        raise ValueError("receipt source or selection policy differs from the preparation plan")
    domains = deployment["domains"]
    if (not domains or set(domains) != set(receipt["domains"]) or set(domains) != set(plan["domains"])
            or not set(domains) <= set(BRIGHT_TEST_COUNTS)):
        raise ValueError("deployed domains must match the frozen bundle")
    planned = {"selection": plan["selection_jsonl"]}
    for domain, spec in plan["domains"].items():
        planned[f"{domain}/examples"] = spec["bright_examples_jsonl"]
        planned[f"{domain}/documents"] = spec["bright_documents_jsonl"]
        if f"{domain}/reasonir_hq" in receipt["inputs"]:
            planned[f"{domain}/reasonir_hq"] = spec["reasonir_hq_jsonl"]
    if set(planned) != set(receipt["inputs"]):
        raise ValueError("receipt input roles differ from the preparation plan")
    for role, asset in receipt["inputs"].items():
        if (not is_digest(asset["sha256"]) or type(asset["records"]) is not int or asset["records"] < 0
                or asset["sha256"] != planned[role]["sha256"]):
            raise ValueError(f"invalid input evidence: {role}")
    for domain, spec in domains.items():
        corpus = require_cluster_storage(spec["corpus_jsonl"])
        if file_sha256(corpus) != receipt["inputs"][f"{domain}/documents"]["sha256"]:
            raise ValueError(f"deployed corpus differs from prepared positive-ID join: {domain}")
    identities = _verify_membership(receipt, domains)
    hashes = receipt["manifest_sha256"]
    if set(hashes) != set(SPLITS):
        raise ValueError("all three frozen manifest hashes are required")
    paths = {}
    for split in SPLITS:
        path = require_cluster_storage(deployment[f"{split}_manifest_path"])
        if path != receipt_path.parent / f"{split}.jsonl":
            raise ValueError(f"{split} manifest is outside the pinned bundle or has a different name")
        if not is_digest(hashes[split]) or file_sha256(path) != hashes[split]:
            raise ValueError(f"{split} manifest differs from the pre-training dataset receipt")
        paths[split] = path
    return FrozenBundle(pinned, paths, hashes, identities)
