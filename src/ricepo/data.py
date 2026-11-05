"""Explicit manifests only: never recreate unpublished author splits silently."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Iterable, Mapping


BRIGHT_TRAIN_COUNTS = dict(zip(
    ("biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow",
     "sustainable_living", "leetcode", "pony", "aops", "theoremqa_questions", "theoremqa_theorems"),
    (414, 404, 412, 424, 275, 418, 426, 420, 94, 384, 0, 340), strict=True))
BRIGHT_TEST_COUNTS = dict(zip(BRIGHT_TRAIN_COUNTS,
    (103, 116, 103, 101, 101, 117, 108, 142, 112, 111, 194, 76), strict=True))


@dataclass(frozen=True)
class QueryExample:
    query_id: str
    domain: str
    query: str
    excluded_ids: frozenset[str]


def canonical_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def content_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: expected object")
            yield value


def project_bright_example(row: Mapping, domain: str) -> QueryExample:
    """Project actor-visible fields; gold/reasoning fields cannot leak through."""
    if domain not in BRIGHT_TEST_COUNTS:
        raise ValueError("unknown BRIGHT domain")
    query_id, query = row["id"], row["query"]
    if not isinstance(query_id, str) or not isinstance(query, str) or not query.strip():
        raise ValueError("invalid query/id")
    excluded = row.get("excluded_ids", [])
    if not isinstance(excluded, (tuple, list)) or not all(isinstance(x, str) for x in excluded):
        raise ValueError("excluded_ids must contain document IDs")
    return QueryExample(query_id, domain, query, frozenset(excluded))


def project_qrels(row: Mapping) -> dict[str, int]:
    """BRIGHT short-document gold, separate from QueryExample."""
    gold = row["gold_ids"]
    if not isinstance(gold, (list, tuple)) or not all(isinstance(x, str) for x in gold):
        raise ValueError("invalid gold_ids")
    return {document_id: 1 for document_id in gold}


def project_corpus_document(row: Mapping) -> tuple[str, str]:
    """Native BRIGHT uses `content`; explicit normalized exports may use `text`."""
    document_id = row.get("id")
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("invalid corpus ID")
    fields = [row[key] for key in ("content", "text") if key in row]
    if not fields or not all(isinstance(value, str) and value == fields[0] for value in fields):
        raise ValueError("missing or conflicting corpus content/text fields")
    return document_id, fields[0]


def join_reasonir_hq(row: Mapping, corpus: Mapping[str, str]) -> tuple[str, dict[str, int]]:
    query = row["query"]
    if not isinstance(query, (list, tuple)) or len(query) != 2 or not isinstance(query[1], str):
        raise ValueError("expected ReasonIR HQ [instruction, query]")
    positive_ids = []
    for pair in row["pos"]:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[1] not in corpus:
            raise ValueError("positive document ID is absent or ambiguous")
        positive_ids.append(pair[1])
    if not positive_ids:
        raise ValueError("missing positive documents")
    # The public negative field contains synthetic text, not official corpus IDs.
    return query[1], {document_id: 1 for document_id in positive_ids}


def validate_split_manifests(splits: Mapping[str, list[dict]]) -> None:
    """Reject query leakage/duplicate IDs; public shared corpora are permitted."""
    seen_ids: dict[tuple[str, str], str] = {}
    seen_queries: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            identity = (row["domain"], row["query_id"])
            normalized = canonical_query(row["query"])
            if not normalized:
                raise ValueError("empty normalized query")
            if identity in seen_ids:
                raise ValueError(f"duplicate query ID: {identity} ({seen_ids[identity]}, {split})")
            if normalized in seen_queries:
                raise ValueError(f"duplicate query text across {seen_queries[normalized]} and {split}")
            seen_ids[identity] = split
            seen_queries[normalized] = split


def verify_counts(rows: Iterable[QueryExample], expected: Mapping[str, int]) -> None:
    counts = {domain: 0 for domain in expected}
    for row in rows:
        if row.domain not in counts:
            raise ValueError("unexpected domain")
        counts[row.domain] += 1
    if counts != dict(expected):
        raise ValueError(f"dataset count mismatch: {counts}")
