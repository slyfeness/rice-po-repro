"""Explicit manifests only: never recreate unpublished author splits silently."""

from dataclasses import dataclass
from collections import defaultdict
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
BRIGHT_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
REASONIR_REVISION = "0275f825929b206d4ead23d34b4f8a50d4eddbc8"


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
    if not isinstance(query_id, str) or not query_id.strip() or not isinstance(query, str) or not query.strip():
        raise ValueError("invalid query/id")
    excluded = row.get("excluded_ids", [])
    if not isinstance(excluded, (tuple, list)) or not all(isinstance(x, str) and x.strip() for x in excluded):
        raise ValueError("excluded_ids must contain document IDs")
    return QueryExample(query_id, domain, query, frozenset(excluded))


def project_qrels(row: Mapping) -> dict[str, int]:
    """BRIGHT short-document gold, separate from QueryExample."""
    gold = row["gold_ids"]
    if not isinstance(gold, (list, tuple)) or not all(isinstance(x, str) and x.strip() for x in gold):
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
    if (not isinstance(query, (list, tuple)) or len(query) != 2
            or not isinstance(query[1], str) or not query[1].strip()):
        raise ValueError("expected ReasonIR HQ [instruction, query]")
    positive_ids = []
    for pair in row["pos"]:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not isinstance(pair[1], str) or pair[1] not in corpus):
            raise ValueError(f"positive document ID is absent or ambiguous: {pair!r}")
        positive_ids.append(pair[1])
    if not positive_ids:
        raise ValueError("missing positive documents")
    # The public negative field contains synthetic text, not official corpus IDs.
    return query[1], {document_id: 1 for document_id in positive_ids}


def identity_record(row: Mapping) -> dict:
    """Label-free membership and source evidence used by the bundle verifier."""
    domain, query_id = row["domain"], row["query_id"]
    if not all(isinstance(x, str) and x.strip() for x in (domain, query_id, row["query"])):
        raise ValueError("expected nonempty domain, query ID and query")
    return {"domain": domain, "query_id": query_id,
            "normalized_query_sha256": content_sha256(canonical_query(row["query"])),
            "source": row.get("source", {})}


class SplitValidationError(ValueError):
    def __init__(self, records):
        self.records = records
        super().__init__("split audit failed: " + json.dumps(records, ensure_ascii=False))


def audit_identities(splits: Mapping[str, list[dict]]) -> list[dict]:
    """Keep distinct official tasks with shared text across BRIGHT test domains."""
    by_id, by_text, by_reasonir_row = defaultdict(list), defaultdict(list), defaultdict(list)
    for split, rows in splits.items():
        for row in rows:
            task = {"split": split, "domain": row["domain"], "query_id": row["query_id"]}
            by_id[row["domain"], row["query_id"]].append(task)
            source = row["source"]
            official = source.get("dataset") == "xlangai/BRIGHT" and source.get("revision") == BRIGHT_REVISION
            by_text[row["normalized_query_sha256"]].append((task, official))
            if source.get("dataset") == "reasonir/reasonir-data":
                by_reasonir_row[source["export_sha256"], source["source_row_index"]].append(task)

    records = []
    def record(kind, tasks, **extra):
        records.append({"kind": kind, "members": sorted(tasks, key=lambda t: (
            t["split"], t["domain"], t["query_id"])), **extra})

    for tasks in by_id.values():
        if len(tasks) > 1:
            record("duplicate query ID", tasks)
    for (digest, index), tasks in by_reasonir_row.items():
        if len(tasks) > 1:
            record("reused ReasonIR source row", tasks, export_sha256=digest, source_row_index=index)
    for digest, group in by_text.items():
        if len(group) == 1:
            continue
        tasks = [task for task, _ in group]
        split_names = {task["split"] for task in tasks}
        if len(split_names) > 1:
            kind = "cross-split duplicate query text"
        elif (split_names == {"test"} and all(official for _, official in group)
              and len({task["domain"] for task in tasks}) == len(tasks)):
            kind = "official shared query text across test domains"
        else:
            kind = "within-split duplicate query text"
        record(kind, tasks, normalized_query_sha256=digest)
    records.sort(key=lambda r: json.dumps(r, sort_keys=True))
    if any(r["kind"] != "official shared query text across test domains" for r in records):
        raise SplitValidationError(records)
    return records


def validate_split_manifests(splits: Mapping[str, list[dict]]) -> list[dict]:
    return audit_identities({split: [identity_record(row) for row in rows] for split, rows in splits.items()})


def verify_counts(rows: Iterable[QueryExample], expected: Mapping[str, int]) -> None:
    counts = {domain: 0 for domain in expected}
    for row in rows:
        if row.domain not in counts:
            raise ValueError("unexpected domain")
        counts[row.domain] += 1
    if counts != dict(expected):
        raise ValueError(f"dataset count mismatch: {counts}")
