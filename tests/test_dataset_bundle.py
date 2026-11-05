"""Source fixtures for a future authorized run. All data below are synthetic."""

import json
from pathlib import Path

import pytest

from ricepo import dataset_bundle, manifest
from ricepo.artifacts import file_sha256
from ricepo.data import BRIGHT_REVISION, REASONIR_REVISION, SplitValidationError


def export(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return {"path": str(path), "sha256": file_sha256(path)}


@pytest.fixture
def build(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    for module in (manifest, dataset_bundle):
        monkeypatch.setattr(module, "require_cluster_storage", lambda path: Path(path).resolve())

    def prepare(*, train_query="train question", test_positive="doc", reasonir_positive="doc",
                reuse_source=False, duplicate_test_id=False):
        domains = {}
        for name in ("theoremqa_questions", "theoremqa_theorems"):
            domains[name] = {
                "bright_documents_jsonl": export(tmp_path / f"{name}-docs.jsonl",
                                                 [{"id": "doc", "content": "Public evidence"}]),
                "bright_examples_jsonl": export(tmp_path / f"{name}-test.jsonl", [
                    {"id": "test-id", "query": "Same official query", "gold_ids": [test_positive]},
                    {"id": "test-id" if duplicate_test_id else "other-id",
                     "query": f"Another query for {name}", "gold_ids": ["doc"]},
                ]),
            }
        train_domain = "theoremqa_theorems"
        domains[train_domain]["reasonir_hq_jsonl"] = export(tmp_path / "hq.jsonl", [
            {"query": ["instruction", train_query], "pos": [["passage", reasonir_positive]]},
            {"query": ["instruction", "dev question"], "pos": [["passage", "doc"]]},
            {"query": ["instruction", "unselected question"], "pos": [["passage", "doc"]]},
        ])
        selections = [
            {"domain": train_domain, "split": "train", "source_row_index": 0, "query_id": "tr"},
            {"domain": train_domain, "split": "development", "source_row_index": 0 if reuse_source else 1,
             "query_id": "dv"},
        ]
        plan = {
            "schema_version": 2, "execution_enabled": True, "count_policy": "report_only",
            "bright_revision": BRIGHT_REVISION, "reasonir_revision": REASONIR_REVISION, "domains": domains,
            "selection_jsonl": export(tmp_path / "selection.jsonl", selections),
            "selection_provenance": {split: {"rule": f"explicit {split} fixture", "filters": [], "seed": None}
                                     for split in ("train", "development")},
        }
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        target = tmp_path / "frozen"
        receipt = manifest.build_manifests(plan_path, target)
        deployment = {"dataset_receipt_path": str(target / "receipt.json"),
                      "dataset_receipt_sha256": file_sha256(target / "receipt.json"),
                      "domains": {name: {"corpus_jsonl": s["bright_documents_jsonl"]["path"]}
                                  for name, s in domains.items()}}
        for split in dataset_bundle.SPLITS:
            deployment[f"{split}_manifest_path"] = str(target / f"{split}.jsonl")
        return deployment, receipt

    return prepare


def test_bundle_fixes_three_manifests_and_keeps_two_official_same_text_tasks(build):
    deployment, receipt = build()
    bundle = dataset_bundle.verify_bundle(deployment)
    assert bundle.receipt_sha256 == deployment["dataset_receipt_sha256"]
    assert set(bundle.manifest_sha256) == set(dataset_bundle.SPLITS)
    assert len(bundle.identities["test"]) == 4
    assert receipt["overlap_audit"][0]["kind"] == "official shared query text across test domains"
    assert len(receipt["overlap_audit"][0]["members"]) == 2
    assert receipt["inputs"]["theoremqa_theorems/reasonir_hq"]["records"] == 3
    assert receipt["counts"]["development"]["theoremqa_theorems"] == 1
    assert receipt["membership"]["development"][0]["source"]["source_row_index"] == 1
    assert receipt["membership"]["development"][0]["source"]["selection_row_index"] == 1
    assert receipt["author_split_identity_verified"] is False
    assert all("qrels" not in entry and "query" not in entry for entry in receipt["membership"]["test"])
    assert receipt["paper_count_matches"] == {"train": False, "test": False}


@pytest.mark.parametrize("split", dataset_bundle.SPLITS)
def test_modified_manifest_rejected_by_original_receipt(build, split):
    deployment, _ = build()
    with Path(deployment[f"{split}_manifest_path"]).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="pre-training dataset receipt"):
        dataset_bundle.verify_bundle(deployment)


def test_unpinned_receipt_and_different_corpus_rejected(build):
    deployment, _ = build()
    with pytest.raises(ValueError, match="precommitted"):
        dataset_bundle.verify_bundle({**deployment, "dataset_receipt_sha256": "a" * 64})
    corpus = next(iter(deployment["domains"].values()))["corpus_jsonl"]
    with Path(corpus).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="prepared positive-ID join"):
        dataset_bundle.verify_bundle(deployment)


def test_frozen_preparation_plan_cannot_be_replaced(build):
    deployment, _ = build()
    plan = Path(deployment["dataset_receipt_path"]).with_name("preparation-plan.json")
    with plan.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="preparation plan differs"):
        dataset_bundle.verify_bundle(deployment)


def test_cross_split_overlap_rejected_with_all_implicated_ids_and_no_bundle(build, tmp_path):
    with pytest.raises(SplitValidationError):
        build(train_query=" Same OFFICIAL  query ")
    assert not (tmp_path / "frozen").exists()
    audit_file, = tmp_path.glob("frozen.overlap-*.json")
    records = json.loads(audit_file.read_bytes())["records"]
    text_collision, = [r for r in records if r["kind"] == "cross-split duplicate query text"]
    assert len(text_collision["members"]) == 3
    assert {m["split"] for m in text_collision["members"]} == {"train", "test"}


def test_duplicate_official_task_id_rejected(build):
    with pytest.raises(SplitValidationError, match="duplicate query ID"):
        build(duplicate_test_id=True)


def test_same_reasonir_source_selected_for_development_and_train_rejected(build):
    with pytest.raises(SplitValidationError, match="reused ReasonIR source row"):
        build(reuse_source=True)


@pytest.mark.parametrize("options, message", [
    ({"test_positive": "lost-test-doc"}, "unresolved corpus IDs"),
    ({"reasonir_positive": "lost-train-doc"}, "absent or ambiguous"),
])
def test_unresolved_positive_is_never_dropped(build, options, message):
    with pytest.raises(ValueError, match=message):
        build(**options)
