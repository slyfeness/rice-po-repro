"""Dataset separation contracts. Authored only; not executed in this delivery."""

from dataclasses import asdict

import pytest

from ricepo.data import (
    canonical_query,
    join_reasonir_hq,
    project_bright_example,
    project_qrels,
    validate_split_manifests,
)


def row(identity, query, domain="biology", **extra):
    return {"query_id": identity, "domain": domain, "query": query, **extra}


def test_train_test_query_leakage_is_detected_after_unicode_case_and_space_normalization():
    train = row("train-1", "  ＤＮＡ  repair\n mechanisms ")
    test = row("test-7", "dna REPAIR mechanisms")
    assert canonical_query(train["query"]) == canonical_query(test["query"])
    with pytest.raises(ValueError, match="duplicate query text"):
        validate_split_manifests({"train": [train], "test": [test]})


def test_same_domain_query_identity_cannot_cross_splits_even_if_text_is_changed():
    with pytest.raises(ValueError, match="duplicate query ID"):
        validate_split_manifests({
            "train": [row("17", "What repairs DNA?")],
            "development": [row("17", "Which repair pathway is activated?")],
        })


def test_domain_qualified_ids_allow_same_local_id_in_distinct_domains():
    validate_split_manifests({
        "train": [row("17", "How does DNA repair work?", domain="biology")],
        "test": [row("17", "How are minerals dated?", domain="earth_science")],
    })


def test_shared_public_corpus_and_gold_document_do_not_imply_query_leakage():
    common = {"corpus_revision": "shared-public-corpus", "gold_ids": ["document-1"]}
    validate_split_manifests({
        "train": [row("a", "What is the enzyme's mechanism?", **common)],
        "test": [row("b", "When does the enzyme fail to bind?", **common)],
    })


def test_duplicate_queries_within_one_split_are_also_rejected():
    with pytest.raises(ValueError, match="duplicate query text"):
        validate_split_manifests({
            "train": [row("a", "A mechanism?"), row("b", "A MECHANISM?")],
        })


def test_projection_separates_gold_and_reasoning_from_actor_visible_example():
    source = {
        "id": "q1",
        "query": "Which mechanism explains the result?",
        "excluded_ids": ["near-duplicate"],
        "gold_ids": ["relevant"],
        "reasoning": "PRIVATE_TO_REWARD_PIPELINE",
        "answer": "DO_NOT_INCLUDE",
    }
    example = project_bright_example(source, "biology")
    assert asdict(example) == {
        "query_id": "q1",
        "domain": "biology",
        "query": source["query"],
        "excluded_ids": frozenset({"near-duplicate"}),
    }
    assert project_qrels(source) == {"relevant": 1}
    assert source["gold_ids"] == ["relevant"]


def test_reasonir_join_uses_positive_ids_and_does_not_promote_synthetic_negatives():
    source = {
        "query": ["retrieval instruction", "A concrete research query"],
        "pos": [["document instruction", "doc-1"]],
        "neg": ["synthetic negative text without an official document ID"],
    }
    query, qrels = join_reasonir_hq(source, {"doc-1": "public document text"})
    assert query == "A concrete research query"
    assert qrels == {"doc-1": 1}


def test_reasonir_positive_missing_from_corpus_is_not_silently_dropped():
    with pytest.raises(ValueError, match="absent or ambiguous"):
        join_reasonir_hq({
            "query": ["instruction", "query"],
            "pos": [["instruction", "missing-id"]],
        }, {"different-id": "document"})
