"""Source fixtures for the shared per-query evaluation path."""

import json
from dataclasses import dataclass

import pytest

from ricepo.assessment import evaluate_queries, summarize_assessment
from ricepo.contracts import RankedDocument, RetrievalResult
from ricepo.data import QueryExample


class TinyRetriever:
    def retrieve(self, query, *, excluded_ids):
        assert query == "original text" and excluded_ids == frozenset({"blocked"})
        return RetrievalResult(query, (RankedDocument("positive", 1.0, "text"),), (), "tiny-v1")


class Reward:
    def score(self, query, ids):
        assert json.loads(query) == ["biology", "q"]
        return float(ids == ("positive",))


def test_retrieval_only_scores_the_original_query_and_requires_no_generator(tmp_path):
    key = '["biology","q"]'
    summary = evaluate_queries(
        examples={key: QueryExample(key, "biology", "original text", frozenset({"blocked"}))},
        expected_ids={("biology", "q")}, split="test", phase="baseline", method="retrieval_only",
        root_seed=None, policy_version="retriever/v1", rollout_depth=None,
        retrievers={"biology": TinyRetriever()}, runners=None, renderer=None, reward=Reward(),
        output=tmp_path / "scores", scope="complete", manifest_sha256="a"*64,
        backend_fingerprints={"biology": "b"*64},
    )
    assert summary["macro"] == 1.0
    rows = json.loads((tmp_path / "scores" / "results-by-query.json").read_text())
    assert rows[0]["generation_seeds"] == []
    assert rows[0]["split"] == "test"


def test_complete_development_refuses_missing_query_before_retrieval(tmp_path):
    with pytest.raises(ValueError, match="membership"):
        evaluate_queries(examples={}, expected_ids={("biology", "missing")}, split="development",
                        phase="development", method="grpo", root_seed=1, policy_version="actor/v1",
                        rollout_depth=5, retrievers={}, runners={}, renderer=object(), reward=None,
                        output=tmp_path / "dev", scope="complete", manifest_sha256="a"*64,
                        backend_fingerprints={"biology":"b"*64})


class EchoRenderer:
    def initial(self, query, docs):
        return (7,) if query == "original text" and docs[0].document_id == "positive" else ()


@dataclass(frozen=True)
class FinalStep:
    retrieval: RetrievalResult


@dataclass(frozen=True)
class Trace:
    policy_version: str
    query_id: str
    steps: tuple
    final_reward: float


class EchoRunner:
    def run(self, *, query_id, old_policy_version, seeds, initial_prefix_token_ids, **kwargs):
        assert len(seeds) == 1 and initial_prefix_token_ids == (7,)
        final = RetrievalResult("expanded", (RankedDocument("positive", 1.0),), (), "tiny-v1")
        return Trace(old_policy_version, query_id, (FinalStep(final),), 1.0)


def test_evaluation_step_seeds_are_paired_across_methods_and_distinct_between_splits(tmp_path):
    key = '["biology","q"]'
    choices = [("prompting", "test"), ("grpo", "test"), ("ricepo", "development")]
    seeds = []
    for method, split in choices:
        target = tmp_path / method
        evaluate_queries(examples={key: QueryExample(key, "biology", "original text", frozenset({"blocked"}))},
                        expected_ids={("biology", "q")}, split=split, phase=split, method=method,
                        root_seed=42, policy_version=method, rollout_depth=1, retrievers={"biology": TinyRetriever()},
                        runners={"biology": EchoRunner()}, renderer=EchoRenderer(), reward=Reward(), output=target,
                        manifest_sha256="a"*64, backend_fingerprints={"biology":"b"*64})
        seeds.append(json.loads((target / "results-by-query.json").read_text())[0]["generation_seeds"])
    assert seeds[0] == seeds[1]
    assert seeds[0] != seeds[2]


def evaluation_rows():
    return [dict(domain="biology", query_id=q, split="test", method="ricepo", policy_version="updated/4",
                 behavior_sha256="a"*64, manifest_sha256="b"*64, backend_version="retrieval/v1",
                 backend_fingerprint="c"*64, retrieval_set_sha256="d"*64, ndcg_at_10=0.25)
            for q in ("q1","q2")]


@pytest.mark.parametrize("field,value", [
    ("method","grpo"), ("policy_version","updated/3"), ("behavior_sha256","e"*64),
    ("manifest_sha256","e"*64), ("split","development"), ("backend_fingerprint","e"*64),
    ("backend_version","other-retriever"), ("retrieval_set_sha256","e"*64),
])
def test_combining_different_run_identities_cannot_form_a_complete_macro(field, value):
    rows = evaluation_rows()
    rows[1][field] = value
    with pytest.raises(ValueError):
        summarize_assessment(rows, {("biology","q1"), ("biology","q2")}, split="test", manifest_sha256="b"*64)


def test_summary_retains_method_policy_manifest_and_domain_backend():
    summary = summarize_assessment(evaluation_rows(), {("biology","q1"), ("biology","q2")},
                                   split="test", manifest_sha256="b"*64)
    assert summary["macro"] == 0.25
    assert summary["policy_version"] == "updated/4" and summary["method"] == "ricepo"
    assert summary["manifest_sha256"] == "b"*64
    assert summary["backend_fingerprints"] == {"biology":"c"*64}


def test_partial_development_rejected_even_if_every_id_is_present():
    rows = [dict(row, split="development") for row in evaluation_rows()]
    with pytest.raises(ValueError, match="development"):
        summarize_assessment(rows, {("biology","q1"), ("biology","q2")}, split="development",
                             manifest_sha256="b"*64, scope="partial_diagnostic")


def test_summarize_cli_rejects_partial_development_before_accessing_assets():
    from ricepo.cli import main
    with pytest.raises(ValueError, match="partial development"):
        main(["summarize", "--input", "absent", "--deployment", "absent", "--output", "unused",
              "--split", "development", "--partial"])
