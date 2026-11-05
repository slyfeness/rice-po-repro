"""Fixed retrieval and reward contracts using small fake providers."""

from types import SimpleNamespace
import sys

import pytest

from ricepo.retrieval import (
    DenseVectorScorer, ExplicitQueryComposer, FixedHybridRetriever, PyTrecReward,
    hybrid_rank, last_token_embeddings, minmax,
)


def test_explicit_query_composition_is_not_guessed():
    assert ExplicitQueryComposer("summary_only", "unused").compose("original", "summary") == "summary"
    assert ExplicitQueryComposer("original_and_summary", "\n").compose("original", "summary") == "original\nsummary"
    with pytest.raises(ValueError, match="resolved"):
        ExplicitQueryComposer("unknown", " ").compose("q", "s")


def test_exclusion_order_changes_normalization_only_when_explicitly_selected():
    dense, sparse = {"a": 0.0, "b": 5.0, "c": 10.0}, {"a": 0.0, "b": 2.0, "c": 100.0}
    kwargs = dict(normalization=minmax, excluded_ids=frozenset({"c"}), top_k=10)
    before = hybrid_rank(dense, sparse, exclude_before_normalization=True, **kwargs)
    after = hybrid_rank(dense, sparse, exclude_before_normalization=False, **kwargs)
    assert [doc.document_id for doc in before] == ["b", "a"]
    assert before[0].score == 1.0
    assert after[0].score == pytest.approx(0.26)


def test_hybrid_fusion_requires_complete_domains_finite_scores_and_explicit_mode():
    kwargs = dict(normalization=minmax, excluded_ids=frozenset(), top_k=2,
                  exclude_before_normalization=False)
    with pytest.raises(ValueError, match="matching complete"):
        hybrid_rank({"a": 1.0}, {"b": 1.0}, **kwargs)
    with pytest.raises(ValueError, match="raw retrieval"):
        hybrid_rank({"a": float("nan")}, {"a": 1.0}, **kwargs)
    with pytest.raises(ValueError, match="explicit boolean"):
        hybrid_rank({"a": 1.0}, {"a": 1.0}, **{**kwargs, "exclude_before_normalization": None})
    with pytest.raises(ValueError, match="changed the scoring domain"):
        hybrid_rank({"a": 1.0}, {"a": 1.0}, **{**kwargs, "normalization": lambda values: {}})


def test_fixed_retriever_snapshots_corpus_and_renders_ranked_observations_only():
    corpus = {"a": "stored original", "b": "second"}
    queries, rendered = [], []

    def scores(query):
        queries.append(query)
        return {"a": 2.0, "b": 1.0}

    def render(documents):
        rendered.append(documents)
        return (31, 32)

    retriever = FixedHybridRetriever(
        dense_score=scores, sparse_score=scores, corpus=corpus, normalization=minmax,
        exclude_before_normalization=False, top_k=1, render_observation=render,
        backend_version="fixed-index-v1",
    )
    corpus["a"] = "external mutation"
    result = retriever.retrieve("explicit composed query", excluded_ids=frozenset())
    assert result.documents[0].text == "stored original"
    assert queries == ["explicit composed query", "explicit composed query"]
    assert result.observation_token_ids == (31, 32)
    assert result.backend_version == "fixed-index-v1"
    assert rendered == [result.documents]


def test_pytrec_reward_preserves_ranking_order_and_keeps_qrels_in_scorer(monkeypatch):
    calls = []

    class Evaluator:
        def __init__(self, qrels, measures):
            assert qrels == {"q": {"b": 2}}
            assert measures == {"ndcg_cut_10"}

        def evaluate(self, run):
            calls.append(run)
            return {"q": {"ndcg_cut_10": 0.42}}

    monkeypatch.setitem(sys.modules, "pytrec_eval", SimpleNamespace(RelevanceEvaluator=Evaluator))
    scorer = PyTrecReward({"q": {"b": 2}})
    assert scorer.score("q", ("a", "b", "c")) == 0.42
    assert calls == [{"q": {"a": 3.0, "b": 2.0, "c": 1.0}}]
    assert scorer.score("q", ()) == 0.0
    with pytest.raises(ValueError, match="duplicate"):
        scorer.score("q", ("a", "a"))
    with pytest.raises(ValueError, match="unknown qrels"):
        scorer.score("other-query", ("a",))


def test_exact_dense_cosine_is_independent_of_corpus_chunk_size():
    np = pytest.importorskip("numpy")
    vectors = np.asarray([[1, 0], [0, 2], [-3, 0]], dtype=np.float32)
    kwargs = dict(document_ids=("a", "b", "c"), vectors=vectors, encode_query=lambda query: [1, 1])
    small = DenseVectorScorer(**kwargs, chunk_size=1)("q")
    whole = DenseVectorScorer(**kwargs, chunk_size=3)("q")
    assert small == pytest.approx(whole)
    assert small == pytest.approx({"a": 2 ** -0.5, "b": 2 ** -0.5, "c": -(2 ** -0.5)})


def test_last_nonpadding_embedding_handles_both_padding_sides_and_requires_eval():
    torch = pytest.importorskip("torch")

    class Model:
        training = False

        def __call__(self, *, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=torch.stack((input_ids.float(), torch.ones_like(input_ids)), dim=-1))

    model = Model()
    ids = torch.tensor([[1, 2, 0], [0, 1, 3]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    vectors = last_token_embeddings(model, ids, mask)
    expected = torch.nn.functional.normalize(torch.tensor([[2.0, 1.0], [3.0, 1.0]]), dim=-1)
    assert torch.allclose(vectors, expected)
    model.training = True
    with pytest.raises(ValueError, match="eval mode"):
        last_token_embeddings(model, ids, mask)
