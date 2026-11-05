"""Authored fixtures; not executed for this delivery."""
import pytest
from ricepo.evaluation import summarize


def test_macro_is_not_query_weighted():
    result = summarize([
        {"domain": "a", "query_id": "1", "ndcg_at_10": 1.0},
        {"domain": "b", "query_id": "1", "ndcg_at_10": 0.0},
        {"domain": "b", "query_id": "2", "ndcg_at_10": 0.0},
    ], ["a", "b"])
    assert result["macro"] == 0.5


def test_incomplete_domains_are_not_silent():
    with pytest.raises(ValueError, match="missing domains"):
        summarize([{"domain": "a", "query_id": "1", "ndcg_at_10": 1.0}], ["a", "b"])
