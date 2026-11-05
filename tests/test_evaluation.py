"""Authored fixtures; not executed for this delivery."""
import pytest
from ricepo.data import BRIGHT_TEST_COUNTS
from ricepo.evaluation import summarize


def test_macro_is_not_query_weighted():
    result = summarize([
        {"domain": "a", "query_id": "1", "ndcg_at_10": 1.0},
        {"domain": "b", "query_id": "1", "ndcg_at_10": 0.0},
        {"domain": "b", "query_id": "2", "ndcg_at_10": 0.0},
    ], [("a", "1"), ("b", "1"), ("b", "2")])
    assert result["macro"] == 0.5
    assert result["expected_query_counts"] == result["query_counts"] == {"a": 1, "b": 2}


def test_one_missing_per_domain_fails_even_with_all_twelve_domains_represented():
    expected = {(domain, str(i)) for domain, count in BRIGHT_TEST_COUNTS.items() for i in range(count)}
    scores = [{"domain": d, "query_id": q, "ndcg_at_10": 1.0} for d, q in expected if q != "0"]
    with pytest.raises(ValueError, match=r"missing expected query IDs \(12\)"):
        summarize(scores, expected)


def test_same_domain_wrong_id_fails_even_with_matching_count():
    with pytest.raises(ValueError, match="unexpected query ID"):
        summarize([{"domain": "a", "query_id": "wrong", "ndcg_at_10": 0}], [("a", "right")])


def test_duplicate_id_does_not_mask_missing_row():
    row = {"domain": "a", "query_id": "1", "ndcg_at_10": 1}
    with pytest.raises(ValueError, match="duplicate query result"):
        summarize([row, row], [("a", "1"), ("a", "2")])


def test_domains_alone_cannot_supply_expected_membership():
    with pytest.raises(ValueError, match="pairs"):
        summarize([], ["ab"])


def test_explicit_partial_diagnostic_reports_omissions_and_separate_metric():
    result = summarize([{"domain": "a", "query_id": "1", "ndcg_at_10": 1}],
                       [("a", "1"), ("a", "2"), ("b", "1")], scope="partial_diagnostic")
    assert result["scope"] == "partial_diagnostic"
    assert result["expected_total"] == 3 and result["evaluated_total"] == 1
    assert result["expected_query_counts"] == {"a": 2, "b": 1}
    assert result["query_counts"] == {"a": 1, "b": 0}
    assert result["missing_ids"] == [["a", "2"], ["b", "1"]]
    assert result["partial_macro"] == 1 and "macro" not in result
