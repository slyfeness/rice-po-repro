"""Comparison contract tests with toy values only; no upstream reference values."""

import hashlib
import json
from pathlib import Path

import pytest

from ricepo import reference_compare as ref
from ricepo.data import content_sha256


def test_numeric_layer_uses_reference_magnitude_and_rejects_shape_changes():
    tolerance = {"atol": 0.01, "rtol": 0.1}
    within = ref._layer_result({"q": [1.09]}, {"q": [1.]}, tolerance)
    assert within["status"] == "matched"
    mismatch = ref._layer_result({"q": [0.1, 1.], "extra": [0.]}, {"q": [0.]}, tolerance)
    assert mismatch["status"] == "mismatch" and mismatch["mismatches"] == 3
    assert mismatch["max_abs_error"] == pytest.approx(0.1)
    assert ref._layer_result([False], [0], None)["status"] == "mismatch"
    assert ref._layer_result(["b", "a"], ["a", "b"], None)["status"] == "mismatch"


def _write(path, obj):
    data = json.dumps(obj, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_partial_external_fixture_reports_missing_layers_without_passing(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "test-only")
    monkeypatch.setattr(ref, "require_cluster_storage", Path)
    case = {"schema_version": 1, "domain": "toy", "document_ids": ["d"], "corpus_sha256": "0" * 64,
            "queries": [{"id": "q", "original_query": "q", "summary": None, "qrels": {"d": 1}, "excluded_ids": []}]}
    layers = {key: {"q": {"d": 1.}} for key in ref.LAYERS}
    layers.update(document_vectors={"d": [1.]}, query_vectors={"q": [1.]}, ranked_ids={"q": ["d"]},
                  ndcg_cut_10={"q": 1.}, token_ids={"documents": {"d": [1]}, "queries": {"q": [2]}},
                  attention_masks={"documents": {"d": [1]}, "queries": {"q": [1]}},
                  rendered_inputs={"documents": {"d": "d"}, "queries": {"q": "q"}})
    local = {"schema_version": 1, "role": "local_observed", "case": case,
             "case_sha256": content_sha256(case), "layers": layers}
    expected = {"schema_version": 1, "role": "upstream_expected", "case_sha256": local["case_sha256"],
                "source": {"repository": "https://github.com/AQ-MedAI/Diver", "revision": ref.DIVER_CODE_REVISION,
                           "retriever_revision": ref.MODEL_REVISION, "producer": "synthetic contract test",
                           "command": "fixture only", "encoding_profile": "toy"},
                "layers": {"ranked_ids": {"q": ["d"]}, "sparse_raw": {"q": {"d": 1.5}}},
                "tolerances": {"sparse_raw": {"rtol": 0, "atol": 0}}}
    local_path, expected_path = tmp_path / "local.json", tmp_path / "expected.json"
    _write(local_path, local)
    digest = _write(expected_path, expected)
    report = ref.compare(local_path, expected_path, digest)
    assert report["all_layers_matched"] is False
    assert report["layers"]["ranked_ids"]["status"] == "matched"
    assert report["layers"]["sparse_raw"]["status"] == "mismatch"
    assert report["layers"]["document_vectors"]["status"] == "missing_upstream_evidence"
    with pytest.raises(ValueError, match="pinned expected"):
        ref.compare(local_path, expected_path, "0" * 64)
    expected["source"]["revision"] = "unpinned"
    with pytest.raises(ValueError, match="pinned Diver"):
        ref.compare(local_path, expected_path, _write(expected_path, expected))


def test_same_observed_file_cannot_be_its_own_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "test-only")
    monkeypatch.setattr(ref, "require_cluster_storage", Path)
    path = tmp_path / "local.json"
    digest = _write(path, {"role": "local_observed"})
    with pytest.raises(ValueError, match="pinned expected"):
        ref.compare(path, path, digest)
