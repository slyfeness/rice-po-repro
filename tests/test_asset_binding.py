"""Source-authored fixtures only. No model assets or upstream scores are included."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ricepo import preparation as prep
from ricepo.asset_identity import local_model_identity, ordered_rows_sha256
from ricepo.artifacts import file_sha256
from ricepo.config import load_config
from ricepo.data import content_sha256
from ricepo.retrieval import DiverEncoder


def _snapshot(root):
    root.mkdir()
    for name in ("config.json", "tokenizer.json", "model-00001.safetensors"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"weight": "model-00001.safetensors"}}), encoding="utf-8")
    (root / "additional_chat_templates").mkdir()
    (root / "additional_chat_templates" / "audit.jinja").write_text("{{ messages }}", encoding="utf-8")
    return root


@pytest.mark.parametrize("name", ["model-00001.safetensors", "config.json", "tokenizer.json",
                                  "additional_chat_templates/audit.jinja"])
def test_same_location_and_name_cannot_hide_payload_changes(tmp_path, name):
    path = _snapshot(tmp_path / "snapshot")
    earlier = local_model_identity(path)
    (path / name).write_text("different bytes", encoding="utf-8")
    later = local_model_identity(path)
    assert earlier["payload_sha256"] != later["payload_sha256"]
    assert earlier["file_sha256"][name] != later["file_sha256"][name]
    assert later["upstream_verification"] == "not_verified"


def test_weights_and_tokenizer_must_exist_not_just_named_revisions(tmp_path):
    path = _snapshot(tmp_path / "snapshot")
    (path / "model-00001.safetensors").unlink()
    with pytest.raises(ValueError, match="missing or unbound"):
        local_model_identity(path)
    (path / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer payloads"):
        local_model_identity(path)


def test_order_hash_binds_both_rows_and_text():
    assert ordered_rows_sha256({"a": "x", "b": "y"}) != ordered_rows_sha256({"b": "y", "a": "x"})
    assert ordered_rows_sha256({"a": "x", "b": "y"}) != ordered_rows_sha256({"a": "y", "b": "x"})


@pytest.fixture
def encoding(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("SLURM_JOB_ID", "test-only")
    monkeypatch.setattr(prep, "require_cluster_storage", lambda p: Path(p))
    path = _snapshot(tmp_path / "snapshot")
    assets = local_model_identity(path)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/implementation_choices.json")
    config["execution"]["enabled"] = True
    encoder = DiverEncoder(SimpleNamespace(training=False), object(), device="cpu", max_length=8, document_prefix="")
    impl = {"hidden_size": 2, "document_prefix": "", "dtype": "float32", "max_length": 8}
    encoder._identity = {"assets": assets, "implementation": impl, "sha256": content_sha256([assets, impl])}
    encoder.encode = lambda rows, is_query: np.asarray([[1., 0.]] * len(rows), dtype=np.float32)
    monkeypatch.setattr(DiverEncoder, "from_local", classmethod(lambda cls, deployment: encoder))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"b","text":"B"}\n{"id":"a","text":"A"}\n', encoding="utf-8")
    target = tmp_path / "encoding-v2"
    spec = {"corpus_jsonl": corpus, "corpus_vectors_npy": target / "v.npy",
            "vector_document_ids_json": target / "ids.json"}
    deployment = {"document_prefix": "", "domains": {"fixture": spec}}
    return config, deployment, spec, encoder, target


def test_atomic_bundle_and_strict_load_detect_changed_weights_order_and_old_receipt(encoding):
    config, deployment, spec, encoder, target = encoding
    receipt = prep.encode_corpus(config, deployment, domain="fixture", batch_size=1)
    assert set(p.name for p in target.iterdir()) == {"v.npy", "ids.json", "v.receipt.json"}
    corpus, ids, vectors, audit = prep.load_vector_bundle("fixture", spec, config["model"], encoder)
    assert ids == list(corpus) == ["b", "a"]
    assert vectors.shape == (2, 2)
    assert audit["encoder_sha256"] == encoder.identity["sha256"]
    assert receipt["upstream_verification"] == "not_verified"
    with pytest.raises(FileExistsError):
        prep.encode_corpus(config, deployment, domain="fixture", batch_size=1)

    original_identity = encoder._identity
    encoder._identity = {**original_identity, "sha256": "f" * 64}
    with pytest.raises(ValueError, match="receipt mismatch"):
        prep.load_vector_bundle("fixture", spec, config["model"], encoder)
    encoder._identity = original_identity
    spec["vector_document_ids_json"].write_text('["a", "b"]', encoding="utf-8")
    receipt["document_ids_sha256"] = file_sha256(spec["vector_document_ids_json"])
    (target / "v.receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="order"):
        prep.load_vector_bundle("fixture", spec, config["model"], encoder)
    (target / "v.receipt.json").write_text('{"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="v2"):
        prep.load_vector_bundle("fixture", spec, config["model"], encoder)


def test_failed_publish_or_encode_never_exposes_a_partial_bundle(encoding, monkeypatch):
    config, deployment, _, _, target = encoding
    def fail(staged, destination):
        assert {p.name for p in staged.iterdir()} == {"v.npy", "ids.json", "v.receipt.json"}
        assert not destination.exists()
        raise OSError("injected publish failure")
    publish = prep._publish
    monkeypatch.setattr(prep, "_publish", fail)
    with pytest.raises(OSError, match="injected"):
        prep.encode_corpus(config, deployment, domain="fixture", batch_size=2)
    assert not target.exists()
    assert not list(target.parent.glob(".pending-*"))
    monkeypatch.setattr(prep, "_publish", publish)
    prep.encode_corpus(config, deployment, domain="fixture", batch_size=2)
    assert target.is_dir()


def test_compare_and_publish_rejects_existing_directory_even_if_empty(tmp_path):
    staged, target = tmp_path / "stage", tmp_path / "target"
    staged.mkdir()
    target.mkdir()
    (staged / "content").write_text("first", encoding="utf-8")
    with pytest.raises(OSError):
        prep._publish(staged, target)
    assert staged.exists() and not list(target.iterdir())
