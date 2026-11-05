"""Offline document encoding and consumption of immutable, directory-published bundles."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .artifacts import atomic_json, file_sha256, require_cluster_storage, require_slurm
from .asset_identity import ordered_rows_sha256
from .audit import attempt, operation, cost_report
from .config import require_execution
from .data import content_sha256, project_corpus_document
from .retrieval import DiverEncoder

MODEL_ID = "AQ-MedAI/Diver-Retriever-4B"
MODEL_REVISION = "ff7f8bc8d1827734dcf2bbfab99f065be618069a"


def _paths(spec):
    corpus = require_cluster_storage(spec["corpus_jsonl"])
    vectors = require_cluster_storage(spec["corpus_vectors_npy"])
    ids = require_cluster_storage(spec["vector_document_ids_json"])
    receipt = vectors.with_suffix(".receipt.json")
    if (ids.parent != vectors.parent or corpus.is_relative_to(vectors.parent)
            or vectors.suffix != ".npy" or ids.suffix != ".json" or ids == receipt
            or vectors.parent.name.startswith(".pending-")):
        raise ValueError("vectors, IDs and receipt require a dedicated version directory separate from the corpus")
    return corpus, vectors, ids, receipt


def _corpus(path):
    corpus = {}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                key, text = project_corpus_document(json.loads(line))
                if key in corpus:
                    raise ValueError("duplicate corpus document ID")
                corpus[key] = text
    if not corpus:
        raise ValueError("empty corpus")
    return corpus, digest.hexdigest()


def _check_model_spec(model_spec):
    if model_spec.get("retriever_id") != MODEL_ID or model_spec.get("retriever_revision") != MODEL_REVISION:
        raise ValueError("requires the pinned original Diver retriever specification")


def _receipt(domain, corpus, corpus_digest, vector_digest, id_digest, encoder, batch_size):
    if not isinstance(encoder, DiverEncoder):
        raise ValueError("a locally loaded DiverEncoder is required for a vector bundle")
    identity = encoder.identity  # Bound by from_local to the actual loaded snapshot and implementation.
    if identity["implementation"]["document_prefix"] != "":
        raise ValueError("raw-document profile requires an empty document prefix")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("missing or invalid encoding batch size")
    return {
        "schema_version": 2, "domain": domain, "documents": len(corpus),
        "corpus_sha256": corpus_digest, "ordered_rows_sha256": ordered_rows_sha256(corpus),
        "ordered_ids_sha256": content_sha256(list(corpus)),
        "vectors_sha256": vector_digest, "document_ids_sha256": id_digest,
        "model_id": MODEL_ID, "declared_model_revision": MODEL_REVISION,
        "encoder_identity": identity, "upstream_verification": "not_verified",
        "encoder_batch_size": batch_size, "vector_dtype": "float32",
        "vector_shape": [len(corpus), identity["implementation"]["hidden_size"]],
    }


def _sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _publish(staged: Path, target: Path) -> None:
    # All publishers use the same sibling lock. This supports shared filesystems
    # that implement flock/rename but reject renameat2(RENAME_NOREPLACE).
    if staged.parent.resolve() != target.parent.resolve():
        raise ValueError("publication requires sibling staging and target directories")
    _sync_directory(staged)
    if os.name == "nt":
        staged.rename(target)
    else:
        import fcntl
        with (target.parent / ("." + target.name + ".publish.lock")).open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if target.exists() or target.is_symlink():
                    raise FileExistsError(target)
                staged.rename(target)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    _sync_directory(target.parent)


def encode_corpus(config: dict, deployment: dict, *, domain: str, batch_size: int) -> dict:
    require_execution(config, "encode-corpus")
    require_slurm()
    _check_model_spec(config["model"])
    if deployment.get("document_prefix") != "" or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("encoding requires raw documents and an explicit positive batch size")
    corpus_file, vector_file, id_file, receipt_file = _paths(deployment["domains"][domain])
    target = vector_file.parent
    if target.exists():
        raise FileExistsError("choose a new, absent vector bundle directory")
    target.parent.mkdir(exist_ok=True, parents=True)

    log_root = target.parent / (target.name + ".encoding-audit")
    with attempt(log_root) as attempt_dir:
        try:
            with operation("asset_write",domain=domain,bundle=str(target)):
                return _encode(deployment, domain, batch_size, corpus_file, vector_file, id_file, receipt_file, target)
        finally:
            atomic_json(attempt_dir / "cost.json", cost_report(log_root))


def _encode(deployment, domain, batch_size, corpus_file, vector_file, id_file, receipt_file, target):
    encoder = DiverEncoder.from_local(deployment)
    import numpy as np
    corpus, corpus_digest = _corpus(corpus_file)
    ids = list(corpus)
    shape = (len(ids), encoder.identity["implementation"]["hidden_size"])
    staged = Path(tempfile.mkdtemp(prefix=".pending-", dir=target.parent))
    try:
        staged_vectors = staged / vector_file.name
        matrix = np.lib.format.open_memmap(staged_vectors, mode="w+", dtype=np.float32, shape=shape)
        try:
            for start in range(0, len(ids), batch_size):
                batch = ids[start:start + batch_size]
                rows = encoder.encode([corpus[key] for key in batch], is_query=False)
                if rows.shape != (len(batch), shape[1]) or rows.dtype != np.float32 or not np.isfinite(rows).all():
                    raise ValueError("encoder returned inconsistent rows, dimensions, precision or nonfinite vectors")
                matrix[start:start + len(batch)] = rows
            matrix.flush()
        finally:
            del matrix
        with staged_vectors.open("rb+") as handle:
            os.fsync(handle.fileno())
        ids_digest = atomic_json(staged / id_file.name, ids)
        receipt = _receipt(domain, corpus, corpus_digest, file_sha256(staged_vectors),
                           ids_digest, encoder, batch_size)
        atomic_json(staged / receipt_file.name, receipt)
        _publish(staged, target)
        return receipt
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def load_vector_bundle(domain: str, spec: dict, model_spec: dict, encoder: DiverEncoder):
    """Check the actual query encoder against the entire corpus encoding before scoring."""
    require_slurm()
    import numpy as np

    _check_model_spec(model_spec)
    corpus_file, vector_file, id_file, receipt_file = _paths(spec)
    if {p.name for p in vector_file.parent.iterdir()} != {vector_file.name, id_file.name, receipt_file.name}:
        raise ValueError("incomplete or unexpected contents of vector bundle directory")
    receipt = json.loads(receipt_file.read_text("utf-8"))
    if not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 2:
        raise ValueError("a complete v2 encoding receipt is required; rebuild legacy bundles")
    corpus, corpus_digest = _corpus(corpus_file)
    ids = json.loads(id_file.read_text("utf-8"))
    if ids != list(corpus):
        raise ValueError("vector row ID order differs from the original corpus")
    vector_digest, id_digest = file_sha256(vector_file), file_sha256(id_file)
    expected = _receipt(domain, corpus, corpus_digest, vector_digest, id_digest, encoder,
                        receipt.get("encoder_batch_size"))
    if content_sha256(receipt) != content_sha256(expected):
        raise ValueError("encoding receipt mismatch: corpus, vector, ordered rows or local encoder identity")
    vectors = np.load(vector_file, mmap_mode="r", allow_pickle=False)
    if list(vectors.shape) != expected["vector_shape"] or vectors.dtype != np.dtype("float32"):
        raise ValueError("matrix dimensions or dtype differ from bound encoder")
    for start in range(0, len(ids), 4096):
        norms = np.linalg.norm(vectors[start:start + 4096], axis=1)
        if not np.isfinite(norms).all() or not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
            raise ValueError("document matrix must contain finite, L2-normalized rows")
    return corpus, ids, vectors, {
        "source_sha256": {"corpus_jsonl": corpus_digest, "corpus_vectors_npy": vector_digest,
                          "vector_document_ids_json": id_digest},
        "encoding_receipt_sha256": file_sha256(receipt_file),
        "encoder_sha256": encoder.identity["sha256"], "encoder_batch_size": receipt["encoder_batch_size"],
        "ordered_rows_sha256": expected["ordered_rows_sha256"],
        "ordered_ids_sha256": expected["ordered_ids_sha256"],
        "upstream_verification": "not_verified", "document_count": len(ids),
        "vector_shape": list(vectors.shape),
    }
