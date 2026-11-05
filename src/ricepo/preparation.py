"""Offline corpus encoding stage, executable only inside an authorized Slurm job."""

import os
from .artifacts import atomic_json, file_sha256, require_cluster_storage, require_slurm
from .config import require_enabled
from .data import project_corpus_document, read_jsonl


def encode_corpus(config: dict, deployment: dict, *, domain: str, batch_size: int) -> dict:
    require_enabled(config)
    require_slurm()
    if (config.get("model", {}).get("retriever_id") != "AQ-MedAI/Diver-Retriever-4B"
            or config["model"].get("retriever_revision") != "ff7f8bc8d1827734dcf2bbfab99f065be618069a"
            or deployment.get("document_prefix") != ""):
        raise ValueError("encoding requires the pinned original Diver model and raw-document profile")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("explicit positive batch size required")
    spec = deployment["domains"][domain]
    corpus_path = require_cluster_storage(spec["corpus_jsonl"])
    vector_path = require_cluster_storage(spec["corpus_vectors_npy"])
    id_path = require_cluster_storage(spec["vector_document_ids_json"])
    receipt_path = vector_path.with_suffix(".receipt.json")
    if any(path.exists() for path in (vector_path, id_path, receipt_path)):
        raise FileExistsError("encoding artifacts already exist; choose a new versioned location")
    model_path = require_cluster_storage(deployment["retriever_model_path"])
    cache_path = require_cluster_storage(deployment["cache_root"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(cache_path / "huggingface")
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from .retrieval import DiverEncoder
    rows = [project_corpus_document(row) for row in read_jsonl(corpus_path)]
    ids = [key for key, _ in rows]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(key, str) or not key for key in ids):
        raise ValueError("empty corpus or invalid/duplicate document IDs")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[deployment["dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                      torch_dtype=dtype).to(deployment["retriever_device"]).eval()
    model.requires_grad_(False)
    encoder = DiverEncoder(model, tokenizer, device=deployment["retriever_device"],
                           max_length=deployment["retriever_max_length"],
                           document_prefix=deployment["document_prefix"])
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = vector_path.with_suffix(".pending.npy")
    # Exclusive reservation avoids replacing a concurrent or interrupted build.
    with temporary.open("xb"):
        pass
    matrix = None
    for start in range(0, len(rows), batch_size):
        vectors = encoder.encode([text for _, text in rows[start:start + batch_size]], is_query=False)
        if not np.isfinite(vectors).all():
            raise ValueError("nonfinite document vector")
        if matrix is None:
            matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                               shape=(len(rows), vectors.shape[1]))
        matrix[start:start + len(vectors)] = vectors
    matrix.flush()
    del matrix
    os.link(temporary, vector_path)
    temporary.unlink()
    ids_digest = atomic_json(id_path, ids)
    receipt = {"schema_version": 1, "domain": domain, "documents": len(rows),
               "corpus_sha256": file_sha256(corpus_path), "vectors_sha256": file_sha256(vector_path),
               "document_ids_sha256": ids_digest, "model_id": config["model"]["retriever_id"],
               "declared_model_revision": config["model"]["retriever_revision"],
               "encoder_dtype": deployment["dtype"], "vector_dtype": "float32",
               "local_model_payload_verified": False,
               "document_prefix": deployment["document_prefix"],
               "max_length": deployment["retriever_max_length"]}
    atomic_json(receipt_path, receipt)
    return receipt
