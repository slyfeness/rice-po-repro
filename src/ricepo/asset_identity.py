"""Content identities for immutable local HF snapshots. A SHA-256 of local bytes
establishes local consistency; upstream provenance requires independent evidence.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

from .artifacts import file_sha256
from .data import content_sha256


def package_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def source_sha256(*objects) -> str:
    payload = "\n".join(inspect.getsource(obj) for obj in objects).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def implementation_class(obj):
    cls = type(obj)
    return {"class": f"{cls.__module__}.{cls.__qualname__}",
            "module_sha256": file_sha256(inspect.getfile(cls))}


def local_model_identity(path: str | Path) -> dict:
    """Hash all payload files, including nested chat templates and shard indices.

    HF snapshot symlinks to blobs on storage are valid. Directory symlinks and
    paths to non-regular assets fail closed. Metadata folders are outside the
    payload; HF models are loaded only from a concrete local snapshot directory.
    """
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("expected a local model snapshot directory")
    storage = Path("/file_storage01/home")
    files = {}
    def raise_error(error):
        raise error
    for base, dirs, names in os.walk(root, onerror=raise_error):
        if Path(base) == root:
            dirs[:] = [d for d in dirs if d not in {".git", ".cache"}]
        if any((Path(base) / name).is_symlink() for name in dirs):
            raise ValueError("directory symlink inside the model snapshot")
        for name in sorted(names):
            item = Path(base) / name
            resolved = item.resolve(strict=True)
            if not resolved.is_file() or (root.is_relative_to(storage) and not resolved.is_relative_to(storage)):
                raise ValueError("model assets must resolve to regular files on storage")
            files[item.relative_to(root).as_posix()] = file_sha256(resolved)

    if "adapter_config.json" in files:
        raise ValueError("standalone model weights required; adapter can redirect the load to an external base")
    if "config.json" not in files or not any(key in files for key in (
        "tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt"
    )):
        raise ValueError("model snapshot requires config and tokenizer payloads")
    weights = {key for key in files if key.endswith((".bin", ".safetensors"))}
    indices = {"model.safetensors.index.json", "pytorch_model.bin.index.json"} & files.keys()
    if not indices and not ({"model.safetensors", "pytorch_model.bin"} & weights):
        raise ValueError("missing loadable HF model weights or shard index")
    for index in indices:
        weight_map = json.loads((root / index).read_text("utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or not set(weight_map.values()) <= weights:
            raise ValueError("weight index references missing or unbound shards")
    return {"schema_version": 1, "file_sha256": files, "payload_sha256": content_sha256(files),
            "upstream_verification": "not_verified"}


def ordered_rows_sha256(corpus: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for doc_id, text in corpus.items():
        # Newlines within either string are escaped by JSON.
        digest.update(json.dumps([doc_id, text], ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8") + b"\n")
    return digest.hexdigest()
