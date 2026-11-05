"""Configure offline caches before importing model, sparse-index or JVM packages."""

import os
from pathlib import Path
import tempfile

from .artifacts import require_cluster_storage


def configure_caches(cache_root: str | Path) -> None:
    root = require_cluster_storage(cache_root)
    locations = {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "huggingface/hub", "HF_DATASETS_CACHE": "datasets",
        "HF_MODULES_CACHE": "modules", "TRANSFORMERS_CACHE": "transformers",
        "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "extensions",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "PYSERINI_CACHE": "pyserini", "GENSIM_DATA_DIR": "gensim",
        "NUMBA_CACHE_DIR": "numba", "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
    }
    for variable, suffix in locations.items():
        destination = require_cluster_storage(root / suffix)
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(destination)
    tempfile.tempdir = os.environ["TMPDIR"]
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    # java.io.tmpdir governs Lucene's temp files and native library extraction.
    option = f'"-Djava.io.tmpdir={tempfile.tempdir}"'
    options = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if not options.endswith(option):
        os.environ["JAVA_TOOL_OPTIONS"] = f"{options} {option}".strip()
