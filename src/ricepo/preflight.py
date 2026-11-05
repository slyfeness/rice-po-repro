"""Future opt-in environment inventory. No installation or measurement at import."""
from importlib.metadata import PackageNotFoundError, version
import platform
import subprocess
import sys
from .artifacts import require_slurm, require_cluster_storage, atomic_json
from .config import require_enabled
from .environment import configure_caches
from .resource_estimate import estimate_resources


def inventory():
    packages = {}
    for name in ("torch","transformers","numpy","tokenizers","safetensors","pytrec-eval",
                 "gensim","pyserini","pyjnius","pytest","vllm"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    def probe(args):
        try:
            p = subprocess.run(args,capture_output=True,text=True,timeout=15,check=False)
            return {"exit_code":p.returncode,"output":(p.stdout+p.stderr).strip()[:2048]}
        except (OSError,subprocess.TimeoutExpired) as exc:
            return {"unavailable":type(exc).__name__}
    gpu = probe(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"])
    cuda = {}
    try:
        import torch
        cuda = {"torch_cuda":torch.version.cuda,"visible_count":torch.cuda.device_count()}
    except (ImportError,OSError,RuntimeError) as exc:
        cuda = {"unavailable":str(exc)}
    return dict(python=sys.version,platform=platform.platform(),packages=packages,torch_cuda=cuda,
                driver_inventory=gpu,jvm=probe(["java","-version"]))


def preflight(config,deployment,output):
    require_enabled(config)
    job = require_slurm()
    configure_caches(deployment["cache_root"])
    assumptions = deployment.get("resource_assumptions")
    report = {"slurm_job_id":job,"environment":inventory(),
              "static_resources":estimate_resources(**assumptions) if assumptions else None,
              "resource_assumptions_provided":assumptions is not None,
              "runtime_validated":False,"four_gpu_validated":False}
    atomic_json(require_cluster_storage(output),report)
    return report
