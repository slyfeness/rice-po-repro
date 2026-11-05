# Dependency status

No dependency installation, package import probe, model initialization, or runtime compatibility check was performed for this source-only delivery. Existing cluster environments must be inspected read-only and reused after a separately authorized experiment stage.

| Component | Required existing dependency |
|---|---|
| Python records/config/data/credit | Python 3.11+ standard library |
| Objective, modeling, updates | PyTorch |
| HF reference runtime and Diver encoding | Transformers, PyTorch, NumPy; model tokenizer dependencies already present |
| Dense vectors | NumPy, precomputed `.npy` matrix and ordered ID file |
| Upstream-equivalent BM25 | Gensim `LuceneBM25Model`, Pyserini and a compatible existing JVM |
| Reward | pytrec_eval |
| Optional generation adapter | vLLM, supplied engine and SamplingParams factory |
| Authored tests | pytest; tensor fixtures additionally require PyTorch |

`pyproject.toml` intentionally does not auto-install large runtime stacks. The pinned Diver requirements file under `third_party/reference/` is provenance material, not a claim that its full stack is suitable for this project or an instruction to alter a shared environment. No fabricated environment lock is supplied. Before a future run, record exact Python/package/CUDA/driver/JVM versions, source commit, asset revisions, tokenizer/chat-template identity, and Slurm allocation in the storage-backed run directory.

The reference runtime uses local asset paths, `local_files_only=True`, and `trust_remote_code=False`; unsupported assets should fail rather than execute remote model code or download a substitute.


## Environment evidence for this checkout

Deploy the source checkout with its `src`, `configs` and `prompts` directories together. A standalone installed wheel is outside this deployment contract. The gated preflight entrypoint records Python, installed package metadata, PyTorch's CUDA version and visible-device count, driver inventory and JVM output. Missing probes receive their own status.

Preparation, retrieval comparison, training and evaluation configure caches before heavy framework/JVM imports. The reference Slurm template requests two GPUs for its two-device topology. A future cluster run must verify the environment, filesystem primitives and memory. Dependency compatibility and peak measurements are pending.
