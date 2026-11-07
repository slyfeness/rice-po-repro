# Actual environment and deployment scope

Existing isolated RICE retrieval overlay and pinned local assets were reused. There was no package installation, model/data download, index build, new system proxy or environment modification in this phase.

| Component | Actual recorded value |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.5.1 |
| NumPy | 1.26.4 |
| Gensim | 4.3.3 |
| Pyserini | 0.22.1 |
| Tokenizers | 0.22.2 |
| Safetensors | 0.8.0 |
| pytrec_eval | Module reports 0.5.11; distribution metadata is absent in the overlay, not a new compatibility lock |
| JVM | Existing Temurin 21.0.12.1+1 |
| GPU | Main training and workload jobs actually received A800-SXM4-80GB devices |
| Driver / CUDA runtime | 580.65.06 / 12.8 |

Each training job uses one process and **two Slurm-assigned devices**, with independent actor/reference/optimizer/output state per method. Two such allocations may overlap up to four GPUs; no physical ID or shared node is assumed. Each requests 16 CPUs and 192G host memory. Dedicated final reload and long-history probability checks use one GPU. GPU allocation timing, failures, initialization, checkpoint I/O and evaluation all count in the actual GPU-hour accounting. The original32 GPU-hour stopping cap was explicitly removed by the user during this phase; maximum concurrent GPUs remains4. CUDA allocator peaks are not total device occupancy; see actual resource evidence.

The frozen behavior is FP32 Actor in eval mode, T=1, top-p=1, top-k=0, identical special-token rules and a 4096-model-token action budget shared by Think and Summary. The model remains in eval mode during probability scoring/training so dropout behavior matches sampling. Existing explicit layer activation recomputation was observed through execution counters in update diagnostics, rather than inferred from a flag. Tool, role and deterministic opener tokens are context with zero direct loss, not detached history.

Training originally executes immutable package `e70e6b2` for GRPO. RICE recovery executes `534b853`; the preceding failed attempt retains `e70e6b2`. Independent batch workload uses `1f42d2b`; long-history real-weight-switch gate uses `534b853`. These are distinct recorded source identities. Later reports do not rewrite the source fingerprint of an earlier run. vLLM conditional-history inference was subsequently exercised with CUDA Graphs and explicit IEEE FP32; four-rank FSDP and complete vLLM/batched-Trainer PPO remain unqualified.

The CPU analysis/tests run within Slurm as well. Public-chart rendering, if listed in `CURVE_PROVENANCE.json`, is presentation of already-redacted aggregate JSON on the workstation using its existing Matplotlib; it does not read private cluster trajectories or execute any model, data processing pipeline or GPU experiment.

CPU174836 also imported the pre-existing vLLM0.19.1 available through the overlay's existing package path. Its distribution metadata reports Torch2.10.0, while the previously recorded imported CUDA build is2.10.0+cu128; metadata and actual CUDA-build reporting are distinct. The inventory alone does not qualify GPU execution; later actual conditional-history GPU gates are reported separately. See `VLLM_STATUS.md`.
