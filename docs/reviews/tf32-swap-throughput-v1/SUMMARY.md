# TF32 SwAP-style scheduling: measured decision

For dedicated inference cards select **S2: 144 sequences, 32768 batched tokens, 0.90 memory fraction**. The measured two-engine/Diver layout retains S0 (64/16384), with 0.80 on the first card and 0.45 on the co-resident card. These remain optional inference deployments; the verified training configuration is unchanged. No optimizer, credit estimation, development evaluation or training run was executed in this phase. Delivery is GitHub-only; no archive was created.

| Candidate | Cache | Tokens | Wall seconds | token/s | Sampled peak MiB |
|---|---|---:|---:|---:|---:|
| S0 (job 175677) | cold | 54429 | 140.13 | 388.43 | 66885 |
| S0 (job 175677) | hot | 54419 | 32.47 | 1675.88 | 66885 |
| S0 (job 175688) | cold | 54452 | 139.42 | 390.55 | 66885 |
| S0 (job 175688) | hot | 54096 | 32.42 | 1668.54 | 66885 |
| S1 (job 175678) | cold | 53889 | 137.04 | 393.23 | 67349 |
| S1 (job 175678) | hot | 53660 | 32.17 | 1668.08 | 67349 |
| S2 (job 175679) | cold | 53438 | 137.14 | 389.67 | 75519 |
| S2 (job 175679) | hot | 53928 | 31.80 | 1695.65 | 75519 |
| S2 (job 175700) | cold | 53893 | 147.92 | 364.35 | 75519 |
| S2 (job 175700) | hot | 53928 | 31.85 | 1692.94 | 75519 |

All three candidates reached their configured concurrency in sampled gauges (S2 hot reached 144); no preemption was recorded. S2 repeated its small hot-cache advantage, while its cold repeat was slower. Its sampled device memory was about 8.4 GiB higher than S0. S0 also repeated consistently and offers more headroom. The S2 choice targets dedicated cards; do not transfer its 0.90 memory fraction to the Diver card. Output token counts differ and are reported, not normalized away.

The complete identical 1152-condition selected S2 single-GPU pool produced 225745 tokens in 366.44s cold (616.05 token/s), and 225472 in 287.60s hot (783.98 token/s). This larger pool does not reproduce the small pool’s nearly 99.8% hot prefix-cache fraction. Do not advertise the small-pool hot rate as online RICE throughput.

## Bottleneck evidence

Native queue, prefill, decode, TTFT and TPOT distributions are in BOTTLENECK_TIMING.csv; initialization, compile, graph capture and profiling durations are separate. Request intervals overlap. Exclusive device-kernel time, hardware-level concurrent kernels and an exclusive global prefill/decode percentage are NOT_MEASURED. No token-ratio estimates were used.

The following are **mean per-request seconds**, weighted by the actual measured request counts; TPOT excludes single-token outputs. They are not an exclusive GPU-time decomposition.

| Full pool | Queue | Prefill | Decode | TTFT | TPOT |
|---|---:|---:|---:|---:|---:|
| Single S2 cold |81.582|3.123|35.708|87.108|0.2491|
| Single S2 hot |56.271|2.844|29.420|59.541|0.2280|
| Four S2 cold |20.995|9.833|22.623|32.569|0.2206|
| Four S2 hot |0.021|0.348|8.275|0.779|0.0469|

Long-prefix cold processing and queueing materially affect makespan. Exact hot replays reduce both prefill latency and decode interference on the smaller pool. Increasing the sequence cap alone did not remove this cost. Lifecycle overhead is measurable; retaining one immutable engine per sampling phase is preferable to repeated initialization, but no end-to-end Trainer speedup is claimed.

## Diver and second engine

The same-process probe failed with a native segfault diagnostic (job 175680, Slurm exit11:0), without established OOM evidence. A single bounded retry separated JVM/Retriever and vLLM frontends and isolated their cache roots; it passed (175686). These two changes do not prove which caused the crash. Diver allocated 7.54 GiB; the co-resident observed peak was 46783 MiB. 48 actual retrievals retained exactly the saved ranks and NDCG, taking 2.986s alone and 3.496s with generation (1.171x latency). This is a fixed 48-request burst, not evidence of sustained all-round concurrent kernel execution.

Same full pool plus 48 real retrievals: one engine 227324 tokens / 399.12s (569.56 token/s); two engines cold 227433 / 253.49s (897.22 token/s). Makespan speedup 1.575x; token-rate speedup 1.575x. Status: PASS.
Two-engine hot replay: 226113 tokens / 233.78s (967.20 token/s).

Real minimal protocol/retrieval and fixed-prefix branch outcomes are in DUAL_ENGINE_RESULTS.json, separately from the offline-prefix speed test. Invalid actions remain terminal failures; none were repaired or resampled. This diagnostic intentionally performs no entropy selection, credit calculation or PPO.

The small online diagnostic took52.36s and ran its two fixed queries sequentially, with five-branch batches. It is a mapping/protocol check, not the complete Trainer wavefront scheduler and not a dual-engine online training throughput result.

Both replicas were restarted from the existing real updated checkpoint, with new IPC/cache lifetimes and identical new payload identities; the old and new payloads differ. No checkpoint or optimizer state was rewritten.

## Four GPUs

- cold: 227348 actual tokens / 131.81s shared wall = **1724.80 token/s**; same-pool makespan speedup 2.780x, token-rate scaling efficiency 70.0%.
- hot: 226845 actual tokens / 33.44s shared wall = **6784.46 token/s**; same-pool makespan speedup 8.602x, token-rate scaling efficiency 216.3%.
All-worker initialization: 187.64s; initialization plus cold pool: 319.45s. Worker loads, individual rates and memory are in FOUR_GPU_RESULTS.json. This is measured pure inference, not four-GPU training. Cache capacity/affinity and output lengths can affect scaling; it is not a compute-only linearity measurement.

The single-card full hot pool cached 77.61% of input tokens; the four smaller worker pools cached approximately99.8%. The 8.6x hot makespan improvement therefore combines four GPUs with a substantially different cache regime. It does **not** demonstrate greater-than-linear compute scaling. Cold token-rate scaling was2.80x/70% efficiency. The largest observed four-card device peak was76583MiB, leaving about5.21GiB on that80GiB device; this is measured headroom for these requests, not a full32K/144-concurrent-action guarantee.

## Correctness, limits and next use

FP32 weights, Triton TF32, TP1, actual EOS/stop tokens, context32768 and T=1/p=1/k-disabled were retained. The 512-token screen is not a 4096-token action certificate. LONG_CONFIRMATION.json records the separate shared4096 two-stage test and masks. Whole-card NVML peaks are sampled, not absolute maxima.

The existing TF32 restricted-pilot numerical/gradient/update evidence was reused as requested. This phase did not repeat PPO calibration, relax its thresholds, qualify BF16, or certify TF32/IEEE backend equivalence. Actual native sampling LPs were retained for every new request. Two/four immutable engine deployment is optional; formal Trainer integration, update-time memory residency and sustained online workload performance remain to be accepted separately.

TRITON_F32_DEFAULT=tf32 does not certify that every FP32 GEMM uses TF32. Native compiler logs also warn that PyTorch FP32 matmul TF32 is not enabled. Per-kernel arithmetic was not profiled and no global PyTorch matmul setting was changed to chase speed.

SwAP’s historical 6339 token/s/GPU used a different precision and workload. The present small hot pool is roughly 26% of that reference, while the complete RICE pool is substantially slower. This is a workload/precision comparison, not evidence of a specific causal speed loss or a promised target rate.

No 32/128-query training or automatic restart follows this result. See SLURM_JOBS.csv for every actual job/terminal state and EVIDENCE_INDEX.json for private artifact/source digests. No raw query, model, trajectory, checkpoint, credentials, account or host identifiers are published.

Actual phase allocation cost (all GPU attempts): **2.402500 GPU-hours**. All phase jobs are terminal; no training was started. See CORRECTNESS_CHECKS.md for the eight actually executed CPU tests, retained invalid actions, hardware/measurement limits, version-switch evidence and sanitized commands.
