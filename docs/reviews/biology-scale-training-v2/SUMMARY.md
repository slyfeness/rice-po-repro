# Biology scale v2: stopped at memory preflight

The explicitly requested GPU1 longest-history test failed with CUDA OOM in backward. No four-query smoke or new 128-query training was started. The old runs remain cancelled and untouched.

| Observed item | Result |
| --- | --- |
| Slurm job | 177080, FAILED, exit 1:0 |
| Allocation elapsed | 149 s, two GPUs, 0.082778 allocated GPU hours |
| Saved history | horizon 16,947; 3,411 action tokens |
| Compute | FP32 Actor/Reference, HF TF32 GEMM, action-position full-vocabulary projection |
| Memory policy | selective CPU saved tensors: device only if horizon ≤5,535 and actions ≤1,889 |
| Resident retrieval model | Diver on logical GPU1, idle during HF |
| HF-interval NVML memory | mean 63,115.45 MiB; sampled peak 80,927 MiB |
| HF-interval utilization | mean 69.58%, peak 100% |
| HF-interval power | mean 211.85 W, peak 415.72 W |
| Optimizer steps / new checkpoints | 0 / 0 |

The failing backward allocation requested 12.84 GiB with 12.80 GiB free, according to rounded CUDA diagnostics. This identifies a capacity failure in the requested co-resident layout. It does not prove that removing a few MiB would make a complete update safe. Completed HF latency and gradient statistics are unavailable because backward did not finish.

The test used the existing saved P0 history, native IDs, action mask, q and advantages. It did not generate tokens, retrieve documents, alter rewards, resume old training or execute optimizer.step. The base runtime was commit `68c1b4aca1938ffe172b0237e2016a47135f3842`; the executed probe is published as `scripts/biology_v2_memory.py`, byte-identical to the executed copy. Private result, error-log and script digests are recorded in MEMORY_PREFLIGHT.json. Raw logs and assets remain private.

Reproducible invocation with private bindings redacted:

```sh
# Within the two-GPU Slurm allocation; existing environment and local assets only.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_F32_DEFAULT=ieee \
python scripts/biology_v2_memory.py --plan "$PRIVATE_SAVED_PLAN" \
  --assets "$PRIVATE_ASSET_PLAN" --output "$NEW_PRIVATE_OUTPUT"
```

The two-rank HF update, dual-engine dispatch and v2 recipe declarations are source-only additions in this review branch. A post-review correction makes DP entropy use the actual frozen behavior-policy identity; its added regression test is not yet executed. A second post-review correction makes the known DP-rank1/isolated-Diver GPU1 placement fail closed before model loading. The v2 recipe declarations are not wired to a runnable scale launcher, deliberately: the required four-device mapping and a safe resource topology do not exist in this implementation.

CPU-only Slurm job 177208 completed before those two corrections: 10 HF-DP tests, 3 prefix-affinity tests and both static v2 recipe expansions passed. It does not validate NCCL, model residency, vLLM, the post-review corrections, or the OOM-prone GPU topology. The preceding CPU job 177205 failed before importing project code because `python` was absent from the job path; the corrected job explicitly used the existing project interpreter.

No production configuration is recommended or activated. The next decision is how to reduce GPU1 residency or backward peak without changing the mathematical objective; this report does not silently move Diver, shorten the history or retry another configuration. Per the user's OOM stop condition, GPU expansion and automatic scale launch stop here.

Old GRPO retains computed/persisted 2/2; old RICE retains 1/0 and an Actor-only P1 snapshot, which is not a complete Adam/RNG recovery point. These progress counts come from the preserved prior evidence. Current sacct confirms both old jobs CANCELLED; squeue access was denied, so no broader queue-visibility claim is made.
