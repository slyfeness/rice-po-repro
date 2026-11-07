# TF32 inference-only experiments

These opt-in drivers reuse local, identity-bound assets. They never create an
optimizer, download assets, or invoke Trainer. Run cluster computation through
Slurm. Keep native requests, outputs and logs private on bulk storage.

Supply `RICE_PROJECT_ROOT`, `RICE_BENCHMARK_ROOT` (containing the frozen private
`MANIFEST.json`), and a short `RICE_RPC_ROOT` on bulk storage. Retrieval drivers
also require `RICE_PARENT_RUN`, the previously bound deployment/recipe/binding
directory, and the existing JVM environment. Every run uses a new job-ID output
directory; an existing directory is rejected.

Actual resource profiles in this experiment were one GPU / 6 CPU / 90G / 30min
for screens and co-residency; two GPU / 16 CPU / 192G / 45min for the dual
comparison; and four GPU / 24 CPU / 384G / 30min for pure inference. All use one
node and one Slurm task, with independent spawned TP1 workers. Supply the
matching `sbatch --gres`, CPU, memory and time overrides when using the common
template. Account, partition and output/error roots come from private bindings.

- `tf32_runtime_screen.py S2 screen`: 144 saved conditions per stage, cold then
  hot cache, at most 512 generated tokens per condition.
- `tf32_runtime_screen.py S2 confirm`: the complete fixed pool, followed by 16
  native two-stage actions with the shared 4096-token model-action budget.
- `tf32_coexist_isolated.py`: one GPU; Diver/JVM stays in the coordinator and a
  separate immutable TP1 frontend uses 0.45 GPU memory fraction.
- `tf32_dual_engine.py`: two Slurm GPUs; same complete pool with one and two
  workers plus actual saved-input retrieval. Then a small real rollout/branch
  diagnostic and restart both replicas from an existing updated snapshot. It
  performs no new PPO update. The branch probe uses the first actual legal
  Summary, explicitly **not** entropy selection or a credit experiment.
- `tf32_four_workers.py`: four independent TP1 workers, one coordinator clock,
  selected S2 (144/32768/0.90), cold/hot pure inference. It is not a distributed
  training implementation.

The manifest contains `models.initial`, `models.updated`, `requests`,
`retrieval_requests`, `candidates`, `screen.requests_per_stage` and
`full_pool.requests_per_stage`. Each condition retains its `request_id`,
`group_key`, native `prefix_token_ids`, `stage`, actual `seed`, `root_seed` and
`remaining_token_budget`. Saved retrieval requests retain their query identity,
composed query, fixed exclusions, expected ranking IDs and NDCG. Public results
include a manifest digest and coverage, never these raw fields.

The original fixed-manifest validator remains available with its original
length-bin requirements. The present real source pool only reaches 11,689
prefix tokens: report this limitation rather than inventing longer histories
to satisfy that older benchmark's three length bins.

Native per-request queue/prefill/decode intervals overlap. They are latency
measurements, not exclusive GPU-kernel accounting. NVML peaks are sampled at
0.2-second intervals. Unknown measurements remain `NOT_MEASURED`. A hot-cache
replay is a benchmark condition, not an independent policy sample or a claim
that an online workload will achieve the same cache hit rate.

Private cluster wrappers used the same experiment bodies with fixed local
bindings. These public versions substitute explicit environment variables for
private paths; execution-evidence digests identify the private originals.
The published helpers additionally preserve the older manifest API and mark
ambiguous source-labelled metric reductions `NOT_MEASURED`. Historical
top-line throughput uses actual native output counts and coordinator time,
not those labelled counters.

`docs/reviews/tf32-swap-throughput-v1/BEST_RUNTIME_CONFIG.json` is an optional
inference specification, not a Trainer configuration. Its `runtime` object can
be passed directly to `VLLMEngine(runtime_options=..., numeric_mode=spec['numeric_mode'],
max_context_tokens=spec['max_model_len'])`, together with the verified local
model path, tokenizer and payload digest. For the two-card layout use each
`measured_dual_layout` worker's nested `runtime` and separate `co_resident`
argument. Sampling/stopping constants remain enforced by `VLLMEngine.sample`
and the two-stage adapter. This is an API mapping, not an automatic launch;
the public specification intentionally retains `execution.enabled=false`.
