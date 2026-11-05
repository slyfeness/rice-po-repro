# Acceptance tiers

These tests and procedures are source-only. No pytest invocation, framework import, model load, Slurm job or numerical comparison was performed in this phase.

| Tier | Entry point | Evidence it can produce |
|---|---|---|
| CPU contracts and toy tensors | `tests/test_rollout.py`, `tests/test_formats.py`, `tests/test_backends.py`, `tests/test_native_evidence.py` | Native spans and generated masks, saved branch prefixes and independent continuations, repeated IDs and repeated document legality, old-policy identity, strict format failures, request IDs and seeds. The new adapter tests use deterministic stub engines through the actual HF/vLLM adapters. The HF case requires installed torch and Transformers. |
| Small real tokenizer | Opt-in tokenizer fixture in `tests/test_opt_in_assets.py` | Exact chat-template and tool observation IDs, preserved native completion IDs, accepted relative spans and expected rejection categories with the supplied real tokenizer. |
| Small real encoder/retriever case | `python -m ricepo.reference_compare` | Input IDs, vectors, component scores, ranks and NDCG against independently pinned upstream expectations. See [retrieval comparison](RETRIEVAL_AUDIT.md). |
| One real GPU and one causal LM | Opt-in entropy fixture in `tests/test_opt_in_assets.py` | Chunked KV-cache entropy versus an independent full-sequence forward at each causal offset, including prompt positions. Both paths use the supplied model, dtype and native IDs. Token zero has an undefined preceding distribution. |
| Two-device reference smoke | Future, separately authorized acceptance through the train and evaluate commands | Finite nonzero optimizer update, checkpoint and restart consistency, fresh-policy rollout, fixed retrieval identity, numerical drift, memory high-water marks and completed stage receipts. Requires frozen data, vectors, local models and a compatible environment. No end-to-end result is available. |
| Distributed program | Future implementation and acceptance | Data partitioning, rank synchronization, loss/gradient equivalence with a one-process run, checkpoint recovery and memory per rank. The current single-process driver rejects a multi-process launch. Four allocated GPUs alone supply no distributed-program evidence. |

The CPU evidence tests check a raw event on disk at parser entry. For each adapter they check the exact request ID, prefix IDs and digest, returned IDs, seed, policy, finish reason, token counts and recorded failure category. The deterministic toy rollout also compares original trajectories and shared branch trajectories under reversed query order and different branch counts. It reuses the seed generator and branch history helpers from the rollout suite. It does not validate numerical equivalence across accelerator kernels.

## Supplying opt-in assets

Both opt-in cases require a Slurm allocation. They skip when their environment-variable pair is absent. Providing only half a pair, an incorrect digest, a missing asset, or an incompatible requested dependency fails the requested case. Each pair consists of a path to a small JSON fixture and the independently pinned SHA-256 of its exact bytes. The case sets `schema_version` to 1 and names a storage-backed `cache_root`. The tests set offline caches before loading any model packages. Keep fixture inputs and pytest temporary/cache directories on storage during a future authorized invocation.

**Tokenizer.** Use `RICEPO_TOKENIZER_FIXTURE` and `RICEPO_TOKENIZER_FIXTURE_SHA256`. Supply a concrete tokenizer-only `tokenizer_directory`, an exact `file_sha256` map with relative paths, and the `system_prompt` path and its `system_prompt_sha256`. The case's `cases` list must include at least one accepted completion and one known format failure. Each row includes:

- `prefill_think`, `observation_top_k`, `observation_token_limit`, the original `query` and an ordered `documents` array with `id` and `text`.
- Independently captured `initial_ids`, `observation_ids`, `completion_ids` and `finish_reason`. Capture the engine's actual native completion IDs. Re-encoding decoded prose would change the test's evidentiary value.
- For an accepted output, `reasoning_span`, `summary_span` as half-open pairs and `summary_text`. For a rejection, `error_code` with the expected parser category. Cases may cover both prefilled and explicit opening-tag modes if compatible with the deployed template.

**Single GPU.** Use `RICEPO_GPU_ENTROPY_FIXTURE` and `RICEPO_GPU_ENTROPY_FIXTURE_SHA256`. Supply `model_directory`, `payload_sha256` from the pinned local file manifest, a concrete CUDA `device`, `dtype` (`float32` or `bfloat16`), `token_ids`, boolean `generated_mask`, `chunk_size`, `atol` and `rtol`. Choose the tolerances before comparing. The short case must contain 3–512 native IDs, include a generated token after an initial non-generated token, and cross a KV-cache chunk boundary. A positive numerical result would apply to that model, hardware, software and case only.
