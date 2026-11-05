# Single-process local reference runtime

`ricepo.hf_runtime.run(config, deployment)` implements the full training loop using prepared local assets. It does not download weights or data and does not construct an experiment budget or dataset split. Both CLI and direct API require explicit execution enablement. The driver additionally requires a Slurm allocation, valid research configuration, one process, two explicit CUDA devices, and storage under `/file_storage01/home` for models, data, outputs, and caches.

**No runtime tests, imports, model launches, builds, or GPU checks have been performed.** This is source awaiting validation. It is not an optimized four-rank driver. The actor and an independent frozen reference occupy `actor_device`; Diver occupies `retriever_device`. Full-sequence policy/reference logits and the current-policy graph can consume substantial memory. Chunked entropy normalization does not establish training-memory feasibility.

## Required local inputs

Copy `configs/deployment.example.json` and replace placeholders. Its numeric resource settings are explicit reference implementation choices, not measured capacity guarantees. `train_manifest_path`, `development_manifest_path`, and `expected_development_domains` are intentionally unresolved. The research config's split paths must resolve to the same files as the deployment. Missing or empty manifests are errors. No sample is claimed to be an author's unpublished split.

Each split is JSONL with this schema:

```json
{"query_id":"q1","domain":"biology","query":"Actor-visible query text","qrels":{"doc-7":1},"excluded_ids":["doc-12"]}
```

Only query text, domain/identity, and fixed exclusion IDs are projected into `QueryExample`. Qrels remain in the offline reward scorer. They are never passed to the prompt renderer, generator, or query encoder. The loader checks duplicate identities, duplicate normalized query text across splits, nonnegative integer judgments including at least one positive judgment, and domain membership. Qrels documents must exist in the matching corpus; positively judged documents cannot be fixed exclusions.

For every domain, supply:

- `corpus_jsonl`: native BRIGHT rows `{"id":"doc-7","content":"Document text"}` or normalized rows with `text` in place of `content`. Corpus preparation, training and holdout use the same projection. IDs must be unique, and if both content fields are present they must match.
- `corpus_vectors_npy`: a floating-point NumPy matrix containing the precomputed Diver corpus vectors. It is memory-mapped and never generated implicitly.
- `vector_document_ids_json`: a JSON list mapping each vector row to its corpus document ID. It must cover the complete corpus exactly once.
- `backend_version`: a pinned deployment identity. Source hashes of corpus, vectors, and vector-row IDs are saved independently.

The `.npy` vector matrix must have the `encode-corpus` sidecar next to it: `vectors.npy` requires `vectors.receipt.json`. Training and holdout both read that sidecar and recompute the corpus, vector, and row-ID hashes. They also require matching domain, document count, configured retriever ID/revision, document prefix, maximum encoder length, encoder precision, and actual float32 vector dtype. Externally prepared vectors must supply the same encoding-receipt schema; the driver neither synthesizes a missing receipt nor presents a declared revision as verified upstream model payload.

Actor and retriever directories must already contain supported Hugging Face model/tokenizer files. Loading uses `local_files_only=True` and `trust_remote_code=False`. Existing torch, Transformers, NumPy, pytrec_eval, Gensim, Pyserini, and JVM dependencies are required. The runtime does not install them. Transformers must expose `generate(use_model_defaults=False)`. Models requiring custom remote code are unsupported by this reference path.

The deployment prompt must match the content hash of `prompt_artifact` in the research config. It may be copied to an explicit control path. Prompting and native parsing both enable prefilled `<think>` handling. Empty summary handling remains the format-policy limitation described in [ROLLOUT_CONTRACT.md](ROLLOUT_CONTRACT.md).

## Sampling and optimization semantics

The reference driver supports temperature 1, top-p 1, top-k 0, no repetition penalty, no beam search, no logit constraints, and a fresh generation configuration with model defaults disabled. It rejects a research configuration requesting a different sampling law. This makes cached teacher-forced old log probabilities correspond to the full model distribution used to sample actions. Termination occurs on `</summary>` or native EOS/length limits; incomplete or malformed output raises, and receives no fabricated reward.

The actor is always in eval mode to disable dropout. Rollout and entropy use inference mode; current-policy scoring during PPO enables gradients. Eval mode does not prevent differentiation. The frozen reference is a separate copy of the initial actor with gradients disabled and remains unchanged across updates.

Actor and reference parameters remain in FP32. This lets AdamW accumulate small `1e-6` updates without rounding each parameter change immediately to BF16. With deployment `dtype: bfloat16`, actor/reference calls use BF16 autocast and the fixed retriever may reside in BF16; `float32` disables autocast. Each policy forward or generation gets a fresh autocast scope, so no cached low-precision weight crosses an optimizer update. The additional FP32 master-parameter and optimizer memory is part of the unverified capacity requirement.

The paper's batch size 128 is interpreted here as **128 queries per optimizer update**, an explicit implementation choice. Every update samples 128 distinct queries without replacement from the training manifest; queries may recur in different updates. At least 128 training queries are required. For each query the driver generates the configured N original trajectories, pools all their summary actions, selects the four highest-entropy anchors, and collects the configured total number of branches including the original. The main profile interprets the appendix's reported five as five *new* branches, for six total. A separately documented sensitivity configuration may instead specify five total (four new plus the original). No branch-count setting is silently overridden.

Every original and counterfactual reaches five retrieval steps. The initial evidence D0 comes from retrieval with the original user query. Subsequent queries follow the config's `summary_only` or `original_and_summary` composition exactly. Dataset exclusions stay fixed per query; earlier retrieval results may recur. Final reward is the last summary's fractional NDCG@10.

The old actor weights remain unchanged throughout rollout, entropy, branching, and credit assignment. All original-trajectory old log probabilities are cached before the first optimizer mutation. Counterfactual trajectories are retained as evidence but never inserted into the optimizer batch. One whole-batch PPO update uses exact categorical reference KL, the declared clipping/reduction settings, actor learning rate `1e-6`, and KL coefficient `0.001`. AdamW betas, epsilon, and weight decay must be explicit. A single training seed is supported per invocation; `total_updates` must be supplied and positive.

The main profile takes exactly one optimizer step per full batch. All current-policy forwards for that step therefore use the same weights that produced the cached old probabilities. The importance ratio should be approximately one, so clipping is ordinarily inactive in this profile. The clipped formula is implemented, but this training schedule is not evidence that clipping altered any numerical update. Multiple PPO epochs or minibatch optimizer steps would be a distinct implementation choice; this reference driver does not add them implicitly.

The model-scoring adapter forwards complete native token sequences without a `.module` bypass. It uses logit position `t-1` to score native token `t`, masks prompt/tool tokens, and keeps their IDs in history. Exact categorical KL is conditional on sampled histories, rather than an exhaustive sum over all possible trajectories.

## Evaluation and durable outputs

After each optimizer update, the driver saves actor/tokenizer weights, optimizer and RNG state, config/source identities, and a completion record containing hashes of checkpoint files. Policy versions are explicit run/iteration identities, not falsely advertised weight hashes; checkpoint hashes establish saved-file identity. There is no implicit resume or overwrite.

Each development query is evaluated with one sampled, five-step trajectory. Expected evaluation domains must be explicitly listed in `expected_development_domains` and must exactly match the development manifest's domain set. For a full BRIGHT evaluation, list all twelve intended domains, including domains with no training examples. A partial-domain run must be identified as such in its experiment configuration; it is not a full benchmark reproduction.

Development seeds are derived per query and held fixed across checkpoint evaluations. They do not advance the training sampling stream. Per-request training and development seeds, initial D0 ranking IDs/scores, and individual development-query rewards are saved for inspection.

The evaluator reports domain means and the domain macro NDCG@10. `best_development_macro` selects the first checkpoint attaining the largest development macro score; `last` selects the final completed checkpoint. The driver does not consult a test split for checkpoint selection. It saves split snapshots, query-batch selections, all original and branch native trajectories, entropy evidence, advantages, old log probabilities, development trajectories, and stage receipts. Checkpoint writing can finish before development evaluation; a failed evaluation prevents a successful iteration receipt.

All work, including sparse-index construction and hashing large corpus/vector/checkpoint files, occurs only after the Slurm guard. HF, torch, CUDA, Triton, Python temporary, and JVM temporary locations are explicitly placed below the configured cache root. Run-time performance, storage consumption, and successful two-device execution remain unverified.

## Held-out checkpoint evaluation

`ricepo evaluate --config CONFIG --deployment DEPLOYMENT --checkpoint-dir CHECKPOINT --test-manifest TEST_JSONL --output-root NEW_TEST_OUTPUT` is a separate authorized-Slurm entry point. The direct API is `ricepo.hf_runtime.evaluate_checkpoint(config, deployment, *, checkpoint_dir, test_manifest, output_root)`. Neither entry point trains or updates a model. Test results are written outside the training run and never enter checkpoint selection.

The checkpoint must be exactly the one already selected in the completed training run's `completed.json`. The evaluator verifies its recorded file set and SHA-256 hashes, frozen policy identity, research configuration, code, training/development manifests, prompt, fixed retriever model/tokenizer assets, corpus, vectors, and row-ID mappings. The three split manifests are jointly checked for repeated domain/query IDs and normalized query text. Corpus overlap across splits remains permissible. Test qrels stay in the scorer.

By default `test_scope: bright_full_12_domains` requires all twelve Appendix A.1 domain-specific test counts, totaling 1,384 queries. The output is labeled *count-matched*: matching these counts alone cannot establish the exact author split. For a separately labeled partial diagnostic, set `test_scope: partial_diagnostic` and explicitly list `expected_test_domains`; its macro must never be presented as the complete BRIGHT benchmark.

Evaluation uses the same fixed retriever construction, renderer, native generator, five-step horizon and final NDCG@10 definition as training, with deterministic per-query seeds. Output includes the selected checkpoint receipt identity, split hashes, original-query initial rankings, full native trajectories, per-query scores, seeds, domain means, macro score, and final evaluation receipt. File hashing and retrieval-index construction may be expensive and require Slurm. This holdout path is also unexecuted.
