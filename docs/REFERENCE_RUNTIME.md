## 2026-11-06 vLLM integration

The opt-in deployment template is `configs/deployment.vllm_two_stage.example.json`, paired with the existing two-stage RICE/GRPO recipes. Execution remains disabled in public research recipes. Set `TRITON_F32_DEFAULT=ieee` before imports, reuse local assets and use Slurm. The Trainer remains a single process with a phase-local inference subprocess on its assigned Actor GPU; the other GPU holds Diver. Two independent allocations are not four-rank training. Do not put this engine on the Retriever GPU without separate capacity validation.

`vllm_two_stage.VLLMTwoStage` consumes native prefixes and a single 4096 sampled-token budget across Think and Summary. Deterministic environment Summary control has no direct policy loss. Stage seeds are explicitly derived from each request seed and backend/stage names; this is a new named RNG implementation, not HF's shared-generator bitwise replay. Temperature, top-p, top-k, penalties and full conditional probabilities stay fixed. Native IDs and scalar sampled log-probabilities are retained before parsing; no stop-string token is re-encoded or repaired.

`vllm_lifecycle.PhaseSeparatedGenerator` offloads HF Actor/Reference and optimizer tensors during generation, verifies an immutable FP32 snapshot against live weights, starts a fresh engine and destroys it before scoring. New-policy snapshots reload into new processes with empty KV caches. Original and counterfactual sampled probabilities must pass full-history HF scoring before PPO. The optimizer, reference, advantage rules and reductions are unchanged. Temporary snapshots/RPC/cache/checkpoint paths must reside on bulk storage. The RPC root must be short enough for UNIX-domain sockets; each worker gets an atomic unique subdirectory.

Internal development evaluation uses the bound sampler. Standalone `holdout` explicitly rejects this backend until its independent sampler lifecycle is implemented; it must not silently use HF. Measured acceptance and limitations are reported in [the current review](reviews/four-gpu-deployment-review/VLLM_RESULTS.md). The named engine currently fixes TP1, FP32, TRITON_ATTN, sequences64, batched tokens16384, max context32768 and pool0.8; these are implementation choices, not recovered author hyperparameters.

---

## 2026-11-06 显式续训与加速入口

新增独立child-run，从完整第一更新检查点加载Actor、优化器、CPU/CUDA RNG，固定Reference仍从初始模型加载。父配置不原地延长；全局更新序号与本地目录序号分别保存。串行续训与可选hf_native_wavefront_v1资格分开；本轮未接通vLLM训练或四rank。精确使用运行绑定源码恢复，不能仅改policy_version字符串。[入口与限制](reviews/continuous-three-update/RUNTIME.md) · [实际batch验收](reviews/continuous-three-update/BATCH_ACCEPTANCE.md) · [vLLM现状](reviews/continuous-three-update/VLLM_STATUS.md)。

---

## 2026-11-06 两阶段受限验收

新增 two_stage_env_summary_v1、actor_precision=float32、eval_layers_v1 和 local_verified_v1；仍为单进程Actor/Reference一设备+Diver另一设备。真实训练包源码对应公开等价提交98267806afc70bf73c26385460918d918b02ea49；恢复必须使用绑定的精确包源码，可配合新scripts/validation/verify_training_checkpoint.py，不能绕过source fingerprint。见[测量及限制](reviews/two-stage-native-v1/SUMMARY.md)。

---

# Reference runtime and recovery contract

The supported deployment is an offline source checkout in a Slurm allocation. The HF training and prompting paths use one process and two explicitly named CUDA devices. D0 evaluation uses the fixed retrieval stack alone. Source review is complete for this revision. Runtime, capacity and recovery acceptance remain pending.

## Resolved inputs

`config.resolve(config, command)` produces a command-specific recipe with source labels, reference values, declared deviations, unresolved fields, a fingerprint and required deployment keys. The CLI exposes `expand`, `baseline`, `train`, `evaluate`, `encode-corpus`, `summarize` and `preflight`. Open `configs/deployment.example.json` for the local paths to fill. Model and data acquisition is separate from these commands. All supplied configs inherit disabled execution.

| Resolved fields | Consumer |
|---|---|
| method, queries_per_update, trajectories_per_query | config dispatch, hf_runtime.rollout_batch, workflow.run_iteration |
| anchors_per_query, anchor_selection, gate_mode, both thresholds, local normalization | workflow._collect_credit, credit.select_anchors, build_advantages, compute_credit |
| total branches, rollout_depth | hf_runtime.collect_branches, RolloutRules, RolloutRunner.branch, assessment.evaluate_queries |
| PPO epochs, optimizer minibatch, microbatch, accumulation, loss reduction, clip, maximum gradient norm | training.ppo_plan_update and objective.ppo_loss |
| optimizer, betas, epsilon, weight decay, LR, reference KL coefficient | HF AdamW construction, conditional KL, PPO plan |
| dev/save intervals, selection | hf_runtime.finish, state.finish_development and selected_so_far |
| query composition, observations, hybrid normalization, exclusions, ranking and metric | PromptRenderer, ExplicitQueryComposer, _domain_retriever, FixedHybridRetriever, PyTrecReward |
| fixed decoder distribution and precision | recipe validation, HFNativeGenerator, actor teacher forcing and entropy |
| training budget, seed and manifests | hf_runtime._train, dataset_bundle.verify_bundle, audit.stable_seed |

The paper reports batch size 128. Its unit is an explicit reproduction decision. The reference treats this as queries per update and takes four original trajectories per query. It uses one epoch, an optimizer minibatch of all 512 original sequences and one sequence per microbatch. Each optimizer minibatch divides by its own global denominator. With sequence_mean the loss is the mean of the per-sequence token sums. The smoke profile changes Q to one and derives a four-trajectory optimizer minibatch. Optional minibatches must evenly divide the trajectory batch. Microbatch size one is supported. The first update can have ratio near one and a nonzero gradient. Zero signal, a zero clipping fraction, or unchanged weights are valid recorded outcomes.

## Data, assets and evaluation

Supply a reviewed version-2 dataset receipt and its independently pinned SHA-256, the three canonical sibling manifest paths, the original domain corpora and complete version-2 vector bundles. Training verifies the three identities before sampling and reads train/dev rows from the verified bytes. The bundle records query separation rules, selection procedures and counts. Real paths and the number of updates are experiment inputs. For the corpus bundle and upstream comparison protocol see [RETRIEVAL_AUDIT.md](RETRIEVAL_AUDIT.md).

`retrieval_only` evaluates D0. `prompting` loads the original actor. `grpo` bypasses entropy and branching. `ricepo` completes all expected original and branch groups before optimization. Every generated evaluation uses one rollout under the same seed schedule and fixed retrieval setup. Development and formal test use assessment.evaluate_queries. Formal holdout verifies the completed run, its previously selected checkpoint, actual files and live policy identity before reporting a score. Scores bind method, behavior, retrieval identities and the split manifest. A partial diagnostic has its own scope, missing IDs, covered counts and partial_macro.

## Artifacts, failure and recovery

`binding.json` binds the config, resolved recipe, code, split receipt, all three manifests, actual local actor/reference/tokenizer assets, index recipes, prompt and sampling behavior. The initial reference model is reloaded from the original bound assets. A SHA-256 of the live actor state and its behavior metadata identifies each policy, including after optimizer updates or checkpoint reload. Hashing large assets and live weights is itself cluster compute. Finite scalar/ratio diagnostics, gradient norms, a fixed parameter probe and the live state comparison separate a step invocation from a weight change.

States proceed through rollout_complete, branches_complete (explicit skip for GRPO), credit_complete, optimizer_step_committed, checkpoint_complete, development_complete and iteration_complete. Cadence skips carry reasons. Development intervals must coincide with checkpoints. The final iteration always saves and evaluates. For best developmental macro, ties select the earliest eligible iteration. The alternative last rule selects the last eligible complete checkpoint.

All events live under their attempt directory. Returned native completion IDs, completion reason, seed, exact prefix and its fingerprint are saved before parsing. A diagnostic preview follows when decoding succeeds. Parsing uses strict uniquely mappable boundaries and aborts on empty content. A malformed or failed sample aborts its group. Retrieval requests retain query, fixed exclusions, prefix/request identity and seed. The reference driver performs no automatic retrieval or generation retries. A regenerated request during supported recovery retains its logical identity, points to the first attempt, and is recorded as replay.

`train --resume` acquires the same exclusive POSIX lock, reconstructs bindings, and accepts only the latest complete checkpoint in contiguous iteration history. Recovery checks the actor payload, optimizer/RNG file, iteration, old receipts, selection and reference identity. It restores optimizer/RNG state, rechecks the live actor, and retries pending development in a fresh attempt before starting the next iteration. A checkpoint published just before its outer receipt may have the receipt completed during recovery. A newer partial/ambiguous iteration stops recovery. Optimizer minibatch events from an interrupted iteration remain evidence; replay through a partially committed update is outside this interface. Successful prior artifacts are never replaced.

The filesystem must support coherent flock across Slurm jobs, same-filesystem directory rename, hard-link publication, and directory fsync. A killed process can leave a started operation or pending directory. Only verified complete receipts are eligible. Filesystem semantics on the target mount require future validation.

## Resource and cost interpretation

Old scoring uses the actor only. The complete causal scorer is retained for comparison. The compact gradient path keeps the full prefix through the final scored predecessor, then pads results to original coordinates. Original trajectories retain their entire final observation. Full-vocabulary work still has a potentially large sequence-by-vocabulary projection and training graph. No training KV cache is detached. Gradient checkpointing/offload and selective output projection are outside the supported plan. The preflight command records observed package/device/JVM versions and, when supplied with dimensions, a separately labeled formula estimate. Future experiments must measure peak allocated/reserved memory and phase times.

Cost evidence includes attempted calls, original/branch continuation tokens, input history processed, entropy/old/current scoring, D0/follow-up retrievals, development/test, identities and checkpoint writes. Counts include replays and failures; unavailable output token counts remain unknown. Times are inclusive host wall times and nested rows overlap. Equal original-query/trajectory/optimization budget and equal measured compute are separate comparisons. The former retains all branch costs on the RICE-PO side.
