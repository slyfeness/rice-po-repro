# First controlled trial — 2026-11-05 America/Chicago

The source under test was bd6ca976565bae53bd16f713e73a1b4acec9edfb. Trial artifacts are under the independent rice-po-repro project's storage-backed reports directory. All cluster computation used Slurm.

| Job | Stage | Result |
|---|---|---|
| CPU-ENV-01 | Existing environment inventory and first test launch | Inventory completed; pytest launcher failed because the Python 3.11 environment lacked pytest |
| CPU-LAUNCH-02 | Reuse installed pure-Python pytest through project-local symlinks | Launcher identified a missing existing py.py dependency; no tests collected |
| CPU-TEST-03 | CPU suite with complete project-local pytest import path | 217 passed, 2 failed, 2 skipped in 77.19 seconds |
| CPU-TEST-04 | CPU suite after publication compatibility fix | 219 passed, 2 skipped in 18.20 seconds; COMPLETED/0 |
| GPU-PROBE-01 | Real local 1.5B actor/reference numerical diagnostic | COMPLETED/0; zero optimizer steps |

Public aliases replace scheduler IDs in the current tree; original IDs and raw reports remain private. Historical commits are not rewritten. The stage review, sanitized commands, evidence digests and limitations are in [reviews/first-smoke/SUMMARY.md](reviews/first-smoke/SUMMARY.md).

The two CPU failures both came from the shared storage rejecting renameat2(RENAME_NOREPLACE) with EINVAL. The publisher now holds a stable sibling flock, checks for any existing destination, and atomically renames the complete staging directory. All cooperating publishers must follow this protocol. The existing empty-target and failure/retry fixtures passed on the mounted filesystem after the change. Shared environments were reused without installation or modification.

The two skipped tests require separately supplied, independently pinned tokenizer/GPU fixtures. They remain skips even though a separate real-model diagnostic was performed.

## Real-model diagnostic

The actor and frozen reference were loaded from the existing pinned local DeepSeek-R1-Distill-Qwen-1.5B snapshot, using FP32 parameters on one allocated A800 80 GB GPU. The job requested two GPUs, with computation on the first logical device. The short input was constructed for numerical diagnostics rather than BRIGHT scoring.

- 14 native input tokens, with the last four treated as scoring positions.
- Whole-vocabulary entropy, KV chunk size 4, compared against a full-sequence forward at identical causal positions. Maximum absolute error 0.000011563301086425781.
- Actor-only old log probabilities versus current policy scoring. Maximum absolute error 0.
- Identical frozen-reference conditional KL. Maximum absolute value 0.
- Backward gradient norm 57.766197204589844, finite and nonzero. Reference parameters had no gradients.
- Zero optimizer steps.
- Peak allocated tensor memory 21,364,740,096 bytes. Peak reserved memory 21,791,506,432 bytes. These are short-case measurements.
- Model identity hashing and model initialization 71.84 seconds. Diagnostic total after framework imports 78.33 seconds. Batch elapsed time 2 minutes 15 seconds, separately recorded by Slurm.

## Full interactive smoke prerequisites

The inspected asset directories contained the actor snapshot but no verified original Diver-4B retriever or BRIGHT/ReasonIR bundle. The reused environment inventory reported missing Gensim, Pyserini, pytrec_eval and JVM. There is no prepared version-2 vector bundle or reviewed three-split recipe for this project. A complete query group, counterfactual retrieval, optimizer update, checkpoint/development cycle and formal test therefore remain pending.

Repository execution templates and runtime_validated/four_gpu_validated fields remain false. The task-specific preflight authorization is stored only in the private cluster control directory. Further acquisition of models/data and preparation of a separate dependency environment require a concrete approved preparation scope.
