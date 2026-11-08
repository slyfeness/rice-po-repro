# HF native wavefront acceptance

The optional `hf_native_wavefront_v1` sampler is connected to Trainer original trajectories and counterfactual branches. It uses the same rollout transition, message history, actual Think/Summary sampler, failure environment and native action mask. Four originals of one query and five new branches of an anchor are batched as independent requests. Each has its own stable seed/RNG; row completion order does not change request identity. Token budgets, temperature1/top-p1/top-k0, original-ID fidelity and frozen behavior policy remain unchanged. KV cache is stage-local and discarded before another call or a weight change.

The actual continuations in this review remain **serial**. The following are separate real asset diagnostics, not an accelerated training result.

## Full method sampling workload: Slurm174719

Fixed next-schedule training query, actual RICE-update-one FP32 weights, Diver+BM25, max5 rounds, 4 originals, 4 entropy anchors and20 new branches. All originals terminate illegally in this particular workload; their earlier valid actions still supply real anchors, of which3 are estimable. No samples were retried or removed. It exercises long native histories, real retrieval and credit, but does not demonstrate four legal complete originals or execute an optimizer step.

| Measurement | Serial | Batch |
|---|---:|---:|
| Actual generated model tokens | 42145 | 42145 |
| Action requests / real retrieval calls | 76 /60 | 76 /60 |
| Generation wall seconds | 973.126198 | 885.689498 |
| Generation token/s | 43.308874 | 47.584396 |
| Whole workload seconds, including non-generation work | 1044.923003 | 942.864561 |
| Maximum native training-history length | 8255 | 8255 |
| Full-history probability checks | 24/24 passed | 24/24 passed |
| Worst sampled versus teacher-forced log-prob error (nats) | 0.000203252 | 0.000238329 |
| Worst per-trajectory p99 error (nats) | 0.000042915 | 0.000043869 |

Original native IDs/masks, selected entropy anchors, branch identities and native branch IDs/masks match exactly in this workload. Actual probabilities are checked at every generated action position against full-history teacher forcing, not just at short prefixes. Legacy result field `within_bf16_tolerance` names the shared checker; the actual Actor here is FP32. Both modes use the same pre-existing absolute-error limits (max.12, mean.02, p99.06 nats), with errors far below those limits. The limits were not loosened for batch mode.

Generation throughput improves1.09872×, and whole-workload wall improves1.10824×. Serial was executed first; batch shared already-loaded models/index and warmer caches. This is one fixed workload, not a repeated throughput distribution or a clean causal estimate of all speed differences. Both modes report peak allocator allocated bytes `[15130699776,8150408704]`, reserved `[46047166464,8174698496]`; these are process high-water marks in one shared model process, not independently cold-isolated memory experiments. No co-resident training reference was loaded in this benchmark.

Count-based padding analysis finds489658 unpadded prefill tokens,528466 padded batch prefill tokens, and42000 useful decoder row slots out of76305 inferred slots (55.04%). Stopped/padded rows can consume work. This suggests limits to short-condition speedups, but does not measure kernel time, FLOPs or actual GPU utilization. No new cache/detach/context-truncation optimization is silently applied.

## Real weight switch on long histories: Slurm174731

Four longest saved training prefixes are selected by length before examining their new outputs, with lengths6956,5956,5902,5869. The same model object loads actual verified GRPO checkpoint-one and checkpoint-three weights, then runs serial and batch with fixed request seeds. Actual parameter identity changes; no past KV state is reused.

| Actual policy | Max native length | Serial / batch token/s | Worst full-history log-prob error (nats) |
|---|---:|---:|---:|
| GRPO update1 | 7309 | 33.772742 /31.405669 | 0.000077248 |
| GRPO update3 | 7337 | 32.215781 /32.786858 | 0.000061989 |

Within each policy, native IDs/action masks agree and all probability gates pass. Across policies, different text is legitimate; equal seeds are not a promise of equal trajectories after learning. These are fixed conditional-history diagnostics, not additional online updates or development selection. The update-one case is slower in batch, so a universal speedup claim would be false.

## Qualification boundary

Sampler wiring and these actual workload gates are complete at the recorded immutable sources. The frozen serial continuation is not changed. A new batch-enabled experiment still needs full online PPO/checkpoint/development and co-resident memory acceptance before formal use, with fresh run identity. Actual tests cover native histories through8255, not every32768-position case, arbitrary model/kernel versions or complete twelve-domain training. vLLM and four-rank FSDP remain outside this acceptance.
