## 2026-11-06 retrieval/backend extension

[Fixed-Summary query composition](reviews/four-gpu-deployment-review/QUERY_COMPOSITION.md) records the different wording in section3, section4.2.2 and Figure5. Original+LF+Summary is an offline candidate only; existing summary-only training and branch rewards are not rewritten. [Native vLLM integration](reviews/four-gpu-deployment-review/VLLM_RESULTS.md) retains the already-authorized two-stage/failure-environment variant, native IDs, full-vocabulary entropy, conditional KL, per-query groups, six outcomes and original advantage/gate formulas. Explicit independent request/stage seeds replace the HF shared RNG implementation for this named backend; identical sampled text is not promised. It remains an independent reproduction, not author-confirmed decoding or failure semantics.

---

## 2026-11-06 连续更新与批量采样

新增显式父检查点续训、按全局序号冻结查询日程、现有32题逐轮退化分析，以及可选HF原生wavefront采样。信用公式、完整词表熵、总体方差、0.05/0.03门控、center_only、优势替换和PPO目标保持。新的unmapped原生ID处理归入明确的独立非法终态环境；不是论文格式修复。见[本轮映射](reviews/continuous-three-update/CHANGES.md)、[恢复证据边界](reviews/continuous-three-update/RECOVERY.md)和[实际批量验收](reviews/continuous-three-update/BATCH_ACCEPTANCE.md)。

---

## 2026-11-06 两阶段受限验收

新增[原生两阶段独立协议](TWO_STAGE_PROTOCOL.md)；其环境边界与失败语义不是论文作者确认设置。compute_credit、优势替换和PPO目标保持，改动映射及真实证据见[本轮修改表](reviews/two-stage-native-v1/CHANGES.md)。

---

# Paper-to-code map

This repository independently implements **RICE-PO** from
[arXiv:2605.26352v3](https://arxiv.org/html/2605.26352v3), with
**DeepSeek-R1-Distill-Qwen-1.5B on BRIGHT** as the primary target.
It is not an author release. The published equations are implementable without
an author repository; unpublished experimental details must instead be named
and frozen as implementation choices. This document describes source-level
correspondence, **not a claim that the code or results have been run**.

Three evidence classes are kept separate:

| Class | Meaning | Where recorded |
| --- | --- | --- |
| Paper fact | Formula, reported constant, or instruction directly stated in v3 | Equation/section locators below and `configs/paper.json` |
| Explicit implementation convention | A concrete way to instantiate an underspecified detail | Configuration evidence and this document; never presented as author code |
| Unavailable experimental artifact | Author split IDs, training recipe details, or original runtime assets not supplied in the paper | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |

The verified Diver and ReasonIR repositories are dependencies or reference
sources. Their existence does not make this repository official RICE-PO code.
Pins and the bounded author-code search are in
[`provenance/sources.json`](../provenance/sources.json).

## Changes relative to d4b490a

The formula primitives below were present at the review baseline. Their callers now consume a resolved recipe. Equation (1) still uses the entire vocabulary and Eq. (2) pools only one query. GRPO stops at the shared trajectory-level advantage. RICE-PO verifies the complete original and branch groups before replacing token advantages. The reference horizon, raw NDCG, original outcome inclusion, population moments, paired reasoning fallback and fixed initial reference follow the original contract.

The variants in `configs/variants/` change one factor. Random anchor selection uses a seeded ordering of the query's candidates and the same K budget, retaining entropy measurement for diagnosis. Five gate variants use the same anchors: both conditions, influence only, residual only, unconditional propagation, or summary replacement alone. Final-depth anchors have R = r and therefore zero residual, and are reported separately. Variant names describe their exact independent definitions.

`modeling.sampled_old_log_probs`, `causal_token_scores`, the compact callback and `training.ppo_plan_update` separate old sampling probabilities, the fixed reference and the current model. The [revision matrix](REVISION_MATRIX.csv) and [runtime contract](REFERENCE_RUNTIME.md) identify their consumers and validation fixtures.

## Objects and one iteration

[`contracts.py`](../src/ricepo/contracts.py) represents native token spans,
trajectories, retrieval steps, anchors, branch outcomes, credit records, and
advantage arrays as explicit records. A `Span` is half-open: `[start, end)`.
A step stores the exact token prefix **before its reasoning**, the generated
token IDs, the reasoning/summary offsets, the retrieval result, and its local
reward. Tool observations and the initial prompt are not policy actions.

[`workflow.run_iteration`](../src/ricepo/workflow.py) organizes the iteration:

1. Collect N original trajectories for each query under one frozen old policy.
2. Compute summary token entropies, then select anchors separately per query.
3. Reuse each selected original outcome and collect independent alternatives
   from that anchor's exact saved prefix; continue alternatives to depth five.
4. Compute local and residual statistics, then replace token advantages on
   selected spans according to the gate.
5. Update the actor using the original trajectories and the final advantage
   arrays. Counterfactual trajectories remain credit-estimation/audit data.
6. Publish the new policy identity and complete stage receipts before another
   iteration uses it.

The old policy must stay identical during original rollout, entropy evaluation,
and branching. The reference policy in the KL term is a separate, fixed object.
The retriever is fixed throughout. Changing the actor between these phases
invalidates the conditional comparisons even if tensor shapes still match.

[`formats.NativeTagParser` and `PromptRenderer`](../src/ricepo/formats.py)
instantiate the native-span and prompt interfaces. They locate content boundaries
without replacing generated token IDs, use Figure 5's instruction body, and
keep retrieval observations separate from qrels. Chat wrapping, prefilled
thinking tags, escaped document text, and content-only span boundaries are
explicit implementation conventions.

## Equations (1)–(11)

Equation references below link to the versioned paper. Function links point to
the independent implementation.

| Equation | Mathematical requirement | Code mapping and invariant |
| --- | --- | --- |
| [(1)](https://arxiv.org/html/2605.26352v3#S4.E1) | Summary score is the mean, over its tokens, of Shannon entropy of the **whole conditional vocabulary distribution**. | [`objective.token_entropy_from_logits`](../src/ricepo/objective.py), [`TransformersEntropyProvider.token_entropies`](../src/ricepo/backends.py), [`credit.select_anchors`](../src/ricepo/credit.py). Teacher forcing uses the saved old-policy history; logit position u−1 predicts token u. Sampled-token negative log probability and a top-k vocabulary entropy are not substitutes. |
| [(2)](https://arxiv.org/html/2605.26352v3#S4.E2) | Pool summary actions across the N trajectories **for the same query**, then take top K. | `credit.select_anchors`; the workflow groups by query before selection. Appendix A.1 fixes K=4 anchors. This is not four anchors per trajectory or four over the entire multi-query batch. |
| [(3)](https://arxiv.org/html/2605.26352v3#S4.E3) | Ideal conditional mean/variance of local summary reward when reasoning is sampled from the same pre-reasoning history. | [`RolloutRunner.branch`](../src/ricepo/rollout.py) and `BranchGroup` define the sampling population; `compute_credit` estimates it. This expectation is not computed exactly. Both reasoning and its induced summary are regenerated. |
| [(4)](https://arxiv.org/html/2605.26352v3#S4.E4) | Include original k=0 and K alternatives: mean = Σr/(K+1), influence variance = Σ(r−mean)²/(K+1). | `credit.compute_credit` / `_moments`; exactly one original outcome and an explicit total branch count are required. The variance denominator is the population denominator, not an unbiased sample correction. |
| [(5)](https://arxiv.org/html/2605.26352v3#S4.E5) | For each branch, residual = final reward − local reward; compute its mean and population variance. Continue to maximum depth and use the **last** summary's retrieval reward. | `RolloutRunner.branch`, `BranchOutcome`, `credit.compute_credit`. Final reward is not best-so-far, accumulated NDCG, or reward from a shared continuation. Every alternative has its own retrieval history after the common prefix. |
| [(6)](https://arxiv.org/html/2605.26352v3#S4.E6) | Propagate iff influence variance ≥ τ_var **and** residual variance ≤ τ_res. | `credit.compute_credit` sets `propagate`; inclusive comparisons use 0.05 and 0.03 from Appendix A.1. Passing only one condition is insufficient. |
| [(7)](https://arxiv.org/html/2605.26352v3#S4.E7) | Accepted reasoning credit copies summary credit. | `credit.build_advantages` uses the same local summary value on the paired reasoning span when the gate passes. Neither variance is added as a reward or used as a learned critic. |
| [(8)](https://arxiv.org/html/2605.26352v3#S4.E8) | Selected summary tokens receive local A_sum; paired reasoning receives A_think; other generated tokens retain A_i. | `credit.build_advantages` initializes the trajectory baseline then **replaces** selected span values. It does not add local credit to A_i. Prompt/tool tokens have false loss masks and zero advantages. |
| [(9)](https://arxiv.org/html/2605.26352v3#S4.E9) | A_think = A_sum when both gates pass; otherwise A_think = A_i. | `credit.build_advantages`. A rejected reasoning span keeps the trajectory baseline; it is not zeroed. The selected summary still gets local credit when propagation is rejected. |
| [(10)](https://arxiv.org/html/2605.26352v3#S4.E10) | Token ratio ρ = π_current(a\|h) / π_old(a\|h). | `objective.ppo_loss` computes exp(current log p − detached old log p). [`training.UpdateItem`](../src/ricepo/training.py) requires matching policy version and native token alignment. Re-tokenized text or probabilities of another rollout policy would change this ratio. |
| [(11)](https://arxiv.org/html/2605.26352v3#S4.E11) | Maximize the clipped token surrogate minus β_KL KL(current\|reference). | `objective.ppo_loss` minimizes its negative and adds the KL penalty. `exact_reference_kl` computes full-vocabulary conditional KL. `training.ppo_plan_update` uses the resolved minibatch denominator. The separate distributed helper retains its global reduction. Appendix A.1 fixes β_KL=0.001; clipping coefficient and numerical reduction are not reported. |

### Advantage conventions and the proposition

The prose preceding (8) calls A_i a group-normalized final reward, without
publishing its exact denominator or stabilizer. The current credit primitive
uses, within one query,

```text
A_i = (R_i − mean(R)) / (population_std(R) + epsilon)
```

The population standard deviation and additive positive epsilon are explicit
implementation conventions. Equal final rewards yield zero A_i.

[Proposition 1 and Appendix A.3](https://arxiv.org/html/2605.26352v3#A1.SS3)
define the local summary advantage as local reward minus its conditional
expectation. The `center_only` option implements its Monte Carlo version:

```text
A_sum = r_original − mean(local branch rewards)
```

The primitive also exposes `center_std` for a separately declared sensitivity
study. Dividing by local standard deviation is not stated in the proposition
and must not be described as mathematically identical to it. The influence and
residual gates always use raw fractional rewards, before any normalization.

The proposition assumes a bound on the **difference of conditional expected
residuals**. A small empirical residual variance in five or six samples does
not prove that bound, nor prove that a specific reasoning sentence caused a
retrieval gain. The variance gate is the paper's practical proxy. Record the
actual branch rewards, variances, and gate decisions so that this approximation
can be examined.

### What “five branches” means

Appendix A.1 reports five Monte Carlo branches per anchor. Section 4.2.2 says to
construct K local branches, labels the original k=0, and equations (4)–(5)
average K+1 outcomes. Those statements leave the experimental counting
convention ambiguous.

The project's declared interpretation is **five new alternatives plus the
original, six outcomes total**. This combines the appendix's sampled count with
the body’s explicit original term; it is not a verified author implementation
detail. The explicit count interface also allows **four new plus original,
five total** as a separately named sensitivity configuration. The anchor count
K=4 is independent of the branch count despite the paper's reuse of K.

### Token alignment, entropy, and loss

[`VLLMGenerator`](../src/ricepo/backends.py) preserves generator-native token
IDs and delegates segmentation to an explicit parser. It does not reconstruct
the prefix from decoded text. Delimiter membership in reasoning/summary spans,
chat serialization, and any model-provided opening thinking tag must be frozen
with the parser and tokenizer revision.

Entropy is in nats. Token-row chunking in `objective.py` retains the full
vocabulary; it is a memory convention, not vocabulary truncation. Entropies
must be evaluated under the same old policy that produced the trajectories.

`sequence_mean` averages each sequence's **token sum**, matching an empirical
expectation over token sums in (11); `token_mean` and `sum` have different
gradient scales. The configured choice applies consistently across ranks.
Exact conditional KL still averages over sampled histories rather than
enumerating all possible trajectories. The optional sampled KL helpers are
approximations, and their usual expectation identity assumes current-policy
samples; old-policy samples introduce additional estimation considerations.
The KL estimator must be identified in every experiment record.

[`modeling.causal_token_scores` and `make_forward`](../src/ricepo/modeling.py)
connect native trajectories to the PPO callback: they shift causal logits,
preserve policy gradients, isolate the frozen reference, and call the supplied
model wrapper. Chunking normalization work does not eliminate the full-sequence
logits or training graph; memory feasibility requires later validation.

## Retrieval and reward contract

The paper fixes Diver-4B+BM25 for all RL methods. This does **not** require the
external Diver pipeline's 14B query generator or its rerankers. Our actor is
the 1.5B model. [`retrieval.py`](../src/ricepo/retrieval.py) isolates query
composition, fixed dense/sparse scores, score fusion, observations, and NDCG.

- Section 4.2.2 uses the generated summary together with the input query;
  §3 and Figure 5 describe the summary as the query. The composer supports
  both interpretations explicitly. `original_and_summary` and its separator
  are a documented adapter choice, not an undisclosed author string format.
- [Diver §2.3](https://arxiv.org/html/2508.07995#S2.SS3) states [0,1] normalization
  and equal dense/sparse weights. The `minmax` implementation, full-domain
  scoring, exclusion timing, constant-score behavior, and deterministic
  document-ID tie breaks are this adapter's specified semantics.
- `last_token_embeddings` uses last nonpadding token pooling and L2
  normalization, following the pinned original Diver model. Encoder text
  prefixes, maximum lengths, index type, dtype, and observation truncation
  must be recorded; inherited Diver defaults do not become RICE-PO facts.
- BRIGHT's benchmark `excluded_ids` are fixed query metadata. They must
  remain fixed across rounds and branches. Documents retrieved in an earlier
  round may appear again; excluding all earlier top-k results would create a
  different environment.
- Full rankings used by `PyTrecReward` are distinct from the short document
  observation shown to the actor. Gold IDs/qrels stay in the reward scorer.
  Gold reasoning, annotations, and NDCG values are not appended to the actor
  prompt.
- Store NDCG@10 in [0,1] without report rounding. Paper tables show scores
  multiplied by 100. Scaling rewards to percentage points before computing
  variance would multiply variances by 10,000 and invalidate the thresholds.

The reviewable [reference choices](../configs/implementation_choices.json)
currently choose `summary_only`, full-corpus min/max after applying fixed
benchmark exclusions, five observed documents with 512 actor-tokenizer tokens
per snippet, and raw-document encoding. These are explicitly labeled choices;
the existence of the profile is not evidence that they reproduce the author's
unpublished retrieval configuration. Corpus encoding is separated into
[`preparation.encode_corpus`](../src/ricepo/preparation.py), which records
artifact identities and must run only in a later authorized compute stage.

## Appendix and evaluation mapping

| Source | Implemented or recorded counterpart | Boundary |
| --- | --- | --- |
| A.1, base models | Model IDs and immutable revisions in `configs/paper.json` and provenance | Focus is 1.5B BRIGHT; the paper's 4B/3B settings and BEIR are not claimed reproduced. |
| A.1, optimization | Learning rate 1e−6, reported batch size 128, KL 0.001 | Total updates, optimizer details, scheduling, sampling N, and how batch size counts queries/trajectories still require an explicit recipe. |
| A.1, credit/rollout | Four anchors; thresholds 0.05/0.03; maximum five retrieval steps | Branch count interpretation is disclosed above. For the primary contract, all original and alternative trajectories reach depth five. |
| A.1 and Table 3 | [`data.BRIGHT_TRAIN_COUNTS`, `BRIGHT_TEST_COUNTS`](../src/ricepo/data.py) | These are published per-domain totals, not a reconstruction of the authors' exact row selection. |
| A.2 and Figure 5 | [`prompts/ricepo_system.txt`](../prompts/ricepo_system.txt), [transcription provenance](../prompts/README.md) | Exact body wording, normalized plain-text layout; chat roles and special tokens are implementation conventions. |
| A.3 | Centered local advantage and residual interpretation above | The theorem's assumption is stronger than a finite-sample variance gate. |
| A.4–A.5, Tables 5–7 | Future controlled gate/anchor ablations can use the same primitive | Matching names does not reproduce an ablation; all other settings, data, seeds, and compute budget must match. |
| A.6, Figure 6 | Stage receipts and future reward traces | No training curve or runtime measurement has been generated here. |

The 1.5B BRIGHT comparison in Table 1 is:

| Method | Paper macro NDCG@10 × 100 |
| --- | ---: |
| Prompting agent | 11.75 |
| GRPO | 24.98 |
| GiGPO | 25.88 |
| HGPO | 25.45 |
| Tree-GRPO | 26.00 |
| RICE-PO | 27.49 |

These are **paper results**, not measurements from this repository.
Evaluation first averages queries within each of the twelve tasks and then
macro-averages task scores; it is not a query-count-weighted average.
[`evaluation.summarize`](../src/ricepo/evaluation.py) implements this aggregation
and rejects an incomplete domain set instead of reporting a partial macro score.
The external Diver baseline's 37.20 uses a different, larger generation
pipeline and is not the target score for the 1.5B RICE-PO setting.

## Static implementation versus demonstrated behavior

Tests in `tests/` specify analytical invariants for entropy, population
moments, both gate boundaries, replacement masks, clipping, native histories,
and independent branches. Their presence is not a passing-test report.
No tests, model inference, preprocessing, training, evaluation, or Slurm jobs
were run in the implementation task. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the artifacts and decisions needed
to make a later experiment independently checkable.
