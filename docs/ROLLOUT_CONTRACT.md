# Rollout contract and implementation choices

The method reference is [RICE-PO v3](https://arxiv.org/html/2605.26352v3), Sections 4.2.2–4.2.3 and Appendix A.

Every generation uses the exact native token prefix saved before the paired reasoning and summary. The runner appends those generated IDs and the retriever's observation IDs without decoding and retokenizing history. Generation spans are relative to that generation; trajectory spans add the saved prefix length. Generated tokens, including format tokens outside selected spans, are trainable. Prompt and tool observation tokens are masked.

Counterfactual branches start at the saved pre-reasoning prefix. They reuse only the preceding steps, and sample a new reasoning–summary pair and subsequent retrieval steps. The original outcome is included exactly once, with its observed local and final rewards. New branches are statistical evidence, not automatically additional policy-update samples.

Section 4.2.2 continues branches to the maximum depth, and Appendix A.1 sets that maximum to five retrieval steps. Credit branching therefore requires an original trajectory already at depth five and takes every new branch to depth five. A shorter original is rejected: extending only its alternatives would compare rewards from different horizons, while extending the original behind the caller's back would alter the training record. `run()` permits an explicitly supplied early-stop rule for other purposes, but early-stopped trajectories cannot enter this full-horizon credit-branch path.

`excluded_ids_by_query` contains fixed dataset exclusion IDs, such as BRIGHT's excluded-document lists. The runner snapshots these sets at construction and passes the same set on every retrieval for the query. Earlier retrieval results are not added to exclusions. Repeated documents across steps or branches are valid; duplicate IDs within one ranked result are rejected.

The runner currently rejects empty reasoning or summary spans and empty summary text. This is an implementation **format policy**, not a claimed paper requirement: Appendix A.2 permits a zero-word summary. The paper does not resolve its entropy, termination, or reward handling sufficiently for this implementation to invent one. This restriction is a fidelity limitation to resolve before claiming exact reproduction of all paper-allowed outputs; no failure reward is fabricated.

Appendix A.1 reports **five Monte Carlo branches** per selected anchor. Section 4.2.2 numbers branches from `k=0` through `K`, with the original at zero. Whether the appendix's five includes the original remains ambiguous. Configuration therefore must select an explicit total count including the original and record the interpretation. The reported five must not be described as an unpublished branch budget.

Final reward is supplied by the configured callback and must be the final summary's fractional NDCG@10 for the paper recipe. The runner rejects nonfinite or out-of-range rewards. It does not silently substitute a penalty when generation, retrieval, or formatting fails.

The optional entropy adapter teacher-forces the same old policy. Logits after native token `t-1` score the distribution before token `t`; token zero has an undefined-history `NaN` sentinel. Chunking retains the KV cache and the full vocabulary. The adapter rejects caller overrides of position/cache/logit-slicing arguments that could invalidate alignment.

These contracts and their tests have been written and statically reviewed only. No runtime, model, GPU, or multi-rank validation is implied.
