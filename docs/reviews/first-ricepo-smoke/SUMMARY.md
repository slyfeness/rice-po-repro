# First RICE-PO smoke review

Latest diagnostic: **original group and entropy anchor completed; branch parsing failed; zero optimizer steps.**

The later functional run at `adc707800239672fa0a1624d54bb26a5b95cda8f` used one Biology query, two original trajectories, one anchor, one new branch and one retrieval round. This is a non-paper diagnostic with a separately named format adapter. The rollout receipt binds the archived trajectories to SHA-256 `f88df33c19ff19bd75cb58fecf214bb7e6d69a411ea44c214e4b9dd688565237`. Two original generations produced 1,081 tokens in total; the branch produced 365. The branch could not map a text boundary to an exact native token offset. No branch group, final credit, update, checkpoint or development result was committed. Elapsed allocation time on the executed attempt: 4m18s, two GPUs. Earlier pre-batch launch failures were separate scheduler events. The sections below describe the earlier strict-reference run.

This review records a bounded, one-query Biology diagnostic using the independent
RICE-PO reference runtime. It is not a benchmark result, a runtime validation
claim, or evidence of four-GPU support. The public material deliberately omits
model files, data, prompts, completions, document rankings, internal paths,
machine names, and raw logs.

## Completed work

- Bound one frozen local actor, one frozen Diver retriever, a Biology-only
  dataset bundle, and its fixed local vector package before sampling.
- Repaired two Transformers 5.5 API-compatibility defects without changing the
  configured sampling law, native token handling, rollout depth, credit logic,
  or PPO objective:
  - `c459a0e`: conditionally supplies `use_model_defaults=False` only to
    releases which explicitly declare that `generate` parameter.
  - `9573901`: requests native `input_ids` from `apply_chat_template` instead
    of accepting the newer default `BatchEncoding` container.
- Executed targeted CPU regression tests and one real frozen-actor generation
  probe. See [TEST_RESULTS.md](TEST_RESULTS.md).
- Entered the first original rollout request of the one-query smoke chain.

## Terminal observation

The first scheduled original trajectory used its pre-registered seed and
generated exactly 1,024 native completion IDs without closing the required
`</summary>` tag. The strict reference parser classified this as
`GenerationTruncated`. Raw native IDs, prefix digest, request identity, seed,
decoded diagnostic preview, failure receipt, and cost receipt were persisted
in private project storage before the process stopped.

No retry, changed seed, changed generation limit, zero-reward substitution,
partial-group credit estimate, branch rollout, optimizer update, checkpoint,
development evaluation, or formal test evaluation was performed.

## Unfinished work

The failure prevents the complete original group required for entropy anchors,
counterfactual branches, credit gating, and a formal PPO update. Resolving it
requires an explicit methodological decision; it cannot be converted into a
successful RICE-PO result by silently extending generation, resampling, or
dropping the failed sample. See [OPEN_ISSUES.md](OPEN_ISSUES.md).
