# Actual test and probe results

These are executed results, not static-test claims. Commands are intentionally
redacted to remove private paths and cluster details.

| Evidence | Command class | Outcome | Notes |
| --- | --- | --- | --- |
| Generation API regression (red) | CPU `pytest` against pre-fix source | expected failure | Old source rejected a generator whose API lacks `use_model_defaults`. |
| Generation API regression (green) | CPU `pytest` against isolated patched source | 1 passed | Confirms the explicit sampling recipe is delivered without the unsupported keyword. |
| Prompt container regression (red) | CPU `pytest` against pre-fix source | 3 failed, 8 passed | Reproduced strings being iterated as token IDs when the modern default container is returned. |
| Prompt container regression (green) | CPU `pytest` against isolated patched source | 11 passed | Covers template-prefilled and non-prefilled render modes. |
| Frozen-actor API probe | one-GPU, one-token generation | passed | Actor accepted the explicit generation recipe without `use_model_defaults`. |
| One-query RICE-PO smoke | two-GPU bounded reference workflow | failed as designed | Strict fail-fast on the first original trajectory: `GenerationTruncated` at 1,024 completion IDs. |

The complete source test suite was not rerun after these two small repairs.
Later functional diagnostic (two GPUs): failed at branch parsing after the
complete original group and entropy selection. A CPU-only offline inspection
of the saved branch IDs with the frozen tokenizer found unique native offsets
for both reasoning boundaries and the beginning of the tagged summary. Only
the summary end was inside a token (ID 3918); both sides contain non-whitespace.
The diagnostic did not generate new text or assign advantages.
No test result above implies a complete training update, retrieval benchmark,
four-GPU validation, or distributed equivalence.

`summary_end_right_inclusive_v1` was added in `2d5827e` after the full/prefix
decode audit established a unique, monotonic native prefix layout. CPU-only
saved-ID verification passed: strict reference parsing rejected, the new
adapter accepted, the selected Summary span ended at the native token boundary,
and the two saved original trajectories remained unchanged. The final targeted
parser source test run passed **16/16**. An immediately preceding **15/16** run
identified and corrected a fixture expectation for preserved trailing
whitespace; it was not a runtime/parser defect.

The subsequent v8 two-GPU smoke failed in its new branch generation with a
separate malformed structure: one closing Think tag, two opening Summary tags,
and no closing Summary tag. This is a real model output and was retained, not
replaced or assigned a reward. The sanitized saved-ID verifier report retains
the original v5 trajectory identity below.
The original-trajectory artifact SHA-256 remained
`f88df33c19ff19bd75cb58fecf214bb7e6d69a411ea44c214e4b9dd688565237`.
No branch reward, advantage, optimizer step, checkpoint, development result,
or completed two-GPU functional update was created.

## 2026-11-05 follow-up results

| Evidence | Command class | Outcome | Notes |
| --- | --- | --- | --- |
| EOS classification regression | CPU `pytest` against immutable patched source | 2 passed | An EOS completion containing incomplete Summary markup remains classified as malformed, not as a normal stop. |
| v5 saved-ID recovery audit | CPU Slurm structural audit | passed | Two originals, one anchor and one saved branch completion had matching query, policy and native prefix identities. |
| v5 offline recovery | two-GPU recovery workflow | completed | Deterministic parse/retrieval/reward recovery, one credit calculation, one real optimizer step, atomic checkpoint and successful reload. The batch had zero advantage signal. |
| Fixed-seed prompt diagnostic | two-GPU, 1 query × 2 seeds × 2 prompts | completed | Strict parser: reference 0/2 (boundary); compact 0/2 (structure). Right-inclusive reparse: reference 2/2 accepted; compact was not selected. |
| Full-five functional smoke | two-GPU, 1 query / 4 originals / depth 5 | failed | A duplicate/nested-tag original completion at depth 2 stopped the incomplete group before entropy, branches, credit or update. |

The parser source test result above is a real executed test result. The full
five failure is a real model-format outcome, not a test failure manufactured by
fixtures. No row in this section establishes a benchmark score, a complete
five-round result, or four-GPU support.
