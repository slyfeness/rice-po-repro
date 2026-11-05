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

`summary_end_overlap_inclusive_v1` was added in `241ccdc` with focused source
tests and refined in `91e2161`/`26d7720` to permit only whitespace beside the
same-token close tag. A later CPU-only, no-generation full/prefix-decode audit
confirmed that prefix decodes are monotonic, but the saved branch's Summary
end equals the close-tag start inside the crossing token; that position has no
native boundary, and the token contains a non-whitespace prefix of the tag.
The adapter and the separately authorized trailing-whitespace rule therefore
both correctly remain inapplicable. The sanitized private report SHA-256 is
`b865f69c324a483b66b053e9bbdd315a0c4ca704d6a4633dac59f352ef6e397e`.
The original-trajectory artifact SHA-256 remained
`f88df33c19ff19bd75cb58fecf214bb7e6d69a411ea44c214e4b9dd688565237`.
No branch reward, advantage, optimizer step, checkpoint, development result,
or two-GPU v7 job was created.
