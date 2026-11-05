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
