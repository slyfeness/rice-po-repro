# Actual execution results

These results were executed once. No historical D0, retriever-regression, prompt, `summary_only`, or earlier development result was rerun for this report.

## RICE-PO

- Completed one 32-query update with 128 original trajectories and 232,005 generated tokens.
- Executed one optimizer step. The state dictionary and fixed parameter probes both changed; maximum observed probe absolute delta was `1.9073486328125e-06`.
- Gradient norm before clipping: `501.1370849609375`; policy loss: `-4.9876344203948975`; reference KL: `0.0` at the first-update reference comparison.
- Old/current probability ratio min/mean/max: `0.9988448024` / `0.9999999471` / `1.0004676580`. With one reference update, no ratio was outside the clip interval and effective clipped fraction was zero; this is expected for a first PPO update, not a skipped update.
- Independently reloaded checkpoint weights matched the committed policy identity.
- Development evaluation covered all expected 32 queries. The failure-adjusted score was `0.2646480573`; 18 queries completed five legal rounds.

## GRPO

- Completed one 32-query update with 128 original trajectories and 232,005 generated tokens.
- Executed one optimizer step. State dictionary and parameter probes changed; maximum observed probe absolute delta was `1.9073486328125e-06`.
- Gradient norm before clipping: `515.257568359375`; policy loss: `-32.643459022045135`; reference KL: `0.0` at the first-update reference comparison.
- Independently reloaded checkpoint weights matched the committed policy identity.
- Development evaluation covered all expected 32 queries. The failure-adjusted score was `0.3140633264`; 17 queries completed five legal rounds.

## RICE-PO credit evidence

- Selected anchors: 128; non-zero local-credit records: 41.
- Summary-token advantage replacements: 72.
- Both paper gate conditions passed: 3 anchors.
- Original trajectories ending in the independent invalid-action terminal: 53 / 128.

The failure-terminal count is evidence of the current protocol limitation. Failed actions were retained as token-native evidence and received the separately defined terminal utility; they were neither re-sampled nor removed from the original trajectory groups.

## Not executed

- No paper batch-128 training update.
- No ten-update candidate training.
- No final test-set result.
- No four-rank/FSDP trainer.
- No fresh vLLM throughput benchmark, retriever regression, D0 calculation, or replay of existing results.
