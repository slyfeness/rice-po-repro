# Open issues and next gate

1. **Completion rate is the blocking result.** The new capacity runs reached the full training, checkpoint, reload, and 32-query development paths, but only 18/32 RICE-PO and 17/32 GRPO development queries completed five legal rounds. Expanding directly to 128 queries × 10 updates would scale this known failure mode, not validate the method.
2. **This is an independent protocol variant.** `original_and_summary`, `two_stage_env_summary_v1`, the compatibility prompt, and `invalid_action_terminal_zero_v1` are explicit reproduction choices or authorized environment semantics, not verified author code/settings.
3. **Credit activation exists but is sparse.** Three joint gate passes and 72 changed Summary-token advantages establish that the local-credit path was exercised. They do not establish a retrieval-quality improvement or adequate non-terminal reasoning propagation.
4. **Development sampling remains single-sample.** This completed capacity run does not implement or validate four samples per development query; it must not be described as such.
5. **No additional experiment was launched from this report.** A revised scale decision needs to address the legal-action completion failure without altering paper credit equations, rewards, thresholds, sampling distribution, or silently selecting successful seeds.
