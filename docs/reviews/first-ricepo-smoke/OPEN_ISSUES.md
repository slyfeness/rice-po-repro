# Open issues and next-stage decision

## Latest functional diagnostic

The mixed-format adapter completed both originals and selected an entropy anchor. The new branch hit `InsideTokenError`; dropping that branch would invalidate its group. The authorized follow-up adapter is now explicitly named `summary_end_overlap_inclusive_v1` in `241ccdc`: only a unique token containing nonempty Summary text plus `</summary>` is wholly assigned to Summary. Strict reference parsing remains unchanged. The saved-ID verification is still pending.

## Earlier strict-reference blocking issue

The frozen actor did not close the strict reasoning/summary format within the
configured one-step generation bound for the first pre-registered original
sample. Because RICE-PO requires complete original trajectory groups, this is a
valid fail-fast outcome rather than a missing value to impute.

## Pending execution

Run the CPU-only saved-ID verifier first. If it passes, submit the separately
labelled two-GPU, one-query v7 functional smoke (two originals, one anchor,
one new branch, depth one). Neither diagnostic is equivalent to the
paper-reference recipe.

The following actions are explicitly not taken:

- retrying random seeds until the format succeeds;
- silently extending the generation bound and calling it the same run;
- retaining only successful original trajectories;
- assigning zero reward to a malformed or truncated output;
- continuing with incomplete branch or credit groups.

Four-GPU execution remains a separate, unvalidated capability. The frozen
four-GPU configuration from another project was not reused for this reference
runtime.
