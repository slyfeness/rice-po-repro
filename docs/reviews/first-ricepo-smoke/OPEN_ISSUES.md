# Open issues and next-stage decision

## Latest functional diagnostic

The mixed-format adapter completed both originals and selected an entropy anchor. The new branch hit `InsideTokenError`; dropping that branch would invalidate its group. The authorized follow-up adapter is now explicitly named `summary_end_overlap_inclusive_v1` in `241ccdc`/`91e2161`/`26d7720`: only a unique token containing nonempty Summary text plus `</summary>` (and at most whitespace beside it) is wholly assigned to Summary. Strict reference parsing remains unchanged. The executed saved-ID inspection found that the actual crossing token contains no close tag, so the authorized adapter correctly cannot accept it.

## Earlier strict-reference blocking issue

The frozen actor did not close the strict reasoning/summary format within the
configured one-step generation bound for the first pre-registered original
sample. Because RICE-PO requires complete original trajectory groups, this is a
valid fail-fast outcome rather than a missing value to impute.

## Blocking decision

The saved branch has a different structure: its Summary end is inside a token
that ends in whitespace, while `</summary>` begins at a later native boundary.
Accepting that output needs a distinct, pre-registered rule and changes the
loss-span ownership question. It is not authorized by the same-token-close
rule. No two-GPU v7 job may be submitted until a compatible rule or a new
pre-registered diagnostic input is authorized. Neither diagnostic is
equivalent to the paper-reference recipe.

The following actions are explicitly not taken:

- retrying random seeds until the format succeeds;
- silently extending the generation bound and calling it the same run;
- retaining only successful original trajectories;
- assigning zero reward to a malformed or truncated output;
- continuing with incomplete branch or credit groups.

Four-GPU execution remains a separate, unvalidated capability. The frozen
four-GPU configuration from another project was not reused for this reference
runtime.
