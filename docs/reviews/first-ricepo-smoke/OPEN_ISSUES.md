# Open issues and next-stage decision

## Latest functional diagnostic

The mixed-format adapter completed both originals and selected an entropy anchor. The original saved branch's unique partial-close-tag boundary was accepted only after the explicitly authorized `summary_end_right_inclusive_v1` adapter (`2d5827e`); strict parsing remains unchanged. The newly generated v8 counterfactual then contained one `</think>`, two `<summary>` openings, and no `</summary>` close. Dropping it would invalidate its group, so the workflow stopped before credit.

## Earlier strict-reference blocking issue

The frozen actor did not close the strict reasoning/summary format within the
configured one-step generation bound for the first pre-registered original
sample. Because RICE-PO requires complete original trajectory groups, this is a
valid fail-fast outcome rather than a missing value to impute.

## Blocking decision

The remaining failure is not a token-boundary ambiguity: the v8 completion has
no closing Summary tag. Accepting it would require inventing a missing tag or
using malformed text as the retrieval query, neither of which is authorized.
Any further attempt needs an explicitly defined format-generation intervention
or a pre-registered input change; it must retain the failed v8 branch and its
seed. Neither diagnostic is equivalent to the paper-reference recipe.

The following actions are explicitly not taken:

- retrying random seeds until the format succeeds;
- silently extending the generation bound and calling it the same run;
- retaining only successful original trajectories;
- assigning zero reward to a malformed or truncated output;
- continuing with incomplete branch or credit groups.

Four-GPU execution remains a separate, unvalidated capability. The frozen
four-GPU configuration from another project was not reused for this reference
runtime.

## Current blocking condition after the full-five attempt

The first full-five original trajectory produced one valid first retrieval
step, then a depth-2 completion with nested/duplicate format tags.  This is
neither the summary-end native-token boundary already covered by
`summary_end_right_inclusive_v1`, nor a retriever, reward, credit, optimizer
or checkpoint fault.  The saved raw completion and request identity remain in
private evidence.  Because four complete original trajectories are required,
the run correctly stopped before formal credit statistics.

The controlled v5 recovery separately proves that the PPO/update/checkpoint
path works, but its zero rewards/advantages do not resolve this on-policy
format-validity requirement.  No further parser exceptions will be added for
missing or duplicate tags.  A future continuation needs a pre-registered
format-generation intervention that preserves an accounted sampling law, or a
different explicitly authorized generation protocol.  It must retain the
failed full-five attempt and cannot resample until a preferred-looking group
appears.
