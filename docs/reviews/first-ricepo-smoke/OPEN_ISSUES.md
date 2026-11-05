# Open issues and next-stage decision

## Blocking issue

The frozen actor did not close the strict reasoning/summary format within the
configured one-step generation bound for the first pre-registered original
sample. Because RICE-PO requires complete original trajectory groups, this is a
valid fail-fast outcome rather than a missing value to impute.

## Decision required

Choose and pre-register a subsequent diagnostic strategy before another formal
RICE-PO attempt. Examples include a separately labelled format-robustness
diagnostic or a documented, non-paper configuration with a different bound.
Neither is automatically equivalent to the paper-reference recipe.

The following actions are explicitly not taken:

- retrying random seeds until the format succeeds;
- silently extending the generation bound and calling it the same run;
- retaining only successful original trajectories;
- assigning zero reward to a malformed or truncated output;
- continuing with incomplete branch or credit groups.

Four-GPU execution remains a separate, unvalidated capability. The frozen
four-GPU configuration from another project was not reused for this reference
runtime.
