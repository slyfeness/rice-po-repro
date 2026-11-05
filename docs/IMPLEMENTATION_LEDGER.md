# Source implementation phase — 2026-11-04

Scope: implement the framework and key files from the paper and publish a public repository. No execution of project code, tests, models, builds, experiment jobs, or benchmarks in this phase.

| Interface | Invariant checked by source review |
|---|---|
| Original rollout → entropy | Same query, policy version, and native generated tokens. Full-vocabulary distributions; no truncated-logprob proxy. |
| Entropy → branches | One pooled top-four selection per query; alternatives begin at the saved history before the paired reasoning. |
| Retrieval → credit | Original and alternatives share a five-step horizon. Per-query benchmark exclusions are fixed across time. Final reward is from the last summary. |
| Credit → update | Population variances, inclusive gate, selected-span replacement, prompt/tool tokens masked, old-policy likelihood and frozen reference separated. |
| Runtime → storage | Immutable stage receipts, source and manifest identities; all large artifacts go to storage-backed paths within Slurm. |
| Publication | Dedicated public repository; no credentials, model/data payloads, cluster logs, trained weights or invented results. |

Static review caught and corrected two method-changing hazards: selecting anchors/normalizing rewards across a mixed-query batch, and removing previously retrieved documents from future rounds. The implementation now groups both selection and advantages by query and keeps benchmark exclusions separate from history. A counterfactual credit group requires the same full horizon for every outcome.

Use `configs/paper.json` for reported constants and `configs/implementation_choices.json` for the independent numerical conventions. No verified author RICE-PO code is claimed. The absent author code does not prevent a paper-based implementation. The unresolved experimental inputs are the train/dev sample manifests, the actual local assets, and the training budget. See [verification status](VALIDATION.md) before interpreting any component as runtime-tested.
