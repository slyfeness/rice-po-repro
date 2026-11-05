# Reproducibility contract

The deliverable is an independent, source-traceable implementation of
[RICE-PO v3](https://arxiv.org/html/2605.26352v3) for a first
**DeepSeek-R1-Distill-Qwen-1.5B / BRIGHT** experiment. It is neither an author
code release nor a claim of reproduced scores. The authors' unpublished code
does not prevent implementing the method. It does prevent asserting that
unreported details match their original run.

**Status on 2026-11-04:** source implementation and documentation only.
No dependency installation, project import, tests, model download/inference,
dataset processing, training, benchmark, distributed run, or cluster job was
executed as part of this task. Static inspection and source-file provenance
are distinct from runtime validation.

## Evidence and frozen inputs

[`provenance/sources.json`](../provenance/sources.json) is the machine-readable
source ledger. Preserve immutable revisions, source URLs, licenses, and
file hashes when changing a dependency.

| Input | Frozen identity / evidence | What this establishes |
| --- | --- | --- |
| Method | arXiv:2605.26352v3, revised 2026-08-06 | Equations (1)–(11), experiment statements, appendix and Figure 5 |
| RICE-PO author code | No verified public repository in the bounded 2026-11-04 search; user confirms no released code | Independent paper implementation, not author-source equivalence |
| Actor | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` | A pinned model choice; no weights were downloaded or run |
| Retriever | `AQ-MedAI/Diver-Retriever-4B@ff7f8bc8d1827734dcf2bbfab99f065be618069a` | Original Diver retriever identity, not a silent switch to a later model |
| Diver source | `AQ-MedAI/Diver@cfdf09ad863540f78a655880ef133c4cb28b3676` | Verified dependency source, not RICE-PO source |
| ReasonIR source | `facebookresearch/ReasonIR@0aac96269e455965949df16520fab72da68ffc22` | Verified reference source, not the authors' training split |
| ReasonIR data | `reasonir/reasonir-data@0275f825929b206d4ead23d34b4f8a50d4eddbc8` | Revision retained from the research freeze; payload not revalidated in this task |
| BRIGHT data | `xlangai/BRIGHT@3066d29c9651a576c8aba4832d249807b181ecae` | Revision retained from the research freeze; payload not revalidated in this task |
| Agent prompt | [Figure 5 transcription](../prompts/README.md) and [text artifact](../prompts/ricepo_system.txt) | Paper-supported wording with documented layout changes; not author chat-template bytes |

Selected third-party source files are kept under
[`third_party/reference/`](../third_party/reference/README.md), with original
licenses and upstream Git blob IDs. They are **reference only, not the runtime
engine**. Do not import those files, install their historical requirements as
an implicit environment recipe, or apply the project's software license to
them. The transcribed Figure 5 prompt retains its CC BY 4.0 attribution.

The model/data revisions above were selected and recorded during the source
research. A future asset acquisition must verify which files actually arrived,
record their hashes and tokenizer/config versions, and never substitute
`main` for an immutable revision.

## Published constants versus implementation decisions

[The equation map](PAPER_TO_CODE.md) identifies exact method requirements.
The following quantities are reported in Appendix A.1:

- Four entropy-selected summary anchors per query's pooled trajectories.
- Five sampled Monte Carlo branches per anchor, with the original-count
  ambiguity explained below.
- Influence variance threshold 0.05; residual variance threshold 0.03.
- Maximum depth five retrieval steps.
- Actor learning rate 1e−6, actor KL coefficient 0.001, training batch size 128.
- Fixed Diver-4B+BM25 retrieval; evaluation by NDCG@10.

Values supported by an equation are not all reported experimental settings.
For example, population variance is explicit in (4)–(5), but a trajectory
standard-deviation denominator, epsilon, PPO clipping value, and generation
temperature are not supplied.

The reviewable implementation recipe must label decisions with a rationale
and their source status. It must not populate absent values under a
`paper_explicit` label. In particular:

[`configs/implementation_choices.json`](../configs/implementation_choices.json)
inherits the paper constants and records the current reviewable choices.
Its training budget and train/development manifests deliberately remain empty;
the three missing inputs are not supplied by pretending to recover author
artifacts. [`config.py`](../src/ricepo/config.py) permits documented independent
choices without requiring verified author code, while keeping execution disabled
until an explicitly enabled, complete configuration is supplied.

| Decision | Reproduction policy |
| --- | --- |
| Branch counting | Primary interpretation: five new samples + one original = six outcomes. A five-total variant is a separately named sensitivity configuration. Never infer the branch count from the anchor count four. |
| Local advantage | `center_only` follows Proposition 1's reward-minus-expectation definition. Local standardization is a separate convention/variant. |
| Trajectory advantage | State population/sample std, additive epsilon, and behavior for constant rewards. Current primitive uses population std + epsilon. |
| Counterfactual loss inclusion | Alternatives estimate credit and are audited separately; they are not automatically added to the original optimization group. |
| Retrieval query | Record whether summary alone or original query + summary is used, including the exact separator. This resolves differing wording between §3/Figure 5 and §4.2.2. |
| Fusion and corpus | Record normalization formula/domain, fixed exclusion timing, BM25 analyzer/parameters, dense pooling and normalization, text prefixes, precision, truncation and tie breaking. |
| Observation | Record ranked-list depth separately from actor-visible top-k documents and token truncation. Neither is implied by NDCG@10. |
| Rollout termination | Primary method contract reaches depth five, with final reward from the last retrieval. Early stopping, best-of-round selection, or reward-based stopping would require a declared variant. |
| Prompt/parser | Hash prompt text and chat serialization; record native-token span rules and model special-token behavior. Prompt word limits do not specify tokenizer limits. |
| Invalid output | Explicit failure/abort policy; do not invent an undocumented format reward or repair training tokens silently. |
| PPO/KL | State clipping, loss reduction, exact or sampled KL, update epochs, optimizer parameters, reference snapshot, precision and distributed gradient normalization. |
| Batch and sampling | State whether batch size 128 counts queries or trajectories; record original trajectories per query N, all decoding settings and RNG streams. |

These choices make an independent experiment well defined. They do not require
an author repository to become executable once the concrete recipe, assets,
runtime integration, and authorized experiment budget are ready.

## Dataset identity: counts are not row identities

Appendix A.1 says that BRIGHT-style training examples are sampled from the
ReasonIR release. Table 3 reports the following totals:

| Domain | Paper train | Paper test |
| --- | ---: | ---: |
| Biology | 414 | 103 |
| Earth science | 404 | 116 |
| Economics | 412 | 103 |
| Psychology | 424 | 101 |
| Robotics | 275 | 101 |
| Stack Overflow | 418 | 117 |
| Sustainable living | 426 | 108 |
| LeetCode | 420 | 142 |
| Pony | 94 | 112 |
| AoPS | 384 | 111 |
| TheoremQA questions | 0 | 194 |
| TheoremQA theorems | 340 | 76 |
| Total | 4,011 | 1,384 |

The paper does not publish row IDs, exact sampling/filtering logic, split seed,
or a standalone split manifest. Matching these twelve train counts is
**not evidence that the authors' exact 4,011 examples were recovered**.
Selecting an HQ subset of ReasonIR is an explicit reconstruction choice unless
the paper or a released manifest establishes that subset. Likewise, the
reported test counts alone do not prove the authors used exactly the same
rows as a particular public BRIGHT revision.

[`data.py`](../src/ricepo/data.py) supplies projections, document joins,
count checks and split-manifest validation. It does not claim to regenerate an
unpublished author split. A future reconstruction must publish a small,
reviewable manifest with:

1. Source dataset revision, file/shard, source row ID, domain, and chosen
   query field, plus a hash of the canonical row representation.
2. Corpus identity and positive document IDs, with join failures and
   exclusions documented. Synthetic negative text is not silently treated as
   a corpus document ID.
3. Selection rule and seed, explicit deduplication/filtering decisions,
   before/after counts, and deterministic ordering.
4. Train/development/test membership and manifest hash. Check both stable
   domain-qualified IDs and normalized query-text overlap; document any
   necessary exceptions instead of silently dropping rows.
5. The distinction between an independently reconstructed training set and
   the fixed public evaluation set.

A development split is an independent experimental design choice; the paper
does not disclose a BRIGHT development split or justify a particular 406-row
holdout. Do not invent an author-equivalent development set. Fix its source,
size, seed, and checkpoint-selection rule before training.

The [explicit manifest builder](../src/ricepo/manifest.py) accepts a local
JSON plan and a JSONL selection list. Each selected training/development row
identifies its split, domain, zero-based ReasonIR HQ source row index, and
query ID. The builder uses every BRIGHT reference example in the declared
domains for test, verifies expected SHA-256 values as it reads each input,
joins qrels against the corresponding domain corpus, checks all three splits
for query leakage, and writes per-domain counts and output hashes. The new
output directory becomes visible after the complete bundle is ready. Its
selection plan, data exports, and outputs stay on cluster storage. The module
docstring specifies the future Slurm entry point and plan fields. No plan or
dataset was executed for this delivery.

The public BRIGHT test set is reserved for final evaluation. Do not tune
thresholds, hyperparameters, reconstruction rules, or checkpoint selection on
its scores. Shared public document corpora are allowed; query/label leakage
across experimental splits is not. Keep qrels and gold reasoning out of
actor-visible examples and document observations.

## A complete future run record

A run should be reconstructible from a frozen recipe and immutable artifacts,
not from an undocumented command history. Before starting it, resolve the
remaining numerical and deployment fields and record:

- Repository commit and clean/dirty state; full resolved configuration and
  all implementation-choice evidence.
- Model, tokenizer, retriever, corpus/index, data manifests, prompt and parser
  hashes, plus the environment package lock and device/runtime versions.
- Explicit training budget: total updates, queries and trajectories, original
  N, alternatives per anchor, generation limits, decoding parameters, batch
  semantics, optimizer, seed streams, and checkpoint selection. There is no
  default `total_updates` inferred from a plotted reward curve.
- Slurm resource specification and separate validation of allocation,
  process startup, environment/model initialization, and actual computation.
  A four-GPU allocation is not evidence of correct four-rank execution.
- Explicit large-asset/cache/output roots under
  `/file_storage01/home/<user>/...`; scripts and light control configuration
  may use the small home filesystem. Do not commit site access details.

All cluster computation—including preprocessing, compilation, indexing,
large checksums, inference, training and evaluation—must use an authorized
Slurm job. The repository's source-authoring and publication task does not
start such a job. A later run can proceed through a bounded validation plan;
author-code availability is not an additional launch prerequisite.

## Audit trail and meaningful validation

[`artifacts.py`](../src/ricepo/artifacts.py) and
[`workflow.py`](../src/ricepo/workflow.py) provide atomic, versioned stage
artifacts. For each iteration retain:

| Stage | Required evidence |
| --- | --- |
| Original rollouts | Query and trajectory IDs, old-policy identity, native tokens/masks, all step prefixes/spans, retrieval queries/rankings, local and final rewards |
| Entropy selection | All candidate summary-token entropies, their means, selected anchor IDs and tie-breaking convention |
| Local branching | Original outcome plus every independent alternative, shared prefix identity, seeds, complete continuations and policy version |
| Credit | Raw local/final reward vectors, local moments, residuals, gate decisions, chosen normalization and final token advantages |
| Update | Native-aligned old/current probabilities, declared KL/reduction, loss and gradient diagnostics, optimizer-step count and new checkpoint identity |
| Evaluation | Per-query run scores, per-domain NDCG@10, twelve-domain macro average, fixed data/retriever identities and selected checkpoint |

A receipt signifies completion of that recorded stage only when its producer
has actually succeeded. It is not evidence of successful future stages, and a
checkpoint name alone does not establish successful weight synchronization.

The written tests target properties that can change the method: entropy of
known categorical distributions, population versus sample moments, both gate
boundaries, local replacement versus addition, masked observations, clipping
signs/gradients, causal alignment, and branch independence. They are **written,
not run**. Future validation should also check a tiny end-to-end case with
fixed assets before a larger budget, then verify distributed equivalence of
the declared objective. No runtime or four-GPU correctness claim is justified
by static source review alone.

## Reporting results

Report raw fractional rewards during training and percentage-scaled NDCG@10
only in result tables. Average queries within each domain, then macro-average
the twelve domains. Do not substitute an overall query-weighted mean.

Compare against the paper's 1.5B target (RICE-PO 27.49, GRPO 24.98, prompting
11.75) only with explicit caveats about any reconstructed split or different
runtime convention. Those figures are published reference values, not local
results. A reproduced directional gain and an exact numerical reproduction
are different claims; state precisely which is supported.

The public repository contains code, templates, paper-derived documentation,
and licensed provenance/reference files. Credentials, private cluster access
settings, personal contact details, model weights, datasets, experiment
trajectories, logs, caches and checkpoints stay outside it. Publish an
experiment's compact non-sensitive manifests/results separately only within
the scope of that experiment.
