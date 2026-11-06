## 2026-11-06 连续更新阶段

本阶段以768f60f为基准，保留已提交的第一更新与私有资产；两种方法在新child-run内继续全局更新2/3，使用同一冻结查询日程。128/32/103数据包和summary_only检索组成不变；四查询诊断批次不能冒充论文128查询训练。两阶段/失败零效用仍是独立约定；详见[数据身份](reviews/continuous-three-update/DATA_AUDIT.md)、[逐轮退化](reviews/continuous-three-update/RETRIEVAL_AUDIT.md)及[局限](reviews/continuous-three-update/OPEN_ISSUES.md)。历史结果与未运行声明保留各自阶段适用范围。

---

## 2026-11-06 两阶段受限验收

本轮实际使用128/32/103冻结包中的4训练查询，所有三组完整32题开发均计入失败，正式测试未执行。非零局部信用与权重变化已核验，开发性能下降；详见[独立复现边界](reviews/two-stage-native-v1/SUMMARY.md)。上游来源核验不因本地指纹一致而升级。

---

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

## Controls in the revised checkout

Before sampling, `verify_bundle` checks the externally pinned receipt, preparation plan, three manifest byte streams and label-free membership records. The selected train/dev manifests must belong to this same bundle. Distinct official test tasks remain identifiable even when different domains share the same query text. The independent train and development counts, source selection and collision policy are recorded separately.

Experiments bind source, resolved recipe, dataset, actual model/tokenizer contents, prompt, retrieval index and behavior settings. D0, prompting, GRPO and RICE-PO share the evaluation component. Index consistency is verified against local bytes. Authentication of the upstream model payload requires independent evidence.

Label a result with its profile, selected checkpoint, exact evaluated IDs and costs. Equal original-query and trajectory budgets include RICE-PO's extra branches and entropy computations. Equal-compute comparisons require measured budgets and accounting for failures and replay. Freeze all choices before accessing test scores. [Runtime contract](REFERENCE_RUNTIME.md) · [Retrieval comparison](RETRIEVAL_AUDIT.md) · [Change ledger](IMPLEMENTATION_LEDGER.md)

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

The [manifest builder](../src/ricepo/manifest.py) takes an offline JSON plan
and an explicit membership JSONL. Each membership row supplies split, domain,
assigned query ID, and a zero-based nonblank record index in the declared ReasonIR
HQ export. A v2 plan also declares a selection rule, filter list, and seed
(integer or null) separately for train and development. The source adapter
takes the query from `query[1]` and joins every `pos[*][1]` to the domain
document ID. A missing document, empty positive set, or positive also marked
excluded aborts preparation. Development uses independently nominated HQ
rows. For test the builder preserves every record of each declared BRIGHT
examples export, including its native ID, `query`, `gold_ids`, and exclusions.
The declared dataset revision and verified export bytes establish the local
input identity. Authenticating the exports' origin requires acquisition records.

Preparation writes train, development, and test JSONL, the original bytes of
`preparation-plan.json` and a v2 `receipt.json` together in a new directory.
The receipt binds plan and input digests, role and
path of each local export, counts before selection, all domain counts after
selection, projection and join rules, the exact selected IDs, source and
selection row indices/digests, normalized query digests and split hashes.
Per-row positive IDs live in the corresponding split manifest's qrels.
Counts include the paper's zero for TheoremQA questions training and separate
booleans for agreement with the paper's train and test counts.

Collision records distinguish repeated domain-qualified IDs, reuse of HQ
source rows, within-split repeated text and train/development/test overlap.
If distinct official tasks in different test domains share normalized text,
both records are retained and reported. Other collisions stop publication
and write a sibling `*.overlap-*.json` audit listing every implicated task.
Each task retains its own corpus, positives and per-domain contribution.

Pin the byte SHA-256 of the preparation receipt in deployment alongside
the receipt path and the three sibling manifest paths. The
[bundle verifier](../src/ricepo/dataset_bundle.py) checks that pin, all three
file digests, per-domain corpus identity, complete source-index coverage,
the overlap audit, and expected task membership. It verifies test bytes
without parsing test judgments. The training runtime receives identities and
hashes, and reads only train/development rows for optimization and selection.
Both selection and final evaluation aggregate against the receipt's exact
domain/query-ID pairs. A missing query in an otherwise represented domain
invalidates a complete macro. Explicit diagnostics publish a separate
`partial_macro`, expected/actual counts and missing IDs. The preparation and
verification interfaces are authored for future authorized Slurm use.

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
