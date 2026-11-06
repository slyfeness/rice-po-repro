# Existing 32-query retrieval diagnosis

This is a descriptive analysis of the existing initial, GRPO-update-1 and RICE-update-1 development trajectories. No new generation, prompt search, query selection or reward modification was used. All 96 saved D0/per-round rankings were rescored against the frozen qrels using the same `PyTrecReward.score` (`ndcg_cut_10`); every stored score agreed. The complete query membership was checked against the 32-query manifest.

The observational unit is one `(domain, query_id, policy)` trajectory, paired across policies. A round exists only after a real legal retrieval action. Missing later rounds are not imputed. `task_score` is the explicitly independent failure-adjusted utility, while `last_valid_ndcg` is the last measured retrieval result, including D0 when no later valid retrieval exists. Query digests are stable pseudonyms, not anonymity guarantees.

| Policy | Legal five-round | All-32 task utility | All-32 last-valid NDCG | D0 on legal subset | Final NDCG on that same legal subset |
|---|---:|---:|---:|---:|---:|
| Initial | 14/32 | 0.159289 | 0.327628 | 0.641635 | 0.364090 |
| GRPO update 1 | 15/32 | 0.047096 | 0.236521 | 0.666488 | 0.100471 |
| RICE update 1 | 13/32 | 0.027714 | 0.184679 | 0.706296 | 0.068219 |

All-32 D0 is 0.634323 for every policy. It is not compared to a differently sized successful subset. On the five queries completed by all three policies, matched D0 is 0.636371 and final NDCG is Initial 0.466667, GRPO 0.086135, RICE 0.100000. This small conditional intersection is supplementary, not the full benchmark.

Both genuine ranking loss and terminal failures matter. The exact descriptive decomposition on the same 32 queries is:

| Policy | D0 minus task utility | D0 minus last-valid NDCG | Last-valid NDCG minus task utility |
|---|---:|---:|---:|
| Initial | 0.475034 | 0.306695 | 0.168339 |
| GRPO update 1 | 0.587227 | 0.397802 | 0.189425 |
| RICE update 1 | 0.606609 | 0.449644 | 0.156965 |

The last column is the arithmetic utility gap caused by the registered failure rule, not a causal estimate or a claim that failed retrieval produced zero NDCG. Most measured loss remains even before that gap is applied.

The earliest decline versus the preceding real ranking occurs at round 1 for 11 Initial, 15 GRPO and 14 RICE queries; at round 2 for 6, 3 and 5; at round 3 for 0, 1 and 1. Queries without a decline are retained. First-round matched comparisons are Initial 29 queries: D0 0.630978 → 0.387516; GRPO 30: 0.609945 → 0.269769; RICE 30: 0.609945 → 0.276444. Later-depth means and their matching query counts/D0 are in `RETRIEVAL_EVIDENCE.json`; changing membership makes them unsuitable as an unqualified time series.

Among legal five-round completions, an intermediate round beats the final round for 4 Initial, 8 GRPO and 5 RICE queries. Exact normalized-summary repetitions occur in 4, 7 and 8 adjacent transitions; exact ranking repetitions in 7, 7 and 8. The original text and private rankings remain linked by SHA-256, without uploading them.

The three queries with different GRPO/RICE task scores explain their aggregate update-one difference. Prefixes below abbreviate full query digests in the JSON evidence:

| Query digest | First differing Summary/ranking/score round | GRPO final task score | RICE final task score | Observation |
|---|---:|---:|---:|---|
| `25c08f2fff73` | 3 | 0.356207 | 0 | Both complete. At round 3 the relevant document is rank 6 for GRPO; RICE's real NDCG becomes 0. |
| `9d4def0f4306` | 2 | 0.333333 | 0 | Both first retrieve a rank-1 positive. GRPO's round-2 positive is rank 4, RICE's rank 6; RICE then fails at round 3. Its last-valid NDCG remains 0.356207, distinct from terminal utility 0. |
| `cb8c01cc054a` | 4 | 0.430677 | 0.500000 | Round 3 is 0.5 for both; at round 4 positives are ranks 4 and 2; final positives are ranks 4 and 3. Both complete. |

Private inspection confirms actual topic drift in some severe-loss cases, not just a parsing artifact. For example, digest `2ab6b43c8f5f` starts at D0=1, then the initial policy changes topic in round 1 and moves further off topic later, with all five real retrieval scores zero. Other cases stay on topic yet lose relevant rank after changing wording. These observations support separate hypotheses of summary drift, low-yield repetition and continuing after a good intermediate ranking. They do not establish that local credit caused a specific sampled wording or that stopping earlier would be an unbiased alternative benchmark.

All 109 Initial, 114 GRPO and 104 RICE legal follow-up requests exactly equal their saved Summary text: this confirms the existing `summary_only` implementation choice. The original query is present in the native initial history, but is not automatically concatenated by the retriever. No switch to `original_and_summary` was made. Mean original-query lexical retention falls from 0.356 to 0.237 for Initial and 0.329 to 0.217 for RICE, across changing round-specific subsets; this is only a lexical proxy, affected by paraphrases/language, not semantic proof.

Qrels spot checks for the three paired cases found every positive ID in the actual fixed corpus and semantically corresponding positive excerpts. The nominal Biology asset includes documents on vision and astronomical brightness; the latter case is not an ID-offset artifact. This spot check does not certify the entire synthetic dataset or establish author-identical samples.

Evidence: CPU Slurm jobs 174717 (analysis plus four targeted tests), 174720 (mechanism summaries), 174721 (parent selection and positive-document spot checks). `RETRIEVAL_EVIDENCE.json` contains per-query/round scores, relevant ranks, paired divergence and private-file hashes; `RETRIEVAL_MECHANISMS.json` contains failure depths and lexical/repetition descriptors. Full query/Summary text, IDs and rankings remain private. No causal or significance claim is made from this single-seed, small diagnostic.
