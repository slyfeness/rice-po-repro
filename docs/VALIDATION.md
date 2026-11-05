# Validation status

**Source-only delivery. No runtime validation has been performed.** The written fixtures are proposed checks, not passing results. There is no local result for Table 1 and no measured GPU memory, throughput, training curve, or retrieval score.

| Check | Delivery status |
|---|---|
| Read method, appendix and Figure 5; map equations to functions | Source review |
| Compare pinned third-party reference files with their upstream Git blobs and SHA-256 records | File identity review |
| Review query grouping, fixed benchmark exclusions, full-depth branches, native causal offsets and PPO loss masks | Source review |
| Python source grammar; JSON/TOML serialization; Git tree and whitespace checks | Static inspection; no project imports |
| Unit/integration tests, including tensor fixtures | **Not run** |
| Dependency compatibility, runtime model/tokenizer load, dense index, BM25/JVM and qrels | **Not run** |
| Reference training, validation, BRIGHT test scores | **Not run** |
| Slurm submission, four-GPU allocation or four-rank program acceptance | **Not run** |

## Suggested future acceptance criteria

1. Under an authorized CPU allocation, run the unit fixtures and freeze package versions. Include grouping of multiple queries, inclusive thresholds, non-generative token masks, both PPO clipping signs, and exact old/reference token alignment.
2. Under a separately authorized small GPU job, compare chunked entropy with a full causal forward on the same native IDs; compare old-policy probabilities with the actual generation distribution. Inspect D0, fixed exclusions, repeated-document legality, and matched branch prefixes manually.
3. Validate corpus/vector identity and the independent data split. Compare top-ten rankings and raw fractional NDCG against the fixed upstream implementation on a small, frozen case. Report which part is an implementation convention.
4. Require a nonzero update with finite loss/gradient, a verified new checkpoint, no gradient on reference parameters and successful fresh-checkpoint rollout. Then separately verify distributed loss/gradient equivalence and four-rank memory if that path is pursued.
5. Only after a fixed budget and independent train/dev/test manifests exist should a result be compared with the paper. Use development for selection and reserve BRIGHT test for the selected checkpoint.

All future compute, including CPU preprocessing and substantial hashing, belongs in Slurm. Publishing this source authorizes none of these stages.
