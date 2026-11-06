# 32-query `original_and_summary` capacity acceptance

This is a new, single-update capacity run, not a rerun of prior `summary_only` results and not a paper-scale result. Both methods began from the same frozen initial actor and used the same fixed 32-query Biology training batch, four original trajectories per query, five-round limit, four-anchor limit, and five newly sampled counterfactual branches per selected RICE-PO anchor.

The retrieval query protocol was the explicitly labelled independent choice `original_and_summary`: original question, newline, generated Summary. It must not be presented as a recovered author setting. The run also used the independently labelled `two_stage_env_summary_v1` protocol and `invalid_action_terminal_zero_v1` failure semantics.

| Method | Original trajectories | PPO steps | Weight-state change | Checkpoint reload | Complete development queries | Failure-adjusted development score |
|---|---:|---:|---|---|---:|---:|
| RICE-PO | 128 | 1 | yes | verified | 18 / 32 | 0.2646480573 |
| GRPO | 128 | 1 | yes | verified | 17 / 32 | 0.3140633264 |

The complete 32-query D0 macro NDCG@10 was 0.6343234050. It is a retrieval baseline, not a failure-adjusted task score. Scores in the table use the explicitly separate zero-utility failure terminal and cannot be relabelled as raw NDCG.

RICE-PO did not merely execute the GRPO fallback: it selected 128 anchors, found 41 non-zero local-credit cases, replaced 72 Summary-token advantages, and passed both paper gates for 3 anchors. Of the 128 original trajectories, 53 ended under the declared invalid-action terminal semantics. These outcomes make the capacity run operationally successful but **not sufficient to authorize a 128-query × 10-update candidate run**: five-round completion was only 18/32 for RICE-PO and 17/32 for GRPO.

The private completion receipts and resolved-config fingerprints are recorded below. Public review material intentionally excludes raw prompts, query identifiers, native token IDs, generations, rankings, checkpoints, logs, usernames, hosts, and filesystem paths.

| Evidence | SHA-256 |
|---|---|
| RICE-PO resolved runtime configuration | `5a443709cf356fbf40f545b78673a4dc3a7574c6b76fa24055e8d00fd7adcf02` |
| GRPO resolved runtime configuration | `f4a73997c68713fc986598087122531fd1d06f38482df7c41e91ec8e9f5e7015` |
| RICE-PO private completion receipt | `2687977cea28f6a68160228a4022ec9f2b195a124dfb5ba908d6719ff129af5a` |
| GRPO private completion receipt | `5b174434029ac0379e94cf8e44b053357cf1879c5b8bc1addd4321e66873f1df` |
| Frozen source archive | `06dbac7041bd73aa1cfca8aab3a89e198f15983844b8ce816f99b092f97fc4a7` |

Source revision: `d1d5bda964010f3e1185cb4566e57a0ae0ad06c3`. RICE-PO recipe fingerprint: `e8c2f2ef3773ccd6b87e05214066783ceffd7f3d30e57cc8cfe7f03ba68f239c`.

Actual resource use was two independent two-GPU Slurm allocations, not a four-rank distributed trainer. RICE-PO elapsed 7,614.265 seconds and GRPO elapsed 3,042.455 seconds. These figures include their actual run lifecycle and must not be interpreted as paper-scale throughput projections.

See [TEST_RESULTS.md](TEST_RESULTS.md) and [OPEN_ISSUES.md](OPEN_ISSUES.md).
