# Two-round protocol and reward diagnostic

The previous five-round trial stopped at depth two of its first original trajectory. It used the adapted prefill prompt (`ricepo_prefilled_think_smoke.txt`). Earlier review text called this the Figure 5 transcription. The transcription is `prompts/ricepo_system.txt`. Use that file with the independent smoke recipe for a new trial. Attribute the fixed-prefix diagnostic to its original prompt.

The former observation renderer appended tool text and a Think marker directly inside the prior assistant turn. The generator also tested for the literal closing Summary string anywhere in the completion. A new optional deployment field `turn_protocol=native_user_observation_v1` frames the retrieved documents as incoming user content, using the frozen tokenizer to start the next assistant turn. The previous generated IDs are retained, including EOS when sampled. One environment EOS is supplied when needed. The new stop check looks for an ordered Think close and tagged Summary close. The role choice is an independent implementation convention. It requires validation on the saved failure.

With identical logits, the original full-vocabulary KL gave value zero and a weighted gradient norm of 1.18e-11 on a fixed CPU input. The first-order PPO implementation now assigns the analytic zero gradient at identical normalized distributions. A CPU regression also compares unequal-distribution gradients with the analytic result. Three local unittest cases passed. These measurements do not identify all sources of the earlier real-model gradient.

## Execution after connectivity was restored

All runs below used source `b5b5407714a8c439b5951b8578f4e2a255f12aa8`. Audit 174191 verified one positive ID for the fixed training query in the frozen corpus. Its ideal-ranking NDCG@10 was 1.0. D0 and all three saved v5 follow-up rankings scored 0.0 with no positive in their first ten positions. Their actual query strings matched the frozen summary-only composer. The historical first action ended with a sampled EOS, followed by inline tool text and one Think prefill. The CPU audit took 46 seconds.

| Slurm job | Experiment | Observed result | Wall time | Peak PyTorch allocation, GiB (actor / second device) |
| --- | --- | --- | --- | --- |
| 174192 | Historical first action plus one second action at the original seed | Second action parsed, 828 new IDs | 2m18s | 9.77 / 0 |
| 174194 | Fresh Figure 5 RICE-PO smoke, 1 query × 4 originals, depth 5, 4 anchors, 5 new branches each | First original completed depths 1 and 2. Depth 3 returned 250 IDs ending in EOS, with an empty reasoning span. The original group remained incomplete. | 4m16s | 16.52 / 7.57 |
| 174196 | Frozen initial-model Prompting on the entire one-query development split | Depth 1 produced 458 IDs, including the tagged summary. The Summary end failed the configured native boundary check. Development coverage was 0/1. | 1m17s | 7.17 / 7.56 |

The latter two single-process jobs overlapped on independent two-GPU allocations on different nodes. The training run recorded 3 generation requests (2,160 output tokens, 11,660 input tokens, 43.52 s), one D0 (1.81 s), and two follow-up retrievals (1.13 s total). Prompting recorded one generation (11.03 s) and one D0 (1.86 s). Timings are host wall time for the named operations. Memory is allocator peak and excludes other device usage.

No complete training trajectory group reached entropy selection, counterfactual credit, PPO or checkpointing in this attempt. Neither the training run nor Prompting produced a complete development NDCG@10. GRPO and the multi-update stage remain unrun. The current frozen bundle has one training and one development query. Previous offline v5 zero-signal results remain separate.

Private evidence SHA-256: saved-rank audit `17ce0cacbb20ab4016c49453bc869dbf0e9877ee95ff801268f0a69da92a3a6d`, replay result `3ea1c33ed99e893a06e93efab5a68f8c753be0f5adc895a88e2ed1278caf5760`, saved-failure span inspection `7e17a1ce1e5e0e5d6ce12dd620a75e00db73d40d9698f80ddd571a0566225d03`.

The two cluster commands below take private run directories as arguments. Use separate Slurm allocations and the project runtime. The audit reads D0 and recovered rankings, checks the qrels against corpus IDs, and scores a positive-ranking control separately. The replay retains the saved first action and uses the original second-round seed once.

```sh
python scripts/audit_saved_turn.py SOURCE_FULL_FIVE V5_RECOVERY NEW_REPORT
python scripts/replay_second_turn.py SOURCE_FULL_FIVE NEW_OUTPUT
```
