# Two-round protocol and reward diagnostic

The previous five-round trial stopped at depth two of its first original trajectory. It used the adapted prefill prompt (`ricepo_prefilled_think_smoke.txt`). Earlier review text called this the Figure 5 transcription. The transcription is `prompts/ricepo_system.txt`. Use that file with the independent smoke recipe for a new trial. Attribute the fixed-prefix diagnostic to its original prompt.

The former observation renderer appended tool text and a Think marker directly inside the prior assistant turn. The generator also tested for the literal closing Summary string anywhere in the completion. A new optional deployment field `turn_protocol=native_user_observation_v1` frames the retrieved documents as incoming user content, using the frozen tokenizer to start the next assistant turn. The previous generated IDs are retained, including EOS when sampled. One environment EOS is supplied when needed. The new stop check looks for an ordered Think close and tagged Summary close. The role choice is an independent implementation convention. It requires validation on the saved failure.

With identical logits, the original full-vocabulary KL gave value zero and a weighted gradient norm of 1.18e-11 on a fixed CPU input. The first-order PPO implementation now assigns the analytic zero gradient at identical normalized distributions. A CPU regression also compares unequal-distribution gradients with the analytic result. Three local unittest cases passed. These measurements do not identify all sources of the earlier real-model gradient.

The SSH endpoint closed the connection before key exchange during this phase. The cluster prefix and reward audits, fixed-seed replay, four-GPU execution, new five-round update and development comparisons are pending. There is no new development NDCG or peak GPU memory measurement.

The two cluster commands below take private run directories as arguments. Use separate Slurm allocations and the project runtime. The audit reads D0 and recovered rankings, checks the qrels against corpus IDs, and scores a positive-ranking control separately. The replay retains the saved first action and uses the original second-round seed once.

```sh
python scripts/audit_saved_turn.py SOURCE_FULL_FIVE V5_RECOVERY NEW_REPORT
python scripts/replay_second_turn.py SOURCE_FULL_FIVE NEW_OUTPUT
```
