# Actual cost and resource accounting

GPU allocation seconds include initialization, hashing/loading, inference, failed jobs, verification, evaluation and I/O; they are not kernel-active seconds. Maximum concurrent allocation was4 GPUs. The user removed the previous32 GPU-hour stopping cap during this phase; measured usage is retained as evidence, not a remaining permission counter. All listed GPU jobs are terminal.

| Method continuation | Allocated elapsed | Actor/Reference allocated peak GiB | Reserved peak GiB | Diver allocated peak GiB |
|---|---:|---:|---:|---:|
| RICEPO | 6962 s | 56.45 | 76.71 | 7.59 |
| GRPO | 3943 s | 51.62 | 78.53 | 7.59 |

RICE includes recovery-backed update2 plus fresh update3 and all32 final development; GRPO includes two fresh updates and final development. The preceding failed RICE allocation adds3194 seconds ×2 GPUs and is not omitted. These are allocator high-water marks, not exclusive per-stage or whole-device/NVML peaks. vLLM parent-process zeros are explicitly not worker memory measurements; optional NVML snapshots were unavailable, so no vLLM worker peak is claimed.

| Inclusive host wall category | RICE child seconds / calls | GRPO child seconds / calls |
|---|---:|---:|
| Original generation | 558.573 / 115 | 1092.199 / 104 |
| Branch generation | 3186.601 / 429 | not executed |
| Full-vocabulary entropy | 99.478 / 27 | not executed |
| Old-probability cache/check | 46.558 / 2 | 38.150 / 2 |
| Update | 275.180 / 2 | 232.912 / 2 |
| Checkpoint writing | 436.019 / 2 | 679.590 / 2 |
| Training D0 | 4.689 / 8 | 6.608 / 8 |
| Training follow-up retrieval | 239.357 / 457 | 46.143 / 83 |
| Complete development | 1480.802 / 1 | 1321.065 / 1 |

Wall categories are nested and must not be added. A completed generator call can return an illegal terminal action; it is not a legal trajectory completion. Cached native replays make no new model tokens; deterministic retrieval and actual partial re-execution still consume measured time. Branch prefix/history reuse is not new generation. Counts below are physical recorded generation events, separate from the action-token counts used by PPO/credit.

| Recorded generation / tool count | RICE successful child | Preserved failed RICE predecessor | GRPO child |
|---|---:|---:|---:|
| Original model tokens | 22274 | 26359 | 45052 |
| New branch model tokens | 126350 | 59587 | 0 |
| Development model tokens | 59233 | 0 | 48537 |
| Training follow-up requests | 457 | 146 | 83 |
| Development D0 requests | 32 | 0 | 32 |
| Development follow-up requests | 96 | 0 | 99 |

The old failed callback lost a partial Summary: its unknown old tokens are not silently zeroed or reconstructed as historical evidence. Recorded token totals spanning that failed attempt are lower bounds, while allocation time is complete. The90-token same-seed recovery action is new physical generation and remains identified as a replay, not a new independent statistical branch.

Additional diagnostics: HF whole sampling workload produces42145 tokens per mode; real weight-switch gate produces1449 and1214 per mode. vLLM TF32/eager, default-TF32 graph and IEEE graph workloads retain their own native counts and failures inEVIDENCE; only the IEEE80-request numerical gate passed the full declared test. Failed IPC/API/wrapper/callback jobs remain charged. CPU preparation/analysis/extraction/ZIP jobs request no GPU.

This phase: 10.128611 GPUh; previous phases: 10.791666; cumulative: 20.920278. Source Slurm root allocations, submit/start/end and exit codes are inEVIDENCE. Pending-only cancelled jobs have0 allocated seconds. There is no claim that all four GPUs were continuously busy, nor a four-rank training validation.
