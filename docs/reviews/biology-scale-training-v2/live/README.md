# Live snapshot — 2026-11-08

Read-only snapshot of the active Biology-scale formal jobs: RICE-PO `177622`
and GRPO `177623`.  It does not establish a completed update.

- `LIVE_SNAPSHOT_*.json` contains Slurm state, persisted metadata-record
  counts, and explicitly labels metrics that the running program did not
  persist as `not_measured`.
- `GPU_UTIL_SUMMARY.csv` aggregates a 65-second, one-second NVML sample from
  only the two GPUs allocated to each named job.  Device labels are local to
  each allocation and do not identify a cluster node.
- `PROGRESS_TIMELINE.csv` is a per-minute metadata-write timeline; it is not a
  token-completion counter.
- `LOG_TAIL_*.txt` contains redacted controller/worker output tails.  Internal
  paths, usernames, and IP addresses were removed.
- `SOURCE_BINDING.json` records the source-commit value observed directly from
  each active worker process environment.

The cancelled auxiliary CPU job `177963` parsed full event payloads too
broadly; it never wrote to either training run.  Job `177992` produced the
published metadata-only snapshot.
