# Four-worker prefix affinity — CPU evidence only

`src/ricepo/inference_scheduling.py::partition_requests` already uses deterministic sticky `group_key` grouping. No scheduler change was made.

**MEASURED input accounting:** the frozen 1,152 requests contain 998 groups and 4,610,132 input tokens. Existing partitions contain 290/288/289/285 requests; no group is split. The actual worker reports a 16-token KV block. CPU accounting finds 1,393,360 distinct chained prefix-block tokens globally, versus 2,267,520 when counted separately per assigned worker. Cross-group shared prefixes exist (665,456 distinct tokens); keeping each group sticky cannot keep every shared ancestor on one card.

**ESTIMATED cache bound:** duplicated per-worker prefix storage adds 874,160 tokens, or 18.96 percentage points of input. An unbounded sequential input-prefix model predicts 69.59% single-worker and 50.63% four-worker cold reuse. The 32-token sensitivity gives 18.96 points too. These are structural bounds, not a simulation of concurrent KV insertion, eviction, generated-token KV or preemption.

**MEASURED historical cache behavior:** single S2 cold reuse was 67.69%; four-worker cold reuse was approximately 50–53%. Cross-group duplication can explain the magnitude of this gap, but does not uniquely attribute every miss. The old partitioner did not incorrectly split a group. See `CACHE_ANALYSIS.json` and the preceding phase's four-GPU evidence.

**NOT RUN:** no new affinity scheduler, GPU affinity A/B or four-card run. The task permits a short GPU comparison when justified; this analysis did not establish a concrete alternative preserving load balance with a measured benefit, so it does not justify consuming another GPU comparison. No claim of recovered throughput is made.
