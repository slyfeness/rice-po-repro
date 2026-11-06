# Retrieval protocol and vLLM acceptance — 2026-11-06

Base: `7e35862688b63e6be7b954e2a2cf51ac753a2518`, existing PR2. User explicitly permits up to4 GPUs and removes the old32 GPU-hour cap. New four-GPU topologies must await human/ChatGPT review before formal training. No paper-scale run is implied.

1. Preserve immutable parents; inspect saved SwAP configurations and actual logs, not memory alone. Publish redacted source identities, original configuration, layout tradeoffs and measured/estimated resources.
2. Fixed-summary offline retrieval A/B on all32 frozen development members for Initial, GRPO1/RICE1 and GRPO3/RICE3. A=Summary; B=original+one LF+Summary, chosen before results. Reproduce A's full ranking and D0 before paired comparisons. No generated output, trajectory, utility or branch credit is rewritten.
3. Native two-stage vLLM adapter, independent named per-request/per-stage seeds, unchanged T1/p1/top-k disabled,4096 shared action budget. Scripted CPU contracts, then one fixed training query with4 originals/top4 anchors/5 new branches and complete sampled-vs-HF probability checks. Stop strings terminate sampled output without modifying logits; original stop-token IDs must survive. Unsupported output rows retain native failure evidence.
4. Phase-separated immutable-model process lifecycle on the existing two-device layout. Fresh process on policy switch prevents stale KV; snapshot must match actual live Actor. Verify real PPO/checkpoint/reload/next-policy sampling only after full rollout numerical acceptance. Fixed Reference and actual sampled old probabilities remain unchanged.
5. Larger-batch/four-GPU production deployment remains conditional on acceptance and the requested topology review. Do not silently mix a new query protocol into existing summary-only checkpoints.

Ruling: start with phase-separated Actor-device inference rather than unproven Retriever/vLLM co-residency. This preserves the current allocation layout, costs extra snapshot/load time, and does not establish the fastest deployment.

Current ledger: fixed-summary job174993 encountered an immutable-progress-file collision after initial completed rankings; those outputs remain intact and are reused by corrected append-only job174994. CPU174995 could not import pytest because the existing project compatibility path was omitted; no dependency was installed. CPU175000 demonstrated the missing adapter. CPU175002/175003 each ran22 targeted tests successfully (overlapping suites, not44 unique tests). Full rollout job175004 is in progress. All further results must be recorded from actual job terminals.

Final:175004 rollout passed;175018/175019 two-update Trainers and complete32 dev passed;175039/175040 independent recovery passed;175043 result audit completed. TaskA complete offline, taskB submitted for review, taskC complete within observed scope. TaskD larger/formal deployment remains review-gated. No new topology or paper-scale training started.
