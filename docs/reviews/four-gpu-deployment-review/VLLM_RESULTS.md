# Native vLLM acceptance — 2026-11-06

## Full rollout: passed, not an optimizer experiment

Slurm175004 completed/0 in1204s on2GPUs (0.668889 GPU-hours). It used the same frozen training query, first-update Actor checkpoint and retrieval assets as the previous HF wavefront workload, with an explicitly new per-stage RNG stream. No prompt, temperature, query composition, reward, gate or advantage formula changed. This is a backend diagnostic, not a continuation or comparable new training curve.

- Four originals all reached five rounds. Four entropy anchors each received five new branches,20 total. Fourteen branches reached the normal endpoint and six reached the authorized invalid-action terminal; none were discarded/retried. The four anchor groups remained estimable because all anchor-local retrievals existed, including branches that failed later.
- All24 complete native histories passed actual sampled LP versus HF teacher scoring. Maximum history9023 tokens; maximum action budget4096 was actually reached once. Mean-error range4.01e-6–8.02e-6 nats, worst history p994.816e-5, maximum error0.0001688004. Ratios range0.9998312–1.0001491. Thresholds stayed mean0.02/p990.06/max0.12; the historical field name `within_bf16_tolerance` does not mean this run used BF16.
- CPU175024 independently verified native action/environment masks, original-prefix matches, all six outcomes including one original per anchor, population variances, original-local-minus-mean advantages, Summary replacement and zero environment advantages. Maximum action4096; failure categories1 native_text_encoding and5 tag_structure. Raw IDs/text remain private; no format repair occurred.
- All four Summary local advantages were nonzero:0.088363,0.003828,0.218410,-0.270400 (record order in evidence). Local variances0.014718/0.033362/0.043706/0.014884 all below0.05, so no reasoning propagation. Residual variances0.020733/0/0.027565/0 all meet0.03. Two depth5 anchors have zero residual by definition; this is not evidence of future stability.

Evidence: [native numerical checks](VLLM_NATIVE_EVIDENCE.json), [independent native/credit audit](VLLM_CREDIT_EVIDENCE.json). Both bind private originals/branches/checkpoint by SHA-256. There was no optimizer step in175004.

## Throughput and memory scope

| Fixed-query method workload | Actual generated tokens | Generation calls including lifecycle (s) | tokens/s | Sampling/retrieval/entropy/credit wall (s) |
|---|---:|---:|---:|---:|
| Earlier HF serial |42145|973.13|43.31|1044.92|
| Earlier HF wavefront |42145|885.69|47.58|942.86|
| vLLM native |33417|454.77|73.48|522.83|

HF serial/batch had identical native outputs. vLLM uses different named stage seeds; its outputs, selected anchors, failed endpoints and token workload differ. Therefore this table is an observed workload comparison, **not an equal-output causal speedup measurement**. The vLLM generation timer includes engine startup within calls; the workload wall excludes subsequent full-history probability scoring. Allocation1204s also includes initial checkpoint verification/loading (substantial shared-storage IO), scoring and shutdown. Max native batch here is4 original/5 branch requests, not64 saturation.

The vLLM worker logs show FP32 model loading6.71GiB, KV pools54.59–54.63GiB and Graph pool0.17GiB; these pool measurements are not full-device peak accounting. Parent PyTorch peak excludes the worker allocator: Actor-side allocated15.36GB/reserved23.15GB and Diver8.15GB/8.17GB for this no-gradient diagnostic. It must not replace the historical training peak near80GiB reserved. Separate prefill/decode kernel timing and synchronized whole-device peak remain unmeasured.

## Trainer integration and remaining gate

Implemented `vllm_two_stage.py`, `vllm_lifecycle.py` and the minimal Trainer dispatch. The Actor GPU alternates immutable inference workers and HF scoring/PPO; Diver keeps its second GPU. Worker startup verifies actual serialized weights against live Actor identity. Every policy change uses a fresh snapshot/process/KV state. All original and branch LP checks precede optimizer commit. The Reference remains the initial frozen model. This is a new backend within the existing allocation, not TP4/FSDP4.

Online acceptance175018 RICE/175019 GRPO uses independent two-GPU jobs, common initial FP32 weights, the same fixed two-query schedule, two updates of one query each,4 originals/max4 anchors/new5, five-round maximum and complete32 development after the second update. It is a new lineage, not mixed into old three-update checkpoints. Both jobs completed two real updates, complete32 development, independent Actor/optimizer/CPU/CUDA RNG reload and actual post-update sampling under new processes. See SUMMARY.md and ONLINE_EVIDENCE.json for exact results; this is not paper-scale performance acceptance.

Public deployment template remains opt-in; public research recipes retain execution.enabled=false. Formal standalone holdout rejects this backend until its independent lifecycle exists; internal development uses the live bound sampler. Four-GPU deployment review and larger-batch testing remain separate gates.
