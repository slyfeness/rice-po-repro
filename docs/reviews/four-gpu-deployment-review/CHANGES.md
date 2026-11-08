# Change map

| Item | Source/function | Category and semantics | Evidence |
|---|---|---|---|
| Native vLLM two-stage generation | vllm_two_stage.VLLMTwoStage / VLLMEngine | Missing backend; native4096 shared budget and masks retained. Named request/stage RNG differs from HF bitwise replay; unwarped T1 sampling law unchanged | CPU contracts;175004 full native probability/mask checks |
| Immutable actual-policy lifecycle | vllm_lifecycle.PhaseSeparatedGenerator | Engineering residency/snapshot/process lifecycle; no objective change | Snapshot actual-weight equality; worker-ready identities; online updates and reloads |
| Worker failure cleanup / RPC collision / raw evidence | WorkerTransport._abort, fresh_rpc_path, VLLMEngine.sample | Deterministic defects found by review; no sampling/reward change |175013 RED,175014 GREEN,175017 real process cleanup |
| Trainer dispatch and numerical gate | hf_runtime._train, collect_branches, update, phase barrier | Uses common rollout/credit/PPO; all original/branch sampled probabilities checked before step; Reference fixed |175018/175019 actual online lineage |
| Unsupported standalone holdout | holdout._actor_runtime | Reject sampler substitution before model load; internal dev stays bound to vLLM |175022, two tests passed |
| Query composition | Saved-summary retrieval controller; QUERY_COMPOSITION.md | Offline candidate only; main summary_only unchanged |174994,900 retrievals, all original A rankings reproduced |
| Four-GPU proposal | FOUR_GPU_CONFIG_REVIEW.md, sanitized SwAP config | Real source extraction and candidate tradeoffs; no four-rank claim or new formal topology |169532 original logs/config digests;175005 inventory;175004 real FP32 pools |
| Deployment/docs | deployment.vllm_two_stage.example.json, runtime/provenance docs | Opt-in local-asset template; execution defaults remain disabled | Static JSON/AST/private-marker/size/diff checks |

Known limits: standalone vLLM holdout, Retriever+engine co-residency, four-rank training, cross-query generation aggregation and paper-scale memory/throughput are not validated. The independent two-stage and failure-zero conventions are retained, not relabeled as author-confirmed semantics. No new dependencies, model/data downloads, automatic CI, license or automatic merge.
