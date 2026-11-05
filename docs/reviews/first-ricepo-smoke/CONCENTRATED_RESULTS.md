# Concentrated repair and controlled execution

Executed source was 6dbdd397e36d65f98271c2d1ea46207af7917e9e. The new body rule is native_body_coverage_v1. It assigns each intersecting native token wholly to its body, extracts the exact character Summary for retrieval, and rejects two bodies sharing one token. Tagged structure and empty-Think acceptance are independent settings. Historical strict parsing remains selectable.

The current run used tagged structure, body coverage, empty Think allowed, native_user_observation_v1, the Figure 5 transcription, summary_only, unwarped sampling (1/1/0), and a 4096-token generation ceiling. Generated format tokens remain actions. Environmental EOS and role tokens remain masked context. Stop detection and parsing share the tagged outer-structure recognizer. Branch seed schedule controlled_v2 removes policy-version dependence. Behavior identity binds parser options, prompt SHA, sampling, weights and protocol.

The baseline CLI accepts --split development or test. Development chooses the bound development manifest, identities and labels. Trained formal test selection still requires the selected checkpoint. The earlier 174196 private development script actually read the frozen development manifest directly. Its failed result had coverage 0/1.

## Executed data and tests

CPU job 174207 built a separate Biology bundle with 128 training, 32 development and 103 official test queries. Selection sorted the SHA256 of fixed seed 20261105 and upstream row index, excluded official test text, required every positive to map into the full Biology corpus, and deduplicated canonical text. The existing corpus vectors were reused. The bundle receipt SHA256 is 56b57c5dc5a22fe521eff09380ad02248372401b5be57760ea540a82ed32b303. Count policy is report_only. Author training membership remains unavailable.

Five focused local CPU tests passed before source publication. An additional zero-advantage AdamW/sign-gradient regression subsequently passed, bringing the relevant total to six. Saved-token CPU job 174210 accepted the prior legal development boundary failure and the saved v5 tagged branch with identical source hashes. The prior third-round output lacked Summary tags and was correctly rejected. Legal empty Think is covered by a constructed tagged fixture. Exact-distribution KL has analytic zero value/gradient, unequal-distribution gradients match the analytic result, and a new zero-decay optimizer leaves a zero-signal fixture unchanged.

## Actual outcomes

| Job | Outcome | Allocated GPU hours | Peak allocator GiB, actor / retriever |
| --- | --- | --- | --- |
| 174212, RICE-PO | Depth 1 accepted. Depth 2 returned 541 native IDs at EOS, with one Summary pair and no closing Think tag. Incomplete first original group. | 0.107778 | 16.30 / 7.56 |
| 174213, Prompting | First development query returned 553 native IDs at EOS with closing Think but no Summary pair. Complete-score coverage 0/32. | 0.057778 | 7.17 / 7.56 |
| 174215, D0 | Complete development coverage 32/32, macro NDCG@10 0.6343234049838116. | 0.040000 | 0 / 7.56 |

The two initial jobs overlapped as independent two-GPU allocations. Total new allocation cost was 0.205556 GPU hours against the 32 GPU-hour ceiling. CPU publication/data/audit jobs allocated zero GPUs. Source failures were retained. No optimizer update, entropy anchor set, counterfactual group, credit record, new checkpoint or train curve was produced in this execution. GRPO and the 3-to-20-update comparison were not started because complete online acceptance failed. The batch templates are disabled candidates.

D0 is the only complete new development metric. Prompting and trained-policy metrics are missing. Neither malformed sample received a synthetic reward. New seeds or dropped branches were not used to complete a group.

The inspected SwAP successful generation settings were four TP1 vLLM replicas, concurrency 144 per GPU, memory utilization 0.90 and max_num_batched_tokens 32768. This execution used the existing HF two-device driver, following the task's first-runtime requirement. RICE-PO integration of that vLLM topology, generation/scoring probability agreement and per-update weight synchronization remain unvalidated. Its throughput was not measured here.

Private full-score artifact SHA256 is c99767d8ee268c7ef430b5e7e22dfc1ba7747065a679f0da34a22b4261322e31. RICE-PO failure SHA256 is 1098be5a36bb431592ee932daedb804d50521181b8cbe1a406157d44d7df4e45. Prompting failure SHA256 is 1ad94886d67943c8bbc97486400724b7385fdba00f9e1394232594ff926136df. Raw data, weights, completions and machine logs remain private.

## Paper interpretation

Fixed depth five is an independent convention for a common counterfactual endpoint. The paper gives maximum depth five without a fully disclosed natural-stop rule. Formal equations (8)-(9) determine replacement in this implementation despite the schematic addition conflict. Summary-only query composition, four originals, six inclusive branch outcomes, query-unit batch interpretation, sampling law and one-pass PPO remain explicit independent choices. Formal batch 128, twelve-domain replication and author sample membership still require further resources or upstream information.
