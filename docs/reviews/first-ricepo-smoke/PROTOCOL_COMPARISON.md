# Native failure classification and paired training-query protocol comparison

Runtime source: c6c9af6c7cad82d6c321ed55066592f19d3e71e9, frozen as one
release before both jobs. No production parser or reward change was made.

## Saved failures

CPU job 174241 decoded the saved IDs with the frozen tokenizer and preserved
the complete native IDs, exact decoded text, original reason and source-file
SHA256 in private per-case evidence. The following counts cover the 28
structure failures. Labels overlap. The remaining length-exhaustion case is
reported separately.

| Label | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 | Total |
| --- | --- | --- | --- | --- | --- | --- |
| Repeated opening Think after prefill | 0 | 0 | 0 | 0 | 0 | 0 |
| Missing closing Think | 0 | 2 | 1 | 0 | 1 | 4 |
| Missing opening Summary | 11 | 2 | 3 | 0 | 1 | 17 |
| Missing closing Summary | 14 | 2 | 3 | 0 | 1 | 20 |
| EOS before a valid outer action | 20 | 4 | 3 | 0 | 1 | 28 |
| Nested tags | 0 | 1 | 0 | 0 | 0 | 1 |
| Duplicate tags | 0 | 1 | 0 | 0 | 0 | 1 |
| Mismatched tag order | 0 | 2 | 0 | 0 | 0 | 2 |
| Other unclassified structure | 0 | 0 | 0 | 0 | 0 | 0 |

The one length-exhaustion failure occurred in round 1 and also contained
duplicate/mismatched tags. Including it yields two duplicate-tag cases and
three mismatched-order cases over all 29 failures. Early EOS co-occurs with
the missing/invalid tags; it is a termination observation, not an independent
disjoint cause.

The Figure 5 text explicitly presents an opening Think tag while the frozen
chat template already supplies it. The existing compatibility prompt states
that the opening tag has been prefilled and requests its closure. This is an
instruction-level mismatch. The saved 28 cases contained no repeated Think
opening, so those failures do not establish repeated prefill as their cause.

## Controlled eight-query result

Eight training queries were selected before generation by a fixed hash rule
over the frozen 128-query training manifest, independently of rewards and
format outcomes. Both conditions used the identical 1.5B weight fingerprint,
per-query/depth seeds, D0 rankings, Diver/BM25 assets, summary_only composition,
temperature 1, top_p 1, top_k 0, 4096-token ceiling, body-coverage parser with
tagged structure and empty Think allowed, and native_user_observation_v1.

Original Figure 5 prompt SHA256:
e2241e76e95dc773dc5250afc68ff352b655ae2e631a0db572855a2f7cebe04f.
Prefill compatibility prompt SHA256:
e7b28d3b6c7aebf97018936e2d2a0750a0cd447d61cb8b4be8935beef57b3423.
The compatibility prompt is an explicitly edited format variant.

| Round | Figure 5 accepted / attempted | Compatibility accepted / attempted |
| --- | --- | --- |
| 1 | 2/8 | 6/8 |
| 2 | 1/2 | 3/6 |
| 3 | 1/1 | 3/3 |
| 4 | 0/1 | 1/3 |
| 5 | 0/0 | 0/1 |
| Complete five-round queries | 0/8 | 0/8 |

Later-round denominators are the surviving attempted queries. They differ by
condition and are not an unconditional paired success estimate. Figure 5 had
seven structural failures and one native-decoding mapping ambiguity at round
2. The compatibility variant had eight structural failures. First-round
acceptance improved by 4/8 in this small fixed comparison; complete five-round
acceptance did not improve.

No condition triggered the complete-online-training gate. No additional
original group, counterfactual group, credit calculation, optimizer step or
checkpoint was created. The restricted batch plan remains unstarted.

| Cost | Figure 5, job 174243 | Compatibility, job 174244 |
| --- | --- | --- |
| Allocation elapsed | 284 s | 303 s |
| Allocated GPUs | 2 | 2 |
| Generation calls / output tokens | 12 / 7,904 | 21 / 10,081 |
| Generation host time | 178.36 s | 224.17 s |
| Follow-up retrieval calls / host time | 4 / 6.74 s | 13 / 7.20 s |
| Peak PyTorch allocator bytes | 7,941,031,424 / 8,125,903,360 | 7,909,591,040 / 8,122,843,648 |

The jobs overlapped as two isolated two-device processes. Pair cost was
0.326111 GPU hours; cumulative task cost is 1.072222 GPU hours against the
existing 32-hour budget.

## Protocol decision and proposed failure environment

Prefill compatibility is the better first-round candidate in these eight
training queries. It has not passed full-episode protocol acceptance. No
further parser exception, seed resampling, new reward or constrained decoding
was introduced.

[INVALID_ACTION_DESIGN.md](../../INVALID_ACTION_DESIGN.md) proposes
invalid_action_terminal_v1 for explicit review. It retains failed generated
IDs, performs no fabricated retrieval, inherits the actual last ranking's
NDCG, accounts for missing Summary/anchors, and specifies the same transitions
for GRPO and RICE-PO. It also documents the high-D0 failure incentive and the
changed anchor-applicability convention. The three-case design simulation
ran on CPU, verifying token retention, no tool call for an invalid action,
observed-state reward inheritance and an absorbing-state retry rejection.
Its scores are synthetic fixtures and never entered real training.
This proposed environment has not been enabled for policy training.

No Qwen3-4B-Thinking-2507 snapshot was found in the inspected model directory,
project assets or default HF cache within the bounded inventory. The primary
1.5B target was retained. A broader server-wide absence was not established.

Private audit report SHA256:
f531fa35dbd211935a909fb229421b6b9bb63697f75325922202ab6c2ba20985.
The enriched evidence index, including full special-token decoding and request
identities in private case files, has SHA256
e8553edbc3eb48ca279c59aac1cbe989943ff2019f8c1e1f7f5e3654dd179bba.
Paired comparison report SHA256:
26b8fa4aca558c3cb688fc6fee2dcf1b05b47173a7f4e6ff6e550eb2fbe0f5ee.

The portable audit and pair drivers are scripts/classify_saved_protocol_failures.py
and scripts/compare_training_protocol.py. They preserve the executed logic and
add explicit main/Slurm guards and environment-selected private source paths.
The pair accepts figure5 or prefill_compat as its single condition argument.
