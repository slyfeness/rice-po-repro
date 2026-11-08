# Independent native two-stage protocol

`two_stage_env_summary_v1` is a user-authorized independent protocol, not an
author-confirmed implementation of Figure 5. The legacy strict parser remains
unchanged. Use `configs/two_stage_ricepo.json` or `two_stage_grpo.json` with the
separate `deployment.two_stage.example.json`, after binding real local assets.
Templates do not enable execution. No downloads or experiments run on import.
Set `compare_grpo_credit_gradients=false` for a GRPO run; the diagnostic is
specifically a RICE update versus its trajectory-only counterfactual on the
same originals, not a second training method inside a GRPO run.

Native history per valid round:

```
prompt/tool/role prefix and prefilled <think>           environment, False
sampled Think body and actually sampled </think>       model action, True
frozen-tokenizer <summary> IDs                         environment, False
sampled Summary body and sampled </summary> or EOS     model action, True
tool observation and subsequent native role boundary  environment, False
```

The sampler does not rebuild generated IDs. Two generate calls share one seeded
RNG scope; their state hashes must agree at the transition. A round shares a
4096 **sampled** token budget across stages; environment tokens consume context
capacity. EOS is never masked or moved. A complete tag crossing a sampled token
does not authorize retaining non-whitespace text after the closing tag. Body
spans use verified whole-native-token ByteLevel coverage and cannot contain an
environment-control interval.

Malformed structure, empty Summary and sampled invalid UTF-8 are rejected with
all native evidence retained. In particular replacement-character decoding is
not an exact Summary. This is an unextractable action under the existing named
failure-zero environment, not a new boundary acceptance rule. A valid UTF-8
decoder/concatenation inconsistency still raises an infrastructure ambiguity;
it is not silently converted into a policy failure.

An identity logits observer records only the selected token's full-vocabulary
conditional log probability. It returns the exact scores tensor unchanged.
Captured probabilities, not a different scoring pass, supply PPO old values;
the full native teacher-forcing pass is an independent numerical gate. All
old/current/reference scoring uses the same actual history. Environment tokens
have no direct loss and remain in the differentiable context. Summary entropy
uses the full conditional vocabulary at real Summary-body positions.

The initial BF16 probability gate failed on real fixed short conditions. The
independent FP32 Actor choice passed; Diver remains BF16 with unchanged asset
identity. This arithmetic choice applies to both newly matched A/B controls and
subsequent training. It changes neither sampling temperature nor probability
normalization. Tolerances were declared before the precision probe; detailed
errors and actual initial ratio ranges are reported, not rounded to unity.

`eval_layers_v1` recomputes decoder layers without changing eval/dropout mode or
detaching context. It exposes counters proving backward recomputation. Real
model loss/gradient comparison is required before use. Logit chunking still does
not prevent a full sequence-vocabulary projection or guarantee memory capacity.

The HF batch candidate has separate per-request generators, left-padding masks
and causal positions, selected-token probabilities, and fresh per-stage KV
state. Finished requests are retained and never resampled. No cache survives a
call or weight change. Its short throughput test does not establish long,
five-round training throughput or vLLM correctness. The reference online path
can remain serial while the candidate is evaluated.

`local_verified_v1` writes and closes the complete actor/optimizer/RNG payload in
an owned node-local temporary directory, actually reloads and verifies it,
copies into shared staging with per-file hashes, then uses the existing atomic
checkpoint publication. Failed local staging and original corrupted assets
remain traceable. The RNG payload includes draw hashes for cross-process
restoration checks. File-system atomic rename and single-writer run locking are
still required.

RICE/GRPO gradient diagnostics operate on identical originals and fixed old
probabilities/reference. The GRPO comparison performs backward only, never an
optimizer step. The actual RICE gradient, its difference from GRPO and dot
product with the actual parameter delta are measured. Missing local rewards
remain undefined; their anchor falls back to trajectory advantages without
dropping branches or inventing retrieval scores.
