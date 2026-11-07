# Explicit HF BF16 weights: FAIL

Executed 2026-11-07 UTC, base `0d6f07ee72a53e5c8fdaa859a7b70516a6c56cab`, existing PR #2 branch. This is an independent precision diagnostic. The verified IEEE FP32 training path is unchanged.

**FAIL: explicit BF16 HF weights do not qualify vLLM BF16/FLASH_ATTN for PPO.** B worsens mean absolute log-probability error by 56.63% relative to A; 14/16 requests worsen, 2 improve. All three frozen aggregate thresholds fail. GPU work stopped after this result: no optimizer step, new generation, synchronization or speed experiment.

## Same-cohort results

All rows compare HF scores against the **same saved vLLM BF16 sampling probabilities**, on 16 native histories / 2,591 action tokens: 9 Think requests / 2,219 tokens and 7 Summary / 372 tokens, including actual EOS and a 9,133-token prefix. No re-encoding, truncation, token repair, filtering or resampling.

| HF computation | Mean absolute log error | p99 | Max | Ratio min–max | Clip fraction | Gate |
|---|---:|---:|---:|---:|---:|---|
| A: FP32 weights + BF16 autocast, reused 175507 | 0.02768450 | 0.16357958 | 0.69478297 | 0.499183–1.340870 | 0.34736% (9/2591) | FAIL |
| B: explicit BF16 weights, no autocast, 175561 | **0.04336277** | **0.33996582** | 0.84131432 | 0.459245–2.319413 | **2.77885% (72/2591)** | **FAIL** |
| C: FP32 IEEE, 175561 | 0.02582462 | 0.14602661 | 0.41146278 | 0.662680–1.341356 | 0.30876% (8/2591) | FAIL |
| Predeclared requirement | ≤0.02 | ≤0.10 | diagnostic | diagnostic | ≤1%, epsilon=0.2 | all three |

[Exact CSV](BF16_WEIGHT_COMPARISON.csv). p99 uses the existing nearest-rank convention; ratio is `exp(HF_log_prob - actual_vLLM_BF16_sampled_log_prob)`. A's ordered cohort and model/tokenizer content identity were checked before reuse. C was scored once because complete token evidence for this exact cohort was unavailable. C failing against **BF16 behavior** does not invalidate the previously qualified IEEE FP32 vLLM/HF path.

## Observable evidence and limits

- B explicitly converts **all 1,777,088,000 parameters** with `model.to(dtype=torch.bfloat16)`. Every parameter equals an independent `from_pretrained(..., dtype=torch.bfloat16)` load. BF16 parameter digest: `ce66378ef5bdd4cb708d6a433a2ee40cb24c416c0fafb3671e4ab517e327bcac`.
- The checkpoint's stored tensors are **already BF16**, covering the same parameter count. A/C expand them to FP32; B retains the BF16 values. Changing HF storage does not recover a different source checkpoint or solve the discrepancy.
- Model/config/tokenizer payload digest: `972c28b609f0a8f8b16d47c488d9f46da6292f50c1ad2936186b1c74716eedaf`. Saved vLLM metadata binds the same local model path and BF16/FLASH_ATTN behavior. Historical engine-resident tensors were not exported: **local snapshot/cast consistency is verified; physical historical engine weight equality is not independently attested**.
- HF uses Qwen2 SDPA dispatch; historical vLLM uses FLASH_ATTN. HF B produces BF16 logits, then converts to FP32 for `log_softmax`, as in A. TF32 is disabled; the model stays in eval mode; complete native history, causal positions `prefix_length-1 ... full_length-2`, EOS and masks are retained. These are different execution paths, not proof of identical kernels.
- B mean error: Think 0.04544836; Summary 0.03092211. The longest-prefix request has 51/371 clipped positions and max error 0.77817059; another single-token Think request has error 0.84131432. Failure is not confined to one request or only Summary.
- Explicit storage is therefore **not a sufficient fix**. Execution/rounding differences remain; the dominant individual operator has **not** been causally isolated. No further kernel/configuration search was conducted.

## Actual execution and checks

| Job | Scope | Terminal state |
|---|---|---|
| 175560 | First single-GPU attempt | FAILED, 2m15s: progress writer reused an immutable receipt name during C; no B or optimizer execution. Partial evidence retained. |
| 175561 | Corrected independent output, C then B | COMPLETED, 44s, exit 0; scientific result **FAIL**. One A800-SXM4-80GB. |
| 175562 | CPU evidence / independent recomputation | COMPLETED, 1s, exit 0. B/C native IDs, prefix binding, EOS/finish reason and full masks checked; aggregates independently reproduced exactly. |

7 targeted CPU tests actually passed locally and in the corrected GPU allocation, no skips. Static AST/config/diff checks are recorded separately. Progress now uses independent per-request files. Current-task GPU allocation cost including failure: `(135+44)/3600 = 0.0497222 GPU-hours`. No active GPU job remains from this task.

Combined scoring peak PyTorch allocated/reserved memory: 17,674,034,688 / 23,781,703,680 bytes; these are not whole-card NVML peaks. B/C scoring intervals were 1.887s/14.623s. Sequential diagnostic timing with different warm/cache conditions is **not a speed benchmark or end-to-end training acceleration**.

Reused environment: PyTorch 2.10.0, Transformers 5.5.1, vLLM 0.19.1, CUDA 12.8, safetensors 0.8.0. A standalone `flash-attn` distribution was not installed; historical vLLM backend selection is recorded separately. No installation/download was performed.

## Reproduction and evidence

[Probe](../../../scripts/validation/compare_bf16_weights.py), [config](../../../configs/bf16_explicit_weight_probe.json), [tests](../../../tests/test_bf16_saved_cohort.py). Public config retains `execution.enabled=false`, `training_qualified=false`. No production Trainer/PPO file changed.

Sanitized command inside a Slurm allocation: `gre/professors`, 1 GPU, 6 CPU, 90G RAM, 20-minute limit, actual device assigned by Slurm:

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRITON_F32_DEFAULT=ieee
export PYTHONPATH="$SOURCE/src:$SOURCE"
"$PYTHON" -m unittest tests.test_bf16_saved_cohort tests.test_precision_probe -v
"$PYTHON" scripts/validation/compare_bf16_weights.py \
  --config "$SOURCE/configs/bf16_explicit_weight_probe.json" \
  --selection "$PRIVATE/A/REQUEST_SELECTION.json" \
  --responses "$PRIVATE/vllm-B/responses" \
  --benchmark "$PRIVATE/vllm-B/BENCHMARK_RESULT.json" \
  --prior-result "$PRIVATE/A/RESULT.json" \
  --model "$LOCAL_MODEL" --output "$NEW_PRIVATE_OUTPUT" --cache "$PRIVATE_CACHE"
```

These environment variables represent explicit private paths bound by the launcher. Output must be a new directory. No automatic entry point or CI added.

[Sanitized evidence / SHA-256 index](PUBLIC_EVIDENCE.json) binds the executed script, archive, frozen selection, input receipt, private per-token output and result. Result SHA-256: `574b2bcadfa3bb09f21209032bb0671029ac0d394848297969534ba0295d3710`. Raw histories, per-token logs, model and job logs remain private. The failed attempt is retained. Existing licenses and notices are unchanged; no new license is chosen.

**NOT RUN after probability FAIL:** gradients, optimizer step, BF16 update rounding, FP32 master/shadow comparison, post-update vLLM sync/KV invalidation, new-policy sampling and speed benchmark. It would be incorrect to claim either successful BF16 training or measured update-rounding failure. Keep IEEE FP32. This task stops here; no 32/128-query training.
