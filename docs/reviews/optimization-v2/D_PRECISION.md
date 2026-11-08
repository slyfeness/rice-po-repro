# Matched BF16 training probe

Decision: **FAIL** for the tested independent configuration. The computation here is HF BF16 autocast with FP32 master weights and FP32 probability normalization, compared against real saved vLLM BF16/FlashAttention action probabilities. This is a new matched-compute probe, not another BF16-sampling-versus-FP32-compute test.

Before model execution, the probe froze its acceptance configuration in commit2747163 and copied it to a private preregistration receipt. The behavior gate requires mean absolute log-probability error≤0.02, p99≤0.10 and initial PPO clipping fraction≤1% at epsilon0.2. Summary entropy rank overlap≥0.75, finite loss/gradients, actual FP32-master weight change and verified fresh-snapshot/KV synchronization would also be required for a training PASS. The old0.02/0.06/0.12 gate remains a historical diagnostic; passing one scalar gate alone would not establish compatibility.

Slurm175507 completed successfully in20 seconds on one assigned GPU. It selected16 saved native Think/Summary conditions by a fixed hash order with longest-history and actual-EOS coverage before model initialization. The same native prefixes and sampled token IDs were scored without re-encoding, truncation, generation or retrieval. The old sampling records came from the repeated B candidate; all probabilities correspond to those actual actions.

Across2591 sampled tokens, mean absolute log error was0.0276844972 and p99 was0.1635795832; both exceed the new preregistered gate. Initial ratio range was0.499182773–1.340870123, mean0.999446320. The actual clipping fraction was0.0034735623. None of the16 request records passed all behavior conditions. The per-request records are in `D_MATCHED_PRECISION_COMPARISON.csv`.

Full-vocabulary entropy was computed at the saved Summary-stage action histories for two HF BF16 variants: FP32 masters under autocast and explicitly BF16 weights. Their fixed-request Top4 entropy ordering overlapped4/4. This is only a diagnostic ranking over the selected requests, not a paper anchor selection across unrelated queries. vLLM's complete distributions were not exported, so this is not evidence of identical vLLM-versus-HF full-vocabulary entropy or same-query anchor sets.

Because the behavior gate failed, the probe stopped before a diagnostic optimizer update. Loss/gradient/update/synchronization are therefore **not run**, not failed or passed. The initial FP32-master invariant was checked before scoring; no optimizer state was created. Existing FP32 training remains the recommended path. No threshold was adjusted after inspecting the result.

Private result SHA-256: `685a90c9977e5f4655201c462db71ea5edad55af583e4b26e70a11e145c83627`. Original prefixes, sampled IDs, model-asset identity, selection and preregistration stay in private experiment storage. Public source tests:2 targeted diagnostics passed locally and in the Slurm job. No unrelated formula, checkpoint or token-chain suite was rerun.
