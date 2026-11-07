# Sanitized execution specimens

These `.txt` files expose the small experiment controllers used for this review, without making them deployment or automatic-execution entry points. Private project/bulk roots and scheduler account/partition become `${PROJECT_ROOT}`, `${BULK_ROOT}`, `${SLURM_ACCOUNT}` and `${SLURM_PARTITION}`. They are literal redactions, not shell expansion instructions for Python. Newlines are normalized to LF. Source code, disabled reusable templates and actual verifier remain in the regular project directories.

`SOURCE_INDEX.json` binds original private controller bytes to each public redacted specimen. Original-controller hashes are also collected from private storage in the final `EVIDENCE.json` index. These identify controller source, not every executed batch wrapper: Slurm saves its submitted wrapper independently. In particular, the provisional report extractor was subsequently corrected/enriched; this directory contains the final extractor, not a claim that it generated earlier provisional reports. The failed extractor and exact provisional predecessor are preserved privately. Frozen model execution releases are not edited by these later reporting changes.

Effective Python invocations inside the submitted Slurm wrappers, with private paths substituted (resource requests are in the wrapper specimens):

```bash
# Parent/asset preflight was completed before either continuation.
python -u ${PROJECT_ROOT}/control/111-continuation-train.py ricepo ${PROJECT_ROOT}/reports/continuation-preflight-174714/complete.json
python -u ${PROJECT_ROOT}/control/111-continuation-train.py grpo ${PROJECT_ROOT}/reports/continuation-preflight-174714/complete.json

# Following the saved native-ID failure, recovery preflight 174746 passed.
python -u ${PROJECT_ROOT}/control/124-native-recovery-train.py ${PROJECT_ROOT}/reports/native-recovery-preflight-174746/complete.json

# Independent process, exact immutable execution source, no optimizer step.
python ${PROJECT_ROOT}/control/verify_continuation_checkpoint.py --run-root ${PROJECT_ROOT}/runs/continuation-grpo-174716 --iteration 1 --device cuda:0 --output ${PROJECT_ROOT}/reports/continuation-grpo-reload-174729.json
python ${PROJECT_ROOT}/control/verify_continuation_checkpoint.py --run-root ${PROJECT_ROOT}/runs/continuation-ricepo-174747 --iteration 1 --device cuda:0 --output ${PROJECT_ROOT}/reports/continuation-ricepo-reload-174748.json
```

The Python verification commands above ran inside one-GPU Slurm allocations, with `PYTHONPATH` set to each method's immutable release, never on the login node. Benchmark controllers 114 and 122 likewise ran in explicitly allocated Slurm jobs. See `TEST_RESULTS.md` for executed jobs and outcomes. These examples are not new job submissions or authorization for unbounded training.
