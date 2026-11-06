# Actual verification, failures and unrun scope

Tests below were actually executed. They are overlapping targeted suites at different source revisions; their counts must not be summed into a unique test count or compared as a scientific progress metric. Final JUnit SHA-256/counts are bound in `EVIDENCE.json`. No missing dependency/asset skip is counted as a pass.

| Slurm job | Executed work | Outcome |
|---|---|---|
| 174714 | CPU actual parent/source/config/data/schedule preflight plus 19 targeted tests | Passed; both immutable first-update parents admitted |
| 174717 | Existing ranking rescoring, exact32 identities and four roundwise tests | Passed; no new generation |
| 174718 | CPU wavefront/rollout/workflow/continuation tests | 44 passed, 0 failed, 0 skipped |
| 174720 / 174721 | Existing-query mechanism analysis / parent selection and qrels-positive spot checks | Completed; descriptive and bounded spot checks |
| 174724 | CPU compile check and 56 targeted tests | Passed at its recorded earlier source |
| 174743 | CPU native-token recovery tests and actual model/tokenizer/cache preflight | 62 passed; identified actual vocabulary mismatch and admitted saved complete action evidence |
| 174746 | Final CPU mandatory saved-Think guards, recovery/native-format/generator/batch/failure/rollout/workflow/continuation | 64 passed, 0 failed, 0 skipped; immutable source `534b853` |
| 174715 | First serial RICE continuation, actual original/branch sampling | **Failed** in tokenizer conversion before any new optimizer step; native partial Summary IDs were not saved by that old callback |
| 174716 | GRPO global updates 2/3, actual optimizer, checkpoints and complete32 development | Completed |
| 174719 | Actual serial/batch four-original/20-new-branch workload, retrieval/credit/full-history probability gates | Passed; native results/identities match; no optimizer step in benchmark |
| 174729 | Independent GRPO final checkpoint/optimizer/CPU-CUDA RNG reload | Passed, restored optimizer step 3; zero additional steps |
| 174731 | Four fixed long prefixes under real policy-1 and policy-3 weights, serial/batch/full-history scoring | Passed at actual lengths through 7337; no sampling from a stale KV cache |
| 174747 | Separate RICE native recovery, updates2/3 and full32 development | Completed; nonzero credit/gradient and actual weights changed at both updates |
| 174748 | Independent RICE final checkpoint/optimizer/CPU-CUDA RNG reload | Passed, actual actor/optimizer/RNG restored atstep3; no new step |
| 174793 | Provisional CPU report extractor | **Failed**: exact comparison used a different floating-point mean/variance accumulation than core trajectory advantage; training itself was unchanged |
| 174796 | Corrected provisional extractor using core `math.fsum` baseline | Passed for then-available evidence; explicitly not the final completed-run report |
| 174836 | CPU import/API inventory of existing vLLM0.19.1 | Passed import only; no engine, GPU inference, throughput or weight synchronization test |

The final extractor additionally requires all three updates per method committed, final development complete, paired global training schedule equal, all GPU jobs terminal and no more than4 GPUs concurrently. Actual GPU hours remain recorded; the user removed the previous32-hour stopping cap during the phase. It independently recomputes six-outcome population variances, original-count-one, center-only local advantages, gate thresholds and actual Summary/Think advantage slices. It checks zero direct advantages at environment positions. That last check verifies the stored loss mask; CPU/model gradient semantics remain covered by their existing targeted tests, not inferred from the JSON alone.

Representative actual test body, executed within CPU Slurm allocation 174746, offline and with existing local assets:

```bash
PYTHONPATH=${PROJECT_ROOT}/pytest-compat:${PROJECT_ROOT}/releases/native-recovery-v2/src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_action_recovery.py tests/test_two_stage_format.py \
  tests/test_two_stage_generator.py tests/test_batched_generation.py \
  tests/test_failure_zero.py tests/test_rollout.py tests/test_workflow.py \
  tests/test_continuation.py --junitxml=${PROJECT_ROOT}/reports/native-recovery-tests-174746.xml
```

Local development RED鈫扜REEN checks separately exercised missing continuation implementation, tampered parent selection/recipe, and rejecting duplicate request identities before paid generation. Their results are engineering diagnostics, not substitutes for the actual cluster/asset gates. Public plotting/AST/JSON/diff/sensitive-file checks are listed separately in publication evidence; a static parse is never described as a runtime test.

**Not run:** full twelve-domain benchmark; paper batch128 training; multi-seed method comparison; official held-out test; full online batched-Trainer PPO update and co-resident reference-memory qualification; 32K worst-case probability/memory acceptance; vLLM integration; four-rank FSDP; arbitrary multi-parent checkpoint continuation. No automatic CI, model downloads, environment installation or unrelated experiment was introduced.

Final additional executions: CPU174831 passed all completed-run, exact paired schedule, credit/advantage and terminal allocation checks. vLLM eager probe174858 generation succeeded but its HF baseline keyword failed;174872 reused its saved native IDs and passed4 probability checks. Graph candidate174880/174881 failed3/80 numerical checks and was rejected. IEEE correction174918/174919 passed80/80 without loosening thresholds;174891 provides the matched16-request HF timing/native evidence.174874 exited0 seconds due to a wrong wrapper filename,174876 failed before generation because an optional worker callback could not serialize; both remain disclosed and charged. No insecure serialization setting was enabled. The corrected callback was removed rather than made a generation prerequisite. Details and remaining unrun Trainer scope are inVLLM_STATUS.md.
