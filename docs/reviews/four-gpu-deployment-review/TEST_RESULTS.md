# Executed checks and known failures

All cluster tests/postprocessing used Slurm and the existing isolated environment; no dependencies were installed. No skipped test is counted as passed.

| Job | Scope | Actual outcome |
|---|---|---|
|174995|Initial CPU launcher|Failed: existing pytest compatibility path omitted; tests did not run|
|175000|Adapter missing regression|Failed collection as expected before implementation|
|175002,175003|Adapter/two-stage contracts|22 passed each; overlapping suites, not44 unique|
|175007|Trainer regression launcher|Failed: incorrect test filename; no tests ran|
|175008|Trainer/adapter regressions|41 passed|
|175013|Review regressions before fix|3 failed,12 deselected: constructor cleanup, unique RPC directory, pre-validation raw persistence|
|175014|Fixed source targeted suite|75 passed,0 failed,0 skipped in5.39s|
|175017|Lifecycle and real spawned-process cleanup|6 passed,0 failed,0 skipped in0.41s; overlaps existing lifecycle tests|
|175022|Standalone sampler mismatch guard|2 passed,0 failed,0 skipped in0.59s|
|175004|Real FP32 vLLM full rollout/credit/LP|COMPLETED,24/24 complete-history numerical checks; no optimizer|
|175024|Saved native histories/credit audit|COMPLETED; independently verified masks/prefixes/moments/replacement|
|175018/175019|RICE/GRPO full Trainer|Running; not yet claimed passed|

Targeted command scope for175014: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PROJECT/pytest-compat:$RELEASE/src:$RELEASE/tests python -m pytest tests/test_vllm_two_stage.py tests/test_two_stage_format.py tests/test_two_stage_masks.py tests/test_vllm_lifecycle.py tests/test_two_stage_training_io.py tests/test_workflow.py tests/test_training_plan.py tests/test_objective.py tests/test_modeling.py tests/test_failure_zero.py -q` (exact source/launcher hashes retained privately).175017 additionally executed the real Linux worker+descendant termination test;175022 ran test_vllm_holdout_guard.py and test_holdout_selection.py. No automatic downloads or CI were added.

Read-only code review found three lifecycle/evidence defects above, all fixed with pre-fix failures and post-fix execution. Actual vLLM stop-token retention, full-history probability agreement, update synchronization and GPU capacity remain runtime gates, not conclusions from CPU tests. Separate stage reports preserve their original tested source versions.
