# Bounded TF32 GEMM diagnostic drivers

These forensic drivers reproduce this phase using already-local private assets. They are not automatic training entry points, portable downloads, or replacements for the standard recipe. Importing the `ricepo` package does not invoke them. All compute must be explicitly launched through Slurm; scripts require `SLURM_JOB_ID`.

Set `RICE_PROJECT_ROOT` to the project's private bulk-storage root and `PYTHONPATH` to an immutable expanded source snapshot. `online_pilot.py` additionally requires `RICE_SOURCE_ROOT` and a short private `RICE_RPC_ROOT`; `restore.py` requires `RICE_RESTORE_RUN`. Existing relative evidence aliases/job IDs identify the frozen inputs; a different deployment must explicitly bind its corresponding local files. No helper downloads assets or chooses new seeds on failure.

Executed order (commands shown without private paths/account/network details):

| Job | Explicit driver / sanitized submission | Purpose |
|---|---|---|
|175836|`sbatch <kernel-probe-launcher>` → `python kernel_probe.py`|Fresh A/B model worker states and tiny actual-model CUDA trace; no trace compression|
|175838|`sbatch <CPU-kernel-analysis-launcher>` → `python analyze_kernels.py`|Summarize kernel names/durations, retain raw traces privately|
|175839|`sbatch <one-GPU-benchmark-launcher> B` → `python benchmark.py B`|All1152 frozen native requests, cold then hot,512 cap|
|175840|Same launcher, `C`|BF16/FlashAttention speed reference only|
|175837|`sbatch <CPU-inventory-launcher>` → `python cpu_inventory.py`|Old operation intervals and unchanged sticky partition analysis|
|175841|`sbatch <CPU-freeze-launcher>` → `python freeze_holdout.py`|Freeze16 independent conditions/two training queries before generation|
|175846|`sbatch <one-GPU-numerical-launcher>` → `python numerical.py`|B actual q versus HF IEEE, full saved remaining budgets|
|175852|`sbatch <CPU-test-launcher>` → `python -m unittest discover -s tests -p test_gemm_diagnostic.py -v`|Four new targeted tests|
|175853|`sbatch <two-GPU-online-launcher>` → `python online_pilot.py`|One conditional two-query RICE update only|
|175854|`sbatch <CPU-cost-launcher>` → `python cost_attribution.py`|Exclusive saved-timing accounting, no model execution|
|175862|`sbatch --dependency=afterok:175853 <one-GPU-restore-launcher>` → `python restore.py`|Independent read-only actor/Adam/RNG restore, no step|

CPU exports175861/175863 use `export_results.py` (the final public copy omits an unavailable failure-kind field). Public copies replace private root literals with environment variables and add Slurm guards. The exact executed private drivers and expanded-source files are hashed in the report's evidence index. CPU exports only release selected aggregates/hashes, not native inputs, model files or logs. Use the final `JOBS.csv` for actual terminal states, including export jobs. No ZIP/TAR or automatic Slurm/CI is produced.
