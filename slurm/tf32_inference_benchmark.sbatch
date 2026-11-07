#!/bin/bash
# Supply partition/account through private sbatch submission arguments.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=90G
#SBATCH --time=00:30:00
# Supply --output/--error paths to an existing private bulk-storage log root.
set -euo pipefail
date -u '+BATCH_ENTRY %Y-%m-%dT%H:%M:%SZ'
: "${RICE_SOURCE_ROOT:?expanded fixed source required}"
: "${RICE_PYTHON:?existing project interpreter required}"
: "${RICE_PROJECT_ROOT:?private bulk-storage project root required}"
: "${RICE_BENCHMARK_ROOT:?frozen native manifest required}"
: "${RICE_RPC_ROOT:?short private bulk-storage IPC root required}"
cd "$RICE_PROJECT_ROOT"
ulimit -c 0
export PYTHONPATH="$RICE_SOURCE_ROOT/src:$RICE_SOURCE_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_F32_DEFAULT=tf32 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=6
export VLLM_RPC_BASE_PATH="$RICE_RPC_ROOT/t$SLURM_JOB_ID"
mkdir -p "$VLLM_RPC_BASE_PATH"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
driver="${1:?screen/coexist/dual/four required}"; shift
case "$driver" in
  screen) file=tf32_runtime_screen.py ;;
  coexist) file=tf32_coexist_isolated.py ;;
  dual) file=tf32_dual_engine.py ;;
  four) file=tf32_four_workers.py ;;
  *) exit 2 ;;
esac
"$RICE_PYTHON" -u "$RICE_SOURCE_ROOT/scripts/benchmarks/$file" "$@"
date -u '+BENCHMARK_TERMINAL %Y-%m-%dT%H:%M:%SZ'
