#!/usr/bin/env bash
# Review template: submit once per method with separate private configurations.
# No automatic paired submission, downloads or execution enablement.
#SBATCH --job-name=ricepo-bounded
#SBATCH --partition=GPU_PARTITION
#SBATCH --account=SLURM_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
# Pass existing absolute bulk-storage --output/--error paths to sbatch.
date -u '+BATCH_ENTRY %Y-%m-%dT%H:%M:%SZ'
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}"
: "${RICEPO_EXECUTION_AUTHORIZED:?explicit authorization required}"
[[ "$RICEPO_EXECUTION_AUTHORIZED" == yes ]] || exit 2
: "${RICEPO_CONTROL_ROOT:?immutable source checkout}"
: "${RICEPO_STORAGE_ROOT:?bulk-storage project root}"
: "${RICEPO_PYTHON:?existing independent environment Python}"
: "${RICEPO_JAVA_HOME:?existing compatible JVM}"
: "${RICEPO_CONFIG:?enabled run-local config with explicit budget and manifests}"
: "${RICEPO_DEPLOYMENT:?private deployment with unique output/cache roots}"
case "$(realpath -m "$RICEPO_STORAGE_ROOT")" in
  /file_storage01/home/*/*) ;;
  *) echo 'bulk storage required' >&2; exit 2 ;;
esac
export JAVA_HOME="$RICEPO_JAVA_HOME" PATH="$RICEPO_JAVA_HOME/bin:$PATH"
export PYTHONPATH="$RICEPO_CONTROL_ROOT/src" OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TMPDIR="$RICEPO_STORAGE_ROOT/tmp/bounded-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
# Preserve Slurm's CUDA_VISIBLE_DEVICES. Private deployment uses logical 0/1.
"$RICEPO_PYTHON" -m ricepo.cli train --config "$RICEPO_CONFIG" --deployment "$RICEPO_DEPLOYMENT"
