#!/usr/bin/env bash
# TEMPLATE ONLY: never submitted by repository creation or CI.
#SBATCH --job-name=ricepo-reference
#SBATCH --partition=gre
#SBATCH --account=professors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --time=48:00:00
#SBATCH --output=/file_storage01/home/%u/rice-po-repro/logs/%x-%j.out
#SBATCH --error=/file_storage01/home/%u/rice-po-repro/logs/%x-%j.err
# Slurm cannot expand shell variables in directives. Pass --output and --error
# as absolute /file_storage01/home/... paths at submission time (see ZHURONG.md).

set -euo pipefail
: "${SLURM_JOB_ID:?must be submitted through Slurm}"
: "${RICEPO_EXECUTION_AUTHORIZED:?explicit future experiment authorization required}"
[[ "$RICEPO_EXECUTION_AUTHORIZED" == "yes" ]] || exit 2
: "${RICEPO_CONTROL_ROOT:?path to this source checkout}"
: "${RICEPO_STORAGE_ROOT:?absolute file_storage01 project data directory}"
: "${RICEPO_PYTHON:?existing environment Python executable}"
: "${RICEPO_CONFIG:?resolved experiment config}"
: "${RICEPO_DEPLOYMENT:?deployment JSON with verified local assets}"
case "$(realpath -m "$RICEPO_STORAGE_ROOT")" in
  /file_storage01/home/*/*) ;;
  *) echo "storage must be below /file_storage01/home/<user>" >&2; exit 2 ;;
esac
mkdir -p "$RICEPO_STORAGE_ROOT/cache" "$RICEPO_STORAGE_ROOT/tmp"
export HF_HOME="$RICEPO_STORAGE_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$RICEPO_STORAGE_ROOT/cache/datasets"
export XDG_CACHE_HOME="$RICEPO_STORAGE_ROOT/cache/xdg"
export TORCH_HOME="$RICEPO_STORAGE_ROOT/cache/torch"
export TRITON_CACHE_DIR="$RICEPO_STORAGE_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$RICEPO_STORAGE_ROOT/cache/vllm"
export TMPDIR="$RICEPO_STORAGE_ROOT/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RICEPO_CONTROL_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=24
cd "$RICEPO_CONTROL_ROOT"
# Sequential HF reference backend: one process, two visible devices.
srun "$RICEPO_PYTHON" -m ricepo.cli train --config "$RICEPO_CONFIG" --deployment "$RICEPO_DEPLOYMENT"
