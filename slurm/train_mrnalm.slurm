#!/bin/bash
#SBATCH --job-name=mrnalm_agghead
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
mkdir -p logs results/runs
export SALUKI_PROJECT_ROOT="${REPO_ROOT}"
export SALUKI_RESULT_ROOT="${REPO_ROOT}/results/runs"

"${PYTHON_BIN}" scripts/models/train_mrnalm_agghead_lora.py

