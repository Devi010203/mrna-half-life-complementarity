#!/bin/bash
#SBATCH --job-name=tabular_model
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_SCRIPT="${MODEL_SCRIPT:-scripts/models/train_xgboost.py}"

cd "${REPO_ROOT}"
mkdir -p logs results/runs
export SALUKI_PROJECT_ROOT="${REPO_ROOT}"
export SALUKI_RESULT_ROOT="${REPO_ROOT}/results/runs"

"${PYTHON_BIN}" "${MODEL_SCRIPT}"

