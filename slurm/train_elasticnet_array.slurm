#!/bin/bash
#SBATCH --job-name=elasticnet
#SBATCH --partition=cpu
#SBATCH --array=0-39
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
mkdir -p logs results/runs
export SALUKI_PROJECT_ROOT="${REPO_ROOT}"
export SALUKI_RESULT_ROOT="${REPO_ROOT}/results/runs"

"${PYTHON_BIN}" scripts/models/train_elasticnet.py

