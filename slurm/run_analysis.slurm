#!/bin/bash
#SBATCH --job-name=half_life_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ANALYSIS_SCRIPT="${ANALYSIS_SCRIPT:-scripts/analysis/late_fusion.py}"

cd "${REPO_ROOT}"
mkdir -p logs results/runs
export SALUKI_PROJECT_ROOT="${REPO_ROOT}"
export SALUKI_RUNS_ROOT="${REPO_ROOT}/results/runs"

"${PYTHON_BIN}" "${ANALYSIS_SCRIPT}"

