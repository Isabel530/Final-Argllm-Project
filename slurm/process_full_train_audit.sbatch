#!/bin/bash
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=uq-apo-full-train-analysis
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
source "${ENV_ACTIVATE}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/analyse_apo_stage.py results/uq_apo/raw/full_train_audit.jsonl \
  --output-dir results/uq_apo/processed/full_train_audit --keep 2
python scripts/analyse_paired_controls.py results/uq_apo/raw/full_train_audit.jsonl \
  --winner-id p_53a4acbe525c \
  --baseline-id p_26953e457823 \
  --output-dir results/uq_apo/processed/full_train_paired
