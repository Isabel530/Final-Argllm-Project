#!/bin/bash
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=uq-apo-qwen-controls-analysis
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
source "${ENV_ACTIVATE}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/analyse_apo_stage.py results/uq_apo/raw/qwen_manual_controls.jsonl \
  --output-dir results/uq_apo/processed/qwen_manual_controls --keep 4
python scripts/analyse_paired_controls.py \
  results/uq_apo/raw/qwen_internal_test.jsonl \
  results/uq_apo/raw/qwen_manual_controls.jsonl \
  --winner-id p_53a4acbe525c \
  --baseline-id p_83adf7318fd5 \
  --baseline-id p_da4eef59b8ac \
  --baseline-id p_0086c8edb3b1 \
  --baseline-id p_26918ada0c0a \
  --output-dir results/uq_apo/processed/qwen_paired_controls
