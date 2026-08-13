#!/bin/bash
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=uq-apo-qwen-test-analysis
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
source "${ENV_ACTIVATE}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/analyse_apo_stage.py results/uq_apo/raw/qwen_internal_test.jsonl \
  --output-dir results/uq_apo/processed/qwen_internal_test --keep 1
python scripts/analyse_uq.py results/uq_apo/raw/qwen_internal_test.jsonl \
  --stage internal_test --output-dir results/uq_apo/processed/qwen_internal_test_uq
python scripts/analyse_apo_stage.py results/uq_apo/raw/mistral_transfer.jsonl \
  --output-dir results/uq_apo/processed/mistral_transfer --keep 3
