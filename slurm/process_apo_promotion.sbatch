#!/bin/bash
#SBATCH --partition=a40
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=uq-apo-promote-select
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
source "${ENV_ACTIVATE}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/analyse_apo_stage.py results/uq_apo/raw/promotion.jsonl \
  --output-dir results/uq_apo/processed/promotion --keep 2
python scripts/select_prompt_candidates.py \
  --candidates results/uq_apo/search/candidates_for_promotion.jsonl \
  --selection results/uq_apo/processed/promotion/selected_prompt_ids.json \
  --output results/uq_apo/candidates/promotion_survivors.jsonl
