#!/bin/bash
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=uq-apo-validate-freeze
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
source "${ENV_ACTIVATE}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/analyse_apo_stage.py results/uq_apo/raw/validation.jsonl \
  --output-dir results/uq_apo/processed/validation_metrics --keep 2
python scripts/analyse_uq.py results/uq_apo/raw/validation.jsonl \
  --stage validation --output-dir results/uq_apo/processed/validation_uq
python scripts/select_final_prompt.py \
  --candidate-summary results/uq_apo/processed/validation_metrics/candidate_summary.csv \
  --pair-uq-summary results/uq_apo/processed/validation_uq/pair_uq_summary.csv \
  --uq-signal sampling_variance \
  --output-dir results/uq_apo/processed/final_selection
python scripts/freeze_final_prompt.py \
  --candidates results/uq_apo/candidates/promotion_survivors.jsonl \
  --selection results/uq_apo/processed/final_selection/selected_prompt_ids.json
