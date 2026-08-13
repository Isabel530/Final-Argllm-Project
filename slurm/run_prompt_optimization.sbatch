#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=uq-apo-search
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
CONFIG="${1:-${PROJECT_ROOT}/configs/uq_apo.yaml}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/results/uq_apo/search}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "Environment activation script not found: ${ENV_ACTIVATE}" >&2
  exit 2
fi

source "${ENV_ACTIVATE}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/vol/bitbucket/xz3725/huggingface_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/vol/bitbucket/xz3725/triton_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "${OUTPUT_DIR}" "${TRITON_CACHE_DIR}"
python -u scripts/run_prompt_optimization.py \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}"
