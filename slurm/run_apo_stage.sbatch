#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=uq-apo-stage
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
STAGE="${1:-}"
CANDIDATES="${2:-}"
OUTPUT="${3:-${PROJECT_ROOT}/results/uq_apo/raw/${STAGE}.jsonl}"

: "${STAGE:?Pass stage as argument 1}"
: "${CANDIDATES:?Pass candidate registry as argument 2}"
case "${STAGE}" in
  search_screening|promotion|train_full|validation|internal_test) ;;
  *) echo "Unsupported stage: ${STAGE}" >&2; exit 2 ;;
esac
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

mkdir -p "$(dirname "${OUTPUT}")" "${TRITON_CACHE_DIR}"
python -u scripts/run_apo_stage.py \
  --candidates "${CANDIDATES}" \
  --stage "${STAGE}" \
  --output "${OUTPUT}"
