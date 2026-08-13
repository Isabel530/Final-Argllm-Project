#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=argjudge
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# With `sbatch --export=NIL`, positional arguments avoid Slurm's unreliable
# login-environment retrieval on the submission host.
PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
STUDY="${1:-${STUDY:-}}"
MODEL_KEY="${2:-${MODEL_KEY:-}}"
OUTPUT_ARGUMENT="${3:-${OUTPUT:-}}"

: "${STUDY:?Pass the study name as argument 1}"
: "${MODEL_KEY:?Pass the model key as argument 2}"
OUTPUT="${OUTPUT_ARGUMENT:-${PROJECT_ROOT}/results/raw/${STUDY}_${MODEL_KEY}.jsonl}"

case "${MODEL_KEY}" in
  mistral_7b|qwen3_8b|gpt_oss_20b) ;;
  *) echo "Unsupported MODEL_KEY: ${MODEL_KEY}" >&2; exit 2 ;;
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

echo "study=${STUDY}"
echo "model=${MODEL_KEY}"
echo "output=${OUTPUT}"
python -u scripts/run_study.py \
  --study "${STUDY}" \
  --models "${MODEL_KEY}" \
  --output "${OUTPUT}"
