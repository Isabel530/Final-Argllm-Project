#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=uq-apo-frozen
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
PURPOSE="${1:-}"
REGISTRY="${2:-}"
OUTPUT="${3:-}"
MANIFEST="${4:-${PROJECT_ROOT}/results/uq_apo/frozen/freeze_manifest.json}"

: "${PURPOSE:?Pass qwen_internal_test or cross_model_transfer as argument 1}"
: "${REGISTRY:?Pass the frozen candidate registry as argument 2}"
: "${OUTPUT:?Pass the raw JSONL output as argument 3}"
case "${PURPOSE}" in
  qwen_internal_test|cross_model_transfer) ;;
  *) echo "Unsupported purpose: ${PURPOSE}" >&2; exit 2 ;;
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
python -u scripts/run_frozen_evaluation.py \
  --purpose "${PURPOSE}" \
  --registry "${REGISTRY}" \
  --freeze-manifest "${MANIFEST}" \
  --output "${OUTPUT}"
