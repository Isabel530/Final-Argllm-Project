#!/bin/bash
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=uq-apo-e11
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/xz3725/final-argllm-project"
ENV_ACTIVATE="${ENV_ACTIVATE:-/vol/bitbucket/xz3725/argumentative-llms/argllm/bin/activate}"
STAGE="${1:-}"
: "${STAGE:?Pass pilot or formal as argument 1}"
case "${STAGE}" in
  pilot|formal) ;;
  *) echo "Unsupported E11 stage: ${STAGE}" >&2; exit 2 ;;
esac

source "${ENV_ACTIVATE}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/vol/bitbucket/xz3725/huggingface_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/vol/bitbucket/xz3725/triton_cache}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${TRITON_CACHE_DIR}" results/uq_apo/raw

python -u scripts/run_e11_webis.py --stage "${STAGE}"
if [[ "${STAGE}" == "pilot" ]]; then
  python scripts/validate_e11_pilot.py
fi
