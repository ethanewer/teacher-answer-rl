#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL_PATH OUTPUT_DIR [LIMIT] [BATCH_SIZE] [EVAL_ARGS...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="$1"
OUTPUT_DIR="$2"
shift 2
LIMIT="${1:-0}"
if [[ $# -gt 0 ]]; then
  shift
fi
BATCH_SIZE="${1:-128}"
if [[ $# -gt 0 ]]; then
  shift
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/NHNHOME/areal_cache/huggingface}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"

cd "$REPO_ROOT"
exec "$REPO_ROOT/.venv/bin/python" -m rlvr_demo.eval_hf_multi_math \
  --model "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  "$@"
