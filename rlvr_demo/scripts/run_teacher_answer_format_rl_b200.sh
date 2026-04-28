#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG_PATH_OR_--config [CONFIG_PATH] [OVERRIDES...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/NHNHOME/areal_cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/NHNHOME/areal_cache/triton}"
export TRITON_CACHE_PATH="${TRITON_CACHE_PATH:-$TRITON_CACHE_DIR}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHONUNBUFFERED=1

export TEACHER_ANSWER_FORMAT_BONUS="${TEACHER_ANSWER_FORMAT_BONUS:-0.5}"
export TEACHER_ANSWER_LENGTH_PENALTY="${TEACHER_ANSWER_LENGTH_PENALTY:-0.2}"

if [[ "${1:-}" == "--config" ]]; then
  shift
fi
CONFIG="${1:-$REPO_ROOT/rlvr_demo/configs/qwen3_06b_multi_math_teacher_answer_final_rl_b200_250.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$REPO_ROOT"
exec "$REPO_ROOT/.venv/bin/python" -m rlvr_demo.train_qwen3_teacher_answer_final_rl --config "$CONFIG" "$@"
