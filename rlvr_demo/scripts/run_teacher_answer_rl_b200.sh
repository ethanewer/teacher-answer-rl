#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/NHNHOME/areal_cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/NHNHOME/areal_cache/triton}"
export TRITON_CACHE_PATH="${TRITON_CACHE_PATH:-/NHNHOME/areal_cache/triton}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == "--config" ]]; then
  shift
fi
CONFIG="${1:-$REPO_ROOT/rlvr_demo/configs/qwen3_06b_multi_math_teacher_answer_rl_b200_250.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$REPO_ROOT"
exec "$REPO_ROOT/.venv/bin/python" -m rlvr_demo.train_qwen3_teacher_answer_rl --config "$CONFIG" "$@"
