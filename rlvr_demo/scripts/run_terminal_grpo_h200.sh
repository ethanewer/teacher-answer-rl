#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_AREAL_VENV="$REPO_ROOT/.venv-megatron"
if [[ ! -x "$DEFAULT_AREAL_VENV/bin/python" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  DEFAULT_AREAL_VENV="$REPO_ROOT/.venv"
fi
AREAL_VENV="${AREAL_VENV:-$DEFAULT_AREAL_VENV}"

export PATH="$AREAL_VENV/bin:$PATH"
export AREAL_LAUNCHER_PYTHON="${AREAL_LAUNCHER_PYTHON:-$AREAL_VENV/bin/python}"
export AREAL_VLLM_PYTHON="${AREAL_VLLM_PYTHON:-$REPO_ROOT/.venv-rollout-vllm/bin/python}"
export AREAL_SGLANG_PYTHON="${AREAL_SGLANG_PYTHON:-$REPO_ROOT/.venv-rollout-sglang/bin/python}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp71s0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp71s0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/wbl-fast/usrs/ee/teacher-answer-rl/triton}"
export TRITON_CACHE_PATH="${TRITON_CACHE_PATH:-/wbl-fast/usrs/ee/teacher-answer-rl/triton}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export AREAL_FORK_READINESS_TIMEOUT="${AREAL_FORK_READINESS_TIMEOUT:-180}"
export PYTHONUNBUFFERED=1

CONFIG="${1:-$REPO_ROOT/rlvr_demo/configs/qwen3_4b_terminal_grpo_h200_1000.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

USE_REMOTE_TASK_SERVICE=0
case "${TERMINAL_GRPO_FORCE_REMOTE_TASK_SERVICE:-1}" in
  1|true|TRUE|yes|YES|on|ON) USE_REMOTE_TASK_SERVICE=1 ;;
esac

if [[ -n "${TERMINAL_TASK_SERVICE_URL:-}" || -n "${TERMINAL_TASK_SERVICE_URL_FILE:-}" ]]; then
  USE_REMOTE_TASK_SERVICE=1
elif [[ "$USE_REMOTE_TASK_SERVICE" != "1" ]] && \
  [[ "${TERMINAL_GRPO_SKIP_DOCKER_PREFLIGHT:-0}" != "1" ]] && \
  { ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; }; then
  USE_REMOTE_TASK_SERVICE=1
fi

if [[ "$USE_REMOTE_TASK_SERVICE" == "1" ]]; then
  if [[ -z "${TERMINAL_TASK_SERVICE_URL:-}" && -z "${TERMINAL_TASK_SERVICE_URL_FILE:-}" ]]; then
    echo "Starting Terminal task service on m7i CPU partition." >&2
    TERMINAL_TASK_SERVICE_URL="$(
      "$SCRIPT_DIR/run_terminal_task_service_slurm_cpu.sh" --wait --print-url
    )"
    export TERMINAL_TASK_SERVICE_URL
  fi
  if [[ -n "${TERMINAL_TASK_SERVICE_URL:-}" ]]; then
    echo "Using remote Terminal task service: $TERMINAL_TASK_SERVICE_URL" >&2
  else
    echo "Using remote Terminal task service URL file: $TERMINAL_TASK_SERVICE_URL_FILE" >&2
  fi
fi

cd "$REPO_ROOT"
exec "$AREAL_VENV/bin/python" -m rlvr_demo.train_qwen3_terminal_grpo --config "$CONFIG" "$@"
