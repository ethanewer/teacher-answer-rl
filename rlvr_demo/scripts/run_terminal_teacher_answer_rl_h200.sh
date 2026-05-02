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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/wbl-fast/usrs/ee/teacher-answer-rl/triton}"
export TRITON_CACHE_PATH="${TRITON_CACHE_PATH:-$TRITON_CACHE_DIR}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export PYTHONUNBUFFERED=1

# The terminal teacher-answer reward should only score teacher commands/task_complete.
export TEACHER_ANSWER_FORMAT_BONUS="${TEACHER_ANSWER_FORMAT_BONUS:-0.0}"
export TEACHER_ANSWER_LENGTH_PENALTY="${TEACHER_ANSWER_LENGTH_PENALTY:-0.0}"
export TERMINAL_EXPERIMENT_WALLCLOCK_START="${TERMINAL_EXPERIMENT_WALLCLOCK_START:-$(python - <<'PY'
import time
print(time.time())
PY
)}"

if [[ "${1:-}" == "--config" ]]; then
  shift
fi
CONFIG="${1:-$REPO_ROOT/rlvr_demo/configs/qwen3_4b_terminal_teacher_answer_rl_h200_1000.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$REPO_ROOT"
exec "$AREAL_VENV/bin/python" -m rlvr_demo.train_qwen3_terminal_teacher_answer_rl --config "$CONFIG" "$@"
