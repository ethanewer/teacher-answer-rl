#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL [SERVED_MODEL_NAME] [PORT] [vllm-args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="$1"
SERVED_MODEL_NAME="${2:-terminal-local}"
PORT="${3:-30080}"
shift || true
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi

VLLM_PYTHON="${VLLM_PYTHON:-$REPO_ROOT/.venv-rollout-vllm/bin/python}"
if [[ ! -x "$VLLM_PYTHON" ]]; then
  echo "vLLM python not found or not executable: $VLLM_PYTHON" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/areal-$(id -un)/.cache/$(id -un)/vllm}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

LOG_DIR="${LOG_DIR:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval/server_logs}"
mkdir -p "$LOG_DIR"

VLLM_ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "${HOST:-0.0.0.0}"
  --port "$PORT"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --max-model-len "${MAX_MODEL_LEN:-40960}"
  --dtype "${DTYPE:-bfloat16}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.86}"
  --uvicorn-log-level "${UVICORN_LOG_LEVEL:-warning}"
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3}"
)

if [[ "${ENABLE_REASONING:-1}" == "1" ]]; then
  VLLM_ARGS+=(--enable-reasoning --reasoning-parser "${REASONING_PARSER:-qwen3}")
fi

cd "$REPO_ROOT"
exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  "${VLLM_ARGS[@]}" \
  "$@" \
  2>&1 | tee "$LOG_DIR/${SERVED_MODEL_NAME}_vllm_$(date -u +%Y%m%dT%H%M%SZ).log"
