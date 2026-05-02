#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 JOB_NAME LITELLM_MODEL [API_BASE] [JOBS_DIR] [write-config-args...]" >&2
  echo "Example: $0 base openai/Qwen3-4B-Thinking-2507 http://127.0.0.1:30000/v1 /wbl-fast/usrs/ee/teacher-answer-rl/harbor_jobs --task fix-git" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JOB_NAME="$1"
LITELLM_MODEL="$2"
shift 2
API_BASE="${1:-http://127.0.0.1:30000/v1}"
if [[ $# -gt 0 ]]; then
  shift
fi
JOBS_DIR="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/harbor_jobs}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_CONFIG_ARGS=("$@")
CONFIG_DIR="${REPO_ROOT}/rlvr_demo/results/terminal_bench_eval_configs"
CONFIG_PATH="${CONFIG_DIR}/${JOB_NAME}.yaml"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HF_HOME="${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for the current Terminal-Bench/Harbor eval workflow." >&2
  echo "This node only has enroot-like runtimes; do not use enroot here because prior FUSE startup hung in D-state." >&2
  exit 2
fi

cd "$REPO_ROOT"
"$REPO_ROOT/.venv/bin/python" -m rlvr_demo.terminal_experiment write-harbor-eval-config \
  --output "$CONFIG_PATH" \
  --job-name "$JOB_NAME" \
  --jobs-dir "$JOBS_DIR" \
  --api-base "$API_BASE" \
  --litellm-model "$LITELLM_MODEL" \
  "${EXTRA_CONFIG_ARGS[@]}"

exec "$REPO_ROOT/.venv/bin/harbor" run --config "$CONFIG_PATH" --yes
