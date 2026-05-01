#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 JOB_NAME LITELLM_MODEL [API_BASE] [JOBS_DIR]" >&2
  echo "Example: $0 base openai/Qwen3-4B-Thinking-2507 http://127.0.0.1:30000/v1 /wbl-fast/usrs/ee/teacher-answer-rl/harbor_jobs" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JOB_NAME="$1"
LITELLM_MODEL="$2"
API_BASE="${3:-http://127.0.0.1:30000/v1}"
JOBS_DIR="${4:-/wbl-fast/usrs/ee/teacher-answer-rl/harbor_jobs}"
CONFIG_DIR="${REPO_ROOT}/rlvr_demo/results/terminal_bench_eval_configs"
CONFIG_PATH="${CONFIG_DIR}/${JOB_NAME}.yaml"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export HF_HOME="${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}"

cd "$REPO_ROOT"
"$REPO_ROOT/.venv/bin/python" -m rlvr_demo.terminal_experiment write-harbor-eval-config \
  --output "$CONFIG_PATH" \
  --job-name "$JOB_NAME" \
  --jobs-dir "$JOBS_DIR" \
  --api-base "$API_BASE" \
  --litellm-model "$LITELLM_MODEL"

exec "$REPO_ROOT/.venv/bin/harbor" run --config "$CONFIG_PATH" --yes
