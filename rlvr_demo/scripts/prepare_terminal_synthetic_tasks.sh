#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUT_DIR="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/terminal_synthetic_tasks/medium}"
if [[ $# -gt 0 ]]; then
  shift
fi
SUBSET="${SYNTHETIC_TASK_SUBSET:-medium}"
SUBSET_ARGS=(--subset "$SUBSET")
for arg in "$@"; do
  case "$arg" in
    --subset|--subset=*|--file|--file=*)
      SUBSET_ARGS=()
      break
      ;;
  esac
done

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export HF_HOME="${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}"

cd "$REPO_ROOT"
exec "$REPO_ROOT/.venv/bin/python" -m rlvr_demo.terminal_experiment prepare-synthetic-tasks \
  --output-dir "$OUTPUT_DIR" \
  "${SUBSET_ARGS[@]}" \
  "$@"
