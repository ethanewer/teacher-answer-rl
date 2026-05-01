#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUT_DIR="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/terminal_synthetic_tasks/medium}"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export HF_HOME="${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}"

cd "$REPO_ROOT"
exec "$REPO_ROOT/.venv/bin/python" -m rlvr_demo.terminal_experiment prepare-synthetic-tasks \
  --output-dir "$OUTPUT_DIR"
