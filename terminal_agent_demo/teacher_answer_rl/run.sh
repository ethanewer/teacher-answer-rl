#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/terminal_agent_demo/scripts/env_h200.sh"

export TEACHER_ANSWER_FORMAT_BONUS="${TEACHER_ANSWER_FORMAT_BONUS:-0.0}"
export TEACHER_ANSWER_LENGTH_PENALTY="${TEACHER_ANSWER_LENGTH_PENALTY:-0.05}"

if [[ "${1:-}" == "--config" ]]; then
  shift
fi
CONFIG="${1:-$REPO_ROOT/terminal_agent_demo/teacher_answer_rl/config.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$REPO_ROOT"
exec "$AREAL_VENV/bin/python" -m terminal_agent_demo.teacher_answer_rl.train --config "$CONFIG" "$@"
