#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/terminal_agent_demo/scripts/env_h200.sh"

if [[ "${1:-}" == "--config" ]]; then
  shift
fi
CONFIG="${1:-$REPO_ROOT/terminal_agent_demo/grpo/config.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

MANIFEST="/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_medium.even_original.synthetic_tasks_manifest.csv"
SUMMARY="${MANIFEST%.csv}.summary.json"
if [[ "$CONFIG" == "$REPO_ROOT/terminal_agent_demo/grpo/config_matched_smoke.yaml" && ! -s "$MANIFEST" ]]; then
  bash "$REPO_ROOT/terminal_agent_demo/grpo/prepare_matched_medium_tasks.sh" "$MANIFEST" "$SUMMARY"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for Terminal-Bench task GRPO." >&2
  exit 2
fi

cd "$REPO_ROOT"
exec "$AREAL_VENV/bin/python" -m terminal_agent_demo.grpo.train --config "$CONFIG" "$@"
