#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/terminal_agent_demo/scripts/env_h200.sh"

if [[ "${1:-}" == "--config" ]]; then
  shift
fi
CONFIG="${1:-$REPO_ROOT/terminal_agent_demo/sft/config.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$REPO_ROOT"
exec "$AREAL_VENV/bin/python" -m terminal_agent_demo.sft.train --config "$CONFIG" "$@"
