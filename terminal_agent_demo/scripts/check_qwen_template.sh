#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/terminal_agent_demo/scripts/env_h200.sh"

OUTPUT="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/qwen_template_append_only_render.txt}"
cd "$REPO_ROOT"
"$AREAL_VENV/bin/python" -m terminal_agent_demo.terminus_tool_calling check-qwen-template \
  --model "${QWEN_TEMPLATE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}" \
  --cache-dir "${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}" \
  --local-files-only \
  --output "$OUTPUT"
