#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/terminal_agent_demo/scripts/env_h200.sh"

OUTPUT="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_medium.even_original.terminus_tool.jsonl}"
SUMMARY="${2:-${OUTPUT%.jsonl}.summary.json}"
INSPECT="${3:-${OUTPUT%.jsonl}.inspect.md}"

mkdir -p "$(dirname "$OUTPUT")"
cd "$REPO_ROOT"
CONVERT_ARGS=(
  --config "${TERMINAL_CORPUS_CONFIG:-skill_based_medium}"
  --row-index-parity even
  --output "$OUTPUT"
  --summary-output "$SUMMARY"
)
if [[ -n "${TERMINAL_CORPUS_LIMIT:-}" ]]; then
  CONVERT_ARGS+=(--limit "$TERMINAL_CORPUS_LIMIT")
fi
"$AREAL_VENV/bin/python" -m terminal_agent_demo.terminus_tool_calling convert-corpus \
  "${CONVERT_ARGS[@]}"
"$AREAL_VENV/bin/python" -m terminal_agent_demo.terminus_tool_calling inspect-converted \
  --input "$OUTPUT" \
  --output "$INSPECT" \
  -n "${INSPECT_N:-5}"
echo "Converted even-index data: $OUTPUT"
echo "Summary: $SUMMARY"
echo "Inspection: $INSPECT"
