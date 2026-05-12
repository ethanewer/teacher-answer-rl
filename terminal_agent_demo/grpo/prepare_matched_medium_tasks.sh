#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/terminal_agent_demo/scripts/env_h200.sh"

OUTPUT="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_medium.even_original.synthetic_tasks_manifest.csv}"
SUMMARY="${2:-${OUTPUT%.csv}.summary.json}"
SFT_JSONL="${SFT_JSONL:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/skill_based_medium.even_original.terminus_tool.jsonl}"
SYNTHETIC_MANIFEST="${SYNTHETIC_MANIFEST:-/wbl-fast/usrs/ee/teacher-answer-rl/terminal_synthetic_tasks/medium/manifest.csv}"

cd "$REPO_ROOT"
"$AREAL_VENV/bin/python" -m terminal_agent_demo.grpo.prepare_matched_tasks \
  --sft-jsonl "$SFT_JSONL" \
  --synthetic-manifest "$SYNTHETIC_MANIFEST" \
  --output "$OUTPUT" \
  --summary-output "$SUMMARY"
