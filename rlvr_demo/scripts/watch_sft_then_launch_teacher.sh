#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

FILEROOT="${FILEROOT:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b}"
SFT_EXPERIMENT="${SFT_EXPERIMENT:-qwen3-8b-terminal-sft-released-fullmix-trajectory-nofilter-b128-2epoch-h200-slurm4}"
TEACHER_EXPERIMENT="${TEACHER_EXPERIMENT:-qwen3-8b-terminal-teacher-answer-rl-released-fullmix-nofilter-b128-s2-5720step-h200}"
TARGET_SFT_STEP="${TARGET_SFT_STEP:-5720}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"
USER_NAME="${USER_NAME:-$(id -un)}"
EVENTS_PATH="$FILEROOT/checkpoints/$USER_NAME/$SFT_EXPERIMENT/trial0/checkpoint_events.jsonl"
TEACHER_EVENTS_PATH="$FILEROOT/checkpoints/$USER_NAME/$TEACHER_EXPERIMENT/trial0/checkpoint_events.jsonl"

find_checkpoint() {
  "$REPO_ROOT/.venv-megatron/bin/python" - "$EVENTS_PATH" "$TARGET_SFT_STEP" <<'PY'
import json
import sys
from pathlib import Path

events_path = Path(sys.argv[1])
target_step = int(sys.argv[2])
if not events_path.exists():
    raise SystemExit(1)
events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
matches = [event for event in events if int(event.get("optimizer_step", -1)) >= target_step]
if not matches:
    raise SystemExit(1)
event = max(matches, key=lambda row: (int(row["optimizer_step"]), int(row["global_step"])))
path = Path(event["checkpoint_path"])
if not path.exists():
    raise SystemExit(1)
print(path)
PY
}

teacher_started() {
  [[ -s "$TEACHER_EVENTS_PATH" ]]
}

cd "$REPO_ROOT"
mkdir -p "$(dirname "$TEACHER_EVENTS_PATH")"

while true; do
  if teacher_started; then
    echo "Teacher checkpoint events already exist at $TEACHER_EVENTS_PATH; not launching a duplicate run." >&2
    exit 0
  fi
  if checkpoint_path="$(find_checkpoint 2>/dev/null)"; then
    export FILEROOT
    export SFT_EXPERIMENT
    export TEACHER_EXPERIMENT
    export DATASET_CONFIG="${DATASET_CONFIG:-full_mix}"
    export TERMINAL_TEACHER_INIT_PATH="$checkpoint_path"
    echo "Launching teacher-answer-RL from final SFT checkpoint: $TERMINAL_TEACHER_INIT_PATH" >&2
    exec bash rlvr_demo/scripts/reproduce_terminal_qwen3_8b_paper_baseline.sh teacher-full
  fi
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') waiting for SFT checkpoint step >= $TARGET_SFT_STEP at $EVENTS_PATH" >&2
  sleep "$CHECK_INTERVAL_SEC"
done
