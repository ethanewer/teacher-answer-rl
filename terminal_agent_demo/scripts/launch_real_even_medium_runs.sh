#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo"
DATA="$RUN_ROOT/data/skill_based_medium.even_original.terminus_tool.jsonl"
SUMMARY="${DATA%.jsonl}.summary.json"
INSPECT="${DATA%.jsonl}.inspect.md"
LAUNCH_LOG_DIR="$RUN_ROOT/launch_logs"
mkdir -p "$LAUNCH_LOG_DIR" "$RUN_ROOT/slurm"

CURRENT_NODE="$(hostname -s)"
EXCLUDE_NODES="${EXCLUDE_NODES:-$CURRENT_NODE}"
SBATCH_EXTRA_ARGS=()
if [[ -n "$EXCLUDE_NODES" ]]; then
  SBATCH_EXTRA_ARGS+=(--exclude="$EXCLUDE_NODES")
fi

cd "$REPO_ROOT"
if [[ ! -s "$DATA" ]]; then
  bash terminal_agent_demo/scripts/prepare_even_medium_data.sh "$DATA" "$SUMMARY" "$INSPECT"
fi

LAUNCH_RECORD="$LAUNCH_LOG_DIR/real_even_medium_$(date -u +%Y%m%d_%H%M%S).txt"
{
  echo "timestamp_utc=$(date -u -Is)"
  echo "submit_host=$(hostname)"
  echo "exclude_nodes=$EXCLUDE_NODES"
  echo "data=$DATA"
  echo "summary=$SUMMARY"
  echo "sft_config=$REPO_ROOT/terminal_agent_demo/sft/config_even_medium_real.yaml"
  echo "teacher_answer_rl_config=$REPO_ROOT/terminal_agent_demo/teacher_answer_rl/config_even_medium_real.yaml"
  echo "sft_sbatch=$REPO_ROOT/terminal_agent_demo/sft/run_even_medium_real.sbatch"
  echo "teacher_answer_rl_sbatch=$REPO_ROOT/terminal_agent_demo/teacher_answer_rl/run_even_medium_real.sbatch"
} | tee "$LAUNCH_RECORD"

SFT_OUTPUT="$(sbatch "${SBATCH_EXTRA_ARGS[@]}" "$REPO_ROOT/terminal_agent_demo/sft/run_even_medium_real.sbatch")"
TA_OUTPUT="$(sbatch "${SBATCH_EXTRA_ARGS[@]}" "$REPO_ROOT/terminal_agent_demo/teacher_answer_rl/run_even_medium_real.sbatch")"

{
  echo "sft_submit_output=$SFT_OUTPUT"
  echo "teacher_answer_rl_submit_output=$TA_OUTPUT"
} | tee -a "$LAUNCH_RECORD"

echo "$SFT_OUTPUT"
echo "$TA_OUTPUT"
echo "Launch record: $LAUNCH_RECORD"
