#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 EVAL_NAME MODEL_NAME [API_BASE] [JOBS_ROOT] [write-config-args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EVAL_NAME="$1"
MODEL_NAME="$2"
shift 2
API_BASE="${1:-http://127.0.0.1:30080/v1}"
if [[ $# -gt 0 ]]; then shift; fi
JOBS_ROOT="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval/harbor_jobs/$EVAL_NAME}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

TASKS=(
  "modernize-scientific-stack"
  "log-summary-date-ranges"
  "multi-source-data-merger"
  "nginx-request-logging"
  "git-leak-recovery"
  "fix-git"
  "constraints-scheduling"
  "vulnerable-secret"
  "regex-log"
  "sqlite-db-truncate"
)

mkdir -p "$JOBS_ROOT"

export PARTITION="${PARTITION:-m7i-cpu}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
export MEM="${MEM:-60G}"
export TIME="${TIME:-08:00:00}"
export N_ATTEMPTS="${N_ATTEMPTS:-5}"
export N_CONCURRENT="${N_CONCURRENT:-5}"
export MAX_TURNS="${MAX_TURNS:-40}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-32768}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-6144}"

for idx in "${!TASKS[@]}"; do
  task="${TASKS[$idx]}"
  "$REPO_ROOT/terminal_agent_demo/eval/run_terminal_bench_eval_slurm_cpu.sh" \
    "${EVAL_NAME}-g${idx}" \
    "$MODEL_NAME" \
    "$API_BASE" \
    "$JOBS_ROOT/g${idx}" \
    --task "$task" \
    --n-attempts "$N_ATTEMPTS" \
    --n-concurrent "$N_CONCURRENT" \
    --max-turns "$MAX_TURNS" \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    "${EXTRA_ARGS[@]}"
done

echo "Submitted split Terminal-Bench eval jobs under $JOBS_ROOT"
echo "Summarize after all jobs finish with:"
echo "  $REPO_ROOT/.venv/bin/python -m terminal_agent_demo.terminal_experiment summarize-harbor --jobs-dir '$JOBS_ROOT' --output '$JOBS_ROOT/summary.csv' --trials-per-task '$N_ATTEMPTS'"
