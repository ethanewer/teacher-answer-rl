#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 EVAL_NAME LITELLM_MODEL [API_BASE] [JOBS_ROOT] [write-config-args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EVAL_NAME="$1"
LITELLM_MODEL="$2"
shift 2
API_BASE="${1:-http://127.0.0.1:30080/v1}"
if [[ $# -gt 0 ]]; then shift; fi
JOBS_ROOT="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/harbor_jobs/$EVAL_NAME}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

TASK_GROUPS=(
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
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-32000}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-8192}"
export OVERRIDE_CPUS="${OVERRIDE_CPUS:-3}"
export OVERRIDE_MEMORY_MB="${OVERRIDE_MEMORY_MB:-10000}"
export LOG_DIR="${LOG_DIR:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-4b-instruct/terminal_bench_eval/slurm_logs}"

for idx in "${!TASK_GROUPS[@]}"; do
  read -r -a tasks <<<"${TASK_GROUPS[$idx]}"
  task_args=()
  for task in "${tasks[@]}"; do
    task_args+=(--task "$task")
  done
  bash "$REPO_ROOT/rlvr_demo/scripts/run_terminal_bench_eval_slurm_cpu.sh" \
    "${EVAL_NAME}-g${idx}" \
    "$LITELLM_MODEL" \
    "$API_BASE" \
    "$JOBS_ROOT/g${idx}" \
    "${task_args[@]}" \
    "${EXTRA_ARGS[@]}"
done

echo "Submitted split Terminal-Bench eval jobs under $JOBS_ROOT"
echo "Summarize after all jobs finish with:"
echo "  $REPO_ROOT/.venv/bin/python -m rlvr_demo.terminal_experiment summarize-harbor --jobs-dir '$JOBS_ROOT' --output '$JOBS_ROOT/summary.csv' --trials-per-task '$N_ATTEMPTS'"
