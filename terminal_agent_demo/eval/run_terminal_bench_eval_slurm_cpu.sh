#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 JOB_NAME MODEL_NAME [API_BASE] [JOBS_DIR] [write-config-args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JOB_NAME="$1"
MODEL_NAME="$2"
shift 2
API_BASE="${1:-http://127.0.0.1:30080/v1}"
if [[ $# -gt 0 ]]; then shift; fi
JOBS_DIR="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval/harbor_jobs/$JOB_NAME}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

PARTITION="${PARTITION:-m7i-cpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-60G}"
TIME="${TIME:-08:00:00}"
LOG_DIR="${LOG_DIR:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval/slurm_logs}"
mkdir -p "$LOG_DIR"

extra_q=""
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  printf -v extra_q ' %q' "${EXTRA_ARGS[@]}"
fi

sbatch \
  --job-name "$JOB_NAME" \
  --partition "$PARTITION" \
  --cpus-per-task "$CPUS_PER_TASK" \
  --mem "$MEM" \
  --time "$TIME" \
  --output "$LOG_DIR/%x-%j.out" \
  --error "$LOG_DIR/%x-%j.err" \
  --wrap="cd '$REPO_ROOT' && OPENAI_API_KEY='${OPENAI_API_KEY:-EMPTY}' HF_HOME='${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}' terminal_agent_demo/eval/run_terminal_bench_eval_harbor.sh '$JOB_NAME' '$MODEL_NAME' '$API_BASE' '$JOBS_DIR'$extra_q"
