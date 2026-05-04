#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 JOB_NAME LITELLM_MODEL [API_BASE] [JOBS_DIR] [write-config-args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JOB_NAME="$1"
LITELLM_MODEL="$2"
shift 2
API_BASE="${1:-http://127.0.0.1:30080/v1}"
if [[ $# -gt 0 ]]; then shift; fi
JOBS_DIR="${1:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/harbor_jobs/$JOB_NAME}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_CONFIG_ARGS=("$@")

LOG_DIR="${LOG_DIR:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b/terminal_bench_eval/slurm_logs}"
mkdir -p "$LOG_DIR"

CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-60G}"
TIME="${TIME:-12:00:00}"
PARTITION="${PARTITION:-m7i-cpu}"
EXCLUDE="${EXCLUDE:-}"
NODELIST="${NODELIST:-}"
DEFAULT_CONFIG_ARGS=(
  --n-attempts "${N_ATTEMPTS:-5}"
  --n-concurrent "${N_CONCURRENT:-3}"
  --max-turns "${MAX_TURNS:-40}"
  --max-input-tokens "${MAX_INPUT_TOKENS:-32000}"
  --max-output-tokens "${MAX_OUTPUT_TOKENS:-8192}"
  --override-cpus "${OVERRIDE_CPUS:-4}"
  --override-memory-mb "${OVERRIDE_MEMORY_MB:-16384}"
)

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required to run Terminal-Bench on the Docker-capable CPU partition." >&2
  exit 2
fi

printf -v extra_q '%q ' "${DEFAULT_CONFIG_ARGS[@]}" "${EXTRA_CONFIG_ARGS[@]}"

SBATCH_ARGS=(-p "$PARTITION" -N1 --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM" --time="$TIME")
if [[ -n "$NODELIST" ]]; then
  SBATCH_ARGS+=(--nodelist="$NODELIST")
fi
if [[ -n "$EXCLUDE" ]]; then
  SBATCH_ARGS+=(--exclude="$EXCLUDE")
fi

sbatch "${SBATCH_ARGS[@]}" \
  -J "$JOB_NAME" \
  --output="$LOG_DIR/${JOB_NAME}-%j.out" \
  --wrap="cd '$REPO_ROOT' && OPENAI_API_KEY='${OPENAI_API_KEY:-EMPTY}' HF_HOME='${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}' rlvr_demo/scripts/run_terminal_bench_eval_harbor.sh '$JOB_NAME' '$LITELLM_MODEL' '$API_BASE' '$JOBS_DIR' ${extra_q}"
