#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

READY_FILE="${TERMINAL_TASK_SERVICE_READY_FILE:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/terminal_task_service/endpoint.json}"
LOG_DIR="${TERMINAL_TASK_SERVICE_LOG_DIR:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent/terminal_task_service/logs}"
PORT="${TERMINAL_TASK_SERVICE_PORT:-39080}"
MAX_WORKERS="${TERMINAL_TASK_SERVICE_MAX_WORKERS:-96}"
MAX_SESSIONS="${TERMINAL_TASK_SERVICE_MAX_SESSIONS:-64}"
MAX_STARTS="${TERMINAL_TASK_SERVICE_MAX_STARTS:-8}"
DEFAULT_OUTPUT_ROOT="/tmp/terminal-task-service-runs-${USER:-$(id -u)}"
OUTPUT_ROOT="${TERMINAL_TASK_SERVICE_OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
PARTITION="${PARTITION:-m7i-cpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-60G}"
TIME="${TIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-terminal-task-service}"
NODELIST="${NODELIST:-}"
EXCLUDE="${EXCLUDE:-}"
WAIT=0
PRINT_URL=0
WAIT_SECONDS="${TERMINAL_TASK_SERVICE_WAIT_SECONDS:-1800}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ready-file)
      READY_FILE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --max-sessions)
      MAX_SESSIONS="$2"
      shift 2
      ;;
    --max-starts)
      MAX_STARTS="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --nodelist)
      NODELIST="$2"
      shift 2
      ;;
    --exclude)
      EXCLUDE="$2"
      shift 2
      ;;
    --wait)
      WAIT=1
      shift
      ;;
    --print-url)
      PRINT_URL=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "$READY_FILE")" "$LOG_DIR"
rm -f "$READY_FILE"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required to start the Terminal task service on $PARTITION." >&2
  exit 2
fi

printf -v repo_q "%q" "$REPO_ROOT"
printf -v python_q "%q" "$REPO_ROOT/.venv/bin/python"
printf -v path_q "%q" "$REPO_ROOT/.venv/bin:$PATH"
printf -v pythonpath_q "%q" "$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
printf -v hf_home_q "%q" "${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}"
printf -v ready_file_q "%q" "$READY_FILE"
printf -v output_root_q "%q" "$OUTPUT_ROOT"
wrap_cmd="cd $repo_q && PATH=$path_q PYTHONPATH=$pythonpath_q HF_HOME=$hf_home_q $python_q -m rlvr_demo.terminal_task_service --host 0.0.0.0 --port $PORT --ready-file $ready_file_q --max-workers $MAX_WORKERS --max-sessions $MAX_SESSIONS --output-root $output_root_q"
if [[ -n "$MAX_STARTS" ]]; then
  wrap_cmd="$wrap_cmd --max-starts $MAX_STARTS"
fi

SBATCH_ARGS=(
  -p "$PARTITION"
  -N 1
  --cpus-per-task="$CPUS_PER_TASK"
  --mem="$MEM"
  --time="$TIME"
)
if [[ -n "$NODELIST" ]]; then
  SBATCH_ARGS+=(--nodelist="$NODELIST")
fi
if [[ -n "$EXCLUDE" ]]; then
  SBATCH_ARGS+=(--exclude="$EXCLUDE")
fi

job_id=$(
  sbatch --parsable \
    "${SBATCH_ARGS[@]}" \
    -J "$JOB_NAME" \
    --output="$LOG_DIR/${JOB_NAME}-%j.out" \
    --wrap="$wrap_cmd"
)

echo "Submitted Terminal task service job $job_id on $PARTITION" >&2
echo "Ready file: $READY_FILE" >&2

if [[ "$WAIT" == "1" ]]; then
  deadline=$((SECONDS + WAIT_SECONDS))
  last_status=0
  while [[ ! -s "$READY_FILE" ]]; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for Terminal task service ready file." >&2
      squeue -j "$job_id" >&2 || true
      exit 1
    fi
    queue_line="$(squeue -j "$job_id" -h 2>/dev/null || true)"
    if [[ -z "$queue_line" ]]; then
      echo "Terminal task service job $job_id is no longer in the queue." >&2
      exit 1
    fi
    if (( SECONDS >= last_status + 30 )); then
      echo "Waiting for Terminal task service job $job_id: $queue_line" >&2
      last_status=$SECONDS
    fi
    sleep 5
  done
fi

if [[ "$PRINT_URL" == "1" ]]; then
  "$REPO_ROOT/.venv/bin/python" - "$READY_FILE" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data["url"])
PY
else
  echo "$job_id"
fi
