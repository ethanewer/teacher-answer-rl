#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 CHECKPOINT JOB_NAME GPU PORT TASK_FILE ATTEMPTS [CONCURRENCY] [MAX_OUTPUT_TOKENS]" >&2
  exit 2
fi

CHECKPOINT="$1"
JOB_NAME="$2"
GPU="$3"
PORT="$4"
TASK_FILE="$5"
ATTEMPTS="$6"
CONCURRENCY="${7:-2}"
MAX_OUTPUT_TOKENS="${8:-4096}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_ROOT="${EVAL_ROOT:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval}"
JOB_ROOT="$EVAL_ROOT/$JOB_NAME"
JOBS_DIR="$JOB_ROOT/harbor_jobs/$JOB_NAME"
SERVER_LOG_DIR="$JOB_ROOT/server_logs"
SERVER_LOG="$SERVER_LOG_DIR/server.log"
MODEL_NAME="terminal-$JOB_NAME"
API_BASE="http://127.0.0.1:$PORT/v1"

if [[ ! -f "$TASK_FILE" ]]; then
  echo "Task file not found: $TASK_FILE" >&2
  exit 2
fi

TASK_ARGS=()
while IFS= read -r task || [[ -n "$task" ]]; do
  [[ -z "$task" || "$task" =~ ^[[:space:]]*# ]] && continue
  TASK_ARGS+=(--task "$task")
done < "$TASK_FILE"

if [[ "${#TASK_ARGS[@]}" -eq 0 ]]; then
  echo "Task file contains no tasks: $TASK_FILE" >&2
  exit 2
fi

mkdir -p "$SERVER_LOG_DIR"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "$REPO_ROOT"
echo "Starting local vLLM server for $JOB_NAME on GPU $GPU, port $PORT"
CUDA_VISIBLE_DEVICES="$GPU" \
LOG_DIR="$SERVER_LOG_DIR" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.78}" \
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}" \
bash terminal_agent_demo/eval/serve_terminal_model_vllm.sh \
  "$CHECKPOINT" "$MODEL_NAME" "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + ${SERVER_TIMEOUT_SECONDS:-900}))
until curl -fsS "$API_BASE/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM server exited early; tail of $SERVER_LOG:" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM server; tail of $SERVER_LOG:" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Server ready; running task-file eval: task_file=$TASK_FILE attempts=$ATTEMPTS concurrency=$CONCURRENCY max_output=$MAX_OUTPUT_TOKENS"
DOCKER_WAIT_SECONDS="${DOCKER_WAIT_SECONDS:-300}" \
bash terminal_agent_demo/eval/run_terminal_bench_eval_harbor.sh \
  "$JOB_NAME" "$MODEL_NAME" "$API_BASE" "$JOBS_DIR" \
  "${TASK_ARGS[@]}" \
  --n-attempts "$ATTEMPTS" \
  --n-concurrent "$CONCURRENCY" \
  --max-turns "${MAX_TURNS:-40}" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-32768}" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --temperature "${TEMPERATURE:-0.2}" \
  --top-p "${TOP_P:-0.8}" \
  --top-k "${TOP_K:-20}"

"$REPO_ROOT/.venv/bin/python" -m terminal_agent_demo.terminal_experiment summarize-harbor \
  --jobs-dir "$JOBS_DIR" \
  --output "$JOB_ROOT/summary.csv" \
  --trials-per-task "$ATTEMPTS"

echo "Summary: $JOB_ROOT/summary.csv"
