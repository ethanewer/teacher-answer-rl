#!/usr/bin/env bash
#SBATCH --job-name=tb-easy10-h200
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval/slurm_logs/%x-%j.out

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: sbatch $0 CHECKPOINT_PATH SERVED_MODEL_NAME JOB_NAME [extra write-config args...]" >&2
  exit 2
fi

REPO_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/AReaL"
CHECKPOINT_PATH="$1"
SERVED_MODEL_NAME="$2"
JOB_NAME="$3"
shift 3
EXTRA_ARGS=("$@")

PORT="${PORT:-$((32000 + SLURM_JOB_ID % 20000))}"
API_BASE="http://127.0.0.1:${PORT}/v1"
EVAL_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval/harbor_jobs/${JOB_NAME}"
SERVER_LOG_DIR="/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval/server_logs"
SERVER_LOG="${SERVER_LOG_DIR}/${JOB_NAME}_server_${SLURM_JOB_ID}.log"

mkdir -p "$EVAL_ROOT" "$SERVER_LOG_DIR"

cd "$REPO_ROOT"
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
echo "checkpoint=$CHECKPOINT_PATH"
echo "model=$SERVED_MODEL_NAME"
echo "job_name=$JOB_NAME"
echo "api_base=$API_BASE"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
MAX_MODEL_LEN=32768 \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.78}" \
TENSOR_PARALLEL_SIZE=1 \
LOG_DIR="$SERVER_LOG_DIR" \
terminal_agent_demo/eval/serve_terminal_model_vllm.sh \
  "$CHECKPOINT_PATH" \
  "$SERVED_MODEL_NAME" \
  "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + ${SERVER_WAIT_SECONDS:-1200}))
until curl -fsS "$API_BASE/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "vLLM server exited before readiness. Last server log lines:" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "vLLM server not ready after ${SERVER_WAIT_SECONDS:-1200}s. Last server log lines:" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

echo "vLLM server ready at $API_BASE"

DOCKER_WAIT_SECONDS="${DOCKER_WAIT_SECONDS:-300}" \
terminal_agent_demo/eval/run_terminal_bench_eval_harbor.sh \
  "$JOB_NAME" \
  "$SERVED_MODEL_NAME" \
  "$API_BASE" \
  "$EVAL_ROOT" \
  --n-attempts "${N_ATTEMPTS:-5}" \
  --n-concurrent "${N_CONCURRENT:-5}" \
  --max-turns "${MAX_TURNS:-40}" \
  --max-input-tokens "${MAX_INPUT_TOKENS:-32768}" \
  --max-output-tokens "${MAX_OUTPUT_TOKENS:-4096}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --top-p "${TOP_P:-0.8}" \
  --top-k "${TOP_K:-20}" \
  "${EXTRA_ARGS[@]}"

"$REPO_ROOT/.venv/bin/python" -m terminal_agent_demo.terminal_experiment summarize-harbor \
  --jobs-dir "$EVAL_ROOT" \
  --output "$EVAL_ROOT/summary.csv" \
  --trials-per-task "${N_ATTEMPTS:-5}"

echo "summary=$EVAL_ROOT/summary.csv"
