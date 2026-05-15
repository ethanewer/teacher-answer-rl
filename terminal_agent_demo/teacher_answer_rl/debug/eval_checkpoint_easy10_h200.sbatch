#!/usr/bin/env bash
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=ta-easy10
#SBATCH --output=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval/slurm_logs/%x-%j.out
#SBATCH --error=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval/slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/AReaL"
EVAL_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval"

: "${CHECKPOINT:?set CHECKPOINT to a checkpoint directory}"
: "${JOB_NAME:?set JOB_NAME to a unique eval job name}"
MODEL_NAME="${MODEL_NAME:-terminal-${JOB_NAME}}"
PORT="${PORT:-$((30080 + SLURM_JOB_ID % 10000))}"
JOBS_DIR="${JOBS_DIR:-$EVAL_ROOT/harbor_jobs/$JOB_NAME}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
MAX_TURNS="${MAX_TURNS:-40}"
N_CONCURRENT="${N_CONCURRENT:-2}"
LLM_KWARGS="${LLM_KWARGS:-}"

mkdir -p "$EVAL_ROOT/server_logs" "$EVAL_ROOT/slurm_logs" "$(dirname "$JOBS_DIR")"
cd "$REPO_ROOT"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.78}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export DOCKER_WAIT_SECONDS="${DOCKER_WAIT_SECONDS:-180}"

SERVER_LOG="$EVAL_ROOT/server_logs/${JOB_NAME}.server.log"
bash terminal_agent_demo/eval/serve_terminal_model_vllm.sh \
  "$CHECKPOINT" "$MODEL_NAME" "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 1200))
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM server exited early. Last log lines:" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM server. Last log lines:" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 5
done

extra=()
if [[ -n "$LLM_KWARGS" ]]; then
  extra+=(--llm-kwargs "$LLM_KWARGS")
fi

bash terminal_agent_demo/eval/run_terminal_bench_eval_harbor.sh \
  "$JOB_NAME" "$MODEL_NAME" "http://127.0.0.1:${PORT}/v1" "$JOBS_DIR" \
  --n-attempts 1 \
  --n-concurrent "$N_CONCURRENT" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --max-turns "$MAX_TURNS" \
  "${extra[@]}"

python - "$JOBS_DIR" <<'PY'
import json
import sys
from pathlib import Path

jobs_dir = Path(sys.argv[1])
result_paths = list(jobs_dir.rglob("result.json"))
for path in result_paths:
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    stats = data.get("stats", {})
    if "n_completed_trials" not in stats:
        continue
    metric = None
    for eval_stats in stats.get("evals", {}).values():
        metrics = eval_stats.get("metrics") or []
        if metrics:
            metric = metrics[0].get("mean")
        break
    print(
        f"RESULT {path}: completed={stats.get('n_completed_trials')} "
        f"errors={stats.get('n_errored_trials')} score={metric}"
    )
PY
