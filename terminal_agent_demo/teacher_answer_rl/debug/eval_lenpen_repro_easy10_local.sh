#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/AReaL"
EVAL_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/terminal_bench_eval"
LOG_ROOT="$EVAL_ROOT/ta_lenpen_repro_local_logs"
CKPT_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/checkpoints/ewer/ta-ref-lenpen-w25-p128-syn08-o2048-s50/trial0/default"

mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"

run_one() {
  local gpu="$1"
  local port="$2"
  local step="$3"
  local job="ta-lenpen-s${step}-easy10-t4096-a1-localrepro1"
  local ckpt="$CKPT_ROOT/epoch0epochstep${step}globalstep${step}"
  local model="terminal-${job}"
  local session="ta_lenpen_${job}"
  local jobs_dir="$EVAL_ROOT/harbor_jobs/$job"
  local server_log="$LOG_ROOT/${job}.server.log"
  local eval_log="$LOG_ROOT/${job}.eval.log"

  if [[ ! -d "$ckpt" ]]; then
    echo "missing checkpoint: $ckpt" >&2
    return 1
  fi

  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES=$gpu LOG_DIR='$LOG_ROOT/server_logs' MAX_MODEL_LEN=32768 GPU_MEMORY_UTILIZATION=0.78 TENSOR_PARALLEL_SIZE=1 bash terminal_agent_demo/eval/serve_terminal_model_vllm.sh '$ckpt' '$model' '$port' >'$server_log' 2>&1"

  local deadline=$((SECONDS + 1200))
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "SERVER_EXITED $job" | tee -a "$eval_log"
      tail -120 "$server_log" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "SERVER_TIMEOUT $job" | tee -a "$eval_log"
      tail -120 "$server_log" >&2 || true
      tmux kill-session -t "$session" 2>/dev/null || true
      return 1
    fi
    sleep 5
  done

  set +e
  DOCKER_WAIT_SECONDS=180 bash terminal_agent_demo/eval/run_terminal_bench_eval_harbor.sh \
    "$job" "$model" "http://127.0.0.1:${port}/v1" "$jobs_dir" \
    --n-attempts 1 \
    --n-concurrent 2 \
    --max-output-tokens 4096 \
    --max-turns 40 \
    >"$eval_log" 2>&1
  local rc=$?
  set -e

  tmux kill-session -t "$session" 2>/dev/null || true

  python - "$jobs_dir" "$job" "$rc" <<'PY'
import json
import sys
from pathlib import Path

jobs_dir = Path(sys.argv[1])
job = sys.argv[2]
rc = int(sys.argv[3])
for path in jobs_dir.rglob("result.json"):
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
        f"{job}\treturn={rc}\tcompleted={stats.get('n_completed_trials')}\t"
        f"errors={stats.get('n_errored_trials')}\tscore={metric}"
    )
PY
  return "$rc"
}

export -f run_one
export REPO_ROOT EVAL_ROOT LOG_ROOT CKPT_ROOT

cat >"$LOG_ROOT/candidates.tsv" <<'TSV'
0	31590	19
1	31591	24
2	31592	39
TSV

while IFS=$'\t' read -r gpu port step; do
  bash -lc "run_one '$gpu' '$port' '$step'" &
done <"$LOG_ROOT/candidates.tsv"

wait
