#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FILEROOT="${FILEROOT:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-qwen3-8b}"
SFT_CONFIG="${SFT_CONFIG:-$REPO_ROOT/rlvr_demo/configs/qwen3_8b_terminal_sft_paper_h200.yaml}"
TEACHER_CONFIG="${TEACHER_CONFIG:-$REPO_ROOT/rlvr_demo/configs/qwen3_8b_terminal_teacher_answer_rl_paper_h200.yaml}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-nvidia/Nemotron-Terminal-Corpus}"
DATASET_CONFIG="${DATASET_CONFIG:-skill_based_medium}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
SEED="${SEED:-7}"
SFT_EXPERIMENT="${SFT_EXPERIMENT:-qwen3-8b-terminal-sft-skill-medium-1k-b128-24step-h200}"
TEACHER_EXPERIMENT="${TEACHER_EXPERIMENT:-qwen3-8b-terminal-teacher-answer-rl-skill-medium-1k-b64-s2-48step-2048-h200}"
RESULT_DIR="${RESULT_DIR:-$FILEROOT/results/qwen3-8b-skill-medium-1k-b128-vs-b64-s2-2048}"
EVAL_DIR="${EVAL_DIR:-$FILEROOT/results/eval_qwen3_8b_skill_medium_1k_b128_b64_s2_2048}"
EVAL_LIMIT_ROWS="${EVAL_LIMIT_ROWS:-1000}"
EVAL_SKIP_TURNS="${EVAL_SKIP_TURNS:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-32}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-40960}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
EVAL_GREEDY="${EVAL_GREEDY:-0}"

DEFAULT_AREAL_VENV="$REPO_ROOT/.venv-megatron"
if [[ ! -x "$DEFAULT_AREAL_VENV/bin/python" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  DEFAULT_AREAL_VENV="$REPO_ROOT/.venv"
fi
AREAL_VENV="${AREAL_VENV:-$DEFAULT_AREAL_VENV}"

export HF_HOME="${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: reproduce_terminal_qwen3_8b_paper_baseline.sh <command>

Commands:
  chat-check       Verify Qwen3-8B chat-template handling on released trajectories.
  data-report      Summarize released-corpus mix and sampled token lengths.
  sft-smoke        Run a 4-step Qwen3-8B SFT smoke test on the mixed corpus.
  sft-full         Run the paper-style Qwen3-8B SFT baseline: 2 epochs, batch 128.
  sft-skill-final  Run the final single-node skill_based_medium SFT comparison.
  teacher-smoke    Run a 4-step Qwen3-8B teacher-answer-RL smoke test.
  teacher-full     Run the Qwen3-8B teacher-answer-RL comparison recipe.
  teacher-skill-final
                   Run the final single-node skill_based_medium teacher-answer-RL comparison.
  eval             Offline-evaluate final SFT, teacher closest to SFT time, and final teacher.
  eval-baselines   Offline-evaluate Qwen3-8B base and released Nemotron-Terminal-8B.
  compile          Compile checkpoint logs, eval metrics, and comparison table.
EOF
}

checkpoint_from_events() {
  local experiment="$1"
  local mode="$2"
  local target_elapsed="${3:-}"
  "$AREAL_VENV/bin/python" - "$FILEROOT/checkpoints/$(id -un)/$experiment/trial0/checkpoint_events.jsonl" "$mode" "$target_elapsed" <<'PY'
import json
import sys
from pathlib import Path

events_path = Path(sys.argv[1])
mode = sys.argv[2]
target = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
events = []
if events_path.exists():
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
if not events:
    raise SystemExit(f"no checkpoint events found at {events_path}")
if mode == "final":
    event = max(events, key=lambda row: (int(row["optimizer_step"]), int(row["global_step"])))
elif mode == "closest":
    if target is None:
        raise SystemExit("closest mode requires target elapsed seconds")
    event = min(
        events,
        key=lambda row: (
            abs(float(row["elapsed_wall_clock_sec"]) - target),
            -int(row["optimizer_step"]),
        ),
    )
else:
    raise SystemExit(f"unknown checkpoint selection mode: {mode}")
print(event["checkpoint_path"])
PY
}

checkpoint_elapsed_from_events() {
  local experiment="$1"
  local mode="$2"
  "$AREAL_VENV/bin/python" - "$FILEROOT/checkpoints/$(id -un)/$experiment/trial0/checkpoint_events.jsonl" "$mode" <<'PY'
import json
import sys
from pathlib import Path

events_path = Path(sys.argv[1])
mode = sys.argv[2]
events = []
if events_path.exists():
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
if not events:
    raise SystemExit(f"no checkpoint events found at {events_path}")
if mode != "final":
    raise SystemExit(f"unknown elapsed selection mode: {mode}")
event = max(events, key=lambda row: (int(row["optimizer_step"]), int(row["global_step"])))
print(event["elapsed_wall_clock_sec"])
PY
}

run_eval_one() {
  local name="$1"
  local checkpoint="$2"
  local gpu="$3"
  local greedy_args=()
  if [[ "$EVAL_GREEDY" == "1" ]]; then
    greedy_args=(--greedy)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$AREAL_VENV/bin/python" -m rlvr_demo.terminal_offline_eval \
    --checkpoint "$checkpoint" \
    --dataset "$DATASET" \
    --dataset-config skill_based_medium \
    --split train \
    --limit-rows "$EVAL_LIMIT_ROWS" \
    --skip-turns "$EVAL_SKIP_TURNS" \
    --limit "$EVAL_LIMIT" \
    --max-length "$EVAL_MAX_LENGTH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output "$EVAL_DIR/$name.json" \
    --predictions-output "$EVAL_DIR/$name.predictions.jsonl" \
    "${greedy_args[@]}"
}

run_comparison_eval() {
  mkdir -p "$EVAL_DIR"
  local sft_final
  local sft_elapsed
  local teacher_closest
  local teacher_final
  sft_final="$(checkpoint_from_events "$SFT_EXPERIMENT" final)"
  sft_elapsed="$(checkpoint_elapsed_from_events "$SFT_EXPERIMENT" final)"
  teacher_closest="$(checkpoint_from_events "$TEACHER_EXPERIMENT" closest "$sft_elapsed")"
  teacher_final="$(checkpoint_from_events "$TEACHER_EXPERIMENT" final)"

  run_eval_one sft_final "$sft_final" 0 &
  local pid_sft=$!
  run_eval_one teacher_closest_wallclock "$teacher_closest" 1 &
  local pid_teacher_closest=$!
  run_eval_one teacher_final "$teacher_final" 2 &
  local pid_teacher_final=$!
  wait "$pid_sft" "$pid_teacher_closest" "$pid_teacher_final"
}

run_baseline_eval() {
  mkdir -p "$EVAL_DIR"
  run_eval_one qwen3_8b_base Qwen/Qwen3-8B 0 &
  local pid_base=$!
  run_eval_one nemotron_terminal_8b nvidia/Nemotron-Terminal-8B 1 &
  local pid_nemotron=$!
  wait "$pid_base" "$pid_nemotron"
}

compile_results() {
  "$AREAL_VENV/bin/python" -m rlvr_demo.terminal_compile_results \
    --fileroot "$FILEROOT" \
    --experiment "$SFT_EXPERIMENT" \
    --experiment "$TEACHER_EXPERIMENT" \
    --eval-dir "$EVAL_DIR" \
    --output-dir "$RESULT_DIR"
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 2
fi
shift || true

cd "$REPO_ROOT"
mkdir -p "$FILEROOT/results" "$FILEROOT/logs"

case "$cmd" in
  chat-check)
    .venv/bin/python -m rlvr_demo.terminal_chat_template_check \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --dataset-config "$DATASET_CONFIG" \
      --split train \
      --max-length "$MAX_LENGTH" \
      --output "$FILEROOT/results/qwen3_8b_chat_template_check.json" \
      "$@"
    ;;
  data-report)
    .venv/bin/python -m rlvr_demo.terminal_data_report \
      --model "$MODEL" \
      --path "$DATASET" \
      --name "$DATASET_CONFIG" \
      --split train \
      --max-length "$MAX_LENGTH" \
      --output "$FILEROOT/results/qwen3_8b_data_report.json" \
      "$@"
    ;;
  sft-smoke)
    bash rlvr_demo/scripts/run_terminal_sft_h200.sh "$SFT_CONFIG" \
      experiment_name=qwen3-8b-terminal-sft-paper-mix-h200-smoke \
      total_train_epochs=1 total_train_steps=4 \
      train_dataset.dataset_kwargs.split_part=null \
      +train_dataset.dataset_kwargs.limit_rows=64 \
      +train_dataset.dataset_kwargs.limit=32 \
      valid_dataset=null \
      train_dataset.batch_size=8 \
      saver.freq_steps=2 \
      evaluator.eval_before_train=false evaluator.freq_steps=999 \
      "$@"
    ;;
  sft-full)
    bash rlvr_demo/scripts/run_terminal_sft_h200.sh "$SFT_CONFIG" "$@"
    ;;
  sft-skill-final)
    bash rlvr_demo/scripts/run_terminal_sft_h200.sh "$SFT_CONFIG" \
      experiment_name="$SFT_EXPERIMENT" \
      train_dataset.dataset_kwargs.name=skill_based_medium \
      +train_dataset.dataset_kwargs.limit_rows=1000 \
      train_dataset.batch_size=128 \
      total_train_epochs=2 total_train_steps=24 \
      saver.freq_steps=4 \
      "$@"
    ;;
  teacher-smoke)
    bash rlvr_demo/scripts/run_terminal_teacher_answer_rl_h200.sh "$TEACHER_CONFIG" \
      experiment_name=qwen3-8b-terminal-teacher-answer-rl-paper-mix-h200-smoke \
      total_train_epochs=1 total_train_steps=4 \
      train_dataset.dataset_kwargs.split_part=null \
      +train_dataset.dataset_kwargs.limit_rows=64 \
      +train_dataset.dataset_kwargs.limit=32 \
      valid_dataset=null \
      train_dataset.batch_size=8 rollout.consumer_batch_size=8 \
      gconfig.n_samples=2 gconfig.max_new_tokens=1024 eval_gconfig.max_new_tokens=1024 actor.max_new_tokens=1024 \
      vllm.max_num_seqs=64 rollout.max_concurrent_rollouts=64 rollout.queue_size=512 \
      saver.freq_steps=2 \
      evaluator.eval_before_train=false evaluator.freq_steps=999 \
      "$@"
    ;;
  teacher-full)
    bash rlvr_demo/scripts/run_terminal_teacher_answer_rl_h200.sh "$TEACHER_CONFIG" "$@"
    ;;
  teacher-skill-final)
    bash rlvr_demo/scripts/run_terminal_teacher_answer_rl_h200.sh "$TEACHER_CONFIG" \
      experiment_name="$TEACHER_EXPERIMENT" \
      train_dataset.dataset_kwargs.name=skill_based_medium \
      +train_dataset.dataset_kwargs.limit_rows=1000 \
      train_dataset.batch_size=64 rollout.consumer_batch_size=64 \
      total_train_epochs=2 total_train_steps=48 \
      gconfig.n_samples=2 gconfig.max_new_tokens=2048 \
      eval_gconfig.max_new_tokens=2048 actor.max_new_tokens=2048 \
      vllm.max_num_seqs=128 rollout.max_concurrent_rollouts=256 rollout.queue_size=2048 \
      saver.freq_steps=8 \
      "$@"
    ;;
  eval)
    run_comparison_eval "$@"
    ;;
  eval-baselines)
    run_baseline_eval "$@"
    ;;
  compile)
    compile_results "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
