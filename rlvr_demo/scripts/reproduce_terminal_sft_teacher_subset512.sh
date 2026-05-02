#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ACTION="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

SFT_EXPERIMENT="${SFT_EXPERIMENT:-qwen3-4b-terminal-sft-tree40960-h200-subset512}"
TEACHER_EXPERIMENT="${TEACHER_EXPERIMENT:-qwen3-4b-terminal-teacher-answer-rl-tree40960-h200-subset512}"
FILEROOT_DEFAULT="/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent"
FILEROOT="${FILEROOT:-$FILEROOT_DEFAULT}"
RESULT_DIR="${RESULT_DIR:-$FILEROOT/results/subset512-sft-vs-teacher-rl}"
EVAL_DIR="${EVAL_DIR:-$FILEROOT/results/eval}"

SFT_CONFIG="${SFT_CONFIG:-$REPO_ROOT/rlvr_demo/configs/qwen3_4b_terminal_sft_h200_1000.yaml}"
TEACHER_CONFIG="${TEACHER_CONFIG:-$REPO_ROOT/rlvr_demo/configs/qwen3_4b_terminal_teacher_answer_rl_h200_1000.yaml}"

TRAIN_LIMIT_ROWS="${TRAIN_LIMIT_ROWS:-1024}"
TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
EVAL_LIMIT_ROWS="${EVAL_LIMIT_ROWS:-4096}"
EVAL_SKIP_TURNS="${EVAL_SKIP_TURNS:-512}"
EVAL_LIMIT="${EVAL_LIMIT:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

DEFAULT_AREAL_VENV="$REPO_ROOT/.venv-megatron"
if [[ ! -x "$DEFAULT_AREAL_VENV/bin/python" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  DEFAULT_AREAL_VENV="$REPO_ROOT/.venv"
fi
AREAL_VENV="${AREAL_VENV:-$DEFAULT_AREAL_VENV}"

export PATH="$AREAL_VENV/bin:$PATH"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  rlvr_demo/scripts/reproduce_terminal_sft_teacher_subset512.sh <action>

Actions:
  chat-check   Verify Qwen3 chat-template and teacher-answer split handling.
  sft          Run the final SFT subset-512 training recipe.
  teacher-rl   Run the final teacher-answer-RL subset-512 training recipe.
  eval         Evaluate final SFT, closest-wall-clock teacher-RL, and final teacher-RL checkpoints.
  compile      Regenerate checkpoint_log.csv/jsonl and comparison_table.json.
  all          Run chat-check, SFT, teacher-RL, eval, and compile in order.

Useful environment overrides:
  SFT_EXPERIMENT, TEACHER_EXPERIMENT, FILEROOT, RESULT_DIR, EVAL_DIR
  TRAIN_LIMIT_ROWS, TRAIN_LIMIT, EVAL_LIMIT_ROWS, EVAL_SKIP_TURNS, EVAL_LIMIT
  CUDA_VISIBLE_DEVICES, AREAL_VENV, HF_HOME

This script reproduces the completed publication-trace run on the matched
512-turn subset. To scale beyond this subset, remove the +limit_rows/+limit
overrides from the sft and teacher-rl commands below and adjust total steps.
EOF
}

chat_check() {
  "$AREAL_VENV/bin/python" -m rlvr_demo.terminal_chat_template_check \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --dataset nvidia/Nemotron-Terminal-Corpus \
    --dataset-config skill_based_medium \
    --split train \
    --max-length 40960 \
    --output "$FILEROOT/results/chat_template_check.json"
}

run_sft() {
  bash "$REPO_ROOT/rlvr_demo/scripts/run_terminal_sft_h200.sh" "$SFT_CONFIG" \
    experiment_name="$SFT_EXPERIMENT" \
    total_train_epochs=1 total_train_steps=8 \
    train_dataset.batch_size=64 \
    train_dataset.dataset_kwargs.split_part=null \
    +train_dataset.dataset_kwargs.limit_rows="$TRAIN_LIMIT_ROWS" \
    +train_dataset.dataset_kwargs.limit="$TRAIN_LIMIT" \
    valid_dataset=null \
    saver.freq_steps=2 \
    evaluator.eval_before_train=false evaluator.freq_steps=999 \
    stats_logger.wandb.mode=disabled "$@"
}

run_teacher_rl() {
  bash "$REPO_ROOT/rlvr_demo/scripts/run_terminal_teacher_answer_rl_h200.sh" "$TEACHER_CONFIG" \
    experiment_name="$TEACHER_EXPERIMENT" \
    total_train_epochs=1 total_train_steps=16 \
    train_dataset.batch_size=32 rollout.consumer_batch_size=32 \
    train_dataset.dataset_kwargs.split_part=null \
    +train_dataset.dataset_kwargs.limit_rows="$TRAIN_LIMIT_ROWS" \
    +train_dataset.dataset_kwargs.limit="$TRAIN_LIMIT" \
    valid_dataset=null \
    gconfig.n_samples=2 gconfig.max_new_tokens="$MAX_NEW_TOKENS" \
    eval_gconfig.max_new_tokens="$MAX_NEW_TOKENS" actor.max_new_tokens="$MAX_NEW_TOKENS" \
    vllm.max_num_seqs=128 rollout.max_concurrent_rollouts=128 rollout.queue_size=2048 \
    saver.freq_steps=2 \
    evaluator.eval_before_train=false evaluator.freq_steps=999 \
    stats_logger.wandb.mode=disabled "$@"
}

checkpoint_path() {
  local experiment="$1"
  local checkpoint_dir="$2"
  echo "$FILEROOT/checkpoints/$(id -un)/$experiment/trial0/default/$checkpoint_dir"
}

run_eval_one() {
  local name="$1"
  local checkpoint="$2"
  local gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$AREAL_VENV/bin/python" -m rlvr_demo.terminal_offline_eval \
    --checkpoint "$checkpoint" \
    --limit-rows "$EVAL_LIMIT_ROWS" \
    --skip-turns "$EVAL_SKIP_TURNS" \
    --limit "$EVAL_LIMIT" \
    --greedy \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output "$EVAL_DIR/$name.json" \
    --predictions-output "$EVAL_DIR/$name.jsonl"
}

run_eval() {
  mkdir -p "$EVAL_DIR"
  run_eval_one sft_final_subset512 \
    "$(checkpoint_path "$SFT_EXPERIMENT" epoch0epochstep7globalstep7)" 0
  run_eval_one teacher_closest_wallclock_subset512 \
    "$(checkpoint_path "$TEACHER_EXPERIMENT" epoch0epochstep1globalstep1)" 1
  run_eval_one teacher_final_subset512 \
    "$(checkpoint_path "$TEACHER_EXPERIMENT" epoch0epochstep15globalstep15)" 2
}

compile_results() {
  "$AREAL_VENV/bin/python" -m rlvr_demo.terminal_compile_results \
    --experiment "$SFT_EXPERIMENT" \
    --experiment "$TEACHER_EXPERIMENT" \
    --eval-dir "$EVAL_DIR" \
    --output-dir "$RESULT_DIR"
}

case "$ACTION" in
  chat-check)
    chat_check "$@"
    ;;
  sft)
    run_sft "$@"
    ;;
  teacher-rl)
    run_teacher_rl "$@"
    ;;
  eval)
    run_eval "$@"
    ;;
  compile)
    compile_results "$@"
    ;;
  all)
    chat_check
    run_sft
    run_teacher_rl
    run_eval
    compile_results
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
