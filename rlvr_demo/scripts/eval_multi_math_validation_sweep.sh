#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 EXPERIMENT_NAME OUTPUT_DIR [LIMIT] [BATCH_SIZE]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

EXPERIMENT_NAME="$1"
OUTPUT_DIR="$2"
LIMIT="${3:-0}"
BATCH_SIZE="${4:-128}"
CHECKPOINT_ROOT="/NHNHOME/areal_runs/qwen3-gsm8k-rlvr/checkpoints/${USER}/${EXPERIMENT_NAME}/trial0/default"

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

mapfile -t checkpoints < <(
  find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'epoch*globalstep*' \
    | awk 'match($0, /globalstep([0-9]+)/) { print substr($0, RSTART + 10) "\t" $0 }' \
    | sort -n -k1,1 \
    | cut -f2-
)
if [[ ${#checkpoints[@]} -eq 0 ]]; then
  echo "No scheduled checkpoints found under $CHECKPOINT_ROOT" >&2
  exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
  name="$(basename "$checkpoint")"
  echo "Evaluating $name on mixed_train_validation"
  "$SCRIPT_DIR/eval_multi_math_hf.sh" \
    "$checkpoint" \
    "$OUTPUT_DIR/$name" \
    "$LIMIT" \
    "$BATCH_SIZE" \
    --benchmarks mixed_train_validation
done
