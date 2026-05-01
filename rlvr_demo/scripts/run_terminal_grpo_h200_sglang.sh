#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec "$SCRIPT_DIR/run_terminal_grpo_h200.sh" \
  "$REPO_ROOT/rlvr_demo/configs/qwen3_4b_terminal_grpo_h200_1000.yaml" \
  rollout.backend=sglang:d4p1t1 \
  experiment_name=qwen3-4b-terminal-grpo-sglang-megatron-h200-1000 \
  "$@"
