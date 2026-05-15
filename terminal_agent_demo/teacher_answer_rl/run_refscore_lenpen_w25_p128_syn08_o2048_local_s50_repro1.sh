#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/AReaL"
CONFIG="$REPO_ROOT/terminal_agent_demo/teacher_answer_rl/config_odd_medium_from_sft_refscore_lenpen_w25_p128_syn08_o2048_local_s50_repro1.yaml"

cd "$REPO_ROOT"
export TEACHER_ANSWER_LENGTH_PENALTY="${TEACHER_ANSWER_LENGTH_PENALTY:-0.05}"
export TEACHER_ANSWER_FORMAT_BONUS="${TEACHER_ANSWER_FORMAT_BONUS:-0.0}"
exec bash terminal_agent_demo/teacher_answer_rl/run.sh "$CONFIG" "$@"
