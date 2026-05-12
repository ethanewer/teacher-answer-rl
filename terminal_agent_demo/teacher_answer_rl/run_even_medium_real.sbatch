#!/usr/bin/env bash
#SBATCH --job-name=term-ta-even
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --output=/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/slurm/%x-%j.out

set -euo pipefail

REPO_ROOT="/wbl-fast/usrs/ee/teacher-answer-rl/AReaL"
CONFIG="$REPO_ROOT/terminal_agent_demo/teacher_answer_rl/config_even_medium_real.yaml"

cd "$REPO_ROOT"
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
echo "config=$CONFIG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

exec bash terminal_agent_demo/teacher_answer_rl/run.sh "$CONFIG"
