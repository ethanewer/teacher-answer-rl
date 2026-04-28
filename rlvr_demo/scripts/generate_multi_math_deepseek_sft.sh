#!/usr/bin/env bash
set -euo pipefail

cd /NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/AReaL

export HF_HOME=/NHNHOME/areal_cache/huggingface
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0

.venv/bin/python -m rlvr_demo.generate_deepseek_multi_math_sft \
  --env /NHNHOME/PROJECT/wbl-workspace/ewer/rl-test/.env \
  --output rlvr_demo/data/deepseek_v4_pro_multi_math_balanced_sft.jsonl \
  --records-per-bucket 512 \
  --concurrency 128 \
  --max-tokens 4096
