#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_AREAL_VENV="$REPO_ROOT/.venv-megatron"
if [[ ! -x "$DEFAULT_AREAL_VENV/bin/python" && -x "$REPO_ROOT/.venv/bin/python" ]]; then
  DEFAULT_AREAL_VENV="$REPO_ROOT/.venv"
fi
AREAL_VENV="${AREAL_VENV:-$DEFAULT_AREAL_VENV}"
AREAL_VLLM_PYTHON="${AREAL_VLLM_PYTHON:-$REPO_ROOT/.venv-rollout-vllm/bin/python}"
AREAL_SGLANG_PYTHON="${AREAL_SGLANG_PYTHON:-$REPO_ROOT/.venv-rollout-sglang/bin/python}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$AREAL_VENV/bin/python" - <<PY
from areal.api.cli_args import SGLangConfig, vLLMConfig

vllm = vLLMConfig(model="Qwen/Qwen3-4B-Thinking-2507", python_executable="$AREAL_VLLM_PYTHON")
sglang = SGLangConfig(model_path="Qwen/Qwen3-4B-Thinking-2507", python_executable="$AREAL_SGLANG_PYTHON")

print("vLLM command:", " ".join(vLLMConfig.build_cmd(vllm, tp_size=1, pp_size=1, host="127.0.0.1", port=30000)[:4]))
print("SGLang command:", " ".join(SGLangConfig.build_cmd(sglang, tp_size=1, base_gpu_id=0, host="127.0.0.1", port=30001)[:4]))
PY

if [[ -x "$AREAL_VLLM_PYTHON" ]]; then
  "$AREAL_VLLM_PYTHON" - <<'PY'
import torch
import vllm
import areal.engine.vllm_ext.areal_vllm_server
print(f"vLLM import ok: torch={torch.__version__} vllm={vllm.__version__}")
PY
else
  echo "Skipping vLLM import smoke: $AREAL_VLLM_PYTHON does not exist." >&2
fi

if [[ -x "$AREAL_SGLANG_PYTHON" ]]; then
  "$AREAL_SGLANG_PYTHON" - <<'PY'
import importlib.metadata
import torch
import sglang
import areal.experimental.inference_service.sglang.launch_server
print(
    "SGLang import ok: "
    f"torch={torch.__version__} sglang={importlib.metadata.version('sglang')}"
)
PY
else
  echo "Skipping SGLang import smoke: $AREAL_SGLANG_PYTHON does not exist." >&2
fi
