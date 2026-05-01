#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BACKEND="${1:-both}"
PYTHON_BIN="${PYTHON_BIN:-3.12}"
USE_UV_MANAGED_PYTHON="${USE_UV_MANAGED_PYTHON:-1}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv is required to install the backend environments from pyproject metadata." >&2
  exit 2
fi

VLLM_VENV="${VLLM_VENV:-$REPO_ROOT/.venv-rollout-vllm}"
SGLANG_VENV="${SGLANG_VENV:-$REPO_ROOT/.venv-rollout-sglang}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

create_venv() {
  local venv_dir="$1"
  if [[ -x "$venv_dir/bin/python" ]]; then
    if ! "$venv_dir/bin/python" - <<'PY' >/dev/null 2>&1
import sqlite3
PY
    then
      echo "Recreating $venv_dir because its Python is missing sqlite3." >&2
      rm -rf "$venv_dir"
    fi
  fi
  if [[ ! -x "$venv_dir/bin/python" ]]; then
    if [[ "$USE_UV_MANAGED_PYTHON" == "1" ]]; then
      "$UV_BIN" venv --managed-python --python "$PYTHON_BIN" --seed "$venv_dir"
    else
      "$PYTHON_BIN" -m venv "$venv_dir"
    fi
  fi
  "$venv_dir/bin/python" - <<'PY'
import sqlite3
PY
  "$venv_dir/bin/python" -m pip install --upgrade pip wheel setuptools
}

install_from_pyproject() {
  local venv_dir="$1"
  local pyproject_path="$2"
  local extra="$3"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  cp "$pyproject_path" "$tmp_dir/pyproject.toml"
  "$UV_BIN" pip install --python "$venv_dir/bin/python" \
    -r "$tmp_dir/pyproject.toml" --extra "$extra" --torch-backend "$UV_TORCH_BACKEND"
  rm -rf "$tmp_dir"
}

install_vllm() {
  create_venv "$VLLM_VENV"
  install_from_pyproject "$VLLM_VENV" "$REPO_ROOT/pyproject.vllm.toml" vllm
  "$UV_BIN" pip install --python "$VLLM_VENV/bin/python" --no-deps -e "$REPO_ROOT"
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$VLLM_VENV/bin/python" - <<'PY'
import torch
import vllm
import areal.engine.vllm_ext.areal_vllm_server
print(f"vLLM env ok: torch={torch.__version__} vllm={vllm.__version__}")
PY
}

install_sglang() {
  create_venv "$SGLANG_VENV"
  install_from_pyproject "$SGLANG_VENV" "$REPO_ROOT/pyproject.toml" sglang
  "$UV_BIN" pip install --python "$SGLANG_VENV/bin/python" --no-deps -e "$REPO_ROOT"
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$SGLANG_VENV/bin/python" - <<'PY'
import importlib.metadata
import torch
import sglang
import areal.experimental.inference_service.sglang.launch_server
print(
    "SGLang env ok: "
    f"torch={torch.__version__} sglang={importlib.metadata.version('sglang')}"
)
PY
}

case "$BACKEND" in
  vllm)
    install_vllm
    ;;
  sglang)
    install_sglang
    ;;
  both)
    install_vllm
    install_sglang
    ;;
  *)
    echo "Usage: $0 [vllm|sglang|both]" >&2
    exit 2
    ;;
esac
