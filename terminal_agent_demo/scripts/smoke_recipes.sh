#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/terminal_agent_demo/scripts/env_h200.sh"

SMOKE_DIR="${SMOKE_DIR:-/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/smoke}"
SMOKE_JSONL="$SMOKE_DIR/skill_based_medium.terminus_tool.smoke8.jsonl"
SMOKE_SUMMARY="$SMOKE_DIR/skill_based_medium.terminus_tool.smoke8.summary.json"
SMOKE_INSPECT="$SMOKE_DIR/skill_based_medium.terminus_tool.smoke8.inspect.md"
QWEN_RENDER="$SMOKE_DIR/qwen_template_smoke.txt"
HARBOR_CONFIG="$SMOKE_DIR/harbor_eval_smoke.yaml"

mkdir -p "$SMOKE_DIR"
cd "$REPO_ROOT"

"$AREAL_VENV/bin/python" -m py_compile \
  terminal_agent_demo/*.py \
  terminal_agent_demo/sft/*.py \
  terminal_agent_demo/teacher_answer_rl/*.py \
  terminal_agent_demo/grpo/*.py \
  terminal_agent_demo/eval/*.py

bash -n \
  terminal_agent_demo/scripts/env_h200.sh \
  terminal_agent_demo/scripts/prepare_converted_data.sh \
  terminal_agent_demo/scripts/check_qwen_template.sh \
  terminal_agent_demo/scripts/smoke_recipes.sh \
  terminal_agent_demo/sft/run.sh \
  terminal_agent_demo/teacher_answer_rl/run.sh \
  terminal_agent_demo/grpo/run.sh \
  terminal_agent_demo/eval/run_terminal_bench_eval_harbor.sh \
  terminal_agent_demo/eval/run_terminal_bench_eval_slurm_cpu.sh \
  terminal_agent_demo/eval/run_terminal_bench_easy10_split_slurm_cpu.sh \
  terminal_agent_demo/eval/serve_terminal_model_vllm.sh

"$AREAL_VENV/bin/python" -m terminal_agent_demo.terminus_tool_calling convert-corpus \
  --config skill_based_medium \
  --limit "${SMOKE_CONVERT_LIMIT:-8}" \
  --output "$SMOKE_JSONL" \
  --summary-output "$SMOKE_SUMMARY"

"$AREAL_VENV/bin/python" -m terminal_agent_demo.terminus_tool_calling inspect-converted \
  --input "$SMOKE_JSONL" \
  --output "$SMOKE_INSPECT" \
  -n "${SMOKE_INSPECT_N:-2}"

"$AREAL_VENV/bin/python" -m terminal_agent_demo.terminus_tool_calling check-qwen-template \
  --model "${QWEN_TEMPLATE_MODEL:-Qwen/Qwen3-4B-Thinking-2507}" \
  --cache-dir "${HF_HOME:-/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache}" \
  --local-files-only \
  --output "$QWEN_RENDER"

"$AREAL_VENV/bin/python" -m terminal_agent_demo.terminal_experiment write-harbor-eval-config \
  --output "$HARBOR_CONFIG" \
  --job-name terminal-agent-demo-smoke \
  --jobs-dir /tmp/terminal-agent-demo-smoke \
  --model-name openai/terminal-local \
  --task fix-git \
  --n-attempts 1 \
  --n-concurrent 1

"$AREAL_VENV/bin/python" - "$SMOKE_JSONL" "$HARBOR_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import yaml
from areal.api.cli_args import GRPOConfig, SFTConfig, load_expr_config
from transformers import AutoTokenizer

from terminal_agent_demo.terminal_agent_data import (
    get_terminal_sft_dataset,
    get_terminal_teacher_answer_rl_dataset,
)
from terminal_agent_demo.terminal_task_grpo import (
    TerminalTaskGRPOConfig,
    get_terminal_synthetic_task_dataset,
)
from terminal_agent_demo.terminus_tool_calling import EXECUTE_COMMANDS_TOOL

smoke_jsonl = Path(sys.argv[1])
harbor_config = Path(sys.argv[2])

configs = []
for name, path, cls in [
    ("sft", "terminal_agent_demo/sft/config.yaml", SFTConfig),
    ("teacher_answer_rl", "terminal_agent_demo/teacher_answer_rl/config.yaml", GRPOConfig),
    ("grpo", "terminal_agent_demo/grpo/config.yaml", TerminalTaskGRPOConfig),
]:
    cfg, _ = load_expr_config(["--config", path], cls)
    configs.append(
        {
            "name": name,
            "experiment_name": cfg.experiment_name,
            "actor_backend": cfg.actor.backend,
            "train_path": str(cfg.train_dataset.path),
            "batch_size": int(cfg.train_dataset.batch_size),
            "max_length": int(cfg.train_dataset.max_length),
        }
    )

model = (
    Path("/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache")
    / "hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots"
)
snapshots = sorted(path for path in model.glob("*") if path.is_dir())
if not snapshots:
    raise SystemExit(f"No local Qwen3 thinking snapshot under {model}")
tokenizer = AutoTokenizer.from_pretrained(
    snapshots[-1],
    local_files_only=True,
    trust_remote_code=True,
)

rows = [json.loads(line) for line in smoke_jsonl.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit("Converted smoke JSONL is empty")
first = rows[0]
user_messages = [msg for msg in first["messages"] if msg["role"] == "user"]
tool_messages = [msg for msg in first["messages"] if msg["role"] == "tool"]
if len(user_messages) != 1:
    raise SystemExit(f"Expected one logical user message, got {len(user_messages)}")
user_content = str(user_messages[0]["content"])
if "Format your response as JSON" in user_content:
    raise SystemExit("Converted prompt still contains old Terminus-2 JSON instructions")
if any(str(msg.get("content", "")).startswith("New Terminal Output:") for msg in tool_messages):
    raise SystemExit("Converted tool response still contains old New Terminal Output prefix")

rendered = tokenizer.apply_chat_template(
    first["messages"],
    tools=[EXECUTE_COMMANDS_TOOL],
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=True,
)
think_snippets = []
for msg in first["messages"]:
    content = str(msg.get("content") or "")
    if msg.get("role") == "assistant" and "<think>" in content and "</think>" in content:
        think_snippets.append(content.split("<think>", 1)[1].split("</think>", 1)[0].strip()[:120])
if think_snippets and not all(snippet in rendered for snippet in think_snippets):
    raise SystemExit("Qwen chat template stripped a converted assistant thinking block")

sft = get_terminal_sft_dataset(
    path=str(smoke_jsonl),
    tokenizer=tokenizer,
    max_length=32768,
    limit=2,
    lazy_tokenize=True,
)
sft_item = sft[0]
teacher = get_terminal_teacher_answer_rl_dataset(
    path=str(smoke_jsonl),
    tokenizer=tokenizer,
    max_length=32768,
    limit=2,
    lazy_tokenize=True,
)
teacher_item = teacher[0]
if '"commands"' not in teacher_item["teacher_answer"]:
    raise SystemExit("Teacher-answer continuation does not contain commands")

grpo = get_terminal_synthetic_task_dataset(
    path="terminal_agent_demo/manifests/terminus_tool_old_layout_smoke.csv",
    limit=1,
    shuffle_records=False,
)
harbor = yaml.safe_load(harbor_config.read_text())
if harbor["agents"][0]["import_path"] != "terminal_agent_demo.terminus_tool_calling:TerminusToolCallingAgent":
    raise SystemExit("Harbor config does not use the terminus tool-calling agent")

print(
    json.dumps(
        {
            "configs": configs,
            "converted_rows": len(rows),
            "logical_user_messages_first_row": len(user_messages),
            "logical_tool_messages_first_row": len(tool_messages),
            "converted_thinking_preserved": True,
            "sft_records": len(sft),
            "sft_loss_tokens_first": int(sum(sft_item["loss_mask"])),
            "teacher_records": len(teacher),
            "teacher_answer_prefix": teacher_item["teacher_answer"][:40],
            "grpo_records": len(grpo),
            "grpo_task": grpo[0]["task_name"],
            "harbor_agent_import": harbor["agents"][0]["import_path"],
        },
        indent=2,
    )
)
PY

find terminal_agent_demo -type d -name __pycache__ -prune -exec rm -rf {} +
echo "Smoke checks passed."
