"""Prompt-only fallback evaluation on selected Terminal-Bench tasks.

This does not execute tasks or run verifiers. It uses the real Terminal-Bench
task instructions and Terminus-2 JSON prompt, then scores the model's first
response for parse/schema/task_complete/command-shape metrics. Use this when
Docker-backed Terminal-Bench execution is unavailable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from openai import OpenAI

from rlvr_demo.terminal_offline_eval import (
    _command_sequence,
    _commands_valid,
    _extract_json_object,
    _task_complete_valid,
    _task_complete_value,
)


TERMINAL_BENCH_TASKS = [
    "fix-git",
    "git-leak-recovery",
    "log-summary-date-ranges",
    "multi-source-data-merger",
    "nginx-request-logging",
    "vulnerable-secret",
    "constraints-scheduling",
    "regex-log",
    "sqlite-db-truncate",
    "modernize-scientific-stack",
]


def _default_prompt_template() -> str:
    return """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes.
- Most bash commands should end with a newline (\\n) to cause them to execute
- For special key sequences, use tmux-style escape sequences such as C-c and C-d
- Do not include extra whitespace before or after the keystrokes unless intended
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid

Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""


def _read_prompt_template(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    candidate = Path(
        ".venv/lib/python3.12/site-packages/terminal_bench/agents/"
        "prompt-templates/terminus-json-plain.txt"
    )
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return _default_prompt_template()


def _task_instruction(dataset_dir: Path, task: str) -> str:
    path = dataset_dir / task / "instruction.md"
    if not path.exists():
        raise FileNotFoundError(f"Missing task instruction: {path}")
    return path.read_text(encoding="utf-8")


def _score_response(text: str) -> dict[str, Any]:
    obj = _extract_json_object(text)
    commands = _command_sequence(obj)
    task_complete = _task_complete_value(obj)
    return {
        "json_parse_valid": obj is not None,
        "commands_schema_valid": _commands_valid(obj),
        "task_complete_valid": _task_complete_valid(obj),
        "task_complete_present": isinstance(obj, dict) and "task_complete" in obj,
        "task_complete_true": task_complete is True,
        "nonempty_commands": bool(commands),
        "command_count": len(commands),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(rows), 1)

    def rate(key: str) -> float:
        return sum(bool(row["scores"][key]) for row in rows) / n

    command_counts = [int(row["scores"]["command_count"]) for row in rows]
    return {
        "num_examples": len(rows),
        "json_parse_valid_rate": rate("json_parse_valid"),
        "commands_schema_valid_rate": rate("commands_schema_valid"),
        "task_complete_valid_rate": rate("task_complete_valid"),
        "task_complete_present_rate": rate("task_complete_present"),
        "task_complete_true_rate": rate("task_complete_true"),
        "nonempty_commands_rate": rate("nonempty_commands"),
        "avg_command_count": fmean(command_counts) if command_counts else 0.0,
        "max_command_count": max(command_counts) if command_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/harbor_datasets/terminal-bench"),
    )
    parser.add_argument("--task", action="append", choices=TERMINAL_BENCH_TASKS)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--terminal-state", default="We need modify the files in /app. No commands have been run yet.")
    parser.add_argument("--prompt-template", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions-output", type=Path)
    args = parser.parse_args()

    tasks = args.task or TERMINAL_BENCH_TASKS
    prompt_template = _read_prompt_template(args.prompt_template)
    client = OpenAI(base_url=args.api_base.rstrip("/") + "/v1", api_key="EMPTY")
    started = time.time()

    rows: list[dict[str, Any]] = []
    for task in tasks:
        instruction = _task_instruction(args.dataset_dir, task)
        prompt = prompt_template.format(
            instruction=instruction,
            terminal_state=args.terminal_state,
        )
        for sample_idx in range(args.n_samples):
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                extra_body={"top_k": args.top_k},
            )
            text = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage is not None else {}
            row = {
                "model": args.model,
                "task": task,
                "sample_idx": sample_idx,
                "prediction": text,
                "scores": _score_response(text),
                "usage": usage,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "task": task,
                        "sample_idx": sample_idx,
                        "scores": row["scores"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    aggregate = {
        "api_base": args.api_base,
        "model": args.model,
        "dataset_dir": str(args.dataset_dir),
        "tasks": tasks,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "elapsed_sec": time.time() - started,
        **_aggregate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
