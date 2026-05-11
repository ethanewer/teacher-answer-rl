"""Service-backed Terminal-Bench eval for the default-agent harness.

This is the non-Harbor path used when Docker is only available behind the
remote terminal task service.  It runs the same default-agent tool protocol used
for training and evaluates with the real Terminal-Bench verifier through the
service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from rlvr_demo.default_agent_terminal_grpo import (
    SYSTEM_PROMPT,
    TOOL_SPECS,
    DefaultAgentTerminalTaskRunner,
    _parse_tagged_tool_calls,
    _read_instruction,
    _tool_call_dict,
)
from rlvr_demo.terminal_task_grpo import TerminalTaskTimeouts


EASY10_TASKS = [
    "modernize-scientific-stack",
    "log-summary-date-ranges",
    "multi-source-data-merger",
    "nginx-request-logging",
    "git-leak-recovery",
    "fix-git",
    "constraints-scheduling",
    "vulnerable-secret",
    "regex-log",
    "sqlite-db-truncate",
]


def _task_instruction(task_path: Path) -> str:
    try:
        return _read_instruction(task_path)
    except FileNotFoundError:
        task_yaml = task_path / "task.yaml"
        if not task_yaml.exists():
            raise
        data = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
        instruction = str(data.get("instruction") or "").strip()
        if not instruction:
            raise FileNotFoundError(f"No instruction in {task_yaml}")
        return instruction


def _assistant_message_from_openai(message: Any) -> dict[str, Any]:
    raw = message.model_dump(exclude_none=True)
    reasoning = str(raw.pop("reasoning_content", "") or "")
    content = str(raw.get("content") or "")
    if reasoning and "<think>" not in content:
        content = f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"
    raw["content"] = content
    if raw.get("tool_calls"):
        return raw
    try:
        tool_calls, cleaned_content = _parse_tagged_tool_calls(content)
    except Exception:
        return raw
    if tool_calls:
        raw["content"] = cleaned_content
        raw["tool_calls"] = tool_calls
    return raw


def _assert_single_user_message(messages: list[dict[str, Any]]) -> None:
    count = sum(1 for message in messages if message.get("role") == "user")
    if count != 1:
        raise RuntimeError(f"default-agent eval expected one user message, got {count}")


def _is_context_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "context length" in text
        or "maximum input length" in text
        or "input tokens" in text and "requested" in text
    )


def _pass_at_k(successes: int, trials: int, k: int) -> float:
    if trials <= 0:
        return 0.0
    failures = trials - successes
    if successes <= 0:
        return 0.0
    if k >= trials:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(trials, k)


async def _run_trial(
    *,
    args: argparse.Namespace,
    client: AsyncOpenAI,
    task_name: str,
    attempt: int,
    executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    task_path = (args.dataset_dir / task_name).resolve()
    data = {
        "task_name": task_name,
        "task_path": str(task_path),
        "instruction": _task_instruction(task_path),
    }
    uid = f"{uuid.uuid4().hex[:8]}-a{attempt}"
    runner = DefaultAgentTerminalTaskRunner(
        output_path=str(args.output_dir / "task_service_runs"),
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        temperature=args.temperature,
        top_p=args.top_p,
        observation_max_chars=args.observation_max_chars,
        task_timeouts=TerminalTaskTimeouts(
            reset_env=args.reset_timeout,
            command=args.command_timeout,
            verifier=args.verifier_timeout,
            cleanup=args.cleanup_timeout,
        ),
        encourage_completion_reward=False,
        executor=executor,
        trajectory_timeout=args.trajectory_timeout,
        task_service_url=args.task_service_url,
    )
    system_prompt = SYSTEM_PROMPT.replace("{cwd}", "/app")
    user_prompt = (
        "Complete the terminal task below in the current workspace.\n\n"
        f"Task ID: {task_name}\n\n"
        f"{data['instruction']}\n\n"
        "Files from the task environment are already available in the current "
        "directory. Treat the current directory as /app. Use the available "
        "tools to inspect files, execute commands, edit code, and write files. "
        "Start by calling a tool unless the task is already complete. "
        "When the task is complete, respond without tool calls."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    started = time.monotonic()
    turn_summaries: list[dict[str, Any]] = []
    error: str | None = None
    reward: float | None = None
    stopped_reason = "max_turns"

    try:
        await runner._remote_reset_env(data, uid)
        for turn in range(args.max_turns):
            _assert_single_user_message(messages)
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    tools=[spec.api_definition() for spec in TOOL_SPECS],
                    max_tokens=args.max_tokens_per_turn,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    extra_body={"top_k": args.top_k},
                    timeout=args.model_timeout,
                )
            except Exception as exc:
                if not _is_context_limit_error(exc):
                    raise
                stopped_reason = "context_limit"
                break
            assistant_message = _assistant_message_from_openai(response.choices[0].message)
            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []
            names = [
                str((_tool_call_dict(tool_call).get("function") or {}).get("name") or "")
                for tool_call in tool_calls
            ]
            turn_summaries.append(
                {
                    "turn": turn + 1,
                    "tool_calls": names,
                    "finish_reason": response.choices[0].finish_reason,
                }
            )
            if not tool_calls:
                stopped_reason = "no_tool_calls"
                break
            tool_results = await asyncio.gather(
                *[runner._execute_tool_call(tool_call) for tool_call in tool_calls]
            )
            messages.extend(tool_results)
        _assert_single_user_message(messages)
        reward = await runner._remote_evaluate_completion()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            await runner._remote_close_env()
        except Exception as exc:
            if error is None:
                error = f"cleanup {type(exc).__name__}: {exc}"

    row: dict[str, Any] = {
        "model": args.model,
        "api_base": args.api_base,
        "task": task_name,
        "attempt": attempt,
        "reward": reward,
        "success": bool(reward is not None and reward >= 1.0),
        "turns": len(turn_summaries),
        "stopped_reason": stopped_reason,
        "error": error,
        "elapsed_sec": time.monotonic() - started,
        "turn_summaries": turn_summaries,
    }
    if args.store_messages:
        row["messages"] = messages
    return row


def _summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    per_task: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)

    for task, task_rows in sorted(grouped.items()):
        rewards = [float(row["reward"]) for row in task_rows if row["reward"] is not None]
        rewards_all = [
            float(row["reward"]) if row["reward"] is not None else 0.0
            for row in task_rows
        ]
        successes = sum(1 for row in task_rows if row.get("success"))
        trials = len(task_rows)
        per_task[task] = {
            "trials": trials,
            "completed": len(rewards),
            "errors": sum(1 for row in task_rows if row.get("error")),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "mean_reward_all_trials": sum(rewards_all) / trials if trials else 0.0,
            "successes": successes,
            "pass_at_1": successes / trials if trials else 0.0,
            "pass_at_2": _pass_at_k(successes, trials, 2),
            "pass_at_4": _pass_at_k(successes, trials, 4),
            "pass_at_5": _pass_at_k(successes, trials, 5),
        }

    rewards = [float(row["reward"]) for row in rows if row["reward"] is not None]
    rewards_all = [
        float(row["reward"]) if row["reward"] is not None else 0.0
        for row in rows
    ]
    task_count = max(len(per_task), 1)
    return {
        "model": args.model,
        "api_base": args.api_base,
        "task_service_url": args.task_service_url,
        "dataset_dir": str(args.dataset_dir),
        "tasks": args.task,
        "n_attempts": args.n_attempts,
        "n_concurrent": args.n_concurrent,
        "max_turns": args.max_turns,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "num_trials": len(rows),
        "completed_trials": len(rewards),
        "errors": sum(1 for row in rows if row.get("error")),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "mean_reward_all_trials": sum(rewards_all) / len(rows) if rows else 0.0,
        "pass_at_1": sum(task["pass_at_1"] for task in per_task.values()) / task_count,
        "pass_at_2": sum(task["pass_at_2"] for task in per_task.values()) / task_count,
        "pass_at_4": sum(task["pass_at_4"] for task in per_task.values()) / task_count,
        "pass_at_5": sum(task["pass_at_5"] for task in per_task.values()) / task_count,
        "per_task": per_task,
    }


async def _amain(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "trials.jsonl"
    summary_path = args.output_dir / "summary.json"
    client = AsyncOpenAI(
        base_url=args.api_base.rstrip("/") + "/v1",
        api_key=args.api_key,
        timeout=args.model_timeout,
        max_retries=args.model_max_retries,
    )
    executor = ThreadPoolExecutor(max_workers=max(args.n_concurrent * 4, 8))
    semaphore = asyncio.Semaphore(args.n_concurrent)
    rows: list[dict[str, Any]] = []
    rows_lock = asyncio.Lock()

    async def guarded(task_name: str, attempt: int) -> None:
        async with semaphore:
            row = await _run_trial(
                args=args,
                client=client,
                task_name=task_name,
                attempt=attempt,
                executor=executor,
            )
            async with rows_lock:
                rows.append(row)
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "task": task_name,
                        "attempt": attempt,
                        "reward": row["reward"],
                        "success": row["success"],
                        "turns": row["turns"],
                        "error": row["error"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if rows_path.exists() and not args.resume:
        rows_path.unlink()
    tasks = [
        guarded(task_name, attempt)
        for task_name in args.task
        for attempt in range(args.n_attempts)
    ]
    await asyncio.gather(*tasks)
    executor.shutdown(wait=False, cancel_futures=True)
    summary = _summarize(rows, args)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-service-url", required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/harbor_datasets/terminal-bench"),
    )
    parser.add_argument("--task", action="append", choices=EASY10_TASKS, default=[])
    parser.add_argument("--n-attempts", type=int, default=5)
    parser.add_argument("--n-concurrent", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tokens-per-turn", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--observation-max-chars", type=int, default=8000)
    parser.add_argument("--trajectory-timeout", type=float, default=3600.0)
    parser.add_argument("--model-timeout", type=float, default=600.0)
    parser.add_argument("--model-max-retries", type=int, default=1)
    parser.add_argument("--reset-timeout", type=float, default=1800.0)
    parser.add_argument("--command-timeout", type=float, default=180.0)
    parser.add_argument("--verifier-timeout", type=float, default=1200.0)
    parser.add_argument("--cleanup-timeout", type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--store-messages", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.task:
        args.task = EASY10_TASKS
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
