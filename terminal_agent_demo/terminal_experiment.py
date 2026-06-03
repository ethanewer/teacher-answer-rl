"""Small CLI utilities for the standalone terminal-agent demo recipes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


TERMINAL_BENCH_EASY10_TASKS = [
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
TERMINAL_BENCH_FULL_SUITE_TASK_COUNT = 89
COMPARABLE_TERMINUS_EVAL_ENV = {
    "TERMINUS_TOOL_ENABLE_TASK_REMINDERS": "0",
    "TERMINUS_TOOL_TASK_REMINDER_ALLOWLIST": "",
    "TERMINUS_TOOL_ENABLE_NO_TOOL_REPAIR": "0",
}


def _model_info(max_input_tokens: int, max_output_tokens: int) -> dict[str, Any]:
    return {
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }


def _cmd_write_harbor_eval_config(args: argparse.Namespace) -> None:
    task_names = args.task or TERMINAL_BENCH_EASY10_TASKS
    config = {
        "job_name": args.job_name,
        "jobs_dir": str(args.jobs_dir),
        "n_attempts": args.n_attempts,
        "n_concurrent_trials": args.n_concurrent,
        "quiet": False,
        "environment": {
            "type": args.environment,
            "force_build": args.environment_force_build,
            "delete": args.environment_delete,
            "override_cpus": args.override_cpus,
            "override_memory_mb": args.override_memory_mb,
        },
        "agents": [
            {
                "import_path": "terminal_agent_demo.terminus_tool_calling:TerminusToolCallingAgent",
                "model_name": args.model_name,
                "kwargs": {
                    "api_base": args.api_base,
                    "temperature": args.temperature,
                    "max_turns": args.max_turns,
                    "max_tokens": args.max_output_tokens,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "record_terminal_session": args.record_terminal_session,
                    "model_info": _model_info(args.max_input_tokens, args.max_output_tokens),
                    "llm_kwargs": args.llm_kwargs,
                    "terminus_env": dict(COMPARABLE_TERMINUS_EVAL_ENV),
                },
            }
        ],
        "datasets": [
            {
                "path": str(args.dataset_path),
                "task_names": task_names,
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(args.output)


def _extract_reward(result: dict[str, Any]) -> float | None:
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    value = rewards.get("reward")
    if value is None:
        return None
    return float(value)


def _cmd_summarize_harbor(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(args.jobs_dir.rglob("result.json")):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "trial_name" not in data or "task_name" not in data:
            continue
        agent_result = data.get("agent_result") or {}
        exception = data.get("exception_info") or {}
        rows.append(
            {
                "job": result_path.parent.parent.name,
                "trial": data.get("trial_name"),
                "task": data.get("task_name"),
                "reward": _extract_reward(data),
                "exception_type": exception.get("exception_type"),
                "n_input_tokens": agent_result.get("n_input_tokens"),
                "n_output_tokens": agent_result.get("n_output_tokens"),
                "path": str(result_path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job",
                "trial",
                "task",
                "reward",
                "exception_type",
                "n_input_tokens",
                "n_output_tokens",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_task: dict[str, list[float]] = {}
    trials_by_task: dict[str, int] = {}
    for row in rows:
        task = str(row["task"])
        trials_by_task[task] = trials_by_task.get(task, 0) + 1
        if row["reward"] is not None:
            by_task.setdefault(task, []).append(float(row["reward"]))
    pass_count = sum(sum(values) for values in by_task.values())
    n_rewarded_trials = sum(len(values) for values in by_task.values())
    trials_per_task = args.trials_per_task or max(trials_by_task.values(), default=0)
    full_suite_denominator = args.full_suite_task_count * trials_per_task
    summary = {
        "n_trials": len(rows),
        "n_rewarded_trials": n_rewarded_trials,
        "n_selected_tasks": len(trials_by_task),
        "selected_tasks": sorted(trials_by_task),
        "pass_count": pass_count,
        "overall_pass_rate": pass_count / max(n_rewarded_trials, 1),
        "selected_subset_pass_rate_including_unrewarded": pass_count / max(len(rows), 1),
        "full_suite_task_count": args.full_suite_task_count,
        "trials_per_task_for_full_suite_lower_bound": trials_per_task,
        "full_suite_lower_bound_denominator": full_suite_denominator,
        "full_suite_lower_bound_pass_rate": pass_count / max(full_suite_denominator, 1),
        "by_task": {
            task: {
                "n": trials_by_task.get(task, len(values)),
                "n_rewarded": len(values),
                "pass_rate": sum(values) / max(len(values), 1),
            }
            for task in sorted(trials_by_task)
            for values in [by_task.get(task, [])]
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _run_text(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def _cmd_preflight(args: argparse.Namespace) -> None:
    del args
    checks: dict[str, Any] = {
        "docker": shutil.which("docker"),
        "uv": shutil.which("uv"),
    }
    code, nvidia = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ]
    )
    checks["nvidia_smi_exit_code"] = code
    checks["gpus"] = nvidia.splitlines() if code == 0 else []
    checks["can_run_terminal_envs"] = bool(checks["docker"])
    print(json.dumps(checks, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    eval_cfg = sub.add_parser("write-harbor-eval-config")
    eval_cfg.add_argument("--output", type=Path, required=True)
    eval_cfg.add_argument("--job-name", required=True)
    eval_cfg.add_argument("--jobs-dir", type=Path, required=True)
    eval_cfg.add_argument("--api-base", default="http://127.0.0.1:30080/v1")
    eval_cfg.add_argument("--model-name", default="openai/terminal-local")
    eval_cfg.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/harbor_datasets/terminal-bench"),
    )
    eval_cfg.add_argument("--environment", default="docker")
    eval_cfg.add_argument("--environment-force-build", action=argparse.BooleanOptionalAction, default=False)
    eval_cfg.add_argument("--environment-delete", action=argparse.BooleanOptionalAction, default=True)
    eval_cfg.add_argument("--task", action="append")
    eval_cfg.add_argument("--n-attempts", type=int, default=5)
    eval_cfg.add_argument("--n-concurrent", type=int, default=5)
    eval_cfg.add_argument("--max-turns", type=int, default=40)
    eval_cfg.add_argument("--max-input-tokens", type=int, default=32768)
    eval_cfg.add_argument("--max-output-tokens", type=int, default=6144)
    eval_cfg.add_argument("--temperature", type=float, default=0.2)
    eval_cfg.add_argument("--top-p", type=float, default=0.8)
    eval_cfg.add_argument("--top-k", type=int, default=20)
    eval_cfg.add_argument("--override-cpus", type=int, default=3)
    eval_cfg.add_argument("--override-memory-mb", type=int, default=10000)
    eval_cfg.add_argument("--record-terminal-session", action=argparse.BooleanOptionalAction, default=True)
    eval_cfg.add_argument("--llm-kwargs", type=json.loads, default={})
    eval_cfg.set_defaults(func=_cmd_write_harbor_eval_config)

    summarize = sub.add_parser("summarize-harbor")
    summarize.add_argument("--jobs-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--full-suite-task-count", type=int, default=TERMINAL_BENCH_FULL_SUITE_TASK_COUNT)
    summarize.add_argument("--trials-per-task", type=int)
    summarize.set_defaults(func=_cmd_summarize_harbor)

    preflight = sub.add_parser("preflight")
    preflight.set_defaults(func=_cmd_preflight)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
