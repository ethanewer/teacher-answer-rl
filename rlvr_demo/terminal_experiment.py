"""Utilities for terminal-agent SFT, teacher-answer-RL, GRPO, and eval runs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import yaml

TERMINAL_BENCH_TASKS = [
    "modernize-scientific-stack",
    "fix-git",
    "git-leak-recovery",
    "log-summary-date-ranges",
    "multi-source-data-merger",
    "nginx-request-logging",
    "vulnerable-secret",
    "prove-plus-comm",
    "constraints-scheduling",
    "pypi-server",
]

SYNTHETIC_TASK_REPO = "nvidia/Nemotron-Terminal-Synthetic-Tasks"
SYNTHETIC_MEDIUM_FILES = [
    "skill_based/medium_shard1.tar.gz",
    "skill_based/medium_shard2.tar.gz",
]


def _cmd_smoke_data(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from rlvr_demo.terminal_agent_data import (
        get_terminal_sft_dataset,
        get_terminal_teacher_answer_rl_dataset,
        split_terminus_teacher_answer,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sft = get_terminal_sft_dataset(
        tokenizer=tokenizer,
        limit_rows=args.limit_rows,
        limit=args.limit,
        split_part=None,
        max_length=args.max_length,
        shuffle_records=False,
    )
    teacher = get_terminal_teacher_answer_rl_dataset(
        tokenizer=tokenizer,
        limit_rows=args.limit_rows,
        limit=args.limit,
        split_part=None,
        max_length=args.max_length,
        shuffle_records=False,
    )

    sample = teacher[0]
    prompt_text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    student_prefix, teacher_answer = split_terminus_teacher_answer(
        sample["student_prefix"] + sample["teacher_answer"]
    )

    duplicate_think = "<think>\n<think>" in prompt_text
    print(
        json.dumps(
            {
                "sft_records": len(sft),
                "teacher_answer_records": len(teacher),
                "prompt_tokens": len(sft[0]["input_ids"]) - sum(sft[0]["loss_mask"]),
                "sft_train_tokens": sum(sft[0]["loss_mask"]),
                "teacher_messages": len(sample["messages"]),
                "student_prefix_chars": len(student_prefix),
                "teacher_answer_chars": len(teacher_answer),
                "teacher_has_commands": '"commands"' in teacher_answer,
                "teacher_has_task_complete": '"task_complete"' in teacher_answer,
                "duplicate_think_in_prompt": duplicate_think,
                "prompt_starts": prompt_text[:120],
            },
            indent=2,
        )
    )
    if duplicate_think:
        raise SystemExit("Qwen chat template rendered duplicate <think> prefix")


def _write_default_task_toml(task_dir: Path) -> bool:
    task_toml = task_dir / "task.toml"
    if task_toml.exists():
        return False
    name = task_dir.name.replace("_", "-")
    task_toml.write_text(
        "\n".join(
            [
                'schema_version = "1.2"',
                "",
                "[task]",
                f'name = "nemotron-terminal/{name}"',
                f'description = "Nemotron terminal synthetic task {name}"',
                "",
                "[environment]",
                "build_timeout_sec = 1200.0",
                "cpus = 2",
                "memory_mb = 8192",
                "storage_mb = 20480",
                "allow_internet = true",
                "",
                "[agent]",
                "timeout_sec = 3600.0",
                "",
                "[verifier]",
                "timeout_sec = 1200.0",
                "",
            ]
        )
    )
    return True


def _discover_task_dirs(root: Path) -> list[Path]:
    tasks: list[Path] = []
    for instruction in root.rglob("instruction.md"):
        task_dir = instruction.parent
        if (task_dir / "environment").exists() and (task_dir / "tests").exists():
            tasks.append(task_dir)
    return sorted(set(tasks))


def _cmd_prepare_synthetic_tasks(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    output_dir = args.output_dir.resolve()
    download_dir = args.download_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    for filename in SYNTHETIC_MEDIUM_FILES:
        tar_path = Path(
            hf_hub_download(
                repo_id=SYNTHETIC_TASK_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=download_dir,
            )
        )
        print(f"Extracting {tar_path} -> {output_dir}")
        with tarfile.open(tar_path) as archive:
            archive.extractall(output_dir)

    task_dirs = _discover_task_dirs(output_dir)
    created = sum(_write_default_task_toml(path) for path in task_dirs)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_name", "path"])
        writer.writeheader()
        for path in task_dirs:
            writer.writerow({"task_name": path.name, "path": str(path)})

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "tasks": len(task_dirs),
                "task_toml_created": created,
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


def _model_info(max_input_tokens: int, max_output_tokens: int) -> dict[str, Any]:
    return {
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }


def _cmd_write_harbor_eval_config(args: argparse.Namespace) -> None:
    config = {
        "job_name": args.job_name,
        "jobs_dir": str(args.jobs_dir),
        "n_attempts": args.n_attempts,
        "n_concurrent_trials": args.n_concurrent,
        "quiet": False,
        "environment": {
            "type": args.environment,
            "delete": True,
            "override_cpus": args.override_cpus,
            "override_memory_mb": args.override_memory_mb,
        },
        "agents": [
            {
                "name": "terminus-2",
                "model_name": args.litellm_model,
                "kwargs": {
                    "parser_name": "json",
                    "api_base": args.api_base,
                    "temperature": 0.6,
                    "max_turns": args.max_turns,
                    "enable_summarize": True,
                    "proactive_summarization_threshold": 8000,
                    "collect_rollout_details": args.collect_rollout_details,
                    "model_info": _model_info(args.max_input_tokens, args.max_output_tokens),
                    "llm_kwargs": {
                        "top_p": 0.95,
                        "top_k": 20,
                        "max_tokens": args.max_output_tokens,
                    },
                },
            }
        ],
        "datasets": [
            {
                "name": "terminal-bench",
                "version": "2.0",
                "task_names": TERMINAL_BENCH_TASKS,
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
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
        if result_path.parent == args.jobs_dir or result_path.parent.parent == args.jobs_dir:
            # Keep trial result files, skip top-level job summaries.
            continue
        try:
            data = json.loads(result_path.read_text())
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
    for row in rows:
        if row["reward"] is None:
            continue
        by_task.setdefault(str(row["task"]), []).append(float(row["reward"]))
    summary = {
        "n_trials": len(rows),
        "n_rewarded_trials": sum(len(values) for values in by_task.values()),
        "overall_pass_rate": (
            sum(sum(values) for values in by_task.values())
            / max(sum(len(values) for values in by_task.values()), 1)
        ),
        "by_task": {
            task: {
                "n": len(values),
                "pass_rate": sum(values) / max(len(values), 1),
            }
            for task, values in sorted(by_task.items())
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
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
    checks: dict[str, Any] = {}
    checks["docker"] = shutil.which("docker")
    checks["singularity"] = shutil.which("singularity") or shutil.which("apptainer")
    checks["enroot"] = shutil.which("enroot")
    checks["uv"] = shutil.which("uv")

    code, nvidia = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ]
    )
    checks["nvidia_smi_exit_code"] = code
    checks["gpus"] = nvidia.splitlines() if code == 0 else []

    code, iface = _run_text(["bash", "-lc", "ip route get 1.1.1.1 | awk '{print $5; exit}'"])
    checks["default_iface_exit_code"] = code
    checks["default_iface"] = iface

    code, glibc = _run_text(["bash", "-lc", "ldd --version | head -1"])
    checks["glibc_exit_code"] = code
    checks["glibc"] = glibc

    checks["can_run_terminal_envs"] = bool(checks["docker"])
    checks["notes"] = []
    if not checks["can_run_terminal_envs"]:
        checks["notes"].append(
            "The current Terminus GRPO and Harbor eval paths use Terminal-Bench's DockerComposeManager and require Docker. Singularity/Apptainer is detected for visibility but is not wired into this workflow."
        )
    if len(checks["gpus"]) != 8:
        checks["notes"].append(
            f"Expected 8 visible GPUs for the H200 configs, found {len(checks['gpus'])}."
        )
    print(json.dumps(checks, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("smoke-data")
    smoke.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    smoke.add_argument("--limit-rows", type=int, default=4)
    smoke.add_argument("--limit", type=int, default=8)
    smoke.add_argument("--max-length", type=int, default=12000)
    smoke.set_defaults(func=_cmd_smoke_data)

    prepare = sub.add_parser("prepare-synthetic-tasks")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--download-dir",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache/nemotron_tasks"),
    )
    prepare.set_defaults(func=_cmd_prepare_synthetic_tasks)

    eval_cfg = sub.add_parser("write-harbor-eval-config")
    eval_cfg.add_argument("--output", type=Path, required=True)
    eval_cfg.add_argument("--job-name", required=True)
    eval_cfg.add_argument("--jobs-dir", type=Path, required=True)
    eval_cfg.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    eval_cfg.add_argument("--litellm-model", default="openai/Qwen3-4B-Thinking-2507")
    eval_cfg.add_argument("--environment", default="docker")
    eval_cfg.add_argument("--n-attempts", type=int, default=5)
    eval_cfg.add_argument("--n-concurrent", type=int, default=10)
    eval_cfg.add_argument("--max-turns", type=int, default=100)
    eval_cfg.add_argument("--max-input-tokens", type=int, default=131072)
    eval_cfg.add_argument("--max-output-tokens", type=int, default=4096)
    eval_cfg.add_argument("--override-cpus", type=int, default=8)
    eval_cfg.add_argument("--override-memory-mb", type=int, default=32768)
    eval_cfg.add_argument("--collect-rollout-details", action="store_true")
    eval_cfg.set_defaults(func=_cmd_write_harbor_eval_config)

    summarize = sub.add_parser("summarize-harbor")
    summarize.add_argument("--jobs-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(func=_cmd_summarize_harbor)

    preflight = sub.add_parser("preflight")
    preflight.set_defaults(func=_cmd_preflight)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
