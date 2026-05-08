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
TERMINAL_BENCH_EASY10_TASKS = [
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
TERMINAL_BENCH_TASK_CHOICES = sorted(
    {
        *TERMINAL_BENCH_TASKS,
        *TERMINAL_BENCH_EASY10_TASKS,
        "break-filter-js-from-html",
        "count-dataset-tokens",
        "sanitize-git-repo",
    }
)
TERMINAL_BENCH_FULL_SUITE_TASK_COUNT = 89

SYNTHETIC_TASK_REPO = "nvidia/Nemotron-Terminal-Synthetic-Tasks"
SYNTHETIC_TASK_FILES_BY_SUBSET = {
    "easy": ["skill_based/easy.tar.gz"],
    "medium": [
        "skill_based/medium_shard1.tar.gz",
        "skill_based/medium_shard2.tar.gz",
    ],
    "mixed": [
        "skill_based/mixed/data_processing.tar.gz",
        "skill_based/mixed/data_science.tar.gz",
        "skill_based/mixed/debugging.tar.gz",
        "skill_based/mixed/file_operations.tar.gz",
        "skill_based/mixed/scientific_computing.tar.gz",
        "skill_based/mixed/security.tar.gz",
    ],
}
SYNTHETIC_TASK_FILES_BY_SUBSET["all"] = (
    SYNTHETIC_TASK_FILES_BY_SUBSET["easy"]
    + SYNTHETIC_TASK_FILES_BY_SUBSET["medium"]
    + SYNTHETIC_TASK_FILES_BY_SUBSET["mixed"]
)


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


def _is_terminal_task_dir(path: Path) -> bool:
    return (
        (path / "instruction.md").is_file()
        and (path / "environment").is_dir()
        and (path / "tests").is_dir()
    )


def _discover_task_dirs(root: Path, max_depth: int = 4) -> list[Path]:
    """Find Nemotron task dirs without walking every file in each task.

    The HF release is a directory archive, not a `datasets` table. Its task dirs
    live a few levels below the extraction root, and each task can contain many
    files under `environment/`. A broad `rglob("instruction.md")` is needlessly
    expensive on shared filesystems, so this prunes once a task directory is
    found and never descends into task payload directories.
    """
    root = root.resolve()
    tasks: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    skip_names = {".git", "__pycache__", "environment", "solution", "tests"}

    while stack:
        path, depth = stack.pop()
        if _is_terminal_task_dir(path):
            tasks.append(path)
            continue
        if depth >= max_depth or not path.is_dir():
            continue
        try:
            children = sorted(
                child
                for child in path.iterdir()
                if child.is_dir() and child.name not in skip_names
            )
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in reversed(children))

    return sorted(set(tasks))


def _synthetic_task_files(args: argparse.Namespace) -> list[str]:
    if args.file:
        return args.file
    filenames: list[str] = []
    for subset in args.subset:
        filenames.extend(SYNTHETIC_TASK_FILES_BY_SUBSET[subset])
    return list(dict.fromkeys(filenames))


def _cmd_prepare_synthetic_tasks(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    args.subset = args.subset or ["medium"]
    output_dir = args.output_dir.resolve()
    download_dir = args.download_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    filenames = _synthetic_task_files(args)
    extracted_files: list[str] = []
    if not args.manifest_only:
        for filename in filenames:
            if args.skip_download:
                tar_path = download_dir / filename
                if not tar_path.exists():
                    raise FileNotFoundError(
                        f"{tar_path} does not exist; rerun without --skip-download"
                    )
            else:
                tar_path = Path(
                    hf_hub_download(
                        repo_id=SYNTHETIC_TASK_REPO,
                        repo_type="dataset",
                        filename=filename,
                        local_dir=download_dir,
                    )
                )
            if not args.skip_extract:
                print(f"Extracting {tar_path} -> {output_dir}")
                with tarfile.open(tar_path) as archive:
                    try:
                        archive.extractall(output_dir, filter="data")
                    except TypeError:
                        archive.extractall(output_dir)
            extracted_files.append(filename)

    task_dirs = _discover_task_dirs(output_dir)
    if args.limit is not None:
        task_dirs = task_dirs[: args.limit]
    created = sum(_write_default_task_toml(path) for path in task_dirs)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_name", "path"])
        writer.writeheader()
        for path in task_dirs:
            task_name = path.relative_to(output_dir).as_posix().replace("/", "__")
            writer.writerow({"task_name": task_name, "path": str(path)})

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "repo": SYNTHETIC_TASK_REPO,
                "subsets": args.subset,
                "files": filenames,
                "files_processed": extracted_files,
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
    task_names = args.task or TERMINAL_BENCH_EASY10_TASKS
    agent_kwargs: dict[str, Any] = {
        "parser_name": "json",
        "api_base": args.api_base,
        "temperature": args.temperature,
        "max_turns": args.max_turns,
        "enable_summarize": args.enable_summarize,
        "proactive_summarization_threshold": args.proactive_summarization_threshold,
        "collect_rollout_details": args.collect_rollout_details,
        "model_info": _model_info(args.max_input_tokens, args.max_output_tokens),
        "interleaved_thinking": args.interleaved_thinking,
        "record_terminal_session": args.record_terminal_session,
        "store_all_messages": args.store_all_messages,
        "llm_kwargs": {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_output_tokens,
        },
    }
    if args.reasoning_effort is not None:
        agent_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.max_thinking_tokens is not None:
        agent_kwargs["max_thinking_tokens"] = args.max_thinking_tokens

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
                "kwargs": agent_kwargs,
            }
        ],
        "datasets": [
            {
                "name": "terminal-bench",
                "version": "2.0",
                "task_names": task_names,
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
    trials_by_task: dict[str, int] = {}
    for row in rows:
        trials_by_task[str(row["task"])] = trials_by_task.get(str(row["task"]), 0) + 1
    pass_count = sum(sum(values) for values in by_task.values())
    n_rewarded_trials = sum(len(values) for values in by_task.values())
    inferred_trials_per_task = max(trials_by_task.values(), default=0)
    trials_per_task = args.trials_per_task or inferred_trials_per_task
    full_suite_denominator = args.full_suite_task_count * trials_per_task
    summary = {
        "n_trials": len(rows),
        "n_rewarded_trials": n_rewarded_trials,
        "n_selected_tasks": len(trials_by_task),
        "selected_tasks": sorted(trials_by_task),
        "pass_count": pass_count,
        "overall_pass_rate": (
            pass_count
            / max(n_rewarded_trials, 1)
        ),
        "selected_subset_pass_rate": (
            pass_count / max(len(rows), 1)
        ),
        "selected_subset_pass_rate_including_unrewarded": (
            pass_count / max(len(rows), 1)
        ),
        "full_suite_task_count": args.full_suite_task_count,
        "trials_per_task_for_full_suite_lower_bound": trials_per_task,
        "full_suite_lower_bound_denominator": full_suite_denominator,
        "full_suite_lower_bound_pass_rate": (
            pass_count / max(full_suite_denominator, 1)
        ),
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
    docker_info_code, docker_info = _run_text(["docker", "info"])
    checks["docker_info_exit_code"] = docker_info_code
    checks["docker_info"] = docker_info.splitlines()[:20] if docker_info else []

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

    checks["can_run_terminal_envs"] = (
        checks["docker"] is not None and docker_info_code == 0
    )
    checks["notes"] = []
    if checks["docker"] is None:
        checks["notes"].append(
            "The current Terminus GRPO and Harbor eval paths use Terminal-Bench's DockerComposeManager and require Docker. Singularity/Apptainer is detected for visibility but is not wired into this workflow."
        )
    elif docker_info_code != 0:
        checks["notes"].append(
            "Docker is installed but the daemon/API is not reachable. Run on a node with Docker socket access or set DOCKER_HOST to a reachable Docker daemon."
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
    prepare.add_argument(
        "--subset",
        action="append",
        choices=sorted(SYNTHETIC_TASK_FILES_BY_SUBSET),
        default=None,
        help="Synthetic task subset to download/extract. Repeatable. Defaults to medium.",
    )
    prepare.add_argument(
        "--file",
        action="append",
        help="Explicit dataset archive path inside the HF repo. Overrides --subset.",
    )
    prepare.add_argument(
        "--skip-download",
        action="store_true",
        help="Use archives already present under --download-dir.",
    )
    prepare.add_argument(
        "--skip-extract",
        action="store_true",
        help="Do not extract archives; only rebuild task.toml files and manifest.",
    )
    prepare.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only scan --output-dir and rebuild the manifest.",
    )
    prepare.add_argument(
        "--limit",
        type=int,
        help="Limit manifest rows after discovery; useful for smoke tests.",
    )
    prepare.set_defaults(func=_cmd_prepare_synthetic_tasks)

    eval_cfg = sub.add_parser("write-harbor-eval-config")
    eval_cfg.add_argument("--output", type=Path, required=True)
    eval_cfg.add_argument("--job-name", required=True)
    eval_cfg.add_argument("--jobs-dir", type=Path, required=True)
    eval_cfg.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    eval_cfg.add_argument("--litellm-model", default="openai/Qwen3-4B-Thinking-2507")
    eval_cfg.add_argument("--environment", default="docker")
    eval_cfg.add_argument("--task", action="append", choices=TERMINAL_BENCH_TASK_CHOICES)
    eval_cfg.add_argument("--n-attempts", type=int, default=5)
    eval_cfg.add_argument("--n-concurrent", type=int, default=10)
    eval_cfg.add_argument("--max-turns", type=int, default=100)
    eval_cfg.add_argument("--max-input-tokens", type=int, default=131072)
    eval_cfg.add_argument("--max-output-tokens", type=int, default=4096)
    eval_cfg.add_argument("--temperature", type=float, default=0.7)
    eval_cfg.add_argument("--top-p", type=float, default=0.8)
    eval_cfg.add_argument("--top-k", type=int, default=20)
    eval_cfg.add_argument("--override-cpus", type=int, default=8)
    eval_cfg.add_argument("--override-memory-mb", type=int, default=32768)
    eval_cfg.add_argument("--collect-rollout-details", action="store_true")
    eval_cfg.add_argument("--enable-summarize", action=argparse.BooleanOptionalAction, default=True)
    eval_cfg.add_argument("--proactive-summarization-threshold", type=int, default=8000)
    eval_cfg.add_argument("--interleaved-thinking", action="store_true")
    eval_cfg.add_argument("--record-terminal-session", action=argparse.BooleanOptionalAction, default=True)
    eval_cfg.add_argument("--store-all-messages", action="store_true")
    eval_cfg.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "default"],
    )
    eval_cfg.add_argument("--max-thinking-tokens", type=int)
    eval_cfg.set_defaults(func=_cmd_write_harbor_eval_config)

    summarize = sub.add_parser("summarize-harbor")
    summarize.add_argument("--jobs-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument(
        "--full-suite-task-count",
        type=int,
        default=TERMINAL_BENCH_FULL_SUITE_TASK_COUNT,
        help="Terminal-Bench full-suite task count for subset lower-bound reporting.",
    )
    summarize.add_argument(
        "--trials-per-task",
        type=int,
        help="Trials per full-suite task; inferred from results when omitted.",
    )
    summarize.set_defaults(func=_cmd_summarize_harbor)

    preflight = sub.add_parser("preflight")
    preflight.set_defaults(func=_cmd_preflight)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
