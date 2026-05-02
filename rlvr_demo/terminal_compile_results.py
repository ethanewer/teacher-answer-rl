"""Compile terminal-agent checkpoint logs and offline eval metrics."""

from __future__ import annotations

import argparse
import csv
import getpass
import json
from pathlib import Path
from typing import Any

import yaml


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _algorithm(experiment_name: str) -> str:
    lowered = experiment_name.lower()
    if "teacher-answer" in lowered:
        return "teacher-answer-rl"
    if "sft" in lowered:
        return "sft"
    return "unknown"


def _metric_for_step(metrics: list[dict[str, Any]], global_step: int) -> dict[str, Any]:
    exact = [row for row in metrics if int(row.get("global_step", -1)) == global_step]
    if exact:
        return dict(exact[-1].get("metrics") or {})
    prior = [
        row for row in metrics if int(row.get("global_step", -1)) <= global_step
    ]
    if prior:
        return dict(prior[-1].get("metrics") or {})
    return {}


def _load_eval_results(eval_dir: Path | None) -> dict[str, dict[str, Any]]:
    by_checkpoint: dict[str, dict[str, Any]] = {}
    if eval_dir is None or not eval_dir.exists():
        return by_checkpoint
    for path in sorted(eval_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        checkpoint = row.get("checkpoint")
        if checkpoint:
            by_checkpoint[str(Path(checkpoint).resolve())] = row
            by_checkpoint[str(checkpoint)] = row
    return by_checkpoint


def _checkpoint_rows(run_dir: Path, log_dir: Path, evals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    config_path = log_dir / "config.yaml"
    if not config_path.exists():
        return []
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    events = _read_jsonl(run_dir / "checkpoint_events.jsonl")
    metrics = _read_jsonl(log_dir / "metrics.jsonl")
    rows = []
    for event in events:
        global_step = int(event["global_step"])
        metric = _metric_for_step(metrics, global_step)
        checkpoint_path = str(Path(event["checkpoint_path"]).resolve())
        eval_result = evals.get(checkpoint_path) or evals.get(event["checkpoint_path"]) or {}
        algorithm = _algorithm(str(config["experiment_name"]))
        batch_size = int(config["train_dataset"]["batch_size"])
        examples_seen = (global_step + 1) * batch_size
        elapsed_wallclock_seconds = event["elapsed_wall_clock_sec"]
        loss_reward_metrics = {
            key: value
            for key, value in metric.items()
            if key.startswith("sft/")
            or key.startswith("teacher_")
            or key in {"ppo/loss", "ppo/approx_kl", "actor/loss"}
        }
        rows.append(
            {
                "algorithm": algorithm,
                "base_model": config["actor"]["path"],
                "dataset": config["train_dataset"]["path"],
                "dataset_split": config["train_dataset"]["split"],
                "dataset_config": (config["train_dataset"].get("dataset_kwargs") or {}).get(
                    "name", ""
                ),
                "split_part": (config["train_dataset"].get("dataset_kwargs") or {}).get(
                    "split_part", ""
                ),
                "max_length": config["train_dataset"].get("max_length"),
                "configured_limit": (config["train_dataset"].get("dataset_kwargs") or {}).get(
                    "limit"
                ),
                "examples_seen": examples_seen,
                "tasks_seen": examples_seen,
                "examples_tasks_seen": examples_seen,
                "optimizer_step": event["optimizer_step"],
                "global_step": global_step,
                "epoch": event["epoch"],
                "epoch_step": event["epoch_step"],
                "fractional_epoch": event["fractional_epoch"],
                "elapsed_wallclock_seconds": elapsed_wallclock_seconds,
                "elapsed_wall_clock_sec": elapsed_wallclock_seconds,
                "timestamp_saved": event["timestamp_saved"],
                "checkpoint_path": event["checkpoint_path"],
                "metrics": loss_reward_metrics,
                "loss_reward_metrics": loss_reward_metrics,
                "loss_reward_metrics_json": json.dumps(loss_reward_metrics, sort_keys=True),
                "json_parse_valid_rate": eval_result.get("json_parse_valid_rate"),
                "commands_schema_valid_rate": eval_result.get("commands_schema_valid_rate"),
                "task_complete_valid_rate": eval_result.get("task_complete_valid_rate"),
                "normalized_command_sequence_similarity": eval_result.get(
                    "normalized_command_sequence_similarity"
                ),
                "command_exact_match_rate": eval_result.get("command_exact_match_rate"),
                "task_complete_prediction_accuracy": eval_result.get(
                    "task_complete_prediction_accuracy"
                ),
            }
        )
    return rows


def _comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_alg: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_alg.setdefault(row["algorithm"], []).append(row)
    for alg_rows in by_alg.values():
        alg_rows.sort(key=lambda row: (float(row["elapsed_wall_clock_sec"]), int(row["optimizer_step"])))
    result = []
    sft_rows = by_alg.get("sft", [])
    teacher_rows = by_alg.get("teacher-answer-rl", [])
    if sft_rows:
        final_sft = sft_rows[-1]
        result.append({"comparison_point": "final_sft", **final_sft})
        if teacher_rows:
            target_elapsed = float(final_sft["elapsed_wall_clock_sec"])
            closest = min(
                teacher_rows,
                key=lambda row: abs(float(row["elapsed_wall_clock_sec"]) - target_elapsed),
            )
            result.append({"comparison_point": "teacher_closest_to_sft_wallclock", **closest})
    if teacher_rows:
        result.append({"comparison_point": "final_teacher_answer_rl", **teacher_rows[-1]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fileroot",
        default="/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent",
    )
    parser.add_argument("--trial", default="trial0")
    parser.add_argument("--experiment", action="append", required=True)
    parser.add_argument("--eval-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    user = getpass.getuser()
    fileroot = Path(args.fileroot)
    evals = _load_eval_results(args.eval_dir)
    rows: list[dict[str, Any]] = []
    for experiment in args.experiment:
        run_dir = fileroot / "checkpoints" / user / experiment / args.trial
        log_dir = fileroot / "logs" / user / experiment / args.trial
        rows.extend(_checkpoint_rows(run_dir, log_dir, evals))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "checkpoint_log.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    csv_path = args.output_dir / "checkpoint_log.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    comparison = _comparison_rows(rows)
    comparison_path = args.output_dir / "comparison_table.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "checkpoint_rows": len(rows),
                "checkpoint_log_jsonl": str(jsonl_path),
                "checkpoint_log_csv": str(csv_path),
                "comparison_table": str(comparison_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
