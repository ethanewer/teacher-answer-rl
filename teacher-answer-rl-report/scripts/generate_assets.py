"""Generate tables and figures for the teacher-answer RL report.

Run from the repository root after evaluations complete:

    AReaL/.venv/bin/python teacher-answer-rl-report/scripts/generate_assets.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "teacher-answer-rl-report"
FIG = OUT / "figures"
TABLE = OUT / "tables"
AREAL_RUNS = Path("/NHNHOME/areal_runs/qwen3-gsm8k-rlvr")
LOG_ROOT = AREAL_RUNS / "logs/ewer"
RESULTS = ROOT / "AReaL/rlvr_demo/results"

ANSI_RE = re.compile(r"\x1b\[[0-9;:]*m")
STEP_RE = re.compile(r"Train step\s+(\d+)/")
GLOBAL_STEP_RE = re.compile(r"globalstep(\d+)")

TRAIN_LOGS = {
    "answer_rl_reasoning_only": LOG_ROOT
    / "qwen3-06b-multi-math-teacher-answer-rl-b200-250-r1/trial0/main.log",
    "answer_rl_final_prompt": LOG_ROOT
    / "qwen3-06b-multi-math-teacher-answer-final-rl-b200-250-r1/trial0/main.log",
    "answer_rl_format_unverified": LOG_ROOT
    / "qwen3-06b-multi-math-teacher-answer-format-unverified-b200-250-r1/trial0/main.log",
    "grpo_baseline": LOG_ROOT
    / "qwen3-06b-multi-math-grpo-b200-250-reviewed-v2/trial0/main.log",
    "deepseek_sft_baseline": LOG_ROOT
    / "qwen3-06b-multi-math-deepseek-sft-b200-250-reviewed-v2/trial0/merged.log",
}

VALIDATION_DIRS = {
    "Answer-RL reasoning-only": RESULTS / "teacher_answer_rl_r1_validation",
    "Answer-RL final-prompt": RESULTS / "teacher_answer_final_rl_r1_validation",
    "Answer-RL format/unfiltered": RESULTS
    / "teacher_answer_format_unverified_r1_validation",
    "GRPO": RESULTS / "postcommit_rerun1_grpo_validation",
    "DeepSeek SFT": RESULTS / "postcommit_rerun1_sft_validation",
}

FULL_RESULT_DIRS = {
    "Base Qwen3-0.6B": RESULTS / "postcommit_rerun1_base_full",
    "GRPO step 149": RESULTS / "postcommit_rerun1_grpo_step149_full",
    "DeepSeek SFT step 249": RESULTS / "postcommit_rerun1_sft_step249_full",
}

BENCHMARKS = [
    ("GSM8K", "gsm8k_test"),
    ("MATH L1/2", "math_test_l12"),
    ("MATH L3", "math_test_l3"),
    ("MATH L4/5", "math_test_l45"),
    ("Overall", "overall"),
]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_areal_stats_log(path: Path) -> pd.DataFrame:
    rows: dict[int, dict[str, Any]] = {}
    current_step: int | None = None
    if not path.exists():
        return pd.DataFrame()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = strip_ansi(raw_line)
            step_match = STEP_RE.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                rows.setdefault(current_step, {"step": current_step})
                continue
            if current_step is None or "│" not in line:
                continue
            fields = [item.strip() for item in line.split("│")[1:-1]]
            for idx in range(0, len(fields) - 1, 2):
                key = fields[idx]
                val = parse_float(fields[idx + 1])
                if key and val is not None:
                    rows.setdefault(current_step, {"step": current_step})[key] = val
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([rows[key] for key in sorted(rows)]).sort_values("step")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def step_from_path(path: Path) -> int:
    match = GLOBAL_STEP_RE.search(str(path))
    if match is None:
        return -1
    return int(match.group(1))


def validation_table() -> pd.DataFrame:
    rows = []
    for method, root in VALIDATION_DIRS.items():
        if not root.exists():
            continue
        for metrics_path in sorted(root.glob("*/overall_metrics.json"), key=step_from_path):
            metrics = read_json(metrics_path)
            rows.append(
                {
                    "method": method,
                    "step": step_from_path(metrics_path),
                    "correct": int(metrics["correct"]),
                    "n": int(metrics["n"]),
                    "accuracy": float(metrics["accuracy"]),
                    "format_rate": float(metrics.get("format_rate", 0.0)),
                    "mean_reward": float(metrics.get("mean_reward", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def discover_full_results() -> dict[str, Path]:
    dirs = dict(FULL_RESULT_DIRS)
    for path in sorted(RESULTS.glob("teacher_answer*_full")):
        if not (path / "overall_metrics.json").exists():
            continue
        label = path.name.replace("teacher_answer_", "Answer-RL ").replace("_", " ")
        dirs[label] = path
    return dirs


def full_test_table() -> pd.DataFrame:
    rows = []
    for method, root in discover_full_results().items():
        if not root.exists():
            continue
        for label, stem in BENCHMARKS:
            path = root / f"{stem}_metrics.json"
            if not path.exists():
                continue
            metrics = read_json(path)
            rows.append(
                {
                    "method": method,
                    "benchmark": label,
                    "correct": int(metrics["correct"]),
                    "n": int(metrics["n"]),
                    "accuracy": float(metrics["accuracy"]),
                    "format_rate": float(metrics.get("format_rate", 0.0)),
                    "mean_reward": float(metrics.get("mean_reward", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def write_training_tables() -> dict[str, pd.DataFrame]:
    train_dfs = {}
    summary_rows = []
    for name, path in TRAIN_LOGS.items():
        df = parse_areal_stats_log(path)
        if df.empty:
            continue
        train_dfs[name] = df
        df.to_csv(TABLE / f"{name}_training_metrics.csv", index=False)
        summary = {"run": name, "rows": len(df), "max_step": int(df["step"].max())}
        for key in [
            "teacher_answer_logp/avg",
            "teacher_added_prefix_len/avg",
            "teacher_context_len/avg",
            "ppo_actor/no_eos_ratios/avg",
            "rollout/reward",
            "sft/loss/avg",
            "timeperf/train_step",
            "timeperf/rollout_postprocess",
        ]:
            if key in df.columns and df[key].notna().any():
                series = df[key].dropna()
                summary[f"{key}:first"] = float(series.iloc[0])
                summary[f"{key}:last"] = float(series.iloc[-1])
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(TABLE / "training_log_summary.csv", index=False)
    return train_dfs


def plot_training_curves(train_dfs: dict[str, pd.DataFrame]) -> None:
    answer_runs = {
        key: val
        for key, val in train_dfs.items()
        if key.startswith("answer_rl_") and "teacher_answer_logp/avg" in val.columns
    }
    if answer_runs:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for name, df in answer_runs.items():
            label = name.replace("answer_rl_", "").replace("_", " ")
            axes[0, 0].plot(df["step"], df["teacher_answer_logp/avg"], label=label)
            if "teacher_added_prefix_len/avg" in df.columns:
                axes[0, 1].plot(df["step"], df["teacher_added_prefix_len/avg"], label=label)
            if "ppo_actor/no_eos_ratios/avg" in df.columns:
                axes[1, 0].plot(df["step"], df["ppo_actor/no_eos_ratios/avg"], label=label)
            if "timeperf/train_step" in df.columns:
                axes[1, 1].plot(df["step"], df["timeperf/train_step"], label=label)
        axes[0, 0].set_title("Teacher Answer Logprob")
        axes[0, 0].set_ylabel("mean log p(answer)")
        axes[0, 1].set_title("Appended Prefix Length")
        axes[1, 0].set_title("No-EOS Ratio")
        axes[1, 1].set_title("Train Step Seconds")
        for axis in axes.ravel():
            axis.set_xlabel("step")
            axis.grid(alpha=0.25)
            axis.legend()
        fig.savefig(FIG / "answer_rl_training_curves.png", dpi=180)
        plt.close(fig)


def plot_validation(df: pd.DataFrame) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("step")
        ax.plot(sub["step"], sub["accuracy"] * 100.0, marker="o", label=method)
    ax.set_xlabel("checkpoint global step")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Mixed Train-Validation Checkpoint Selection")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(FIG / "validation_accuracy.png", dpi=180)
    plt.close(fig)


def plot_full_results(df: pd.DataFrame) -> None:
    if df.empty:
        return
    overall = df[df["benchmark"] == "Overall"].copy()
    if overall.empty:
        return
    overall = overall.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.barh(overall["method"], overall["accuracy"] * 100.0)
    ax.set_xlabel("official test accuracy (%)")
    ax.set_title("Full Mixed-Math Test Accuracy")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG / "full_test_overall_accuracy.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)

    train_dfs = write_training_tables()
    plot_training_curves(train_dfs)

    val_df = validation_table()
    val_df.to_csv(TABLE / "validation_results.csv", index=False)
    plot_validation(val_df)

    full_df = full_test_table()
    full_df.to_csv(TABLE / "full_test_results.csv", index=False)
    plot_full_results(full_df)

    if not val_df.empty:
        best = val_df.sort_values(["method", "accuracy", "step"]).groupby("method").tail(1)
        best.to_csv(TABLE / "best_validation_checkpoints.csv", index=False)


if __name__ == "__main__":
    main()
