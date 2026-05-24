from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/wbl-fast/usrs/ee/teacher-answer-rl")
OUT_DIR = ROOT / "figures"
RUN_NAME = "grpo-openthoughts-easy-from-sft-b8-s8-o1024-t8-trajectory-valid-nofilter-nokl-s45"
METRICS_PATH = (
    ROOT
    / "areal_runs/terminal-agent-demo/logs/ewer"
    / RUN_NAME
    / "trial0/metrics.jsonl"
)
CSV_PATH = OUT_DIR / "default_grpo_train_eval_vs_time.csv"


def strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def moving_average(values: list[float], window: int = 5) -> list[float]:
    averaged: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        averaged.append(sum(values[start : idx + 1]) / (idx - start + 1))
    return averaged


def read_metrics_jsonl() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for line in METRICS_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        metrics = record["metrics"]
        train_reward = metrics.get("ppo_actor/task_reward/avg")
        if train_reward is None:
            continue

        step = int(record.get("optimizer_step", record["global_step"] + 1))
        hours = float(record["elapsed_wall_clock_sec"]) / 3600.0
        eval_reward = metrics.get("eval-rollout/reward")
        rows.append(
            {
                "step": step,
                "elapsed_hours": hours,
                "train_reward_avg": float(train_reward),
                "eval_subset_reward": "" if eval_reward is None else float(eval_reward),
                "eval_subset_score": "" if eval_reward is None else 100.0 * float(eval_reward),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | int | str]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "elapsed_hours",
                "train_reward_avg",
                "eval_subset_reward",
                "eval_subset_score",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv() -> list[dict[str, float | int | str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "step": int(row["step"]),
                    "elapsed_hours": float(row["elapsed_hours"]),
                    "train_reward_avg": float(row["train_reward_avg"]),
                    "eval_subset_reward": row["eval_subset_reward"],
                    "eval_subset_score": row["eval_subset_score"],
                }
            )
        return rows


def main() -> None:
    if METRICS_PATH.exists():
        rows = read_metrics_jsonl()
        write_csv(rows)
    else:
        rows = read_csv()

    xs = [float(row["elapsed_hours"]) for row in rows]
    train_rewards = [float(row["train_reward_avg"]) for row in rows]
    train_ma = moving_average(train_rewards)
    eval_rows = [row for row in rows if row["eval_subset_score"] != ""]
    eval_xs = [float(row["elapsed_hours"]) for row in eval_rows]
    eval_scores = [float(row["eval_subset_score"]) for row in eval_rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#111827",
            "axes.labelcolor": "#111827",
            "xtick.color": "#111827",
            "ytick.color": "#111827",
            "text.color": "#111827",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, (ax_train, ax_eval) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.2),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )

    ax_train.plot(
        xs,
        train_rewards,
        color="#10b981",
        marker="o",
        markersize=3.6,
        linewidth=1.15,
        alpha=0.55,
        label="Per-step train reward",
    )
    ax_train.plot(
        xs,
        train_ma,
        color="#065f46",
        linewidth=2.4,
        label="5-step moving average",
    )
    ax_train.set_title("Default GRPO Reward and TB Subset Eval vs Time", fontsize=16, pad=14)
    ax_train.set_ylabel("Train reward", fontsize=12)
    ax_train.set_ylim(0, max(train_rewards) * 1.18)
    ax_train.grid(True, which="major", color="#d1d5db", linewidth=0.8, alpha=0.75)
    ax_train.spines["top"].set_visible(False)
    ax_train.spines["right"].set_visible(False)
    ax_train.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#d1d5db")

    ax_eval.plot(
        eval_xs,
        eval_scores,
        color="#2563eb",
        marker="s",
        markersize=6.0,
        linewidth=2.2,
        label="TB subset eval score",
    )
    for row, score in zip(eval_rows, eval_scores):
        ax_eval.annotate(
            f"s{row['step']}",
            (float(row["elapsed_hours"]), score),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#2563eb",
        )
    ax_eval.set_xlabel("RL training time (hours)", fontsize=12)
    ax_eval.set_ylabel("Subset eval score", fontsize=12)
    ax_eval.set_ylim(0, max(eval_scores) * 1.22)
    ax_eval.grid(True, which="major", color="#d1d5db", linewidth=0.8, alpha=0.75)
    ax_eval.spines["top"].set_visible(False)
    ax_eval.spines["right"].set_visible(False)
    ax_eval.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#d1d5db")

    ax_eval.text(
        0.01,
        -0.33,
        "Recipe: b8 prompts x 8 rollouts, no KL, trajectory rewards. "
        "Eval score is eval-rollout/reward x 100 on the held-out easy subset.",
        transform=ax_eval.transAxes,
        fontsize=9,
        color="#4b5563",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "svg", "pdf"):
        out_path = OUT_DIR / f"default_grpo_train_eval_vs_time.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        if ext == "svg":
            strip_trailing_whitespace(out_path)


if __name__ == "__main__":
    main()
