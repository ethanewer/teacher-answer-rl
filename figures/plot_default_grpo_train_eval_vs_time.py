from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/wbl-fast/usrs/ee/teacher-answer-rl")
OUT_DIR = ROOT / "figures"
RUN_NAME = "grpo-easy-from-sft-b12-s4-o1024-t25-individual-interleaved-meanonly-lr7e7-s70"
METRICS_PATH = (
    ROOT
    / "areal_runs/terminal-agent-demo/logs/ewer"
    / RUN_NAME
    / "trial0/metrics.jsonl"
)
CSV_PATH = OUT_DIR / "default_grpo_train_eval_vs_time.csv"
SFT_BASELINE_SCORE = 17.0
EVAL_POINTS = {
    19: (
        ROOT
        / "areal_runs/terminal-agent-demo/terminal_bench_eval"
        / "grpo-meanonly-b12s4-s19-full20-a5-c1-clean-20260527-r2"
    ),
    39: (
        ROOT
        / "areal_runs/terminal-agent-demo/terminal_bench_eval"
        / "grpo-meanonly-b12s4-s39-full20-a5-c1-clean-20260527-r1"
    ),
}
EXPECTED_EVAL_ATTEMPTS = 100
TB20_TASKS = [
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
    "sparql-university",
    "write-compressor",
    "fix-code-vulnerability",
    "git-multibranch",
    "hf-model-inference",
    "large-scale-text-editing",
    "merge-diff-arc-agi-task",
    "openssl-selfsigned-cert",
    "portfolio-optimization",
    "pytorch-model-cli",
]


def strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def moving_average(values: list[float], window: int = 10) -> list[float]:
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
        rows.append(
            {
                "step": step,
                "elapsed_hours": hours,
                "train_reward_avg": float(train_reward),
            }
        )
    return rows


def read_external_tb20_score(eval_root: Path) -> float | None:
    if not eval_root.exists():
        return None

    task_results: dict[str, list[tuple[float, str, float]]] = {task: [] for task in TB20_TASKS}
    for path in eval_root.glob("*/harbor_jobs/*/*/*/result.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        task = (data.get("task_name") or data.get("task_id") or "").split("__")[0]
        if task not in task_results:
            continue

        exception_info = data.get("exception_info")
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        if exception_info or reward is not None:
            task_results[task].append((path.stat().st_mtime, str(path), float(reward or 0.0)))

    capped_rewards: list[float] = []
    for task in TB20_TASKS:
        results = sorted(task_results[task])[:5]
        capped_rewards.extend(reward for _, _, reward in results)

    if len(capped_rewards) != EXPECTED_EVAL_ATTEMPTS:
        return None
    return 100.0 * sum(capped_rewards) / len(capped_rewards)


def write_csv(rows: list[dict[str, float | int | str]]) -> None:
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "elapsed_hours",
                "train_reward_avg",
                "eval_tb20_score",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        eval_scores = {
            step: read_external_tb20_score(eval_root)
            for step, eval_root in EVAL_POINTS.items()
        }
        writer.writerow(
            {
                "step": 0,
                "elapsed_hours": 0.0,
                "train_reward_avg": "",
                "eval_tb20_score": SFT_BASELINE_SCORE,
            }
        )
        for row in rows:
            eval_score = ""
            score = eval_scores.get(int(row["step"]))
            if score is not None:
                eval_score = score
            writer.writerow({**row, "eval_tb20_score": eval_score})


def read_csv() -> list[dict[str, float | int | str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    "step": int(row["step"]),
                    "elapsed_hours": float(row["elapsed_hours"]),
                    "train_reward_avg": (
                        "" if row["train_reward_avg"] == "" else float(row["train_reward_avg"])
                    ),
                    "eval_tb20_score": row["eval_tb20_score"],
                }
            )
        return rows


def main() -> None:
    if METRICS_PATH.exists():
        metric_rows = read_metrics_jsonl()
        write_csv(metric_rows)
        rows = read_csv()
    else:
        rows = read_csv()

    train_rows = [row for row in rows if row["train_reward_avg"] != ""]
    xs = [float(row["elapsed_hours"]) for row in train_rows]
    train_rewards = [float(row["train_reward_avg"]) for row in train_rows]
    train_ma = moving_average(train_rewards)
    eval_rows = [row for row in rows if row["eval_tb20_score"] != ""]
    eval_xs = [float(row["elapsed_hours"]) for row in eval_rows]
    eval_scores = [float(row["eval_tb20_score"]) for row in eval_rows]

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
        label="10-step moving average",
    )
    ax_train.set_title("Default GRPO Reward and TB20 Eval vs Time", fontsize=16, pad=14)
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
        label="External TB20 eval score",
    )
    for row, score in zip(eval_rows, eval_scores):
        ax_eval.annotate(
            "SFT" if int(row["step"]) == 0 else f"s{row['step']}",
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
        "Recipe: b12 prompts x 4 rollouts, individual-turn exports, interleaved grouped rollouts, "
        "group mean-only reward normalization. Eval is 20 Terminal-Bench tasks x 5 attempts.",
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
