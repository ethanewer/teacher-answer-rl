from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/wbl-fast/usrs/ee/teacher-answer-rl")
LOG_DIR = ROOT / "areal_runs/terminal-agent-demo/long_sweep_logs"
OUT_DIR = ROOT / "figures"

INITIAL_LOG = LOG_DIR / "grpo_easy_s75_3h.log"
CONTINUATION_LOG = LOG_DIR / "grpo_easy_s49_plus25_timeout1200b_3h.log"

ANSI_RE = re.compile(r"\x1b\[[0-9;:]*[A-Za-z]")
STEP_RE = re.compile(r"Step (\d+)/")
METRIC = "ppo_actor/task_reward/avg"


def strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def parse_metric(path: Path) -> list[tuple[int, float]]:
    current_step: int | None = None
    points: list[tuple[int, float]] = []
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = ANSI_RE.sub("", raw_line)
        step_match = STEP_RE.search(line)
        if "Train step" in line and step_match:
            current_step = int(step_match.group(1))

        if METRIC not in line or current_step is None:
            continue

        cells = [cell.strip() for cell in line.split("│") if cell.strip()]
        values = {cells[i]: cells[i + 1] for i in range(0, len(cells) - 1, 2)}
        if METRIC in values:
            points.append((current_step, float(values[METRIC])))
    return points


def read_committed_csv(path: Path) -> list[tuple[int, float, str, int]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            (
                int(row["global_step"]),
                float(row["task_reward_avg"]),
                row["phase"],
                int(row["local_step"]),
            )
            for row in reader
        ]


def moving_average(values: list[float], window: int = 5) -> list[float]:
    averaged = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        averaged.append(sum(values[start : idx + 1]) / (idx - start + 1))
    return averaged


def main() -> None:
    chained_points: list[tuple[int, float, str, int]] = []
    csv_path = OUT_DIR / "grpo_task_reward_vs_training_step.csv"
    if INITIAL_LOG.exists() and CONTINUATION_LOG.exists():
        initial = parse_metric(INITIAL_LOG)
        continuation = parse_metric(CONTINUATION_LOG)

        # The continuation starts from the step-49 checkpoint. Use steps 1-49
        # from the first run, then map continuation local steps 1-25 to global
        # 50-74.
        chained_points.extend((step, reward, "initial", step) for step, reward in initial if step <= 49)
        chained_points.extend((49 + step, reward, "continuation", step) for step, reward in continuation)

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["global_step", "task_reward_avg", "phase", "local_step"])
            writer.writerows(chained_points)
    else:
        chained_points = read_committed_csv(csv_path)

    steps = [p[0] for p in chained_points]
    rewards = [p[1] for p in chained_points]
    avg_rewards = moving_average(rewards, window=5)

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

    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    ax.plot(
        steps,
        rewards,
        color="#10b981",
        marker="o",
        markersize=3.8,
        linewidth=1.2,
        alpha=0.58,
        label="Per-step task reward avg",
    )
    ax.plot(
        steps,
        avg_rewards,
        color="#065f46",
        linewidth=2.4,
        label="5-step moving average",
    )
    ax.axvline(49.5, color="#6b7280", linestyle="--", linewidth=1.1)
    ax.text(
        50.2,
        max(rewards) * 0.96,
        "model-only continuation\nfrom step-49 checkpoint",
        fontsize=8.5,
        color="#4b5563",
        va="top",
    )

    ax.set_title("GRPO Training Reward by Step", fontsize=16, pad=14)
    ax.set_xlabel("GRPO training step", fontsize=12)
    ax.set_ylabel("Average task reward", fontsize=12)
    ax.set_xlim(0, max(steps) + 2)
    ax.set_ylim(0, max(rewards) * 1.12)
    ax.grid(True, which="major", color="#d1d5db", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#d1d5db")
    ax.text(
        0.01,
        -0.16,
        f"Metric: {METRIC}. Steps 1-49 from the initial GRPO run; steps 50-74 from the continuation run.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "svg", "pdf"):
        out_path = OUT_DIR / f"grpo_task_reward_vs_training_step.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        if ext == "svg":
            strip_trailing_whitespace(out_path)


if __name__ == "__main__":
    main()
