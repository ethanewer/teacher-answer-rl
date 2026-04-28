"""Generate figures and tables for the RLVR progress report.

Run from the repository root:

    AReaL/.venv/bin/python progress-report/scripts/generate_assets.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "progress-report"
FIG = OUT / "figures"
TABLE = OUT / "tables"
AREAL_RUNS = Path("/NHNHOME/areal_runs/qwen3-gsm8k-rlvr")
LOG_ROOT = AREAL_RUNS / "logs/ewer"
RESULTS = ROOT / "AReaL/rlvr_demo/results"

ANSI_RE = re.compile(r"\x1b\[[0-9;:]*m")
STEP_RE = re.compile(r"Train step\s+(\d+)/")


AIME_GROUPS = {
    "Base Qwen3-1.7B": [
        "aime2026_qwen3_17b_base_prompt_v2_seed7",
        "aime2026_qwen3_17b_base_prompt_v2_seed13",
        "aime2026_qwen3_17b_base_prompt_v2_seed21",
    ],
    "GRPO original": [
        "aime2026_qwen3_17b_hardmath_correct_grpo_step299_seed7",
        "aime2026_qwen3_17b_hardmath_correct_grpo_step299_seed13",
        "aime2026_qwen3_17b_hardmath_correct_grpo_step299_seed21",
    ],
    "GRPO rerun": [
        "aime2026_qwen3_17b_hardmath_correct_grpo_repro1_step299_seed7",
        "aime2026_qwen3_17b_hardmath_correct_grpo_repro1_step299_seed13",
        "aime2026_qwen3_17b_hardmath_correct_grpo_repro1_step299_seed21",
    ],
    "Rollout-SFT original": [
        "aime2026_qwen3_17b_rollout_rft_sft_step199_seed7",
        "aime2026_qwen3_17b_rollout_rft_sft_step199_seed13",
        "aime2026_qwen3_17b_rollout_rft_sft_step199_seed21",
    ],
    "Rollout-SFT rerun": [
        "aime2026_qwen3_17b_rollout_rft_sft_repro1_step199_seed7",
        "aime2026_qwen3_17b_rollout_rft_sft_repro1_step199_seed13",
        "aime2026_qwen3_17b_rollout_rft_sft_repro1_step199_seed21",
    ],
}

MIXED_GROUPS = {
    "Base Qwen3-0.6B": "reviewed_v2_base_full",
    "GRPO step 99": "reviewed_v2_grpo_step100_full",
    "DeepSeek SFT step 49": "reviewed_v2_deepseek_sft_step50_full",
}

TRAIN_LOGS = {
    "aime_grpo_original": LOG_ROOT / "qwen3-17b-aime-hardmath-correct-grpo-b200-dev-300-r1/trial0/main.log",
    "aime_grpo_rerun": LOG_ROOT / "qwen3-17b-aime-hardmath-correct-grpo-b200-dev-300-repro1/trial0/main.log",
    "aime_sft_original": LOG_ROOT / "qwen3-17b-aime-rollout-rft-sft-b200-300-r1/trial0/merged.log",
    "aime_sft_rerun": LOG_ROOT / "qwen3-17b-aime-rollout-rft-sft-b200-300-repro1/trial0/merged.log",
    "mixed_grpo": LOG_ROOT / "qwen3-06b-multi-math-grpo-b200-250-reviewed-v2/trial0/main.log",
    "mixed_sft": LOG_ROOT / "qwen3-06b-multi-math-deepseek-sft-b200-250-reviewed-v2/trial0/merged.log",
}


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_areal_stats_log(path: Path) -> pd.DataFrame:
    """Parse AReaL pretty-table StatsLogger blocks into one row per train step."""
    rows: dict[int, dict[str, Any]] = {}
    current_step: int | None = None
    if not path.exists():
        raise FileNotFoundError(path)
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
        raise ValueError(f"No StatsLogger rows parsed from {path}")
    return pd.DataFrame([rows[key] for key in sorted(rows)]).sort_values("step")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def aime_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    agg_rows: list[dict[str, Any]] = []
    per_question: dict[str, dict[tuple[int, str], bool]] = {}
    for group, dirs in AIME_GROUPS.items():
        group_correct = 0
        group_n = 0
        keyed: dict[tuple[int, str], bool] = {}
        for dirname in dirs:
            metrics = read_json(RESULTS / dirname / "aime_2026_metrics.json")
            seed = int(metrics["seed"])
            correct = int(metrics["correct"])
            n = int(metrics["n"])
            group_correct += correct
            group_n += n
            seed_rows.append(
                {
                    "group": group,
                    "seed": seed,
                    "correct": correct,
                    "n": n,
                    "accuracy": correct / n,
                }
            )
            with (RESULTS / dirname / "aime_2026_predictions.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    keyed[(seed, row["question"])] = bool(row["correct"])
        lo, hi = wilson(group_correct, group_n)
        agg_rows.append(
            {
                "group": group,
                "correct": group_correct,
                "n": group_n,
                "accuracy": group_correct / group_n,
                "wilson95_low": lo,
                "wilson95_high": hi,
            }
        )
        per_question[group] = keyed

    base = per_question["Base Qwen3-1.7B"]
    paired_rows = []
    for group, keyed in per_question.items():
        if group == "Base Qwen3-1.7B":
            continue
        candidate_only = 0
        base_only = 0
        both_correct = 0
        both_wrong = 0
        for key in sorted(base):
            b = base[key]
            c = keyed[key]
            if b and c:
                both_correct += 1
            elif b and not c:
                base_only += 1
            elif c:
                candidate_only += 1
            else:
                both_wrong += 1
        discordant = candidate_only + base_only
        paired_rows.append(
            {
                "candidate": group,
                "candidate_only_correct": candidate_only,
                "base_only_correct": base_only,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "discordant": discordant,
                "two_sided_sign_test_p": binom_two_sided(candidate_only, discordant),
            }
        )
    return pd.DataFrame(seed_rows), pd.DataFrame(agg_rows), pd.DataFrame(paired_rows)


def mixed_math_table() -> pd.DataFrame:
    benchmarks = [
        ("GSM8K", "gsm8k_test"),
        ("MATH L1/2", "math_test_l12"),
        ("MATH L3", "math_test_l3"),
        ("MATH L4/5", "math_test_l45"),
        ("Overall", "overall"),
    ]
    rows = []
    for group, dirname in MIXED_GROUPS.items():
        for label, stem in benchmarks:
            metrics = read_json(RESULTS / dirname / f"{stem}_metrics.json")
            rows.append(
                {
                    "group": group,
                    "benchmark": label,
                    "correct": int(metrics["correct"]),
                    "n": int(metrics["n"]),
                    "accuracy": float(metrics["accuracy"]),
                    "format_rate": float(metrics.get("format_rate", 0.0)),
                    "mean_reward": float(metrics.get("mean_reward", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def write_tables() -> dict[str, pd.DataFrame]:
    seed_df, agg_df, paired_df = aime_tables()
    mixed_df = mixed_math_table()
    train_dfs = {name: parse_areal_stats_log(path) for name, path in TRAIN_LOGS.items()}
    summary_rows = []
    for name, df in train_dfs.items():
        summary = {"run": name, "steps_parsed": int(df["step"].max()), "rows": len(df)}
        for key in [
            "ppo_actor/update/actor_loss/avg",
            "rollout/reward",
            "eval-rollout/reward",
            "sft/loss/avg",
            "sft/ppl/avg",
            "timeperf/train_step",
        ]:
            if key in df.columns:
                summary[f"{key}:first"] = float(df[key].dropna().iloc[0])
                summary[f"{key}:last"] = float(df[key].dropna().iloc[-1])
        summary_rows.append(summary)

    TABLE.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(TABLE / "aime2026_seed_results.csv", index=False)
    agg_df.to_csv(TABLE / "aime2026_aggregate_results.csv", index=False)
    paired_df.to_csv(TABLE / "aime2026_paired_tests.csv", index=False)
    mixed_df.to_csv(TABLE / "mixed_math_test_results.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(TABLE / "training_log_summary.csv", index=False)
    for name, df in train_dfs.items():
        keep = [col for col in df.columns if col == "step" or col.endswith("/avg") or col in {"rollout/reward", "eval-rollout/reward", "timeperf/train_step"}]
        df[keep].to_csv(TABLE / f"{name}_training_metrics.csv", index=False)
    return {
        "aime_seed": seed_df,
        "aime_agg": agg_df,
        "aime_paired": paired_df,
        "mixed": mixed_df,
        **train_dfs,
    }


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_line(ax, df: pd.DataFrame, key: str, label: str, *, window: int = 9) -> None:
    if key not in df.columns:
        return
    ax.plot(df["step"], smooth(df[key], window), label=label, linewidth=2)


def make_figures(data: dict[str, pd.DataFrame]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # AIME aggregate score.
    agg = data["aime_agg"].copy()
    agg["accuracy_pct"] = 100 * agg["accuracy"]
    yerr = [
        100 * (agg["accuracy"] - agg["wilson95_low"]),
        100 * (agg["wilson95_high"] - agg["accuracy"]),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    colors = ["#777777", "#2f6fba", "#7da9d6", "#c96567", "#d99a9a"]
    ax.bar(agg["group"], agg["accuracy_pct"], color=colors, yerr=yerr, capsize=4)
    ax.set_ylabel("AIME-2026 accuracy (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, max(22, float((100 * agg["wilson95_high"]).max()) + 2))
    ax.tick_params(axis="x", rotation=25)
    for idx, row in agg.iterrows():
        ax.text(idx, row["accuracy_pct"] + 0.7, f"{int(row['correct'])}/{int(row['n'])}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG / "aime2026_aggregate_accuracy.png")
    plt.close(fig)

    # GRPO training curves.
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True)
    for name, label in [
        ("aime_grpo_original", "GRPO original"),
        ("aime_grpo_rerun", "GRPO rerun"),
    ]:
        plot_line(axes[0], data[name], "ppo_actor/update/actor_loss/avg", label, window=15)
        plot_line(axes[1], data[name], "rollout/reward", f"{label} rollout", window=15)
        plot_line(axes[1], data[name], "eval-rollout/reward", f"{label} dev eval", window=3)
    axes[0].set_ylabel("Actor loss avg")
    axes[1].set_ylabel("Reward")
    axes[1].set_xlabel("Training step")
    axes[0].set_title("AIME-targeted GRPO training curves")
    axes[0].legend()
    axes[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "aime_grpo_training_curves.png")
    plt.close(fig)

    # SFT training loss.
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=True)
    for name, label in [
        ("aime_sft_original", "Rollout-SFT original"),
        ("aime_sft_rerun", "Rollout-SFT rerun"),
    ]:
        plot_line(axes[0], data[name], "sft/loss/avg", label, window=15)
        plot_line(axes[1], data[name], "sft/ppl/avg", label, window=15)
    axes[0].set_ylabel("SFT loss avg")
    axes[1].set_ylabel("Perplexity avg")
    axes[1].set_xlabel("Training step")
    axes[0].set_title("AIME rollout-SFT training curves")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "aime_sft_training_curves.png")
    plt.close(fig)

    # Mixed math results.
    mixed = data["mixed"].copy()
    order = ["GSM8K", "MATH L1/2", "MATH L3", "MATH L4/5", "Overall"]
    groups = list(MIXED_GROUPS)
    x = range(len(order))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for offset, group in enumerate(groups):
        vals = [
            100 * float(mixed[(mixed["group"] == group) & (mixed["benchmark"] == benchmark)]["accuracy"].iloc[0])
            for benchmark in order
        ]
        ax.bar([idx + (offset - 1) * width for idx in x], vals, width=width, label=group)
    ax.set_xticks(list(x), order)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Mixed-difficulty full-test accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "mixed_math_accuracy.png")
    plt.close(fig)

    # Mixed training curves.
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.0), sharex=False)
    plot_line(axes[0], data["mixed_grpo"], "ppo_actor/update/actor_loss/avg", "GRPO actor loss", window=11)
    plot_line(axes[0], data["mixed_grpo"], "rollout/reward", "GRPO rollout reward", window=11)
    plot_line(axes[0], data["mixed_grpo"], "eval-rollout/reward", "GRPO dev reward", window=3)
    plot_line(axes[1], data["mixed_sft"], "sft/loss/avg", "DeepSeek SFT loss", window=11)
    plot_line(axes[1], data["mixed_sft"], "sft/ppl/avg", "DeepSeek SFT ppl", window=11)
    axes[0].set_title("Mixed-math GRPO training")
    axes[0].set_xlabel("Training step")
    axes[1].set_title("Mixed-math SFT training")
    axes[1].set_xlabel("Training step")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "mixed_math_training_curves.png")
    plt.close(fig)


def main() -> None:
    data = write_tables()
    make_figures(data)
    print(f"Wrote report assets under {OUT}")


if __name__ == "__main__":
    main()
