from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent

SERIES = {
    "Hand-crafted TA-RL": {
        "color": "#2563eb",
        "marker": "o",
        "points": [
            (0.00, 17, "SFT"),
            (0.20, 27, "README short"),
            (1.06, 3, "s449"),
            (2.13, 7, "s899"),
            (3.08, 3, "s1299"),
        ],
    },
    "Likelihood TA-RL": {
        "color": "#dc2626",
        "marker": "s",
        "points": [
            (0.00, 17, "SFT"),
            (0.20, 27, "README short"),
            (1.12, 9, "s449"),
            (2.25, 0, "s899"),
            (3.28, 3, "s1299"),
        ],
    },
    "GRPO": {
        "color": "#059669",
        "marker": "^",
        "points": [
            (0.00, 17, "SFT"),
            (0.82, 1, "s19"),
            (1.42, 24, "README best"),
            (2.02, 5, "s49"),
            (3.00, 7, "s74"),
        ],
    },
}

SFT_BASELINE = (0.0, 17, "SFT")


def strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def main() -> None:
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

    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=180)

    for name, spec in SERIES.items():
        xs = [p[0] for p in spec["points"]]
        ys = [p[1] for p in spec["points"]]
        ax.plot(
            xs,
            ys,
            label=name,
            color=spec["color"],
            marker=spec["marker"],
            markersize=7,
            linewidth=2.4,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        for x, y, label in spec["points"]:
            dy = 1.25 if y < 24 else -2.1
            ax.annotate(
                label,
                (x, y),
                xytext=(0, dy * 5),
                textcoords="offset points",
                ha="center",
                va="bottom" if dy > 0 else "top",
                fontsize=8,
                color=spec["color"],
            )

    ax.scatter(
        [SFT_BASELINE[0]],
        [SFT_BASELINE[1]],
        color="#111827",
        marker="D",
        s=82,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )

    ax.set_title("Terminal-Bench Performance vs RL Training Time", fontsize=16, pad=14)
    ax.set_xlabel("RL training time (hours)", fontsize=12)
    ax.set_ylabel("Full eval score (out of 100)", fontsize=12)
    ax.set_xlim(0, 3.45)
    ax.set_ylim(-1, 31)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.grid(True, which="major", color="#d1d5db", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#d1d5db")

    ax.text(
        0.01,
        -0.15,
        "Eval: 20 tasks x 5 attempts = 100 trials. Short README points are reused full-eval results.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "svg", "pdf"):
        out_path = OUT_DIR / f"tb_perf_vs_rl_training_time.{ext}"
        fig.savefig(out_path, bbox_inches="tight")
        if ext == "svg":
            strip_trailing_whitespace(out_path)


if __name__ == "__main__":
    main()
