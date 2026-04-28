"""Summarize AReaL eval-rollout JSONL dumps for the GSM8K demo."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize_step(step_dir: Path) -> dict[str, float | int]:
    rewards: list[float] = []
    gen_lens: list[int] = []
    correct = 0
    strict_format = 0
    no_eos = 0

    files = sorted(step_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                reward = float(row["reward"])
                rewards.append(reward)
                gen_lens.append(int(row.get("seqlen", 0)) - int(row.get("prompt_len", 0)))
                correct += reward >= 1.0 - 1e-6
                strict_format += abs(reward - 1.1) < 1e-4 or abs(reward - 0.1) < 1e-4
                no_eos += "<|im_end|>" not in row.get("completion", "")

    n = len(rewards)
    if n == 0:
        raise ValueError(f"No eval rows found under {step_dir}")

    return {
        "step": int(step_dir.name),
        "n": n,
        "mean_reward": statistics.mean(rewards),
        "correct": correct,
        "accuracy": correct / n,
        "strict_format": strict_format,
        "strict_format_rate": strict_format / n,
        "avg_gen_len": statistics.mean(gen_lens),
        "no_eos": no_eos,
        "no_eos_rate": no_eos / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_rollout_dir", type=Path)
    args = parser.parse_args()

    step_dirs = sorted(
        [path for path in args.eval_rollout_dir.iterdir() if path.is_dir()],
        key=lambda path: int(path.name),
    )
    for item in (summarize_step(step_dir) for step_dir in step_dirs):
        print(
            "step={step:3d} n={n} mean_reward={mean_reward:.6f} "
            "correct={correct}/{n} acc={accuracy:.4f} "
            "strict={strict_format_rate:.4f} avg_gen_len={avg_gen_len:.1f} "
            "no_eos={no_eos_rate:.4f}".format(**item)
        )


if __name__ == "__main__":
    main()
