"""Summarize AIME-2026 multi-seed results with paired comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rlvr_demo.multi_math_data import question_hash


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        nargs="+",
        required=True,
        metavar=("LABEL", "RESULT_DIR"),
        help="Label followed by one or more result directories. The first group is the baseline.",
    )
    parser.add_argument("--markdown", action="store_true", help="Also print compact markdown tables.")
    return parser.parse_args()


def _wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _binom_two_sided_pvalue(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dir(path: Path) -> tuple[int, dict[tuple[int, str], bool]]:
    metrics = _read_json(path / "aime_2026_metrics.json")
    seed = int(metrics["seed"])
    rows: dict[tuple[int, str], bool] = {}
    with (path / "aime_2026_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (seed, question_hash(str(row["question"])))
            if key in rows:
                raise ValueError(f"duplicate seed/question in {path}: {key}")
            rows[key] = bool(row["correct"])
    return seed, rows


def _load_group(label: str, paths: list[Path]) -> dict[str, Any]:
    by_key: dict[tuple[int, str], bool] = {}
    seeds: list[int] = []
    for path in paths:
        seed, rows = _load_dir(path)
        seeds.append(seed)
        overlap = set(by_key) & set(rows)
        if overlap:
            raise ValueError(f"group {label} repeats {len(overlap)} seed/question keys")
        by_key.update(rows)
    correct = sum(1 for value in by_key.values() if value)
    n = len(by_key)
    lo, hi = _wilson_interval(correct, n)
    return {
        "label": label,
        "seeds": seeds,
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "wilson_95_ci": [lo, hi],
        "by_key": by_key,
        "paths": [str(path) for path in paths],
    }


def _paired(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_keys = set(base["by_key"])
    candidate_keys = set(candidate["by_key"])
    if base_keys != candidate_keys:
        return {
            "baseline": base["label"],
            "candidate": candidate["label"],
            "paired": False,
            "missing_from_candidate": len(base_keys - candidate_keys),
            "extra_in_candidate": len(candidate_keys - base_keys),
        }

    candidate_only = 0
    baseline_only = 0
    both_correct = 0
    both_wrong = 0
    for key in sorted(base_keys):
        base_correct = bool(base["by_key"][key])
        candidate_correct = bool(candidate["by_key"][key])
        if base_correct and candidate_correct:
            both_correct += 1
        elif base_correct and not candidate_correct:
            baseline_only += 1
        elif candidate_correct:
            candidate_only += 1
        else:
            both_wrong += 1
    discordant = candidate_only + baseline_only
    return {
        "baseline": base["label"],
        "candidate": candidate["label"],
        "paired": True,
        "candidate_only_correct": candidate_only,
        "baseline_only_correct": baseline_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "two_sided_sign_test_p": _binom_two_sided_pvalue(candidate_only, discordant),
    }


def _public_group(group: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in group.items() if key != "by_key"}


def _print_markdown(groups: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> None:
    print("\n| Group | Seeds | Correct | Accuracy | Wilson 95% CI |")
    print("| --- | --- | ---: | ---: | --- |")
    for group in groups:
        lo, hi = group["wilson_95_ci"]
        print(
            f"| {group['label']} | {group['seeds']} | {group['correct']}/{group['n']} | "
            f"{100.0 * group['accuracy']:.2f}% | "
            f"{100.0 * lo:.2f}% to {100.0 * hi:.2f}% |"
        )

    print("\n| Candidate vs baseline | Candidate-only | Baseline-only | Sign-test p |")
    print("| --- | ---: | ---: | ---: |")
    for item in comparisons:
        if not item["paired"]:
            print(
                f"| {item['candidate']} | unpaired "
                f"(missing {item['missing_from_candidate']}, extra {item['extra_in_candidate']}) |  |  |"
            )
            continue
        print(
            f"| {item['candidate']} vs {item['baseline']} | "
            f"{item['candidate_only_correct']} | {item['baseline_only_correct']} | "
            f"{item['two_sided_sign_test_p']:.4f} |"
        )


def main() -> None:
    args = _parse_args()
    if any(len(item) < 2 for item in args.group):
        raise ValueError("each --group needs a label and at least one result directory")
    groups = [_load_group(item[0], [Path(path) for path in item[1:]]) for item in args.group]
    base = groups[0]
    comparisons = [_paired(base, group) for group in groups[1:]]
    report = {
        "benchmark": "aime_2026",
        "baseline": base["label"],
        "groups": [_public_group(group) for group in groups],
        "paired_comparisons": comparisons,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.markdown:
        _print_markdown(groups, comparisons)


if __name__ == "__main__":
    main()
