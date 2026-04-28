"""Audit AIME-2026 train/eval split hygiene for the AIME-targeted recipes."""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rlvr_demo.multi_math_data import (
    load_math_records,
    normalize_question,
    question_hash,
    record_bucket,
)


GRPO_TRAIN_SOURCES: list[dict[str, Any]] = [
    {"name": "aime_historical", "split": "train", "max_year": 2021},
    {"name": "aime_recent", "split": "train", "min_year": 2022, "max_year": 2023},
    {"name": "math", "split": "train", "levels": ["Level 3", "Level 4", "Level 5"]},
]

AIME_2024_SOURCE = [{"name": "aime_historical", "split": "train", "min_year": 2024, "max_year": 2024}]
AIME_2025_SOURCE = [{"name": "aime_2025", "split": "train"}]
AIME_2026_SOURCE = [{"name": "aime_2026", "split": "train"}]


def _jsonl_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows_by_hash: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question") or "").strip()
            if not question:
                continue
            rows_by_hash.setdefault(
                question_hash(question),
                {
                    "question": question,
                    "source": str(row.get("source") or "jsonl"),
                    "level": str(row.get("level") or ""),
                },
            )
    return list(rows_by_hash.values())


def _hashes(rows: list[dict[str, Any]]) -> set[str]:
    return {question_hash(str(row["question"])) for row in rows}


def _bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(record_bucket(row) for row in rows))


def _aime_year_bounds(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    years = [
        int(row["contest_year"])
        for row in rows
        if row.get("contest_year") is not None and str(row.get("source")) == "aime"
    ]
    if not years:
        return {"min": None, "max": None}
    return {"min": min(years), "max": max(years)}


def _fuzzy_report(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    max_ratio = 0.0
    max_pair: dict[str, str] | None = None
    high_pairs: list[dict[str, Any]] = []
    normalized_test = [(row, normalize_question(str(row["question"]))) for row in test_rows]
    for train_row in train_rows:
        train_text = normalize_question(str(train_row["question"]))
        for test_row, test_text in normalized_test:
            ratio = difflib.SequenceMatcher(None, train_text, test_text).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
                max_pair = {
                    "train_question": str(train_row["question"])[:240],
                    "test_question": str(test_row["question"])[:240],
                }
            if ratio >= threshold:
                high_pairs.append(
                    {
                        "ratio": ratio,
                        "source": train_row.get("source", ""),
                        "level": train_row.get("level", ""),
                        "train_question": str(train_row["question"])[:240],
                        "test_question": str(test_row["question"])[:240],
                    }
                )
    high_pairs.sort(key=lambda item: float(item["ratio"]), reverse=True)
    return {
        "max_ratio": round(max_ratio, 6),
        "pairs_at_or_above_threshold": len(high_pairs),
        "top_pairs": high_pairs[:5],
        "max_pair": max_pair,
    }


def _comparison(
    name: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    train_hashes = _hashes(train_rows)
    test_hashes = _hashes(test_rows)
    return {
        "name": name,
        "unique_questions": len(train_hashes),
        "bucket_counts": _bucket_counts(train_rows),
        "exact_overlap": len(train_hashes & test_hashes),
        "fuzzy": _fuzzy_report(train_rows, test_rows, threshold),
    }


def _collect_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    subset = report["rft_subset_check"]
    if int(subset["rft_questions_not_in_grpo_train"]) != 0:
        failures.append("RFT-SFT prompts are not a subset of the GRPO training prompts")

    comparisons: list[dict[str, Any]] = list(report["against_aime_2026"])
    for group in report["dev_holdout_checks"].values():
        comparisons.extend(group)

    for item in comparisons:
        name = str(item["name"])
        exact_overlap = int(item["exact_overlap"])
        fuzzy_count = int(item["fuzzy"]["pairs_at_or_above_threshold"])
        if exact_overlap:
            failures.append(f"{name} has {exact_overlap} exact held-out overlaps")
        if fuzzy_count:
            failures.append(f"{name} has {fuzzy_count} fuzzy held-out overlaps")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--rft-jsonl",
        type=Path,
        default=Path("rlvr_demo/data/qwen3_17b_hardmath_grpo_correct_rollout_sft_max2.jsonl"),
    )
    parser.add_argument(
        "--deepseek-aime-jsonl",
        type=Path,
        default=Path("rlvr_demo/data/deepseek_v4_pro_aime_1983_2025_sft.jsonl"),
    )
    parser.add_argument("--fuzzy-threshold", type=float, default=0.90)
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="Exit nonzero on any exact overlap, fuzzy overlap, or RFT subset violation.",
    )
    args = parser.parse_args()

    grpo_train = load_math_records(GRPO_TRAIN_SOURCES, args.seed)
    grpo_by_hash = {question_hash(str(row["question"])): row for row in grpo_train}
    grpo_train = list(grpo_by_hash.values())

    rft_sft = _jsonl_questions(args.rft_jsonl)
    rft_hashes = _hashes(rft_sft)
    grpo_hashes = _hashes(grpo_train)

    deepseek_aime = _jsonl_questions(args.deepseek_aime_jsonl)
    aime_2024 = load_math_records(AIME_2024_SOURCE, args.seed)
    aime_2025 = load_math_records(AIME_2025_SOURCE, args.seed)
    aime_2026 = load_math_records(AIME_2026_SOURCE, args.seed)

    report = {
        "seed": args.seed,
        "fuzzy_threshold": args.fuzzy_threshold,
        "train_sources": {
            "grpo": GRPO_TRAIN_SOURCES,
            "sft_rollout_distillation": {
                "prompt_source": "verifier-correct rollouts from the GRPO training prompt set",
                "jsonl": str(args.rft_jsonl),
            },
        },
        "source_date_bounds": {
            "grpo_aime_years": _aime_year_bounds(grpo_train),
            "math_source_note": "MATH train split from the 2021 Hendrycks et al. competition-math dataset",
            "aime_2026_eval_rows": len(aime_2026),
        },
        "rft_subset_check": {
            "rft_unique_questions": len(rft_hashes),
            "grpo_train_unique_questions": len(grpo_hashes),
            "rft_questions_not_in_grpo_train": len(rft_hashes - grpo_hashes),
        },
        "against_aime_2026": [
            _comparison("grpo_train_prompts", grpo_train, aime_2026, args.fuzzy_threshold),
            _comparison("rollout_rft_sft_prompts", rft_sft, aime_2026, args.fuzzy_threshold),
            _comparison("deepseek_aime_sft_prompts", deepseek_aime, aime_2026, args.fuzzy_threshold),
        ],
        "dev_holdout_checks": {
            "against_aime_2024": [
                _comparison("grpo_train_prompts", grpo_train, aime_2024, args.fuzzy_threshold),
                _comparison("rollout_rft_sft_prompts", rft_sft, aime_2024, args.fuzzy_threshold),
            ],
            "against_aime_2025": [
                _comparison("grpo_train_prompts", grpo_train, aime_2025, args.fuzzy_threshold),
                _comparison("rollout_rft_sft_prompts", rft_sft, aime_2025, args.fuzzy_threshold),
            ],
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    failures = _collect_failures(report)
    if args.fail_on_overlap and failures:
        print(json.dumps({"audit_failures": failures}, indent=2, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
