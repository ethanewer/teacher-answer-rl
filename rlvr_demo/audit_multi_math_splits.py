"""Audit mixed-math train/validation/test split hygiene."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from rlvr_demo.model_paths import resolve_hf_snapshot
from rlvr_demo.multi_math_data import (
    DEEPSEEK_VALIDATION_HOLDOUT,
    GRPO_VALIDATION_HOLDOUT,
    SFT_VALIDATION_HOLDOUT,
    TEST_SOURCES,
    TRAIN_SOURCES,
    _apply_balanced_holdout_partition,
    _dedupe_records,
    build_multi_math_prompt,
    load_math_records,
    question_hash,
    record_bucket,
)


def _counter(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(record_bucket(row) for row in records))


def _hashes(records: list[dict[str, Any]]) -> set[str]:
    return {question_hash(str(row["question"])) for row in records}


def _partition(
    records: list[dict[str, Any]],
    holdout_per_bucket: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = _apply_balanced_holdout_partition(
        records,
        split_part="train",
        holdout_per_bucket=holdout_per_bucket,
        seed=seed,
    )
    validation = _apply_balanced_holdout_partition(
        records,
        split_part="validation",
        holdout_per_bucket=holdout_per_bucket,
        seed=seed,
    )
    return train, validation


def _deepseek_rows(path: Path, require_correct: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows_by_hash: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if require_correct:
                if row.get("status") != "ok" or row.get("teacher_correct") is not True:
                    continue
            elif row.get("status") not in {"ok", "wrong"}:
                continue
            if not str(row.get("teacher_prediction", "")).strip():
                continue
            rows_by_hash.setdefault(question_hash(str(row["question"])), row)
    return list(rows_by_hash.values())


def _token_filtered_count(records: list[dict[str, Any]], tokenizer, max_prompt_tokens: int) -> int:
    count = 0
    for row in records:
        prompt = [
            {
                "role": "user",
                "content": build_multi_math_prompt(str(row["question"])),
            }
        ]
        input_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if len(input_ids) <= max_prompt_tokens:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--deepseek-jsonl",
        type=Path,
        default=Path("rlvr_demo/data/deepseek_v4_pro_multi_math_balanced_sft.jsonl"),
    )
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="Exit nonzero if any final train/validation/test split overlap remains.",
    )
    args = parser.parse_args()

    train_raw = load_math_records(TRAIN_SOURCES, args.seed)
    test_raw = load_math_records(TEST_SOURCES, args.seed)
    train_unique = _dedupe_records(train_raw)
    test_unique = _dedupe_records(test_raw)
    test_hashes = _hashes(test_unique)
    clean_train = [row for row in train_unique if question_hash(str(row["question"])) not in test_hashes]

    grpo_train, grpo_valid = _partition(clean_train, GRPO_VALIDATION_HOLDOUT, args.seed)
    sft_train, sft_valid = _partition(clean_train, SFT_VALIDATION_HOLDOUT, args.seed)
    grpo_valid_hashes = _hashes(grpo_valid)

    deepseek_rows = _deepseek_rows(args.deepseek_jsonl)
    deepseek_filtered = [
        row for row in deepseek_rows if question_hash(str(row["question"])) not in grpo_valid_hashes
    ]
    deepseek_train, deepseek_valid = _partition(
        deepseek_filtered,
        DEEPSEEK_VALIDATION_HOLDOUT,
        args.seed,
    )
    deepseek_unfiltered_rows = _deepseek_rows(args.deepseek_jsonl, require_correct=False)
    deepseek_unfiltered_filtered = [
        row
        for row in deepseek_unfiltered_rows
        if question_hash(str(row["question"])) not in grpo_valid_hashes
    ]
    deepseek_unfiltered_train, deepseek_unfiltered_valid = _partition(
        deepseek_unfiltered_filtered,
        DEEPSEEK_VALIDATION_HOLDOUT,
        args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(resolve_hf_snapshot(args.model), trust_remote_code=True)
    report = {
        "seed": args.seed,
        "raw_counts": {
            "train_rows": len(train_raw),
            "train_unique": len(train_unique),
            "test_rows": len(test_raw),
            "test_unique": len(test_unique),
            "train_test_overlap": len(_hashes(train_unique) & test_hashes),
            "clean_train_unique": len(clean_train),
        },
        "bucket_counts": {
            "clean_train": _counter(clean_train),
            "test": _counter(test_unique),
            "grpo_train": _counter(grpo_train),
            "grpo_validation": _counter(grpo_valid),
            "sft_train": _counter(sft_train),
            "sft_validation": _counter(sft_valid),
            "deepseek_verified": _counter(deepseek_rows),
            "deepseek_after_shared_validation_filter": _counter(deepseek_filtered),
            "deepseek_train": _counter(deepseek_train),
            "deepseek_validation": _counter(deepseek_valid),
            "deepseek_unfiltered": _counter(deepseek_unfiltered_rows),
            "deepseek_unfiltered_after_shared_validation_filter": _counter(
                deepseek_unfiltered_filtered
            ),
            "deepseek_unfiltered_train": _counter(deepseek_unfiltered_train),
            "deepseek_unfiltered_validation": _counter(deepseek_unfiltered_valid),
        },
        "overlap_checks": {
            "grpo_train_vs_validation": len(_hashes(grpo_train) & _hashes(grpo_valid)),
            "grpo_train_vs_test": len(_hashes(grpo_train) & test_hashes),
            "grpo_validation_vs_test": len(_hashes(grpo_valid) & test_hashes),
            "sft_train_vs_validation": len(_hashes(sft_train) & _hashes(sft_valid)),
            "sft_train_vs_test": len(_hashes(sft_train) & test_hashes),
            "sft_validation_vs_test": len(_hashes(sft_valid) & test_hashes),
            "deepseek_verified_vs_shared_validation": len(_hashes(deepseek_rows) & grpo_valid_hashes),
            "deepseek_filtered_vs_shared_validation": len(_hashes(deepseek_filtered) & grpo_valid_hashes),
            "deepseek_train_vs_validation": len(_hashes(deepseek_train) & _hashes(deepseek_valid)),
            "deepseek_train_vs_test": len(_hashes(deepseek_train) & test_hashes),
            "deepseek_validation_vs_test": len(_hashes(deepseek_valid) & test_hashes),
            "deepseek_train_vs_shared_validation": len(_hashes(deepseek_train) & grpo_valid_hashes),
            "deepseek_validation_vs_shared_validation": len(
                _hashes(deepseek_valid) & grpo_valid_hashes
            ),
            "deepseek_unfiltered_filtered_vs_shared_validation": len(
                _hashes(deepseek_unfiltered_filtered) & grpo_valid_hashes
            ),
            "deepseek_unfiltered_train_vs_validation": len(
                _hashes(deepseek_unfiltered_train) & _hashes(deepseek_unfiltered_valid)
            ),
            "deepseek_unfiltered_train_vs_test": len(
                _hashes(deepseek_unfiltered_train) & test_hashes
            ),
            "deepseek_unfiltered_validation_vs_test": len(
                _hashes(deepseek_unfiltered_valid) & test_hashes
            ),
            "deepseek_unfiltered_train_vs_shared_validation": len(
                _hashes(deepseek_unfiltered_train) & grpo_valid_hashes
            ),
            "deepseek_unfiltered_validation_vs_shared_validation": len(
                _hashes(deepseek_unfiltered_valid) & grpo_valid_hashes
            ),
        },
        "token_filtered_counts": {
            "grpo_train_prompt_le_1536": _token_filtered_count(grpo_train, tokenizer, 1536),
            "grpo_validation_prompt_le_1536": _token_filtered_count(grpo_valid, tokenizer, 1536),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    final_overlap_keys = [
        "grpo_train_vs_validation",
        "grpo_train_vs_test",
        "grpo_validation_vs_test",
        "sft_train_vs_validation",
        "sft_train_vs_test",
        "sft_validation_vs_test",
        "deepseek_filtered_vs_shared_validation",
        "deepseek_train_vs_validation",
        "deepseek_train_vs_test",
        "deepseek_validation_vs_test",
        "deepseek_train_vs_shared_validation",
        "deepseek_validation_vs_shared_validation",
        "deepseek_unfiltered_filtered_vs_shared_validation",
        "deepseek_unfiltered_train_vs_validation",
        "deepseek_unfiltered_train_vs_test",
        "deepseek_unfiltered_validation_vs_test",
        "deepseek_unfiltered_train_vs_shared_validation",
        "deepseek_unfiltered_validation_vs_shared_validation",
    ]
    failures = {
        key: int(report["overlap_checks"][key])
        for key in final_overlap_keys
        if int(report["overlap_checks"][key]) != 0
    }
    if args.fail_on_overlap and failures:
        print(json.dumps({"audit_failures": failures}, indent=2, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
