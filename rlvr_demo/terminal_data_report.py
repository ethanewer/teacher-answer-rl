"""Report usable split sizes for Nemotron terminal-agent experiments."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from areal.utils.hf_utils import load_hf_tokenizer
from rlvr_demo.terminal_agent_data import (
    TERMINAL_CORPUS,
    TERMINAL_CORPUS_CONFIG,
    _history_messages,
    _partition_refs,
    _prepare_terminal_rows_and_refs,
    split_terminus_teacher_answer,
)


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return sorted(values)[idx]


def _summarize_lengths(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=TERMINAL_CORPUS)
    parser.add_argument("--name", default=TERMINAL_CORPUS_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--holdout-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=40960)
    parser.add_argument("--length-sample-turns", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.time()
    rows, refs = _prepare_terminal_rows_and_refs(
        path=args.path,
        name=args.name,
        split=args.split,
        seed=args.seed,
        limit_rows=None,
        shuffle_rows=False,
    )
    teacher_refs = []
    teacher_answers: dict[tuple[int, int], str] = {}
    for ref in refs:
        row_idx = int(ref["row_idx"])
        assistant_idx = int(ref["assistant_idx"])
        content = str(rows[row_idx]["messages"][assistant_idx]["content"]).strip()
        try:
            _, teacher_answer = split_terminus_teacher_answer(content)
        except Exception:
            continue
        teacher_refs.append(ref)
        teacher_answers[(row_idx, assistant_idx)] = teacher_answer

    train_refs = _partition_refs(
        refs,
        split_part="train",
        holdout_size=args.holdout_size,
        seed=args.seed,
        shuffle_records=True,
    )
    valid_refs = _partition_refs(
        refs,
        split_part="validation",
        holdout_size=args.holdout_size,
        seed=args.seed,
        shuffle_records=False,
    )
    teacher_train_refs = _partition_refs(
        teacher_refs,
        split_part="train",
        holdout_size=args.holdout_size,
        seed=args.seed,
        shuffle_records=True,
    )
    teacher_valid_refs = _partition_refs(
        teacher_refs,
        split_part="validation",
        holdout_size=args.holdout_size,
        seed=args.seed,
        shuffle_records=False,
    )

    sample = list(train_refs)
    random.Random(args.seed).shuffle(sample)
    sample = sample[: args.length_sample_turns]
    tokenizer = load_hf_tokenizer(args.model)
    sft_lengths: list[int] = []
    sft_target_lengths: list[int] = []
    for ref in sample:
        row = rows[int(ref["row_idx"])]
        assistant_idx = int(ref["assistant_idx"])
        messages = _history_messages(
            row["messages"],
            assistant_idx,
            strip_prior_assistant_thinking=True,
        )
        assistant = str(row["messages"][assistant_idx]["content"]).strip()
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": assistant}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        sft_lengths.append(len(full_ids))
        sft_target_lengths.append(max(0, len(full_ids) - len(prompt_ids)))

    teacher_sample = list(teacher_train_refs)
    random.Random(args.seed).shuffle(teacher_sample)
    teacher_sample = teacher_sample[: args.length_sample_turns]
    teacher_prompt_lengths: list[int] = []
    teacher_answer_lengths: list[int] = []
    for ref in teacher_sample:
        row_idx = int(ref["row_idx"])
        assistant_idx = int(ref["assistant_idx"])
        row = rows[row_idx]
        messages = _history_messages(
            row["messages"],
            assistant_idx,
            strip_prior_assistant_thinking=True,
        )
        teacher_answer = teacher_answers[(row_idx, assistant_idx)]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        teacher_prompt_lengths.append(len(prompt_ids))
        teacher_answer_lengths.append(
            len(tokenizer.encode(teacher_answer, add_special_tokens=False))
        )

    report = {
        "dataset": args.path,
        "config": args.name,
        "split": args.split,
        "base_model": args.model,
        "seed": args.seed,
        "holdout_size": args.holdout_size,
        "max_length": args.max_length,
        "raw_rows": len(rows),
        "raw_trainable_turns": len(refs),
        "unique_trajectories": len({ref["source_id"] for ref in refs}),
        "sft_train_turns_before_length_filter": len(train_refs),
        "sft_validation_turns_before_length_filter": len(valid_refs),
        "teacher_answer_train_turns_before_length_filter": len(teacher_train_refs),
        "teacher_answer_validation_turns_before_length_filter": len(teacher_valid_refs),
        "sampled_sft_full_length_tokens": _summarize_lengths(sft_lengths),
        "sampled_sft_target_tokens": _summarize_lengths(sft_target_lengths),
        "sampled_teacher_prompt_tokens": _summarize_lengths(teacher_prompt_lengths),
        "sampled_teacher_answer_tokens": _summarize_lengths(teacher_answer_lengths),
        "sampled_sft_over_max_length": sum(x > args.max_length for x in sft_lengths),
        "sampled_teacher_prompt_over_max_length": sum(
            x > args.max_length for x in teacher_prompt_lengths
        ),
        "elapsed_sec": time.time() - started,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
