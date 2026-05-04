"""Estimate non-thinking Qwen3 terminal SFT token counts after stripping think blocks."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from rlvr_demo.terminal_agent_data import (
    TERMINAL_CORPUS,
    TERMINAL_CORPUS_FULL_MIX_CONFIGS,
)


THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
STRIP_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
ADAPTER_FILES = (
    "dataset_adapters/math.parquet",
    "dataset_adapters/code.parquet",
    "dataset_adapters/swe.parquet",
)


def _coerce_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for raw in row.get("conversations") or []:
        role = str(raw.get("role", "")).strip()
        content = str(raw.get("content", ""))
        if role in {"system", "user", "assistant", "tool"} and content:
            messages.append({"role": role, "content": content})
    return messages


def _strip_think(content: str) -> str:
    return STRIP_THINK_RE.sub("", content).lstrip()


def _first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _encode_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _trajectory_counts(row: dict[str, Any], tokenizer) -> dict[str, int]:
    counts = defaultdict(int)
    for msg in _coerce_messages(row):
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            counts["assistant_turns"] += 1
            header_len = _encode_len(tokenizer, "<|im_start|>assistant\n")
            body_orig = _encode_len(tokenizer, content.strip() + "<|im_end|>\n")
            body_strip = _encode_len(
                tokenizer, _strip_think(content).strip() + "<|im_end|>\n"
            )
            counts["total_orig"] += header_len + body_orig
            counts["trained_orig"] += body_orig
            counts["total_strip"] += header_len + body_strip
            counts["trained_strip"] += body_strip
            for match in THINK_RE.finditer(content):
                counts["think_inner_chars"] += len(match.group(1))
                counts["think_inner_tokens"] += _encode_len(tokenizer, match.group(1))
                counts["think_block_tokens"] += _encode_len(tokenizer, match.group(0))
            obj = _first_json_object(content)
            if obj is not None:
                analysis = str(obj.get("analysis", ""))
                plan = str(obj.get("plan", ""))
                counts["analysis_plan_chars"] += len(analysis) + len(plan)
                counts["analysis_plan_tokens"] += _encode_len(
                    tokenizer, f"{analysis}\n{plan}"
                )
            continue

        if role == "tool":
            text = (
                f"<|im_start|>user\n<tool_response>\n{content}\n"
                "</tool_response><|im_end|>\n"
            )
        else:
            text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        msg_len = _encode_len(tokenizer, text)
        counts["total_orig"] += msg_len
        counts["total_strip"] += msg_len
    return dict(counts)


def _sample_indices(row_count: int, sample_count: int) -> list[int]:
    sample_count = min(sample_count, row_count)
    if sample_count <= 1:
        return [0] if row_count else []
    return sorted(
        set(round(idx * (row_count - 1) / (sample_count - 1)) for idx in range(sample_count))
    )


def _new_stats() -> dict[str, Any]:
    stats: dict[str, Any] = defaultdict(float)
    stats["total_strip_lengths"] = []
    stats["trained_strip_lengths"] = []
    return stats


def _accumulate(stats: dict[str, Any], counts: dict[str, int]) -> None:
    stats["rows"] += 1
    stats["total_strip_lengths"].append(counts["total_strip"])
    stats["trained_strip_lengths"].append(counts["trained_strip"])
    for key, value in counts.items():
        stats[key] += value


def _quantile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(math.ceil(fraction * len(values))) - 1)]


def _summarize(config: str, row_count: int, stats: dict[str, Any]) -> dict[str, Any]:
    rows = int(stats["rows"])
    scale = row_count / rows if rows else 0.0

    def mean(key: str) -> float:
        return float(stats[key]) / rows if rows else 0.0

    def estimate(key: str) -> float:
        return float(stats[key]) * scale

    total_strip_lengths = list(stats["total_strip_lengths"])
    return {
        "config": config,
        "row_count": row_count,
        "sample_rows": rows,
        "mean_total_orig_tokens": mean("total_orig"),
        "mean_total_strip_tokens": mean("total_strip"),
        "mean_trained_orig_tokens": mean("trained_orig"),
        "mean_trained_strip_tokens": mean("trained_strip"),
        "mean_removed_total_tokens": mean("total_orig") - mean("total_strip"),
        "mean_think_inner_tokens": mean("think_inner_tokens"),
        "mean_think_inner_chars": mean("think_inner_chars"),
        "mean_analysis_plan_tokens": mean("analysis_plan_tokens"),
        "mean_analysis_plan_chars": mean("analysis_plan_chars"),
        "mean_assistant_turns": mean("assistant_turns"),
        "p50_total_strip_tokens": _quantile(total_strip_lengths, 0.50),
        "p95_total_strip_tokens": _quantile(total_strip_lengths, 0.95),
        "p99_total_strip_tokens": _quantile(total_strip_lengths, 0.99),
        "max_total_strip_tokens_sample": max(total_strip_lengths, default=0),
        "over_32768_total_strip_fraction": (
            sum(1 for value in total_strip_lengths if value > 32768) / rows
            if rows
            else 0.0
        ),
        "estimated_total_orig_tokens": estimate("total_orig"),
        "estimated_total_strip_tokens": estimate("total_strip"),
        "estimated_trained_strip_tokens": estimate("trained_strip"),
        "estimated_removed_total_tokens": estimate("total_orig")
        - estimate("total_strip"),
        "estimated_think_inner_tokens": estimate("think_inner_tokens"),
        "estimated_analysis_plan_tokens": estimate("analysis_plan_tokens"),
    }


def _iter_dataset_sample(
    dataset: str,
    config: str,
    sample_rows: int,
) -> tuple[int, Iterable[dict[str, Any]]]:
    hf_dataset = load_dataset(dataset, name=config, split="train")
    row_count = len(hf_dataset)
    indices = _sample_indices(row_count, sample_rows)
    return row_count, (dict(hf_dataset[int(index)]) for index in indices)


def _adapter_summary(dataset: str, tokenizer, sample_rows: int) -> dict[str, Any]:
    stats = _new_stats()
    total_rows = 0
    per_file_sample = max(1, sample_rows // len(ADAPTER_FILES))
    for file_path in ADAPTER_FILES:
        local_path = hf_hub_download(dataset, file_path, repo_type="dataset")
        parquet_file = pq.ParquetFile(local_path)
        row_count = parquet_file.metadata.num_rows
        total_rows += row_count
        wanted = set(_sample_indices(row_count, per_file_sample))
        seen = 0
        got = 0
        for batch in parquet_file.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                if seen in wanted:
                    _accumulate(stats, _trajectory_counts(row, tokenizer))
                    got += 1
                seen += 1
            if got >= len(wanted):
                break
    return _summarize("dataset_adapters", total_rows, stats)


def _aggregate(label: str, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = sum(summary["row_count"] for summary in summaries)
    output = {
        "config": label,
        "row_count": row_count,
        "sample_rows": sum(summary["sample_rows"] for summary in summaries),
    }
    for key in (
        "estimated_total_orig_tokens",
        "estimated_total_strip_tokens",
        "estimated_trained_strip_tokens",
        "estimated_removed_total_tokens",
        "estimated_think_inner_tokens",
        "estimated_analysis_plan_tokens",
    ):
        output[key] = sum(summary[key] for summary in summaries)
    output["mean_total_strip_tokens"] = output["estimated_total_strip_tokens"] / row_count
    output["mean_trained_strip_tokens"] = (
        output["estimated_trained_strip_tokens"] / row_count
    )
    output["mean_removed_total_tokens"] = (
        output["estimated_removed_total_tokens"] / row_count
    )
    output["mean_think_inner_tokens"] = (
        output["estimated_think_inner_tokens"] / row_count
    )
    output["mean_analysis_plan_tokens"] = (
        output["estimated_analysis_plan_tokens"] / row_count
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset", default=TERMINAL_CORPUS)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(TERMINAL_CORPUS_FULL_MIX_CONFIGS),
        help="Dataset configs to sample. dataset_adapters is sampled from parquet.",
    )
    parser.add_argument("--sample-rows", type=int, default=4096)
    parser.add_argument("--adapter-sample-rows", type=int, default=6144)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    summaries: list[dict[str, Any]] = []
    for config in args.configs:
        if config == "dataset_adapters":
            summary = _adapter_summary(args.dataset, tokenizer, args.adapter_sample_rows)
        else:
            row_count, rows = _iter_dataset_sample(args.dataset, config, args.sample_rows)
            stats = _new_stats()
            for row in rows:
                _accumulate(stats, _trajectory_counts(row, tokenizer))
            summary = _summarize(config, row_count, stats)
        summaries.append(summary)

    by_config = {summary["config"]: summary for summary in summaries}
    aggregates = []
    skill_names = [
        "skill_based_easy",
        "skill_based_medium",
        "skill_based_mixed",
    ]
    if all(name in by_config for name in skill_names):
        aggregates.append(
            _aggregate("skill_based_all", [by_config[name] for name in skill_names])
        )
    full_mix_names = ["dataset_adapters", *skill_names]
    if all(name in by_config for name in full_mix_names):
        aggregates.append(
            _aggregate("full_mix", [by_config[name] for name in full_mix_names])
        )
    report = {
        "model": args.model,
        "dataset": args.dataset,
        "chat_template_contains_think": "<think>" in (tokenizer.chat_template or ""),
        "summaries": summaries,
        "aggregates": aggregates,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
