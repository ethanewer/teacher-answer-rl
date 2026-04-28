"""Mixed GSM8K/MATH dataset helpers for Qwen3 math RLVR and SFT."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from rlvr_demo.qwen3_gsm8k_data import extract_gold_answer


MULTI_MATH_PROMPT_TEMPLATE = """Please solve the math problem step by step. Use <think> tags to show your reasoning process, then provide the final answer.

Problem:
{question}

IMPORTANT: Put the final answer after `Final answer:`. For exact values, use a simplified exact expression or LaTeX. Do not include units or explanatory text after the final answer.

Format:
<think>
Your step-by-step reasoning here...
</think>
Final answer: [answer only]"""

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_WHITESPACE_RE = re.compile(r"\s+")
_BUCKET_ORDER = ("gsm8k", "math_l12", "math_l3", "math_l45")
TRAIN_SOURCES = [{"name": "gsm8k", "split": "train"}, {"name": "math", "split": "train"}]
TEST_SOURCES = [{"name": "gsm8k", "split": "test"}, {"name": "math", "split": "test"}]
GRPO_VALIDATION_HOLDOUT = {"gsm8k": 128, "math_l12": 64, "math_l3": 64, "math_l45": 64}
SFT_VALIDATION_HOLDOUT = GRPO_VALIDATION_HOLDOUT
DEEPSEEK_VALIDATION_HOLDOUT = {"gsm8k": 32, "math_l12": 32, "math_l3": 32, "math_l45": 32}


def normalize_question(question: str) -> str:
    return _WHITESPACE_RE.sub(" ", question.strip()).casefold()


def question_hash(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def build_multi_math_prompt(question: str) -> str:
    return MULTI_MATH_PROMPT_TEMPLATE.format(question=question.strip())


def _extract_braced(text: str, start: int) -> str | None:
    depth = 0
    chars: list[str] = []
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
            if depth == 1:
                continue
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        if depth >= 1:
            chars.append(char)
    return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the last top-level ``\\boxed{...}`` answer from a MATH solution."""
    matches = list(_BOXED_RE.finditer(solution))
    for match in reversed(matches):
        answer = _extract_braced(solution, match.end() - 1)
        if answer:
            return answer
    return solution.strip()


def _coerce_sources(sources: Any, default_source: str | None) -> list[dict[str, Any]]:
    if sources is None:
        if default_source is None:
            raise ValueError("dataset_kwargs.sources must be set for mixed math datasets")
        return [{"name": default_source}]
    if not isinstance(sources, list):
        raise TypeError("dataset_kwargs.sources must be a list of source specs")
    return [dict(item) for item in sources]


def _select_limit(dataset: Dataset, limit: int | None, seed: int, shuffle: bool) -> Dataset:
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError(f"limit must be positive when set, got {limit}")
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(min(limit, len(dataset))))


def _source_records(spec: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    name = str(spec["name"]).lower()
    limit = spec.get("limit")
    split = str(spec.get("split", "train"))
    shuffle_limit = bool(spec.get("shuffle_limit", False))
    records: list[dict[str, Any]] = []

    if name in {"gsm8k", "openai/gsm8k"}:
        dataset = load_dataset("openai/gsm8k", name="main", split=split)
        dataset = _select_limit(dataset, limit, seed, shuffle_limit)
        for row in dataset:
            question = str(row["question"]).strip()
            answer = extract_gold_answer(str(row["answer"]))
            solution = str(row["answer"]).split("####", 1)[0].strip()
            records.append(
                {
                    "source": "gsm8k",
                    "level": "gsm8k",
                    "question": question,
                    "answer": answer,
                    "solution": solution,
                }
            )
        return records

    if name in {"math", "math-lighteval", "digitallearninggmbh/math-lighteval"}:
        dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
        levels = spec.get("levels")
        if levels:
            allowed = {str(level) for level in levels}
            dataset = dataset.filter(lambda row: str(row["level"]) in allowed)
        types = spec.get("types")
        if types:
            allowed_types = {str(item) for item in types}
            dataset = dataset.filter(lambda row: str(row["type"]) in allowed_types)
        dataset = _select_limit(dataset, limit, seed, shuffle_limit)
        for row in dataset:
            question = str(row["problem"]).strip()
            solution = str(row["solution"]).strip()
            records.append(
                {
                    "source": "math",
                    "level": str(row["level"]),
                    "question": question,
                    "answer": extract_boxed_answer(solution),
                    "solution": solution,
                    "type": str(row["type"]),
                }
            )
        return records

    raise ValueError(f"Unsupported math source: {name}")


def load_math_records(sources: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, source in enumerate(sources):
        source_seed = seed + idx * 1009
        records.extend(_source_records(source, source_seed))
    return records


def load_heldout_hashes(sources: Iterable[dict[str, Any]], seed: int) -> set[str]:
    return {question_hash(row["question"]) for row in load_math_records(sources, seed)}


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in records:
        hsh = question_hash(row["question"])
        if hsh in seen:
            continue
        seen.add(hsh)
        unique.append(row)
    return unique


def load_clean_train_records(seed: int) -> list[dict[str, Any]]:
    """Load deduped official train rows after removing exact official test overlap."""
    records = _dedupe_records(load_math_records(TRAIN_SOURCES, seed))
    heldout = load_heldout_hashes(TEST_SOURCES, seed)
    return [row for row in records if question_hash(str(row["question"])) not in heldout]


def record_bucket(row: dict[str, Any]) -> str:
    """Return the difficulty bucket used for balanced train-validation splits."""
    bucket = str(row.get("bucket", ""))
    if bucket in _BUCKET_ORDER:
        return bucket
    if row.get("source") == "gsm8k":
        return "gsm8k"
    level = str(row.get("level", ""))
    if level in {"Level 1", "Level 2"}:
        return "math_l12"
    if level == "Level 3":
        return "math_l3"
    if level in {"Level 4", "Level 5"}:
        return "math_l45"
    return "other"


def _coerce_holdout_per_bucket(holdout_per_bucket: Any) -> dict[str, int]:
    if holdout_per_bucket is None:
        return {}
    return {
        str(bucket): int(count)
        for bucket, count in dict(holdout_per_bucket).items()
        if int(count) > 0
    }


def _balanced_holdout_hashes(
    records: list[dict[str, Any]],
    holdout_per_bucket: dict[str, int],
    seed: int,
) -> set[str]:
    by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in _BUCKET_ORDER}
    for row in records:
        bucket = record_bucket(row)
        if bucket in by_bucket:
            by_bucket[bucket].append(row)

    selected: set[str] = set()
    for idx, bucket in enumerate(_BUCKET_ORDER):
        n = holdout_per_bucket.get(bucket, 0)
        if n <= 0:
            continue
        rows = list(by_bucket[bucket])
        random.Random(seed + idx * 1009).shuffle(rows)
        for row in rows[: min(n, len(rows))]:
            selected.add(question_hash(str(row["question"])))
    return selected


def _apply_balanced_holdout_partition(
    records: list[dict[str, Any]],
    split_part: str | None,
    holdout_per_bucket: Any,
    seed: int,
) -> list[dict[str, Any]]:
    if split_part is None:
        return records
    if split_part not in {"train", "validation"}:
        raise ValueError("split_part must be 'train' or 'validation'")
    counts = _coerce_holdout_per_bucket(holdout_per_bucket)
    if not counts:
        raise ValueError("balanced_holdout requires non-empty holdout_per_bucket")
    holdout_hashes = _balanced_holdout_hashes(records, counts, seed)
    if split_part == "validation":
        return [row for row in records if question_hash(str(row["question"])) in holdout_hashes]
    return [row for row in records if question_hash(str(row["question"])) not in holdout_hashes]


def balanced_train_validation_hashes(
    seed: int,
    holdout_per_bucket: dict[str, int] | None = None,
) -> set[str]:
    """Return the shared train-split validation hashes used for checkpoint selection."""
    return _balanced_holdout_hashes(
        load_clean_train_records(seed),
        GRPO_VALIDATION_HOLDOUT if holdout_per_bucket is None else holdout_per_bucket,
        seed,
    )


def _apply_holdout_partition(
    records: list[dict[str, Any]],
    split_part: str | None,
    holdout_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if split_part is None:
        return records
    if split_part not in {"train", "validation"}:
        raise ValueError("split_part must be 'train' or 'validation'")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    holdout = min(holdout_size, len(shuffled))
    if split_part == "validation":
        return shuffled[:holdout]
    return shuffled[holdout:]


def get_multi_math_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    sources: list[dict[str, Any]] | None = None,
    eval_sources: list[dict[str, Any]] | None = None,
    seed: int = 1,
    limit: int | None = None,
    shuffle_limit: bool = False,
    shuffle_records: bool = False,
    split_part: str | None = None,
    holdout_size: int = 512,
    balanced_holdout: bool = False,
    holdout_per_bucket: dict[str, int] | None = None,
    **_: Any,
) -> Dataset:
    """Load mixed math RL rows while preventing train/test question overlap."""
    del path
    default_source = "gsm8k" if split in {"train", "test"} else None
    source_specs = _coerce_sources(sources, default_source)
    records = _dedupe_records(load_math_records(source_specs, seed))

    if eval_sources is not None:
        heldout = load_heldout_hashes(eval_sources, seed)
        before = len(records)
        records = [row for row in records if question_hash(row["question"]) not in heldout]
        removed = before - len(records)
        if removed:
            print(f"Removed {removed} train rows that overlapped held-out eval questions.")

    if balanced_holdout:
        records = _apply_balanced_holdout_partition(
            records,
            split_part=split_part,
            holdout_per_bucket=holdout_per_bucket,
            seed=seed,
        )
    else:
        records = _apply_holdout_partition(records, split_part, holdout_size, seed)

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when set, got {limit}")
        if shuffle_limit:
            random.Random(seed).shuffle(records)
        records = records[:limit]
    elif shuffle_records:
        random.Random(seed).shuffle(records)

    rows: list[dict[str, Any]] = []
    for row in records:
        messages = [{"role": "user", "content": build_multi_math_prompt(row["question"])}]
        rows.append(
            {
                "messages": messages,
                "answer": row["answer"],
                "question": row["question"],
                "source": row.get("source", ""),
                "level": row.get("level", ""),
                "type": row.get("type", ""),
            }
        )

    dataset = Dataset.from_list(rows)
    if max_length is not None:

        def filter_length(sample: dict[str, Any]) -> bool:
            input_ids = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            return len(input_ids) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


def get_named_eval_dataset(
    name: str,
    tokenizer,
    max_length: int | None,
    limit: int | None,
    seed: int,
) -> Dataset:
    if name == "mixed_train_validation":
        return get_multi_math_dataset(
            path="mixed_math",
            split="train",
            tokenizer=tokenizer,
            max_length=max_length,
            sources=TRAIN_SOURCES,
            eval_sources=TEST_SOURCES,
            seed=seed,
            limit=limit,
            shuffle_limit=True,
            split_part="validation",
            balanced_holdout=True,
            holdout_per_bucket=GRPO_VALIDATION_HOLDOUT,
        )

    if name == "gsm8k_test":
        sources = [{"name": "gsm8k", "split": "test", "limit": limit, "shuffle_limit": False}]
    elif name == "math_test_all":
        sources = [{"name": "math", "split": "test", "limit": limit, "shuffle_limit": False}]
    elif name == "math_test_l12":
        sources = [
            {
                "name": "math",
                "split": "test",
                "levels": ["Level 1", "Level 2"],
                "limit": limit,
                "shuffle_limit": False,
            }
        ]
    elif name == "math_test_l3":
        sources = [
            {
                "name": "math",
                "split": "test",
                "levels": ["Level 3"],
                "limit": limit,
                "shuffle_limit": False,
            }
        ]
    elif name == "math_test_l45":
        sources = [
            {
                "name": "math",
                "split": "test",
                "levels": ["Level 4", "Level 5"],
                "limit": limit,
                "shuffle_limit": False,
            }
        ]
    else:
        raise ValueError(f"Unknown eval dataset name: {name}")
    return get_multi_math_dataset(
        path="mixed_math",
        split="test",
        tokenizer=tokenizer,
        max_length=max_length,
        sources=sources,
        seed=seed,
    )


def get_multi_math_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    sources: list[dict[str, Any]] | None = None,
    eval_sources: list[dict[str, Any]] | None = None,
    seed: int = 1,
    limit: int | None = None,
    shuffle_limit: bool = False,
    shuffle_records: bool = False,
    split_part: str | None = None,
    holdout_size: int = 512,
    balanced_holdout: bool = False,
    holdout_per_bucket: dict[str, int] | None = None,
    **_: Any,
) -> Dataset:
    """Load mixed math SFT rows from official train-split solutions."""
    rl_dataset = get_multi_math_dataset(
        path=path,
        split=split,
        tokenizer=tokenizer,
        max_length=None,
        sources=sources,
        eval_sources=eval_sources,
        seed=seed,
        limit=limit,
        shuffle_limit=shuffle_limit,
        shuffle_records=shuffle_records,
        split_part=split_part,
        holdout_size=holdout_size,
        balanced_holdout=balanced_holdout,
        holdout_per_bucket=holdout_per_bucket,
    )

    solution_by_hash = {
        question_hash(row["question"]): row["solution"]
        for row in load_math_records(_coerce_sources(sources, None), seed)
    }

    records: list[dict[str, list[int]]] = []
    for row in rl_dataset:
        question = str(row["question"])
        solution = solution_by_hash.get(question_hash(question), "")
        assistant = {
            "role": "assistant",
            "content": f"<think>\n{solution.strip()}\n</think>\nFinal answer: {row['answer']}",
        }
        messages = [{"role": "user", "content": build_multi_math_prompt(question)}]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [*messages, assistant],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            full_text = tokenizer.apply_chat_template(
                [*messages, assistant],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_ids) <= len(prompt_ids):
            continue
        if max_length is not None and len(full_ids) > max_length:
            continue
        loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
        records.append({"input_ids": full_ids, "loss_mask": loss_mask})

    if not records:
        raise ValueError("No usable mixed math SFT records found")
    return Dataset.from_list(records)


def _tokenize_sft_row(
    prompt: str,
    completion: str,
    tokenizer,
    max_length: int | None,
) -> dict[str, list[int]] | None:
    messages = [{"role": "user", "content": prompt}]
    assistant = {"role": "assistant", "content": completion}
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [*messages, assistant],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = tokenizer.apply_chat_template(
            [*messages, assistant],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(full_ids) <= len(prompt_ids):
        return None
    if max_length is not None and len(full_ids) > max_length:
        return None
    loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return {"input_ids": full_ids, "loss_mask": loss_mask}


def get_deepseek_multi_math_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    seed: int = 1,
    limit: int | None = None,
    split_part: str | None = None,
    holdout_size: int = 128,
    balanced_holdout: bool = False,
    holdout_per_bucket: dict[str, int] | None = None,
    exclude_train_validation: bool = False,
    exclude_holdout_per_bucket: dict[str, int] | None = None,
    require_correct: bool = True,
    shuffle_records: bool = True,
    **_: Any,
) -> Dataset:
    """Load DeepSeek-generated mixed-math SFT JSONL and tokenize it for AReaL SFT."""
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"DeepSeek SFT data file does not exist: {jsonl_path}")
    if split_part is None:
        split_part = "validation" if split == "validation" else "train"
    if split_part not in {"train", "validation"}:
        raise ValueError("split_part must be 'train' or 'validation'")

    rows_by_hash: dict[str, dict[str, Any]] = {}
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            if require_correct and row.get("teacher_correct") is not True:
                continue
            question = str(row.get("question", ""))
            completion = str(row.get("completion", "")).strip()
            if not question or not completion:
                continue
            rows_by_hash.setdefault(question_hash(question), row)

    rows = list(rows_by_hash.values())
    if exclude_train_validation:
        exclude_hashes = balanced_train_validation_hashes(
            seed=seed,
            holdout_per_bucket=(
                GRPO_VALIDATION_HOLDOUT
                if exclude_holdout_per_bucket is None
                else _coerce_holdout_per_bucket(exclude_holdout_per_bucket)
            ),
        )
        rows = [row for row in rows if question_hash(str(row["question"])) not in exclude_hashes]

    if balanced_holdout:
        counts = _coerce_holdout_per_bucket(holdout_per_bucket)
        if not counts:
            raise ValueError("balanced_holdout requires non-empty holdout_per_bucket")
        holdout_hashes = _balanced_holdout_hashes(
            rows,
            counts,
            seed,
        )
        if split_part == "validation":
            rows = [row for row in rows if question_hash(str(row["question"])) in holdout_hashes]
        else:
            rows = [row for row in rows if question_hash(str(row["question"])) not in holdout_hashes]
    else:
        rng = random.Random(seed)
        if shuffle_records:
            rng.shuffle(rows)

        holdout = min(holdout_size, len(rows))
        if split_part == "validation":
            rows = rows[:holdout]
        else:
            rows = rows[holdout:]

    if shuffle_records:
        random.Random(seed).shuffle(rows)

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when set, got {limit}")
        rows = rows[:limit]

    records: list[dict[str, list[int]]] = []
    for row in rows:
        prompt = str(row.get("prompt") or build_multi_math_prompt(str(row["question"])))
        tokenized = _tokenize_sft_row(prompt, str(row["completion"]), tokenizer, max_length)
        if tokenized is not None:
            records.append(tokenized)

    if not records:
        raise ValueError(f"No usable DeepSeek mixed math SFT records found in {jsonl_path}")
    return Dataset.from_list(records)
