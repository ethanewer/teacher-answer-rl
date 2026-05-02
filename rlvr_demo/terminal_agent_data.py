"""Dataset and parsing helpers for Terminus-style terminal-agent experiments."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator
from typing import Any

from datasets import Dataset, load_dataset


TERMINAL_CORPUS = "nvidia/Nemotron-Terminal-Corpus"
TERMINAL_CORPUS_CONFIG = "skill_based_medium"

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
_COMMAND_KEY_PATTERNS = (
    '"commands"',
    '\n"commands"',
    '\n  "commands"',
    '\n    "commands"',
    '\r\n"commands"',
    '\r\n  "commands"',
    '\r\n    "commands"',
)


class TerminalAnswerSplitError(ValueError):
    """Raised when a Terminus assistant message cannot be split at commands."""


def normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip()).casefold()


def stable_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def strip_thinking_block(content: str) -> str:
    """Remove explicit Qwen/Terminus thinking from a previous assistant turn."""
    return _THINK_RE.sub("", content, count=1).lstrip()


def _coerce_messages(conversations: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for raw in conversations:
        role = str(raw.get("role", "")).strip()
        content = str(raw.get("content", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _history_messages(
    messages: list[dict[str, str]],
    assistant_idx: int,
    strip_prior_assistant_thinking: bool,
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for msg in messages[:assistant_idx]:
        if strip_prior_assistant_thinking and msg["role"] == "assistant":
            history.append({**msg, "content": strip_thinking_block(msg["content"])})
        else:
            history.append(dict(msg))
    return history


def _line_start_for_index(text: str, index: int) -> int:
    return max(text.rfind("\n", 0, index), text.rfind("\r", 0, index)) + 1


def _find_top_level_json_key(text: str, key: str) -> int | None:
    """Return the index of a top-level JSON key in the first object in text.

    The Terminus responses often contain a thinking block followed by a JSON object.
    A plain substring search can hit prose inside ``analysis`` or ``plan``. This
    scanner only accepts string keys at object depth 1, outside quoted strings.
    """
    object_start = text.find("{")
    if object_start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    idx = object_start
    while idx < len(text):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
                raw_key = text[string_start + 1 : idx]
                lookahead = idx + 1
                while lookahead < len(text) and text[lookahead].isspace():
                    lookahead += 1
                if depth == 1 and lookahead < len(text) and text[lookahead] == ":":
                    try:
                        decoded = json.loads(text[string_start : idx + 1])
                    except json.JSONDecodeError:
                        decoded = raw_key
                    if decoded == key:
                        return string_start
            idx += 1
            continue

        if char == '"':
            in_string = True
            string_start = idx
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth <= 0:
                return None
        idx += 1
    return None


def split_terminus_teacher_answer(content: str) -> tuple[str, str]:
    """Split a Terminus response into student reasoning prefix and teacher answer.

    The split point is the top-level ``"commands"`` field in the first JSON object.
    The reasoning prefix retains everything before that key, including the trailing
    comma/newline after ``"plan"``. The teacher answer starts on the ``commands``
    line and includes ``task_complete`` and the closing brace.
    """
    command_key_idx = _find_top_level_json_key(content, "commands")
    if command_key_idx is None:
        raise TerminalAnswerSplitError("assistant response has no top-level commands key")

    teacher_start = _line_start_for_index(content, command_key_idx)
    student_prefix = content[:teacher_start]
    teacher_answer = content[teacher_start:].rstrip()

    if not student_prefix.strip() or not teacher_answer.strip():
        raise TerminalAnswerSplitError("empty student prefix or teacher answer")
    if '"commands"' not in teacher_answer:
        raise TerminalAnswerSplitError("teacher answer does not contain commands")
    return student_prefix, teacher_answer


def iter_terminal_turns(
    row: dict[str, Any],
    strip_prior_assistant_thinking: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield one trainable assistant turn per Terminal-Corpus row."""
    messages = _coerce_messages(row.get("conversations") or [])
    task = str(row.get("task", ""))
    episode = str(row.get("episode", ""))
    source_id = stable_hash(f"{task}\n{episode}\n{json.dumps(messages, sort_keys=True)}")

    for idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        history = _history_messages(messages, idx, strip_prior_assistant_thinking)
        if not history or history[-1]["role"] != "user":
            continue
        assistant_content = msg["content"].strip()
        if not assistant_content:
            continue
        yield {
            "messages": history,
            "assistant": assistant_content,
            "task": task,
            "episode": episode,
            "source_id": source_id,
            "turn_idx": idx,
            "agent": str(row.get("agent", "")),
            "model": str(row.get("model", "")),
            "model_provider": str(row.get("model_provider", "")),
        }


def _load_terminal_rows(
    path: str,
    name: str,
    split: str,
    seed: int,
    limit_rows: int | None,
    shuffle_rows: bool,
) -> list[dict[str, Any]]:
    dataset = load_dataset(path=path, name=name, split=split)
    if shuffle_rows:
        dataset = dataset.shuffle(seed=seed)
    if limit_rows is not None:
        if limit_rows <= 0:
            raise ValueError(f"limit_rows must be positive when set, got {limit_rows}")
        dataset = dataset.select(range(min(limit_rows, len(dataset))))
    return [dict(row) for row in dataset]


def _partition_turns(
    turns: list[dict[str, Any]],
    split_part: str | None,
    holdout_size: int,
    seed: int,
    shuffle_records: bool,
    shuffle_source_groups: bool = False,
) -> list[dict[str, Any]]:
    if split_part is None:
        selected = list(turns)
    else:
        if split_part not in {"train", "validation"}:
            raise ValueError("split_part must be 'train' or 'validation'")
        by_source: dict[str, dict[str, Any]] = {}
        for turn in turns:
            by_source.setdefault(str(turn["source_id"]), turn)
        source_ids = sorted(by_source)
        random.Random(seed).shuffle(source_ids)
        holdout_ids = set(source_ids[: min(holdout_size, len(source_ids))])
        if split_part == "validation":
            selected = [turn for turn in turns if turn["source_id"] in holdout_ids]
        else:
            selected = [turn for turn in turns if turn["source_id"] not in holdout_ids]

    if shuffle_records:
        random.Random(seed).shuffle(selected)
    elif shuffle_source_groups:
        grouped: dict[str, list[dict[str, Any]]] = {}
        group_order: list[str] = []
        for turn in selected:
            source_id = str(turn["source_id"])
            if source_id not in grouped:
                grouped[source_id] = []
                group_order.append(source_id)
            grouped[source_id].append(turn)
        random.Random(seed).shuffle(group_order)
        selected = [turn for source_id in group_order for turn in grouped[source_id]]
    return selected


def _tokenize_sft_turn(
    turn: dict[str, Any],
    tokenizer,
    max_length: int | None,
    enable_thinking: bool,
) -> dict[str, list[int]] | None:
    prompt_ids = tokenizer.apply_chat_template(
        turn["messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_ids = tokenizer.apply_chat_template(
        [*turn["messages"], {"role": "assistant", "content": turn["assistant"]}],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        prompt_text = tokenizer.apply_chat_template(
            turn["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        full_text = tokenizer.apply_chat_template(
            [*turn["messages"], {"role": "assistant", "content": turn["assistant"]}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) <= len(prompt_ids):
        return None
    if max_length is not None and len(full_ids) > max_length:
        return None
    loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return {"input_ids": list(full_ids), "loss_mask": loss_mask}


def _terminal_turns(
    path: str,
    name: str,
    split: str,
    seed: int,
    limit_rows: int | None,
    strip_prior_assistant_thinking: bool,
    shuffle_rows: bool,
) -> list[dict[str, Any]]:
    rows = _load_terminal_rows(path, name, split, seed, limit_rows, shuffle_rows)
    turns: list[dict[str, Any]] = []
    for row in rows:
        turns.extend(
            iter_terminal_turns(
                row,
                strip_prior_assistant_thinking=strip_prior_assistant_thinking,
            )
        )
    return turns


def get_terminal_sft_dataset(
    path: str = TERMINAL_CORPUS,
    split: str = "train",
    tokenizer=None,
    max_length: int | None = None,
    name: str = TERMINAL_CORPUS_CONFIG,
    seed: int = 1,
    limit: int | None = None,
    limit_rows: int | None = None,
    split_part: str | None = None,
    holdout_size: int = 512,
    shuffle_rows: bool = False,
    shuffle_records: bool = True,
    shuffle_source_groups: bool = False,
    strip_prior_assistant_thinking: bool = True,
    enable_thinking: bool = True,
    **_: Any,
) -> Dataset:
    """Load Terminal-Corpus turns and tokenize them for AReaL SFT."""
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    turns = _terminal_turns(
        path=path,
        name=name,
        split=split,
        seed=seed,
        limit_rows=limit_rows,
        strip_prior_assistant_thinking=strip_prior_assistant_thinking,
        shuffle_rows=shuffle_rows,
    )
    turns = _partition_turns(
        turns,
        split_part,
        holdout_size,
        seed,
        shuffle_records,
        shuffle_source_groups=shuffle_source_groups,
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when set, got {limit}")
        turns = turns[:limit]

    records: list[dict[str, list[int]]] = []
    for turn in turns:
        tokenized = _tokenize_sft_turn(turn, tokenizer, max_length, enable_thinking)
        if tokenized is not None:
            records.append(tokenized)

    if not records:
        raise ValueError("No usable terminal SFT records found")
    return Dataset.from_list(records)


def get_terminal_teacher_answer_rl_dataset(
    path: str = TERMINAL_CORPUS,
    split: str = "train",
    tokenizer=None,
    max_length: int | None = None,
    name: str = TERMINAL_CORPUS_CONFIG,
    seed: int = 1,
    limit: int | None = None,
    limit_rows: int | None = None,
    split_part: str | None = None,
    holdout_size: int = 512,
    shuffle_rows: bool = False,
    shuffle_records: bool = True,
    shuffle_source_groups: bool = False,
    strip_prior_assistant_thinking: bool = True,
    enable_thinking: bool = True,
    **_: Any,
) -> Dataset:
    """Load Terminal-Corpus turns for teacher-answer likelihood RL."""
    turns = _terminal_turns(
        path=path,
        name=name,
        split=split,
        seed=seed,
        limit_rows=limit_rows,
        strip_prior_assistant_thinking=strip_prior_assistant_thinking,
        shuffle_rows=shuffle_rows,
    )

    split_turns: list[dict[str, Any]] = []
    for turn in turns:
        try:
            student_prefix, teacher_answer = split_terminus_teacher_answer(
                str(turn["assistant"])
            )
        except TerminalAnswerSplitError:
            continue
        item = dict(turn)
        item["student_prefix"] = student_prefix
        item["teacher_answer"] = teacher_answer
        split_turns.append(item)

    split_turns = _partition_turns(
        split_turns,
        split_part,
        holdout_size,
        seed,
        shuffle_records,
        shuffle_source_groups=shuffle_source_groups,
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when set, got {limit}")
        split_turns = split_turns[:limit]

    records: list[dict[str, Any]] = []
    for turn in split_turns:
        messages = list(turn["messages"])
        if max_length is not None and tokenizer is not None:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            if len(input_ids) > max_length:
                continue
        records.append(
            {
                "messages": messages,
                "teacher_answer": turn["teacher_answer"],
                "student_prefix": turn["student_prefix"],
                "task": turn["task"],
                "episode": turn["episode"],
                "source_id": turn["source_id"],
                "turn_idx": turn["turn_idx"],
                "agent": turn["agent"],
                "model": turn["model"],
                "model_provider": turn["model_provider"],
            }
        )

    if not records:
        raise ValueError("No usable terminal teacher-answer RL records found")
    return Dataset.from_list(records)


def terminal_command_key_patterns(tokenizer) -> list[list[int]]:
    """Tokenize command-key patterns used to locate generated command payloads."""
    patterns: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for text in _COMMAND_KEY_PATTERNS:
        ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        if ids and ids not in seen:
            patterns.append(list(ids))
            seen.add(ids)
    return patterns
