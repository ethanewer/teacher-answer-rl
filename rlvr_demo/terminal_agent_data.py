"""Dataset and parsing helpers for Terminus-style terminal-agent experiments."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from torch.utils.data import Dataset as TorchDataset


TERMINAL_CORPUS = "nvidia/Nemotron-Terminal-Corpus"
TERMINAL_CORPUS_CONFIG = "skill_based_medium"
TERMINAL_CORPUS_FULL_MIX_CONFIGS = (
    "dataset_adapters",
    "skill_based_easy",
    "skill_based_medium",
    "skill_based_mixed",
)
TERMINAL_CORPUS_RELEASED_SYNTHETIC_CONFIGS = (
    "skill_based_easy",
    "skill_based_medium",
    "skill_based_mixed",
)

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
_TERMINAL_CORPUS_ADAPTER_FILES = (
    "dataset_adapters/math.parquet",
    "dataset_adapters/code.parquet",
    "dataset_adapters/swe.parquet",
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


def _normalize_config_names(name: str | Sequence[str]) -> list[str]:
    if isinstance(name, str):
        if "," in name:
            names = [part.strip() for part in name.split(",")]
        elif name == "full_mix":
            names = list(TERMINAL_CORPUS_FULL_MIX_CONFIGS)
        elif name in {
            "skill_based_all",
            "synthetic_released",
            "synthetic_released_no_filter",
        }:
            names = list(TERMINAL_CORPUS_RELEASED_SYNTHETIC_CONFIGS)
        else:
            names = [name]
    else:
        names = [str(part).strip() for part in name]
    names = [part for part in names if part]
    if not names:
        raise ValueError("At least one dataset config name is required")
    return names


def _iter_adapter_rows(
    path: str,
    limit_rows: int | None,
) -> Iterator[dict[str, Any]]:
    """Yield dataset-adapter rows directly from parquet files.

    `datasets==4.8.5` with the current `pyarrow` fails to materialize the
    adapter config because the split has nested list<struct<...>> columns in
    single large row groups. Reading batches through `pyarrow.parquet` avoids
    that conversion path and still consumes the released HF parquet files.
    """
    emitted = 0
    for file_path in _TERMINAL_CORPUS_ADAPTER_FILES:
        local_path = hf_hub_download(path, file_path, repo_type="dataset")
        parquet_file = pq.ParquetFile(local_path)
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                row["_dataset_config"] = "dataset_adapters"
                row["_dataset_file"] = file_path
                yield row
                emitted += 1
                if limit_rows is not None and emitted >= limit_rows:
                    return


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
    """Split a Terminus response into generated-prefix target and teacher answer.

    The split point is the top-level ``"commands"`` field in the first JSON object.
    The prefix is the span the student should generate from a fresh assistant turn
    during teacher-answer-RL. The teacher answer starts on the ``commands`` line
    and includes ``task_complete`` and the closing brace.
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
    name: str | Sequence[str],
    split: str,
    seed: int,
    limit_rows: int | None,
    shuffle_rows: bool,
) -> list[dict[str, Any]]:
    config_names = _normalize_config_names(name)
    if limit_rows is not None and limit_rows <= 0:
        raise ValueError(f"limit_rows must be positive when set, got {limit_rows}")

    rows: list[dict[str, Any]] = []
    remaining = limit_rows
    for config_name in config_names:
        if config_name == "dataset_adapters" and path == TERMINAL_CORPUS and split == "train":
            before = len(rows)
            rows.extend(_iter_adapter_rows(path, remaining))
            if remaining is not None:
                remaining -= len(rows) - before
        else:
            dataset = load_dataset(path=path, name=config_name, split=split)
            if shuffle_rows:
                dataset = dataset.shuffle(seed=seed)
            if remaining is not None:
                take = min(remaining, len(dataset))
                dataset = dataset.select(range(take))
                remaining -= take
            for row in dataset:
                item = dict(row)
                item["_dataset_config"] = config_name
                rows.append(item)
        if remaining == 0:
            break

    if shuffle_rows and len(config_names) > 1:
        random.Random(seed).shuffle(rows)
    return rows


def _iter_terminal_rows(
    path: str,
    name: str | Sequence[str],
    split: str,
    seed: int,
    limit_rows: int | None,
    shuffle_rows: bool,
) -> Iterator[dict[str, Any]]:
    """Yield released terminal rows without forcing a full intermediate list."""
    if shuffle_rows:
        yield from _load_terminal_rows(path, name, split, seed, limit_rows, shuffle_rows)
        return

    config_names = _normalize_config_names(name)
    if limit_rows is not None and limit_rows <= 0:
        raise ValueError(f"limit_rows must be positive when set, got {limit_rows}")

    remaining = limit_rows
    for config_name in config_names:
        if config_name == "dataset_adapters" and path == TERMINAL_CORPUS and split == "train":
            emitted = 0
            for row in _iter_adapter_rows(path, remaining):
                emitted += 1
                yield row
            if remaining is not None:
                remaining -= emitted
                if remaining == 0:
                    break
            continue

        dataset = load_dataset(path=path, name=config_name, split=split)
        take = len(dataset) if remaining is None else min(remaining, len(dataset))
        for idx in range(take):
            item = dict(dataset[idx])
            item["_dataset_config"] = config_name
            yield item
        if remaining is not None:
            remaining -= take
            if remaining == 0:
                break


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


def _partition_refs(
    refs: list[dict[str, Any]],
    split_part: str | None,
    holdout_size: int,
    seed: int,
    shuffle_records: bool,
    shuffle_source_groups: bool = False,
) -> list[dict[str, Any]]:
    if split_part is None:
        selected = list(refs)
    else:
        if split_part not in {"train", "validation"}:
            raise ValueError("split_part must be 'train' or 'validation'")
        source_ids = sorted({str(ref["source_id"]) for ref in refs})
        random.Random(seed).shuffle(source_ids)
        holdout_ids = set(source_ids[: min(holdout_size, len(source_ids))])
        if split_part == "validation":
            selected = [ref for ref in refs if ref["source_id"] in holdout_ids]
        else:
            selected = [ref for ref in refs if ref["source_id"] not in holdout_ids]

    if shuffle_records:
        random.Random(seed).shuffle(selected)
    elif shuffle_source_groups:
        grouped: dict[str, list[dict[str, Any]]] = {}
        group_order: list[str] = []
        for ref in selected:
            source_id = str(ref["source_id"])
            if source_id not in grouped:
                grouped[source_id] = []
                group_order.append(source_id)
            grouped[source_id].append(ref)
        random.Random(seed).shuffle(group_order)
        selected = [ref for source_id in group_order for ref in grouped[source_id]]
    return selected


def _prepare_terminal_rows_and_refs(
    path: str,
    name: str | Sequence[str],
    split: str,
    seed: int,
    limit_rows: int | None,
    shuffle_rows: bool,
    require_teacher_answer: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    refs: list[dict[str, Any]] = []
    for raw_idx, raw in enumerate(
        _iter_terminal_rows(path, name, split, seed, limit_rows, shuffle_rows), start=1
    ):
        messages = _coerce_messages(raw.get("conversations") or [])
        if not messages:
            continue
        task = str(raw.get("task", ""))
        episode = str(raw.get("episode", ""))
        source_id = stable_hash(f"{task}\n{episode}\n{json.dumps(messages, sort_keys=True)}")
        row_idx = len(rows)
        rows.append(
            {
                "messages": messages,
                "task": task,
                "episode": episode,
                "source_id": source_id,
                "agent": str(raw.get("agent", "")),
                "model": str(raw.get("model", "")),
                "model_provider": str(raw.get("model_provider", "")),
            }
        )
        for assistant_idx, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            if assistant_idx == 0 or messages[assistant_idx - 1]["role"] != "user":
                continue
            if not msg["content"].strip():
                continue
            if require_teacher_answer:
                try:
                    split_terminus_teacher_answer(msg["content"].strip())
                except TerminalAnswerSplitError:
                    continue
            refs.append(
                {
                    "row_idx": row_idx,
                    "assistant_idx": assistant_idx,
                    "source_id": source_id,
                }
            )
        if raw_idx % 25000 == 0:
            print(
                f"[terminal_agent_data] prepared {raw_idx} rows, "
                f"{len(refs)} assistant turns",
                flush=True,
            )
    print(
        f"[terminal_agent_data] prepared {len(rows)} usable rows, "
        f"{len(refs)} assistant turns",
        flush=True,
    )
    return rows, refs


class _TerminalTurnDataset(TorchDataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        refs: list[dict[str, Any]],
        strip_prior_assistant_thinking: bool,
    ) -> None:
        self.rows = rows
        self.refs = refs
        self.strip_prior_assistant_thinking = strip_prior_assistant_thinking

    def __len__(self) -> int:
        return len(self.refs)

    def _turn_for_ref(self, ref: dict[str, Any]) -> dict[str, Any]:
        row = self.rows[int(ref["row_idx"])]
        assistant_idx = int(ref["assistant_idx"])
        messages = row["messages"]
        return {
            "messages": _history_messages(
                messages,
                assistant_idx,
                self.strip_prior_assistant_thinking,
            ),
            "assistant": messages[assistant_idx]["content"].strip(),
            "task": row["task"],
            "episode": row["episode"],
            "source_id": row["source_id"],
            "turn_idx": assistant_idx,
            "agent": row["agent"],
            "model": row["model"],
            "model_provider": row["model_provider"],
        }


class TerminalSFTLazyDataset(_TerminalTurnDataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        refs: list[dict[str, Any]],
        tokenizer,
        max_length: int | None,
        strip_prior_assistant_thinking: bool,
        enable_thinking: bool,
    ) -> None:
        super().__init__(rows, refs, strip_prior_assistant_thinking)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        for offset in range(len(self.refs)):
            ref = self.refs[(idx + offset) % len(self.refs)]
            tokenized = _tokenize_sft_turn(
                self._turn_for_ref(ref),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                enable_thinking=self.enable_thinking,
            )
            if tokenized is not None:
                return tokenized
        raise IndexError("No tokenizable terminal SFT records found")


class TerminalTeacherAnswerLazyDataset(_TerminalTurnDataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        refs: list[dict[str, Any]],
        strip_prior_assistant_thinking: bool,
    ) -> None:
        super().__init__(rows, refs, strip_prior_assistant_thinking)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for offset in range(len(self.refs)):
            turn = self._turn_for_ref(self.refs[(idx + offset) % len(self.refs)])
            try:
                student_prefix, teacher_answer = split_terminus_teacher_answer(
                    str(turn["assistant"])
                )
            except TerminalAnswerSplitError:
                continue
            return {
                "messages": turn["messages"],
                "teacher_answer": teacher_answer,
                "student_prefix": student_prefix,
                "task": turn["task"],
                "episode": turn["episode"],
                "source_id": turn["source_id"],
                "turn_idx": turn["turn_idx"],
                "agent": turn["agent"],
                "model": turn["model"],
                "model_provider": turn["model_provider"],
            }
        raise IndexError("No usable terminal teacher-answer RL records found")


def _prepare_terminal_rows_only(
    path: str,
    name: str | Sequence[str],
    split: str,
    seed: int,
    limit_rows: int | None,
    shuffle_rows: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_idx, raw in enumerate(
        _iter_terminal_rows(path, name, split, seed, limit_rows, shuffle_rows), start=1
    ):
        messages = _coerce_messages(raw.get("conversations") or [])
        if not any(msg["role"] == "assistant" for msg in messages):
            continue
        task = str(raw.get("task", ""))
        episode = str(raw.get("episode", ""))
        rows.append(
            {
                "messages": messages,
                "task": task,
                "episode": episode,
                "source_id": stable_hash(
                    f"{task}\n{episode}\n{json.dumps(messages, sort_keys=True)}"
                ),
                "agent": str(raw.get("agent", "")),
                "model": str(raw.get("model", "")),
                "model_provider": str(raw.get("model_provider", "")),
            }
        )
        if raw_idx % 25000 == 0:
            print(
                f"[terminal_agent_data] prepared {raw_idx} rows, "
                f"{len(rows)} usable trajectories",
                flush=True,
            )
    print(
        f"[terminal_agent_data] prepared {len(rows)} usable trajectories",
        flush=True,
    )
    return rows


def _partition_rows(
    rows: list[dict[str, Any]],
    split_part: str | None,
    holdout_size: int,
    seed: int,
    shuffle_records: bool,
) -> list[dict[str, Any]]:
    if split_part is None:
        selected = list(rows)
    else:
        if split_part not in {"train", "validation"}:
            raise ValueError("split_part must be 'train' or 'validation'")
        source_ids = sorted({str(row["source_id"]) for row in rows})
        random.Random(seed).shuffle(source_ids)
        holdout_ids = set(source_ids[: min(holdout_size, len(source_ids))])
        if split_part == "validation":
            selected = [row for row in rows if row["source_id"] in holdout_ids]
        else:
            selected = [row for row in rows if row["source_id"] not in holdout_ids]

    if shuffle_records:
        random.Random(seed).shuffle(selected)
    return selected


def _encode_text(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _tokenize_sft_trajectory(
    row: dict[str, Any],
    tokenizer,
    max_length: int | None,
    truncate_long: bool,
) -> dict[str, list[int]] | None:
    """Tokenize a complete Terminus trajectory with all assistant spans supervised.

    Qwen3's HF chat template strips thinking content from prior assistant turns.
    For paper-scale SFT, we need one trajectory row with every assistant response
    supervised, so this serializer mirrors Qwen's ChatML tokens while preserving
    the released assistant content and applying the loss mask only to assistant
    content plus the assistant end-of-message token.
    """
    input_ids: list[int] = []
    loss_mask: list[int] = []
    for msg in row["messages"]:
        role = msg["role"]
        content = str(msg["content"])
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        if role == "assistant":
            header_ids = _encode_text(tokenizer, "<|im_start|>assistant\n")
            body_ids = _encode_text(tokenizer, content.strip() + "<|im_end|>\n")
            input_ids.extend(header_ids)
            loss_mask.extend([0] * len(header_ids))
            input_ids.extend(body_ids)
            loss_mask.extend([1] * len(body_ids))
        elif role == "tool":
            # Match Qwen's no-tools fallback for tool messages closely enough for
            # the terminal corpus. Current released rows are user/assistant only.
            text = f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            ids = _encode_text(tokenizer, text)
            input_ids.extend(ids)
            loss_mask.extend([0] * len(ids))
        else:
            ids = _encode_text(tokenizer, f"<|im_start|>{role}\n{content}<|im_end|>\n")
            input_ids.extend(ids)
            loss_mask.extend([0] * len(ids))

    if not any(loss_mask):
        return None
    if max_length is not None and len(input_ids) > max_length:
        if not truncate_long:
            return None
        input_ids = input_ids[:max_length]
        loss_mask = loss_mask[:max_length]
        if not any(loss_mask):
            return None
    return {"input_ids": input_ids, "loss_mask": loss_mask}


class TerminalSFTTrajectoryLazyDataset(TorchDataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        max_length: int | None,
        truncate_long: bool,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncate_long = truncate_long

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        for offset in range(len(self.rows)):
            row = self.rows[(idx + offset) % len(self.rows)]
            tokenized = _tokenize_sft_trajectory(
                row,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                truncate_long=self.truncate_long,
            )
            if tokenized is not None:
                return tokenized
        raise IndexError("No tokenizable terminal SFT trajectories found")


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
    name: str | Sequence[str],
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
    name: str | Sequence[str] = TERMINAL_CORPUS_CONFIG,
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
    lazy_tokenize: bool = False,
    sft_format: str = "turn",
    truncate_long: bool = False,
    **_: Any,
) -> Dataset:
    """Load Terminal-Corpus turns and tokenize them for AReaL SFT."""
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    if sft_format not in {"turn", "trajectory"}:
        raise ValueError("sft_format must be 'turn' or 'trajectory'")
    if sft_format == "trajectory":
        rows = _prepare_terminal_rows_only(
            path=path,
            name=name,
            split=split,
            seed=seed,
            limit_rows=limit_rows,
            shuffle_rows=shuffle_rows,
        )
        rows = _partition_rows(rows, split_part, holdout_size, seed, shuffle_records)
        if limit is not None:
            if limit <= 0:
                raise ValueError(f"limit must be positive when set, got {limit}")
            rows = rows[:limit]
        if not rows:
            raise ValueError("No usable terminal SFT trajectories found")
        return TerminalSFTTrajectoryLazyDataset(
            rows=rows,
            tokenizer=tokenizer,
            max_length=max_length,
            truncate_long=truncate_long,
        )

    if lazy_tokenize:
        rows, refs = _prepare_terminal_rows_and_refs(
            path=path,
            name=name,
            split=split,
            seed=seed,
            limit_rows=limit_rows,
            shuffle_rows=shuffle_rows,
        )
        refs = _partition_refs(
            refs,
            split_part,
            holdout_size,
            seed,
            shuffle_records,
            shuffle_source_groups=shuffle_source_groups,
        )
        if limit is not None:
            if limit <= 0:
                raise ValueError(f"limit must be positive when set, got {limit}")
            refs = refs[:limit]
        if not refs:
            raise ValueError("No usable terminal SFT records found")
        return TerminalSFTLazyDataset(
            rows=rows,
            refs=refs,
            tokenizer=tokenizer,
            max_length=max_length,
            strip_prior_assistant_thinking=strip_prior_assistant_thinking,
            enable_thinking=enable_thinking,
        )

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
    name: str | Sequence[str] = TERMINAL_CORPUS_CONFIG,
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
    lazy_tokenize: bool = False,
    **_: Any,
) -> Dataset:
    """Load Terminal-Corpus turns for teacher-answer likelihood RL."""
    if lazy_tokenize:
        rows, refs = _prepare_terminal_rows_and_refs(
            path=path,
            name=name,
            split=split,
            seed=seed,
            limit_rows=limit_rows,
            shuffle_rows=shuffle_rows,
            require_teacher_answer=True,
        )
        refs = _partition_refs(
            refs,
            split_part,
            holdout_size,
            seed,
            shuffle_records,
            shuffle_source_groups=shuffle_source_groups,
        )
        if limit is not None:
            if limit <= 0:
                raise ValueError(f"limit must be positive when set, got {limit}")
            refs = refs[:limit]
        if not refs:
            raise ValueError("No usable terminal teacher-answer RL records found")
        return TerminalTeacherAnswerLazyDataset(
            rows=rows,
            refs=refs,
            strip_prior_assistant_thinking=strip_prior_assistant_thinking,
        )

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
