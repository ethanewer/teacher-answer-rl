"""Datasets for Terminus tool-calling terminal-agent recipes.

The expected input is JSONL produced by
``python -m terminal_agent_demo.terminus_tool_calling convert-corpus``.  Each
row contains a Qwen-compatible chat ``messages`` list where there is one real
user task message, assistant turns call ``execute_commands``, and observations
are role=tool messages.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from terminal_agent_demo.terminus_tool_calling import (
    EXECUTE_COMMANDS_TOOL,
    TERMINUS_TOOL_SYSTEM_PROMPT,
    TerminusToolPayloadError,
    parse_execute_commands_arguments,
)


_COMMAND_KEY_PATTERNS = (
    '"commands"',
    '\n"commands"',
    '\n  "commands"',
    '\n    "commands"',
    '\n      "commands"',
    '\r\n"commands"',
    '\r\n  "commands"',
    '\r\n    "commands"',
    '\r\n      "commands"',
)


class TerminalToolDataError(ValueError):
    """Raised when converted terminal-agent data is malformed."""


def _jsonl_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    return offsets


def _read_jsonl_row(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline()
    return json.loads(line.decode("utf-8"))


def _partition_items(
    items: list[Any],
    *,
    split_part: str | None,
    holdout_size: int,
    seed: int,
    shuffle_records: bool,
) -> list[Any]:
    selected = list(items)
    if split_part is not None:
        if split_part not in {"train", "validation"}:
            raise ValueError("split_part must be 'train' or 'validation'")
        shuffled = list(selected)
        random.Random(seed).shuffle(shuffled)
        holdout = set(shuffled[: min(holdout_size, len(shuffled))])
        if split_part == "validation":
            selected = [item for item in selected if item in holdout]
        else:
            selected = [item for item in selected if item not in holdout]
    if shuffle_records:
        random.Random(seed).shuffle(selected)
    return selected


def _limit_items(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return items
    if limit <= 0:
        raise ValueError(f"limit must be positive when set, got {limit}")
    return items[:limit]


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TerminalToolDataError("row has no messages list")
    if sum(1 for msg in messages if msg.get("role") == "user") != 1:
        raise TerminalToolDataError("converted trajectory must contain exactly one user message")
    copied = [dict(msg) for msg in messages]
    if copied and copied[0].get("role") == "system":
        copied[0]["content"] = TERMINUS_TOOL_SYSTEM_PROMPT
    return copied


def _apply_chat_template(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    enable_thinking: bool,
):
    return tokenizer.apply_chat_template(
        messages,
        tools=[EXECUTE_COMMANDS_TOOL],
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def _tokenize_sft_trajectory(
    row: dict[str, Any],
    tokenizer,
    *,
    max_length: int | None,
    truncate_long: bool,
    enable_thinking: bool,
) -> dict[str, list[int]] | None:
    messages = _messages(row)
    try:
        full_ids = list(
            _apply_chat_template(
                tokenizer,
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
        )
    except Exception as exc:
        raise TerminalToolDataError(f"failed to render converted trajectory: {exc}") from exc

    loss_mask = [0] * len(full_ids)
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_ids = list(
            _apply_chat_template(
                tokenizer,
                messages[:msg_idx],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
        turn_ids = list(
            _apply_chat_template(
                tokenizer,
                messages[: msg_idx + 1],
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
        )
        if full_ids[: len(turn_ids)] != turn_ids or turn_ids[: len(prefix_ids)] != prefix_ids:
            prefix_text = _apply_chat_template(
                tokenizer,
                messages[:msg_idx],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            turn_text = _apply_chat_template(
                tokenizer,
                messages[: msg_idx + 1],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            full_text = _apply_chat_template(
                tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            if not turn_text.startswith(prefix_text) or not full_text.startswith(turn_text):
                return None
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
            turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            if len(loss_mask) != len(full_ids):
                loss_mask = [0] * len(full_ids)
        for pos in range(len(prefix_ids), len(turn_ids)):
            if pos < len(loss_mask):
                loss_mask[pos] = 1

    if not any(loss_mask):
        return None
    if max_length is not None and len(full_ids) > max_length:
        if not truncate_long:
            return None
        full_ids = full_ids[:max_length]
        loss_mask = loss_mask[:max_length]
        if not any(loss_mask):
            return None
    return {"input_ids": full_ids, "loss_mask": loss_mask}


class TerminalToolSFTLazyDataset(TorchDataset):
    def __init__(
        self,
        path: Path,
        offsets: list[int],
        tokenizer,
        max_length: int | None,
        truncate_long: bool,
        enable_thinking: bool,
    ) -> None:
        self.path = path
        self.offsets = offsets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncate_long = truncate_long
        self.enable_thinking = enable_thinking

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        for offset_idx in range(len(self.offsets)):
            offset = self.offsets[(idx + offset_idx) % len(self.offsets)]
            row = _read_jsonl_row(self.path, offset)
            tokenized = _tokenize_sft_trajectory(
                row,
                self.tokenizer,
                max_length=self.max_length,
                truncate_long=self.truncate_long,
                enable_thinking=self.enable_thinking,
            )
            if tokenized is not None:
                return tokenized
        raise IndexError("No tokenizable Terminus tool-calling SFT trajectories found")


def _normalize_teacher_turn_filter(teacher_turn_filter: str | None) -> str:
    if teacher_turn_filter is None:
        return "all"
    value = str(teacher_turn_filter).strip().lower().replace("-", "_")
    if not value or value in {"0", "false", "none", "null", "off", "all", "any"}:
        return "all"
    if value in {"task_complete", "complete", "completed", "final"}:
        return "task_complete"
    if value in {"not_task_complete", "incomplete", "intermediate"}:
        return "not_task_complete"
    raise ValueError(
        "teacher_turn_filter must be one of all, task_complete, or not_task_complete"
    )


def _assistant_tool_payload(msg: dict[str, Any]):
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    call = tool_calls[0]
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    try:
        return parse_execute_commands_arguments(function.get("arguments") or "{}")
    except TerminusToolPayloadError:
        return None


def _assistant_matches_turn_filter(
    msg: dict[str, Any],
    teacher_turn_filter: str | None,
) -> bool:
    turn_filter = _normalize_teacher_turn_filter(teacher_turn_filter)
    if turn_filter == "all":
        return True
    payload = _assistant_tool_payload(msg)
    if payload is None:
        return False
    if turn_filter == "task_complete":
        return payload.task_complete
    if turn_filter == "not_task_complete":
        return not payload.task_complete
    raise AssertionError(f"unexpected teacher_turn_filter: {turn_filter}")


def _assistant_indices(
    row: dict[str, Any],
    *,
    teacher_turn_filter: str | None = None,
) -> list[int]:
    return [
        idx
        for idx, msg in enumerate(_messages(row))
        if msg.get("role") == "assistant"
        and msg.get("tool_calls")
        and _assistant_matches_turn_filter(msg, teacher_turn_filter)
    ]


def _line_start_for_index(text: str, idx: int) -> int:
    newline = max(text.rfind("\n", 0, idx), text.rfind("\r", 0, idx))
    return 0 if newline < 0 else newline + 1


def _split_tool_teacher_answer(
    tokenizer,
    messages: list[dict[str, Any]],
    assistant_idx: int,
    *,
    enable_thinking: bool,
    teacher_answer_start: str = "commands",
) -> tuple[str, str]:
    prefix_text = _apply_chat_template(
        tokenizer,
        messages[:assistant_idx],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    turn_text = _apply_chat_template(
        tokenizer,
        messages[: assistant_idx + 1],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    if not turn_text.startswith(prefix_text):
        raise TerminalToolDataError("assistant turn does not extend generation prompt")
    assistant_text = turn_text[len(prefix_text) :]
    command_starts = [
        start
        for pattern in _COMMAND_KEY_PATTERNS
        if (start := assistant_text.find(pattern)) >= 0
    ]
    if teacher_answer_start == "commands":
        if not command_starts:
            raise TerminalToolDataError("assistant tool call has no commands key")
        # Match the rollout workflow's stop-pattern semantics. Pretty-printed
        # arguments stop at the preceding newline/indentation pattern; compact
        # arguments stop directly at the "commands" key.
        teacher_start = min(command_starts)
    elif teacher_answer_start == "tool_call":
        teacher_start = assistant_text.find("<tool_call>")
        if teacher_start < 0:
            raise TerminalToolDataError("assistant turn has no tool_call block")
    else:
        raise TerminalToolDataError(
            f"unsupported teacher_answer_start: {teacher_answer_start}"
        )
    student_prefix = assistant_text[:teacher_start]
    teacher_answer = assistant_text[teacher_start:].rstrip()
    if not student_prefix.strip() or not teacher_answer.strip():
        raise TerminalToolDataError("empty student prefix or teacher answer")
    return student_prefix, teacher_answer


class TerminalToolTeacherAnswerLazyDataset(TorchDataset):
    def __init__(
        self,
        path: Path,
        refs: list[tuple[int, int]],
        tokenizer,
        max_length: int | None,
        enable_thinking: bool,
        teacher_answer_start: str,
    ) -> None:
        self.path = path
        self.refs = refs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.teacher_answer_start = teacher_answer_start

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for offset_idx in range(len(self.refs)):
            offset, assistant_idx = self.refs[(idx + offset_idx) % len(self.refs)]
            row = _read_jsonl_row(self.path, offset)
            messages = _messages(row)
            try:
                student_prefix, teacher_answer = _split_tool_teacher_answer(
                    self.tokenizer,
                    messages,
                    assistant_idx,
                    enable_thinking=self.enable_thinking,
                    teacher_answer_start=self.teacher_answer_start,
                )
            except TerminalToolDataError:
                continue
            history = messages[:assistant_idx]
            if self.max_length is not None:
                input_ids = _apply_chat_template(
                    self.tokenizer,
                    history,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                if len(input_ids) > self.max_length:
                    continue
            return {
                "messages": history,
                "teacher_answer": teacher_answer,
                "student_prefix": student_prefix,
                "source_task": row.get("source_task"),
                "source_trial_name": row.get("source_trial_name"),
                "source_model": row.get("source_model"),
                "turn_idx": assistant_idx,
            }
        raise IndexError("No usable Terminus tool-calling teacher-answer records found")


_TEACHER_REF_CACHE_VERSION = 2


def _teacher_turn_filter_slug(teacher_turn_filter: str | None) -> str:
    return _normalize_teacher_turn_filter(teacher_turn_filter)


def _default_teacher_refs_cache_path(
    path: Path,
    teacher_turn_filter: str | None,
) -> Path:
    slug = _teacher_turn_filter_slug(teacher_turn_filter)
    suffix = ".teacher_refs.v2.json" if slug == "all" else f".teacher_refs.v2.{slug}.json"
    return path.with_suffix(path.suffix + suffix)


def _resolve_teacher_refs_cache_path(
    path: Path,
    cache: str | bool | None,
    teacher_turn_filter: str | None,
) -> Path | None:
    if cache is None:
        return _default_teacher_refs_cache_path(path, teacher_turn_filter)
    if isinstance(cache, bool):
        return _default_teacher_refs_cache_path(path, teacher_turn_filter) if cache else None
    cache_str = str(cache).strip()
    if not cache_str or cache_str.lower() in {"0", "false", "none", "null", "off"}:
        return None
    if cache_str.lower() in {"1", "true", "auto", "default", "on"}:
        return _default_teacher_refs_cache_path(path, teacher_turn_filter)
    return Path(cache_str).expanduser()


def _load_teacher_refs_cache(
    path: Path,
    cache_path: Path,
    offsets: list[int],
    teacher_turn_filter: str | None,
) -> list[tuple[int, int]] | None:
    try:
        stat = path.stat()
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != _TEACHER_REF_CACHE_VERSION:
        return None
    if payload.get("path") != str(path):
        return None
    if payload.get("teacher_turn_filter", "all") != _normalize_teacher_turn_filter(
        teacher_turn_filter
    ):
        return None
    if payload.get("size") != stat.st_size or payload.get("mtime_ns") != stat.st_mtime_ns:
        return None
    selected_offsets = payload.get("offsets")
    if selected_offsets != offsets:
        return None
    refs = payload.get("refs")
    if not isinstance(refs, list):
        return None
    try:
        return [(int(offset), int(assistant_idx)) for offset, assistant_idx in refs]
    except (TypeError, ValueError):
        return None


def _write_teacher_refs_cache(
    path: Path,
    cache_path: Path,
    offsets: list[int],
    refs: list[tuple[int, int]],
    teacher_turn_filter: str | None,
) -> None:
    tmp_path: Path | None = None
    try:
        stat = path.stat()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _TEACHER_REF_CACHE_VERSION,
            "path": str(path),
            "teacher_turn_filter": _normalize_teacher_turn_filter(teacher_turn_filter),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "offsets": offsets,
            "refs": refs,
        }
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            delete=False,
        ) as handle:
            json.dump(payload, handle)
            handle.write("\n")
            tmp_path = Path(handle.name)
        tmp_path.replace(cache_path)
    except OSError:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _build_teacher_refs(
    path: Path,
    offsets: list[int],
    teacher_turn_filter: str | None,
) -> list[tuple[int, int]]:
    refs: list[tuple[int, int]] = []
    for offset in offsets:
        row = _read_jsonl_row(path, offset)
        for assistant_idx in _assistant_indices(
            row,
            teacher_turn_filter=teacher_turn_filter,
        ):
            refs.append((offset, assistant_idx))
    return refs


def _build_or_load_teacher_refs(
    path: Path,
    offsets: list[int],
    cache: str | bool | None,
    teacher_turn_filter: str | None,
) -> list[tuple[int, int]]:
    cache_path = _resolve_teacher_refs_cache_path(path, cache, teacher_turn_filter)
    if cache_path is not None:
        cached = _load_teacher_refs_cache(
            path,
            cache_path,
            offsets,
            teacher_turn_filter,
        )
        if cached is not None:
            return cached
    refs = _build_teacher_refs(path, offsets, teacher_turn_filter)
    if cache_path is not None:
        _write_teacher_refs_cache(path, cache_path, offsets, refs, teacher_turn_filter)
    return refs


def get_terminal_sft_dataset(
    path: str,
    split: str = "train",
    tokenizer=None,
    max_length: int | None = None,
    seed: int = 1,
    limit: int | None = None,
    limit_rows: int | None = None,
    split_part: str | None = None,
    holdout_size: int = 512,
    shuffle_records: bool = False,
    enable_thinking: bool = True,
    lazy_tokenize: bool = True,
    sft_format: str = "trajectory",
    truncate_long: bool = True,
    **_: Any,
):
    """Load converted Terminus tool-calling trajectories for SFT."""
    del split
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    if sft_format != "trajectory":
        raise ValueError("converted Terminus tool-calling SFT supports sft_format=trajectory")
    jsonl_path = Path(path).expanduser().resolve()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"converted JSONL does not exist: {jsonl_path}")
    offsets = _jsonl_offsets(jsonl_path)
    offsets = _partition_items(
        offsets,
        split_part=split_part,
        holdout_size=holdout_size,
        seed=seed,
        shuffle_records=shuffle_records,
    )
    offsets = _limit_items(offsets, limit_rows)
    offsets = _limit_items(offsets, limit)
    if not offsets:
        raise ValueError("No converted Terminus tool-calling SFT trajectories found")
    dataset = TerminalToolSFTLazyDataset(
        jsonl_path,
        offsets,
        tokenizer,
        max_length=max_length,
        truncate_long=truncate_long,
        enable_thinking=enable_thinking,
    )
    if lazy_tokenize:
        return dataset
    records = [dataset[idx] for idx in range(len(dataset))]
    return Dataset.from_list(records)


def get_terminal_teacher_answer_rl_dataset(
    path: str,
    split: str = "train",
    tokenizer=None,
    max_length: int | None = None,
    seed: int = 1,
    limit: int | None = None,
    limit_rows: int | None = None,
    split_part: str | None = None,
    holdout_size: int = 512,
    shuffle_records: bool = False,
    enable_thinking: bool = True,
    teacher_answer_start: str = "commands",
    teacher_turn_filter: str | None = None,
    teacher_refs_cache: str | bool | None = None,
    lazy_tokenize: bool = True,
    **_: Any,
):
    """Load converted Terminus tool-calling turns for teacher-answer RL."""
    del split
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    jsonl_path = Path(path).expanduser().resolve()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"converted JSONL does not exist: {jsonl_path}")
    offsets = _jsonl_offsets(jsonl_path)
    offsets = _limit_items(offsets, limit_rows)
    refs = _build_or_load_teacher_refs(
        jsonl_path,
        offsets,
        teacher_refs_cache,
        teacher_turn_filter,
    )
    refs = _partition_items(
        refs,
        split_part=split_part,
        holdout_size=holdout_size,
        seed=seed,
        shuffle_records=shuffle_records,
    )
    refs = _limit_items(refs, limit)
    if not refs:
        raise ValueError("No converted Terminus tool-calling teacher-answer records found")
    dataset = TerminalToolTeacherAnswerLazyDataset(
        jsonl_path,
        refs,
        tokenizer,
        max_length=max_length,
        enable_thinking=enable_thinking,
        teacher_answer_start=teacher_answer_start,
    )
    if lazy_tokenize:
        return dataset
    records = [dataset[idx] for idx in range(len(dataset))]
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
