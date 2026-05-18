"""RL workflows for Terminus tool-calling terminal-agent data."""

from __future__ import annotations

import uuid
import json
import re
from collections import Counter
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.utils import stats_tracker
from areal.utils.perf_tracer import atrace_session_phase, session_context

from terminal_agent_demo.terminal_agent_data import terminal_command_key_patterns
from terminal_agent_demo.teacher_answer_rl.reward import (
    _metadata_vector,
    _tokenize_text,
    teacher_answer_reward_postprocess,
)
from terminal_agent_demo.terminus_tool_calling import (
    EXECUTE_COMMANDS_TOOL,
    ParsedPayload,
    TerminusToolPayloadError,
    parse_execute_commands_arguments,
)


COMMAND_STOP_STRINGS = (
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


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _find_subsequence(tokens: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(tokens):
        return None
    for start in range(len(tokens) - len(pattern) + 1):
        if tokens[start : start + len(pattern)] == pattern:
            return start
    return None


def _pad_rows(rows: list[list[Any]], pad_value: Any) -> list[list[Any]]:
    max_len = max(len(row) for row in rows)
    return [row + [pad_value] * (max_len - len(row)) for row in rows]


def _tool_call_start_token_len(tokenizer, output_text: str, output_len: int) -> int:
    start = output_text.find("<tool_call>")
    if start < 0:
        return output_len
    return min(len(tokenizer.encode(output_text[:start], add_special_tokens=False)), output_len)


def _tool_call_end_token_len(tokenizer, output_text: str, output_len: int) -> int | None:
    start = output_text.find("<tool_call>")
    if start < 0:
        return None
    end = output_text.find("</tool_call>", start)
    if end < 0:
        return None
    end += len("</tool_call>")
    return min(len(tokenizer.encode(output_text[:end], add_special_tokens=False)), output_len)


def _parse_generated_tool_call(output_text: str) -> bool:
    return _parse_tool_call_payload(output_text) is not None


def _parse_tool_call_payload(output_text: str) -> ParsedPayload | None:
    start = output_text.find("<tool_call>")
    if start < 0 or "</tool_call>" not in output_text[start:]:
        return None
    decoder = json.JSONDecoder()
    payload_text = output_text[start + len("<tool_call>") :].lstrip()
    try:
        payload, _ = decoder.raw_decode(payload_text)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("name") != "execute_commands":
        return None
    try:
        return parse_execute_commands_arguments(payload.get("arguments", {}))
    except TerminusToolPayloadError:
        return None


def _parse_partial_teacher_payload(text: str) -> ParsedPayload | None:
    full_payload = _parse_tool_call_payload(text)
    if full_payload is not None:
        return full_payload

    start = text.find('"commands"')
    if start < 0:
        return None
    fragment = text[start:]
    end = fragment.find("</tool_call>")
    if end >= 0:
        fragment = fragment[:end]
    fragment = fragment.strip()
    decoder = json.JSONDecoder()
    for candidate in ("{" + fragment, '{"analysis":"","plan":",' + fragment):
        try:
            payload, _ = decoder.raw_decode(candidate)
        except Exception:
            continue
        try:
            return parse_execute_commands_arguments(payload)
        except TerminusToolPayloadError:
            continue
    return None


_COMMAND_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+-]+")
_KEYSTROKES_RE = re.compile(r'"keystrokes"\s*:\s*"((?:\\.|[^"\\])*)"')


def _command_text_from_payload(payload: ParsedPayload | None) -> str:
    if payload is None:
        return ""
    return "\n".join(
        command.keystrokes.strip()
        for command in payload.commands
        if command.keystrokes.strip()
    )


def _partial_command_list(text: str) -> list[str]:
    """Extract keystroke strings even when the surrounding tool JSON is partial."""
    payload = _parse_partial_teacher_payload(text)
    if payload is not None:
        return [
            command.keystrokes
            for command in payload.commands
            if command.keystrokes.strip()
        ]

    commands: list[str] = []
    for match in _KEYSTROKES_RE.finditer(text):
        encoded = '"' + match.group(1) + '"'
        try:
            decoded = json.loads(encoded)
        except Exception:
            decoded = match.group(1)
        decoded = str(decoded).strip()
        if decoded:
            commands.append(decoded)
    return commands


def _partial_command_text(text: str) -> str:
    return "\n".join(command.strip() for command in _partial_command_list(text))


def _normal_command_text(command: str) -> str:
    return " ".join(command.strip().split())


def _previous_command_list(data: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for message in data.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for raw_call in message.get("tool_calls") or []:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            if function.get("name") != "execute_commands":
                continue
            try:
                payload = parse_execute_commands_arguments(
                    function.get("arguments", {})
                )
            except TerminusToolPayloadError:
                continue
            commands.extend(
                command.keystrokes
                for command in payload.commands
                if command.keystrokes.strip()
            )
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                commands.extend(_partial_command_list(content))
    return commands


def _command_repeat_fraction(
    generated_commands: list[str],
    previous_commands: list[str],
) -> float:
    generated = [
        _normal_command_text(command)
        for command in generated_commands
        if command.strip()
    ]
    if not generated:
        return 0.0
    previous = {
        _normal_command_text(command)
        for command in previous_commands
        if command.strip()
    }
    if not previous:
        return 0.0
    return sum(1.0 for command in generated if command in previous) / len(generated)


def _command_token_f1(generated_text: str, teacher_text: str) -> float:
    if not generated_text or not teacher_text:
        return 0.0
    generated_tokens = Counter(
        token.lower() for token in _COMMAND_TOKEN_RE.findall(generated_text)
    )
    teacher_tokens = Counter(
        token.lower() for token in _COMMAND_TOKEN_RE.findall(teacher_text)
    )
    if not generated_tokens or not teacher_tokens:
        return 0.0
    overlap = sum(
        min(generated_tokens[token], teacher_tokens[token])
        for token in generated_tokens.keys() & teacher_tokens.keys()
    )
    if overlap <= 0:
        return 0.0
    precision = overlap / sum(generated_tokens.values())
    recall = overlap / sum(teacher_tokens.values())
    return float(2.0 * precision * recall / max(precision + recall, 1e-12))


def _teacher_answer_score_mask(
    answer_ids: list[int],
    command_key_patterns: list[list[int]],
    score_mode: str,
) -> list[int]:
    mode = score_mode.strip().lower().replace("-", "_")
    if mode in {"", "all", "full"}:
        return [1] * len(answer_ids)
    if mode not in {"commands_onward", "commands"}:
        raise ValueError(f"unsupported teacher_answer_score_mode: {score_mode}")
    starts = [
        start
        for pattern in command_key_patterns
        if (start := _find_subsequence(answer_ids, pattern)) is not None
    ]
    if not starts:
        return [1] * len(answer_ids)
    start = min(starts)
    return [0] * start + [1] * (len(answer_ids) - start)


def _optional_tools(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = data.get("tools")
    return tools if isinstance(tools, list) and tools else None


class GenericToolActionLikelihoodWorkflow(RolloutWorkflow):
    """Domain-general teacher-action likelihood reward.

    This workflow samples only the student's prefix before the next serialized
    tool call, then lets reward postprocessing score the corpus teacher action
    under that sampled prefix. It does not parse tool arguments or use
    terminal-specific action similarity rewards.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = True,
        teacher_answer_score_mode: str = "all",
        generic_stop_strings: list[str] | tuple[str, ...] | str | None = None,
    ):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            self.tokenizer = load_hf_tokenizer(self.tokenizer)
        if generic_stop_strings is None:
            stop_strings = ["<tool_call>"]
        elif isinstance(generic_stop_strings, str):
            stop_strings = [generic_stop_strings]
        else:
            stop_strings = [str(item) for item in generic_stop_strings]
        stop = list(gconfig.stop or [])
        for stop_string in stop_strings:
            if stop_string and stop_string not in stop:
                stop.append(stop_string)
        self.gconfig = gconfig.new(stop=stop).new_with_stop_and_pad_token_ids(
            self.tokenizer
        )
        self.enable_thinking = enable_thinking
        self.teacher_answer_score_mode = teacher_answer_score_mode

    def _input_ids(self, data: dict[str, Any]) -> list[int]:
        kwargs: dict[str, Any] = dict(
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        tools = _optional_tools(data)
        if tools is not None:
            kwargs["tools"] = tools
        return list(self.tokenizer.apply_chat_template(data["messages"], **kwargs))

    def _teacher_answer_metadata(
        self,
        seq_len: int,
        data: dict[str, Any],
        context_len: int,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
        teacher_answer = str(data["teacher_answer"]).rstrip()
        answer_ids = _tokenize_text(self.tokenizer, teacher_answer)
        score_mode = self.teacher_answer_score_mode.strip().lower().replace("-", "_")
        if score_mode in {"", "all", "full"}:
            score_ids = [1] * len(answer_ids)
        else:
            raise ValueError(
                "GenericToolActionLikelihoodWorkflow only supports "
                f"teacher_answer_score_mode=all/full, got {self.teacher_answer_score_mode!r}"
            )
        prefix_values, prefix_mask = _metadata_vector([], seq_len)
        answer_values, answer_mask = _metadata_vector(answer_ids, seq_len)
        score_values, _ = _metadata_vector(score_ids, seq_len)
        context_len = max(0, min(context_len, seq_len))
        context_mask = [1] * context_len + [0] * (seq_len - context_len)
        return (
            prefix_values,
            prefix_mask,
            answer_values,
            answer_mask,
            score_values,
            context_mask,
        )

    @session_context()
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        input_ids = self._input_ids(data)
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        versions = [-1] * resp.input_len + resp.output_versions
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        context_len = resp.input_len + resp.output_len

        (
            prefix_values,
            prefix_mask,
            answer_values,
            answer_mask,
            score_values,
            context_mask,
        ) = self._teacher_answer_metadata(len(seq), data, context_len=context_len)

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=0.0)

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(0.0, dtype=torch.float32),
            "teacher_answer_prefix_ids": torch.tensor(prefix_values, dtype=torch.int32),
            "teacher_answer_prefix_mask": torch.tensor(prefix_mask, dtype=torch.bool),
            "teacher_answer_ids": torch.tensor(answer_values, dtype=torch.int32),
            "teacher_answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
            "teacher_answer_score_mask": torch.tensor(score_values, dtype=torch.bool),
            "teacher_context_mask": torch.tensor(context_mask, dtype=torch.bool),
        }
        return {key: value.unsqueeze(0) for key, value in res.items()}


class TerminalToolTeacherAnswerRLWorkflow(RolloutWorkflow):
    """Teacher-answer RL for the ``execute_commands`` tool-call payload.

    The rollout samples the student's current assistant prefix up to the
    ``commands`` key inside the tool arguments. PPO optimizes only that sampled
    prefix. The reward pass appends the converted teacher continuation from the
    corpus, starting at ``commands`` and including ``task_complete`` plus the
    assistant end marker, then uses its average log-probability as the scalar
    reward for the sampled prefix.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = True,
        teacher_answer_score_mode: str = "all",
    ):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            self.tokenizer = load_hf_tokenizer(self.tokenizer)
        stop = list(gconfig.stop or [])
        for command_stop in COMMAND_STOP_STRINGS:
            if command_stop not in stop:
                stop.append(command_stop)
        self.gconfig = gconfig.new(stop=stop).new_with_stop_and_pad_token_ids(
            self.tokenizer
        )
        self.enable_thinking = enable_thinking
        self.teacher_answer_score_mode = teacher_answer_score_mode
        self.command_key_patterns = terminal_command_key_patterns(self.tokenizer)

    def _input_ids(self, data: dict[str, Any]) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                data["messages"],
                tools=[EXECUTE_COMMANDS_TOOL],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )

    def _commands_output_start(self, output_tokens: list[int]) -> int | None:
        starts = [
            start
            for pattern in self.command_key_patterns
            if (start := _find_subsequence(output_tokens, pattern)) is not None
        ]
        if not starts:
            return None
        return min(starts)

    async def _teacher_answer_metadata(
        self,
        seq_len: int,
        data: dict[str, Any],
        context_len: int,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
        teacher_answer = str(data["teacher_answer"]).rstrip()
        answer_ids = _tokenize_text(self.tokenizer, teacher_answer)
        score_ids = _teacher_answer_score_mask(
            answer_ids,
            self.command_key_patterns,
            self.teacher_answer_score_mode,
        )
        prefix_values, prefix_mask = _metadata_vector([], seq_len)
        answer_values, answer_mask = _metadata_vector(answer_ids, seq_len)
        score_values, _ = _metadata_vector(score_ids, seq_len)
        context_len = max(0, min(context_len, seq_len))
        context_mask = [1] * context_len + [0] * (seq_len - context_len)
        return (
            prefix_values,
            prefix_mask,
            answer_values,
            answer_mask,
            score_values,
            context_mask,
        )

    @session_context()
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        input_ids = self._input_ids(data)
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        command_start = self._commands_output_start(resp.output_tokens)
        optimized_output_len = resp.output_len if command_start is None else command_start
        context_len = resp.input_len + optimized_output_len

        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        versions = [-1] * resp.input_len + resp.output_versions
        loss_mask = (
            [0] * resp.input_len
            + [1] * optimized_output_len
            + [0] * max(resp.output_len - optimized_output_len, 0)
        )

        (
            prefix_values,
            prefix_mask,
            answer_values,
            answer_mask,
            score_values,
            context_mask,
        ) = await self._teacher_answer_metadata(len(seq), data, context_len=context_len)

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=0.0)

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(0.0, dtype=torch.float32),
            "teacher_answer_prefix_ids": torch.tensor(prefix_values, dtype=torch.int32),
            "teacher_answer_prefix_mask": torch.tensor(prefix_mask, dtype=torch.bool),
            "teacher_answer_ids": torch.tensor(answer_values, dtype=torch.int32),
            "teacher_answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
            "teacher_answer_score_mask": torch.tensor(score_values, dtype=torch.bool),
            "teacher_context_mask": torch.tensor(context_mask, dtype=torch.bool),
        }
        return {key: value.unsqueeze(0) for key, value in res.items()}


class TerminalToolFullTurnTeacherAnswerRLWorkflow(RolloutWorkflow):
    """Full-turn TA-RL with separate thinking and syntax training views.

    The student samples a complete assistant turn. The teacher-answer reward is
    computed on the corpus teacher continuation after the sampled thinking
    prefix. A second, low-weight view can train the whole sampled turn for basic
    tool-call syntax without letting syntax rewards dominate reasoning.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = True,
        syntax_reward_weight: float = 0.1,
        valid_syntax_reward: float = 0.0,
        invalid_syntax_reward: float = -0.1,
        emit_syntax_view: bool = True,
        emit_valid_syntax_view: bool = False,
        teacher_answer_score_mode: str = "all",
        teacher_loss_span: str = "thinking",
        syntax_reward_on_teacher_row: bool = False,
        supervise_teacher_answer: bool = False,
        teacher_answer_supervised_weight: float = 0.0,
        teacher_answer_supervised_score_mode: str | None = None,
        teacher_answer_supervised_max_prefix_tokens: int | None = None,
        teacher_reward_requires_valid_syntax: bool = False,
        drop_invalid_teacher_row_loss: bool = False,
        syntax_thinking_length_penalty_weight: float = 0.0,
        syntax_output_length_penalty_weight: float = 0.0,
        syntax_length_penalty_requires_invalid: bool = False,
        teacher_command_match_reward_weight: float = 0.0,
        teacher_command_presence_reward_weight: float = 0.0,
        teacher_command_newline_reward_weight: float = 0.0,
        teacher_completion_match_reward_weight: float = 0.0,
        teacher_completion_requires_valid_syntax: bool = False,
        teacher_empty_completion_reward_weight: float = 0.0,
        teacher_command_count_reward_weight: float = 0.0,
        teacher_repeated_command_penalty_weight: float = 0.0,
        tool_call_scaffold_reward_weight: float = 0.0,
        invalid_teacher_loss_span: str = "default",
    ):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            self.tokenizer = load_hf_tokenizer(self.tokenizer)
        self.gconfig = gconfig.new(stop=[]).new_with_stop_and_pad_token_ids(
            self.tokenizer
        )
        self.enable_thinking = enable_thinking
        self.syntax_reward_weight = float(syntax_reward_weight)
        self.valid_syntax_reward = float(valid_syntax_reward)
        self.invalid_syntax_reward = float(invalid_syntax_reward)
        self.emit_syntax_view = bool(emit_syntax_view)
        self.emit_valid_syntax_view = bool(emit_valid_syntax_view)
        self.teacher_answer_score_mode = teacher_answer_score_mode
        self.teacher_loss_span = teacher_loss_span.strip().lower().replace("-", "_")
        if self.teacher_loss_span not in {"thinking", "tool_call", "full_turn"}:
            raise ValueError(f"unsupported teacher_loss_span: {teacher_loss_span}")
        self.syntax_reward_on_teacher_row = bool(syntax_reward_on_teacher_row)
        self.supervise_teacher_answer = bool(supervise_teacher_answer)
        self.teacher_answer_supervised_weight = float(teacher_answer_supervised_weight)
        self.teacher_answer_supervised_score_mode = (
            teacher_answer_supervised_score_mode or teacher_answer_score_mode
        )
        if teacher_answer_supervised_max_prefix_tokens is None:
            self.teacher_answer_supervised_max_prefix_tokens = None
        else:
            self.teacher_answer_supervised_max_prefix_tokens = max(
                0, int(teacher_answer_supervised_max_prefix_tokens)
            )
        self.teacher_reward_requires_valid_syntax = bool(
            teacher_reward_requires_valid_syntax
        )
        self.drop_invalid_teacher_row_loss = bool(drop_invalid_teacher_row_loss)
        self.syntax_thinking_length_penalty_weight = float(
            syntax_thinking_length_penalty_weight
        )
        self.syntax_output_length_penalty_weight = float(
            syntax_output_length_penalty_weight
        )
        self.syntax_length_penalty_requires_invalid = bool(
            syntax_length_penalty_requires_invalid
        )
        self.teacher_command_match_reward_weight = float(
            teacher_command_match_reward_weight
        )
        self.teacher_command_presence_reward_weight = float(
            teacher_command_presence_reward_weight
        )
        self.teacher_command_newline_reward_weight = float(
            teacher_command_newline_reward_weight
        )
        self.teacher_completion_match_reward_weight = float(
            teacher_completion_match_reward_weight
        )
        self.teacher_completion_requires_valid_syntax = bool(
            teacher_completion_requires_valid_syntax
        )
        self.teacher_empty_completion_reward_weight = float(
            teacher_empty_completion_reward_weight
        )
        self.teacher_command_count_reward_weight = float(
            teacher_command_count_reward_weight
        )
        self.teacher_repeated_command_penalty_weight = float(
            teacher_repeated_command_penalty_weight
        )
        self.tool_call_scaffold_reward_weight = float(
            tool_call_scaffold_reward_weight
        )
        self.invalid_teacher_loss_span = (
            invalid_teacher_loss_span.strip().lower().replace("-", "_")
        )
        if self.invalid_teacher_loss_span not in {
            "default",
            "none",
            "thinking",
            "tool_call",
            "full_turn",
            "output",
        }:
            raise ValueError(
                f"unsupported invalid_teacher_loss_span: {invalid_teacher_loss_span}"
            )
        self.command_key_patterns = terminal_command_key_patterns(self.tokenizer)

    def _input_ids(self, data: dict[str, Any]) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                data["messages"],
                tools=[EXECUTE_COMMANDS_TOOL],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )

    def _metadata_rows(
        self,
        seq_len: int,
        data: dict[str, Any],
        context_lens: list[int],
        n_rows: int,
        include_teacher_answer: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher_answer = str(data["teacher_answer"]).rstrip()
        answer_ids = _tokenize_text(self.tokenizer, teacher_answer)
        score_ids = _teacher_answer_score_mask(
            answer_ids,
            self.command_key_patterns,
            self.teacher_answer_score_mode,
        )
        empty_values, empty_mask = _metadata_vector([], seq_len)
        answer_values, answer_mask = _metadata_vector(answer_ids, seq_len)
        score_values, _ = _metadata_vector(score_ids, seq_len)

        prefix_values = [empty_values for _ in range(n_rows)]
        prefix_masks = [empty_mask for _ in range(n_rows)]
        if include_teacher_answer:
            answer_values_rows = [answer_values] + [empty_values for _ in range(n_rows - 1)]
            answer_mask_rows = [answer_mask] + [empty_mask for _ in range(n_rows - 1)]
            answer_score_mask_rows = [score_values] + [empty_mask for _ in range(n_rows - 1)]
        else:
            answer_values_rows = [empty_values for _ in range(n_rows)]
            answer_mask_rows = [empty_mask for _ in range(n_rows)]
            answer_score_mask_rows = [empty_mask for _ in range(n_rows)]
        context_rows = []
        for context_len in context_lens:
            context_len = max(0, min(context_len, seq_len))
            context_rows.append([1] * context_len + [0] * (seq_len - context_len))
        return (
            torch.tensor(prefix_values, dtype=torch.int32),
            torch.tensor(prefix_masks, dtype=torch.bool),
            torch.tensor(answer_values_rows, dtype=torch.int32),
            torch.tensor(answer_mask_rows, dtype=torch.bool),
            torch.tensor(answer_score_mask_rows, dtype=torch.bool),
            torch.tensor(context_rows, dtype=torch.bool),
        )

    @session_context()
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        input_ids = self._input_ids(data)
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        output_tokens_no_stop = resp.output_tokens_without_stop
        output_text = self.tokenizer.decode(
            output_tokens_no_stop,
            skip_special_tokens=False,
        )
        thinking_len = _tool_call_start_token_len(
            self.tokenizer,
            output_text,
            resp.output_len,
        )
        full_turn_len = _tool_call_end_token_len(
            self.tokenizer,
            output_text,
            resp.output_len,
        )
        output_keep_len = full_turn_len if full_turn_len is not None else resp.output_len
        output_tokens = resp.output_tokens[:output_keep_len]
        output_logprobs = resp.output_logprobs[:output_keep_len]
        output_versions = resp.output_versions[:output_keep_len]
        output_len = len(output_tokens)
        thinking_len = min(thinking_len, output_len)
        context_len = resp.input_len + thinking_len
        generated_payload = _parse_tool_call_payload(output_text)
        syntax_ok = generated_payload is not None
        teacher_answer = str(data["teacher_answer"]).rstrip()
        teacher_payload = _parse_partial_teacher_payload(teacher_answer)
        teacher_commands = _partial_command_list(teacher_answer)
        generated_commands = _partial_command_list(output_text)
        previous_commands = _previous_command_list(data)
        teacher_command_text = "\n".join(
            command.strip() for command in teacher_commands
        )
        generated_command_text = "\n".join(
            command.strip() for command in generated_commands
        )
        command_overlap = _command_token_f1(generated_command_text, teacher_command_text)
        command_match_reward = (
            self.teacher_command_match_reward_weight * command_overlap
        )
        teacher_has_command = bool(teacher_command_text)
        generated_has_command = bool(generated_command_text)
        command_presence_reward = 0.0
        if teacher_has_command and self.teacher_command_presence_reward_weight:
            command_presence_reward = self.teacher_command_presence_reward_weight * (
                1.0 if generated_has_command else -1.0
            )
        command_newline_reward = 0.0
        if teacher_has_command and self.teacher_command_newline_reward_weight:
            executable_commands = [
                command for command in generated_commands if command.strip()
            ]
            if executable_commands:
                newline_ratio = sum(
                    1.0 for command in executable_commands if command.endswith("\n")
                ) / len(executable_commands)
                command_newline_reward = (
                    self.teacher_command_newline_reward_weight * newline_ratio
                )
            else:
                command_newline_reward = -self.teacher_command_newline_reward_weight
        teacher_task_complete = bool(
            teacher_payload.task_complete if teacher_payload is not None else False
        )
        generated_task_complete = bool(
            generated_payload.task_complete if generated_payload is not None else False
        )
        completion_match_reward = 0.0
        if self.teacher_completion_match_reward_weight and (
            syntax_ok or not self.teacher_completion_requires_valid_syntax
        ):
            completion_match_reward = self.teacher_completion_match_reward_weight * (
                1.0 if generated_task_complete == teacher_task_complete else -1.0
            )
        empty_completion_reward = 0.0
        if self.teacher_empty_completion_reward_weight and teacher_task_complete:
            teacher_empty_completion = not teacher_has_command
            generated_empty_completion = generated_task_complete and not generated_has_command
            if teacher_empty_completion:
                empty_completion_reward = self.teacher_empty_completion_reward_weight * (
                    1.0 if generated_empty_completion else -1.0
                )
        command_count_reward = 0.0
        if self.teacher_command_count_reward_weight and teacher_has_command:
            teacher_count = len([command for command in teacher_commands if command.strip()])
            generated_count = len([command for command in generated_commands if command.strip()])
            if teacher_count > 0:
                count_similarity = 1.0 - min(
                    abs(generated_count - teacher_count) / max(float(teacher_count), 1.0),
                    1.0,
                )
                command_count_reward = (
                    self.teacher_command_count_reward_weight * count_similarity
                )
        command_repeat_fraction = _command_repeat_fraction(
            generated_commands,
            previous_commands,
        )
        repeated_command_penalty = (
            self.teacher_repeated_command_penalty_weight
            * command_repeat_fraction
        )
        tool_call_start = output_text.find("<tool_call>")
        if tool_call_start >= 0:
            tool_call_fragment = output_text[tool_call_start:]
        else:
            tool_call_fragment = output_text
        tool_call_scaffold_progress = (
            float(tool_call_start >= 0)
            + float("</tool_call>" in tool_call_fragment)
            + float("execute_commands" in tool_call_fragment)
            + float('"commands"' in tool_call_fragment)
        ) / 4.0
        tool_call_scaffold_reward = self.tool_call_scaffold_reward_weight * (
            2.0 * tool_call_scaffold_progress - 1.0
        )
        max_new_tokens = max(float(getattr(self.gconfig, "max_new_tokens", 1) or 1), 1.0)
        syntax_base_reward = self.syntax_reward_weight * (
            self.valid_syntax_reward if syntax_ok else self.invalid_syntax_reward
        )
        use_syntax_length_penalty = (
            not self.syntax_length_penalty_requires_invalid or not syntax_ok
        )
        syntax_thinking_length_penalty = (
            self.syntax_thinking_length_penalty_weight
            * (float(thinking_len) / max_new_tokens)
            * float(use_syntax_length_penalty)
        )
        syntax_output_length_penalty = (
            self.syntax_output_length_penalty_weight
            * (float(output_len) / max_new_tokens)
            * float(use_syntax_length_penalty)
        )
        syntax_reward = (
            syntax_base_reward
            - syntax_thinking_length_penalty
            - syntax_output_length_penalty
        )
        include_teacher_answer = (
            syntax_ok or not self.teacher_reward_requires_valid_syntax
        )
        supervised_answer_ids: list[int] = []
        supervised_answer_mask: list[int] = []
        if self.supervise_teacher_answer and self.teacher_answer_supervised_weight != 0.0:
            supervised_answer_ids = _tokenize_text(self.tokenizer, teacher_answer)
            supervised_answer_mask = _teacher_answer_score_mask(
                supervised_answer_ids,
                self.command_key_patterns,
                self.teacher_answer_supervised_score_mode,
            )

        if self.teacher_loss_span == "thinking":
            teacher_seq = resp.input_tokens + output_tokens[:thinking_len]
            teacher_logprobs = [0.0] * resp.input_len + output_logprobs[:thinking_len]
            teacher_versions = [-1] * resp.input_len + output_versions[:thinking_len]
            teacher_mask = [0] * resp.input_len + [1] * thinking_len
        else:
            teacher_seq = resp.input_tokens + output_tokens
            teacher_logprobs = [0.0] * resp.input_len + output_logprobs
            teacher_versions = [-1] * resp.input_len + output_versions
            if self.teacher_loss_span == "tool_call":
                teacher_mask = [0] * (resp.input_len + thinking_len) + [1] * (
                    output_len - thinking_len
                )
            else:
                teacher_mask = [0] * resp.input_len + [1] * output_len
        if not syntax_ok and self.invalid_teacher_loss_span != "default":
            if self.invalid_teacher_loss_span == "none":
                teacher_mask = [0] * len(teacher_mask)
            elif self.invalid_teacher_loss_span == "thinking":
                teacher_mask = (
                    [0] * resp.input_len
                    + [1] * thinking_len
                    + [0] * max(output_len - thinking_len, 0)
                )
            elif self.invalid_teacher_loss_span == "tool_call":
                teacher_mask = [0] * (resp.input_len + thinking_len) + [1] * (
                    output_len - thinking_len
                )
            else:
                teacher_mask = [0] * resp.input_len + [1] * output_len
        if not syntax_ok and self.drop_invalid_teacher_row_loss:
            teacher_mask = [0] * len(teacher_mask)
        seq_rows = [teacher_seq]
        logprob_rows = [teacher_logprobs]
        version_rows = [teacher_versions]
        masks = [teacher_mask]
        supervised_masks = [[0] * len(teacher_seq)]
        supervised_weight_masks = [[0.0] * len(teacher_seq)]
        rewards = [
            command_match_reward
            + command_presence_reward
            + command_newline_reward
            + completion_match_reward
            + empty_completion_reward
            + command_count_reward
            - repeated_command_penalty
            + tool_call_scaffold_reward
            + (syntax_reward if self.syntax_reward_on_teacher_row else 0.0)
        ]

        supervised_token_count = 0
        if supervised_answer_ids:
            supervised_prefix_len = thinking_len
            if self.teacher_answer_supervised_max_prefix_tokens is not None:
                supervised_prefix_len = min(
                    supervised_prefix_len,
                    self.teacher_answer_supervised_max_prefix_tokens,
                )
            supervised_context_len = resp.input_len + supervised_prefix_len
            max_tokens = int(getattr(self.gconfig, "max_tokens", 0) or 0)
            if max_tokens > 0:
                answer_budget = max(max_tokens - supervised_context_len, 0)
                supervised_answer_ids = supervised_answer_ids[:answer_budget]
                supervised_answer_mask = supervised_answer_mask[:answer_budget]
            supervised_token_count = int(sum(supervised_answer_mask))
            if supervised_answer_ids:
                supervised_seq = (
                    resp.input_tokens
                    + output_tokens[:supervised_prefix_len]
                    + supervised_answer_ids
                )
                supervised_logprobs = (
                    [0.0] * resp.input_len
                    + output_logprobs[:supervised_prefix_len]
                    + [0.0] * len(supervised_answer_ids)
                )
                supervised_versions = (
                    [-1] * resp.input_len
                    + output_versions[:supervised_prefix_len]
                    + [-1] * len(supervised_answer_ids)
                )
                seq_rows.append(supervised_seq)
                logprob_rows.append(supervised_logprobs)
                version_rows.append(supervised_versions)
                masks.append([0] * len(supervised_seq))
                supervised_masks.append(
                    [0] * supervised_context_len + supervised_answer_mask
                )
                supervised_weight_masks.append(
                    [0.0] * supervised_context_len
                    + [
                        self.teacher_answer_supervised_weight * float(v)
                        for v in supervised_answer_mask
                    ]
                )
                rewards.append(0.0)

        if self.emit_syntax_view and (self.emit_valid_syntax_view or not syntax_ok):
            syntax_seq = resp.input_tokens + output_tokens
            syntax_logprobs = [0.0] * resp.input_len + output_logprobs
            syntax_versions = [-1] * resp.input_len + output_versions
            seq_rows.append(syntax_seq)
            logprob_rows.append(syntax_logprobs)
            version_rows.append(syntax_versions)
            masks.append([0] * resp.input_len + [1] * output_len)
            supervised_masks.append([0] * len(syntax_seq))
            supervised_weight_masks.append([0.0] * len(syntax_seq))
            rewards.append(syntax_reward)
            if self.tool_call_scaffold_reward_weight:
                rewards[-1] += tool_call_scaffold_reward

        n_rows = len(masks)
        row_lens = [len(row) for row in seq_rows]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        seq_rows = _pad_rows(seq_rows, pad_id)
        logprob_rows = _pad_rows(logprob_rows, 0.0)
        version_rows = _pad_rows(version_rows, -1)
        masks = _pad_rows(masks, 0)
        supervised_masks = _pad_rows(supervised_masks, 0)
        supervised_weight_masks = _pad_rows(supervised_weight_masks, 0.0)
        attention_rows = [[1] * row_len for row_len in row_lens]
        attention_rows = _pad_rows(attention_rows, 0)
        context_lens = [resp.input_len + thinking_len] + row_lens[1:]
        seq_len = len(seq_rows[0])
        (
            prefix_values,
            prefix_mask,
            answer_values,
            answer_mask,
            score_mask,
            context_mask,
        ) = self._metadata_rows(
            seq_len,
            data,
            context_lens=context_lens,
            n_rows=n_rows,
            include_teacher_answer=include_teacher_answer,
        )

        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=float(syntax_reward)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_syntax_base_reward=float(syntax_base_reward)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_syntax_thinking_length_penalty=float(
                syntax_thinking_length_penalty
            )
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_syntax_output_length_penalty=float(
                syntax_output_length_penalty
            )
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_syntax_length_penalty_enabled=float(use_syntax_length_penalty)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_syntax_ok=float(syntax_ok)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_answer_reward_enabled=float(include_teacher_answer)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_command_match_reward=float(command_match_reward)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_command_presence_reward=float(command_presence_reward)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_command_newline_reward=float(command_newline_reward)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_command_overlap=float(command_overlap)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_command_parse_ok=float(bool(teacher_command_text))
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_generated_command_found=float(generated_has_command)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_task_complete=float(teacher_task_complete)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_generated_task_complete=float(generated_task_complete)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_completion_match_reward=float(completion_match_reward)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_completion_requires_valid_syntax=float(
                self.teacher_completion_requires_valid_syntax
            )
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_empty_completion_reward=float(empty_completion_reward)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_command_count_reward=float(command_count_reward)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_repeated_command_fraction=float(command_repeat_fraction)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_repeated_command_penalty=float(repeated_command_penalty)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_tool_call_scaffold_progress=float(tool_call_scaffold_progress)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_tool_call_scaffold_reward=float(tool_call_scaffold_reward)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_invalid_row_loss_dropped=float(
                (not syntax_ok) and self.drop_invalid_teacher_row_loss
            )
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_thinking_len=float(thinking_len)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_output_trimmed=float(output_keep_len < resp.output_len)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_generation_len=float(resp.output_len)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_kept_generation_len=float(output_len)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_optimized_row_len=float(sum(teacher_mask))
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_loss_span_full_turn=float(self.teacher_loss_span == "full_turn")
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_loss_span_tool_call=float(self.teacher_loss_span == "tool_call")
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_invalid_loss_span_output=float(
                self.invalid_teacher_loss_span in {"full_turn", "output"}
            )
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_supervised_tokens=float(supervised_token_count)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_supervised_prefix_len=float(
                min(
                    thinking_len,
                    self.teacher_answer_supervised_max_prefix_tokens
                    if self.teacher_answer_supervised_max_prefix_tokens is not None
                    else thinking_len,
                )
            )
        )

        return {
            "input_ids": torch.tensor(seq_rows, dtype=torch.int32),
            "loss_mask": torch.tensor(masks, dtype=torch.int32),
            "supervised_loss_mask": torch.tensor(supervised_masks, dtype=torch.int32),
            "supervised_loss_weight_mask": torch.tensor(
                supervised_weight_masks,
                dtype=torch.float32,
            ),
            "logprobs": torch.tensor(logprob_rows, dtype=torch.float32),
            "versions": torch.tensor(version_rows, dtype=torch.int32),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.bool),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "teacher_answer_prefix_ids": prefix_values,
            "teacher_answer_prefix_mask": prefix_mask,
            "teacher_answer_ids": answer_values,
            "teacher_answer_mask": answer_mask,
            "teacher_answer_score_mask": score_mask,
            "teacher_context_mask": context_mask,
        }


__all__ = [
    "GenericToolActionLikelihoodWorkflow",
    "TerminalToolTeacherAnswerRLWorkflow",
    "TerminalToolFullTurnTeacherAnswerRLWorkflow",
    "teacher_answer_reward_postprocess",
]
