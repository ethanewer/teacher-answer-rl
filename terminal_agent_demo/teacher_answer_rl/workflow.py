"""RL workflows for Terminus tool-calling terminal-agent data."""

from __future__ import annotations

import uuid
import json
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
    start = output_text.find("<tool_call>")
    if start < 0 or "</tool_call>" not in output_text[start:]:
        return False
    decoder = json.JSONDecoder()
    payload_text = output_text[start + len("<tool_call>") :].lstrip()
    try:
        payload, _ = decoder.raw_decode(payload_text)
    except Exception:
        return False
    if not isinstance(payload, dict) or payload.get("name") != "execute_commands":
        return False
    try:
        parse_execute_commands_arguments(payload.get("arguments", {}))
    except TerminusToolPayloadError:
        return False
    return True


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
        syntax_ok = _parse_generated_tool_call(output_text)
        max_new_tokens = max(float(getattr(self.gconfig, "max_new_tokens", 1) or 1), 1.0)
        syntax_base_reward = self.syntax_reward_weight * (
            self.valid_syntax_reward if syntax_ok else self.invalid_syntax_reward
        )
        syntax_thinking_length_penalty = (
            self.syntax_thinking_length_penalty_weight
            * (float(thinking_len) / max_new_tokens)
        )
        syntax_output_length_penalty = (
            self.syntax_output_length_penalty_weight
            * (float(output_len) / max_new_tokens)
        )
        syntax_reward = (
            syntax_base_reward
            - syntax_thinking_length_penalty
            - syntax_output_length_penalty
        )
        include_teacher_answer = (
            syntax_ok or not self.teacher_reward_requires_valid_syntax
        )
        teacher_answer = str(data["teacher_answer"]).rstrip()
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
        if not syntax_ok and self.drop_invalid_teacher_row_loss:
            teacher_mask = [0] * len(teacher_mask)
        seq_rows = [teacher_seq]
        logprob_rows = [teacher_logprobs]
        version_rows = [teacher_versions]
        masks = [teacher_mask]
        supervised_masks = [[0] * len(teacher_seq)]
        supervised_weight_masks = [[0.0] * len(teacher_seq)]
        rewards = [syntax_reward if self.syntax_reward_on_teacher_row else 0.0]

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
            teacher_syntax_ok=float(syntax_ok)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_answer_reward_enabled=float(include_teacher_answer)
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
    "TerminalToolTeacherAnswerRLWorkflow",
    "TerminalToolFullTurnTeacherAnswerRLWorkflow",
    "teacher_answer_reward_postprocess",
]
