"""RL workflows for Terminus tool-calling terminal-agent data."""

from __future__ import annotations

import uuid
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
from terminal_agent_demo.terminus_tool_calling import EXECUTE_COMMANDS_TOOL


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


def _find_subsequence(tokens: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(tokens):
        return None
    for start in range(len(tokens) - len(pattern) + 1):
        if tokens[start : start + len(pattern)] == pattern:
            return start
    return None


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
    ) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
        teacher_answer = str(data["teacher_answer"]).rstrip()
        answer_ids = _tokenize_text(self.tokenizer, teacher_answer)
        prefix_values, prefix_mask = _metadata_vector([], seq_len)
        answer_values, answer_mask = _metadata_vector(answer_ids, seq_len)
        context_len = max(0, min(context_len, seq_len))
        context_mask = [1] * context_len + [0] * (seq_len - context_len)
        return prefix_values, prefix_mask, answer_values, answer_mask, context_mask

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
            "teacher_context_mask": torch.tensor(context_mask, dtype=torch.bool),
        }
        return {key: value.unsqueeze(0) for key, value in res.items()}


__all__ = [
    "TerminalToolTeacherAnswerRLWorkflow",
    "teacher_answer_reward_postprocess",
]
