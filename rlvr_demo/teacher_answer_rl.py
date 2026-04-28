"""Teacher-answer reward learning for math reasoning.

The algorithm samples student reasoning on-policy, then scores each sampled
trajectory by the current student's log-probability of a teacher-provided answer
conditioned on the question and sampled reasoning. The teacher supplies only a
text answer; teacher reasoning is not used.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import InferenceEngine, ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.infra.rpc.rtensor import RTensor
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

from rlvr_demo.multi_math_data import (
    DEEPSEEK_VALIDATION_HOLDOUT,
    GRPO_VALIDATION_HOLDOUT,
    _balanced_holdout_hashes,
    _coerce_holdout_per_bucket,
    balanced_train_validation_hashes,
    build_multi_math_prompt,
    question_hash,
)

logger = logging.getLogger("TeacherAnswerRL")


TEACHER_ANSWER_REASONING_PROMPT_TEMPLATE = """Please solve the math problem step by step. Use <think> tags for your reasoning.

Problem:
{question}

IMPORTANT: Do not write `Final answer:` and do not write a separate final answer line. Output only the reasoning block:
<think>
Your step-by-step reasoning here...
</think>"""

ANSWER_PREFIX = "\nFinal answer:"
ANSWER_PREFIX_CANDIDATES = ("\nFinal answer:", "Final answer:")


def build_teacher_answer_reasoning_prompt(question: str) -> str:
    return TEACHER_ANSWER_REASONING_PROMPT_TEMPLATE.format(question=question.strip())


def build_teacher_answer_prompt(question: str, prompt_style: str) -> str:
    if prompt_style == "reasoning_only":
        return build_teacher_answer_reasoning_prompt(question)
    if prompt_style == "final_answer":
        return build_multi_math_prompt(question)
    raise ValueError("prompt_style must be one of: reasoning_only, final_answer")


def _tokenize_text(tokenizer: PreTrainedTokenizerFast, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _find_subsequence(tokens: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(tokens):
        return None
    last = len(tokens) - len(pattern) + 1
    for start in range(last):
        if tokens[start : start + len(pattern)] == pattern:
            return start
    return None


def _metadata_vector(ids: list[int], length: int) -> tuple[list[int], list[int]]:
    if len(ids) > length:
        ids = ids[:length]
    values = ids + [0] * (length - len(ids))
    mask = [1] * len(ids) + [0] * (length - len(ids))
    return values, mask


class TeacherAnswerRLWorkflow(RolloutWorkflow):
    """Roll out reasoning-only samples for teacher-answer RL."""

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
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.answer_prefix_ids = _tokenize_text(self.tokenizer, ANSWER_PREFIX)
        self.answer_prefix_candidates = [
            _tokenize_text(self.tokenizer, text) for text in ANSWER_PREFIX_CANDIDATES
        ]

    def _input_ids(self, data: dict[str, Any]) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                data["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        )

    @trace_session("teacher_answer_metadata")
    async def _teacher_answer_metadata(
        self,
        seq_len: int,
        data: dict[str, Any],
        context_len: int | None = None,
        prefix_ids: list[int] | None = None,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
        teacher_answer = str(data["teacher_answer"]).strip()
        answer_ids = _tokenize_text(self.tokenizer, f" {teacher_answer}")
        prefix_values, prefix_mask = _metadata_vector(
            self.answer_prefix_ids if prefix_ids is None else prefix_ids,
            seq_len,
        )
        answer_values, answer_mask = _metadata_vector(answer_ids, seq_len)
        if context_len is None:
            context_len = seq_len
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

        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        (
            prefix_values,
            prefix_mask,
            answer_values,
            answer_mask,
            context_mask,
        ) = await self._teacher_answer_metadata(len(seq), data)

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


class TeacherAnswerFinalPromptRLWorkflow(TeacherAnswerRLWorkflow):
    """Roll out eval-style answers while training only the reasoning/prefix span."""

    def _answer_context_output_len(self, output_tokens: list[int]) -> int | None:
        best: int | None = None
        best_len = 0
        for pattern in self.answer_prefix_candidates:
            start = _find_subsequence(output_tokens, pattern)
            if start is None:
                continue
            if best is None or start < best:
                best = start
                best_len = len(pattern)
        if best is None:
            return None
        return best + best_len

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

        answer_context_output_len = self._answer_context_output_len(resp.output_tokens)
        if answer_context_output_len is None:
            context_len = len(seq)
            prefix_ids = self.answer_prefix_ids
            optimized_output_len = resp.output_len
        else:
            context_len = resp.input_len + answer_context_output_len
            prefix_ids = []
            optimized_output_len = answer_context_output_len

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
        ) = await self._teacher_answer_metadata(
            len(seq),
            data,
            context_len=context_len,
            prefix_ids=prefix_ids,
        )

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


def _pad_2d(
    rows: list[torch.Tensor],
    pad_value: int | bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_len = max(int(row.numel()) for row in rows)
    out = torch.full((len(rows), max_len), pad_value, dtype=dtype, device=rows[0].device)
    for idx, row in enumerate(rows):
        out[idx, : row.numel()] = row.to(dtype=dtype)
    return out


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, RTensor):
        return value.to_local()
    return value


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _build_scoring_batch(traj: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    input_ids = _as_tensor(traj["input_ids"])
    attention_mask = _as_tensor(traj["attention_mask"]).bool()
    prefix_ids = _as_tensor(traj["teacher_answer_prefix_ids"])
    prefix_mask = _as_tensor(traj["teacher_answer_prefix_mask"]).bool()
    answer_ids = _as_tensor(traj["teacher_answer_ids"])
    answer_mask = _as_tensor(traj["teacher_answer_mask"]).bool()
    context_mask = _as_tensor(traj.get("teacher_context_mask", traj["attention_mask"])).bool()

    scoring_ids: list[torch.Tensor] = []
    scoring_loss_masks: list[torch.Tensor] = []
    for idx in range(input_ids.shape[0]):
        base_limit = int(attention_mask[idx].sum().item())
        base = input_ids[idx, :base_limit][context_mask[idx, :base_limit]]
        prefix = prefix_ids[idx][prefix_mask[idx]]
        answer = answer_ids[idx][answer_mask[idx]]
        seq = torch.cat([base, prefix, answer], dim=0)
        mask = torch.cat(
            [
                torch.zeros(base.numel() + prefix.numel(), dtype=torch.int32, device=seq.device),
                torch.ones(answer.numel(), dtype=torch.int32, device=seq.device),
            ],
            dim=0,
        )
        scoring_ids.append(seq)
        scoring_loss_masks.append(mask)

    attention_rows = [
        torch.ones(row.numel(), dtype=torch.bool, device=row.device)
        for row in scoring_ids
    ]
    padded_ids = _pad_2d(scoring_ids, 0, torch.int32)
    padded_loss_mask = _pad_2d(scoring_loss_masks, 0, torch.int32)
    padded_attention_mask = _pad_2d(attention_rows, False, torch.bool)
    return {
        "input_ids": padded_ids,
        "loss_mask": padded_loss_mask,
        "attention_mask": padded_attention_mask,
    }


def teacher_answer_reward_postprocess(
    trainer,
    rollout_batch: list[dict[str, Any]],
    global_step: int,
) -> None:
    """Replace placeholder rewards with teacher-answer log-probability rewards."""
    del global_step
    format_bonus = _env_float("TEACHER_ANSWER_FORMAT_BONUS", 0.0)
    length_penalty = _env_float("TEACHER_ANSWER_LENGTH_PENALTY", 0.0)
    max_new_tokens = max(float(getattr(trainer.config.gconfig, "max_new_tokens", 1) or 1), 1.0)
    scoring_batches = [_build_scoring_batch(traj) for traj in rollout_batch]
    scoring_logps = trainer.actor.compute_logp(scoring_batches)
    if scoring_logps is None:
        raise RuntimeError("actor.compute_logp returned None for teacher-answer scoring")

    all_rewards: list[torch.Tensor] = []
    all_adjusted_rewards: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []
    all_context_lengths: list[torch.Tensor] = []
    all_prefix_lengths: list[torch.Tensor] = []
    all_format_found: list[torch.Tensor] = []
    all_optimized_lengths: list[torch.Tensor] = []
    for traj, scoring_batch, logp in zip(rollout_batch, scoring_batches, scoring_logps):
        logp = _as_tensor(logp)
        answer_mask = torch.roll(
            scoring_batch["loss_mask"].to(logp.device).float(),
            shifts=-1,
            dims=-1,
        )
        lengths = answer_mask.sum(dim=-1).clamp(min=1.0)
        rewards = (logp * answer_mask).sum(dim=-1) / lengths
        context_mask = _as_tensor(traj.get("teacher_context_mask", traj["attention_mask"]))
        prefix_mask = _as_tensor(traj["teacher_answer_prefix_mask"])
        loss_mask = _as_tensor(traj["loss_mask"]).to(rewards.device).float()
        prefix_lengths = prefix_mask.float().sum(dim=-1).to(rewards.device)
        format_found = (prefix_lengths <= 0).float()
        optimized_lengths = loss_mask.sum(dim=-1)
        adjusted_rewards = (
            rewards
            + format_bonus * format_found
            - length_penalty * (optimized_lengths / max_new_tokens)
        )
        traj["rewards"] = adjusted_rewards.to(dtype=torch.float32)
        all_rewards.append(rewards.detach().float().cpu())
        all_adjusted_rewards.append(adjusted_rewards.detach().float().cpu())
        all_lengths.append(lengths.detach().float().cpu())
        all_context_lengths.append(context_mask.float().sum(dim=-1).detach().cpu())
        all_prefix_lengths.append(prefix_lengths.detach().float().cpu())
        all_format_found.append(format_found.detach().float().cpu())
        all_optimized_lengths.append(optimized_lengths.detach().float().cpu())

    rewards_cat = torch.cat(all_rewards)
    adjusted_rewards_cat = torch.cat(all_adjusted_rewards)
    lengths_cat = torch.cat(all_lengths)
    context_lengths_cat = torch.cat(all_context_lengths)
    prefix_lengths_cat = torch.cat(all_prefix_lengths)
    format_found_cat = torch.cat(all_format_found)
    optimized_lengths_cat = torch.cat(all_optimized_lengths)
    stats_tracker.denominator(
        teacher_answer_n_seqs=torch.ones_like(rewards_cat, dtype=torch.bool)
    )
    stats_tracker.stat(
        teacher_answer_logp=rewards_cat,
        teacher_answer_reward=adjusted_rewards_cat,
        teacher_answer_len=lengths_cat,
        teacher_context_len=context_lengths_cat,
        teacher_added_prefix_len=prefix_lengths_cat,
        teacher_format_found=format_found_cat,
        teacher_optimized_len=optimized_lengths_cat,
        denominator="teacher_answer_n_seqs",
    )


def _load_teacher_rows(
    path: str,
    require_correct: bool,
    teacher_answer_field: str,
) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Teacher-answer JSONL does not exist: {jsonl_path}")

    rows_by_hash: dict[str, dict[str, Any]] = {}
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if require_correct:
                if row.get("status") != "ok" or row.get("teacher_correct") is not True:
                    continue
            elif row.get("status") not in {"ok", "wrong"}:
                continue
            question = str(row.get("question", "")).strip()
            teacher_answer = str(row.get(teacher_answer_field, "")).strip()
            if not question or not teacher_answer:
                continue
            rows_by_hash.setdefault(question_hash(question), row)
    return list(rows_by_hash.values())


def get_deepseek_teacher_answer_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    seed: int = 1,
    limit: int | None = None,
    split_part: str | None = None,
    balanced_holdout: bool = True,
    holdout_per_bucket: dict[str, int] | None = None,
    exclude_train_validation: bool = True,
    exclude_holdout_per_bucket: dict[str, int] | None = None,
    require_correct: bool = True,
    teacher_answer_field: str = "teacher_prediction",
    shuffle_records: bool = True,
    prompt_style: str = "reasoning_only",
    **_: Any,
) -> Dataset:
    """Load teacher-answer rows for answer-likelihood RL."""
    if split_part is None:
        split_part = "validation" if split == "validation" else "train"
    if split_part not in {"train", "validation"}:
        raise ValueError("split_part must be 'train' or 'validation'")

    rows = _load_teacher_rows(path, require_correct, teacher_answer_field)

    if exclude_train_validation:
        exclude_hashes = balanced_train_validation_hashes(
            seed=seed,
            holdout_per_bucket=(
                GRPO_VALIDATION_HOLDOUT
                if exclude_holdout_per_bucket is None
                else _coerce_holdout_per_bucket(exclude_holdout_per_bucket)
            ),
        )
        rows = [
            row for row in rows if question_hash(str(row["question"])) not in exclude_hashes
        ]

    if balanced_holdout:
        counts = _coerce_holdout_per_bucket(
            DEEPSEEK_VALIDATION_HOLDOUT
            if holdout_per_bucket is None
            else holdout_per_bucket
        )
        holdout_hashes = _balanced_holdout_hashes(rows, counts, seed)
        if split_part == "validation":
            rows = [
                row for row in rows if question_hash(str(row["question"])) in holdout_hashes
            ]
        else:
            rows = [
                row for row in rows if question_hash(str(row["question"])) not in holdout_hashes
            ]

    if shuffle_records:
        import random

        random.Random(seed).shuffle(rows)

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when set, got {limit}")
        rows = rows[:limit]

    records: list[dict[str, Any]] = []
    for row in rows:
        question = str(row["question"]).strip()
        prompt = build_teacher_answer_prompt(question, prompt_style)
        messages = [{"role": "user", "content": prompt}]
        if max_length is not None:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if len(input_ids) > max_length:
                continue
        records.append(
            {
                "messages": messages,
                "question": question,
                "answer": str(row.get("answer", "")).strip(),
                "teacher_answer": str(row[teacher_answer_field]).strip(),
                "source": str(row.get("source", "")),
                "level": str(row.get("level", "")),
                "bucket": str(row.get("bucket", "")),
            }
        )

    if not records:
        raise ValueError(f"No usable teacher-answer RL records found in {path}")
    return Dataset.from_list(records)
