"""Teacher-answer reward postprocessing used by terminal-agent recipes."""

from __future__ import annotations

import os
from typing import Any

import torch

from areal.infra.rpc.rtensor import RTensor
from areal.utils import stats_tracker


def _tokenize_text(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _metadata_vector(ids: list[int], length: int) -> tuple[list[int], list[int]]:
    if len(ids) > length:
        ids = ids[:length]
    values = ids + [0] * (length - len(ids))
    mask = [1] * len(ids) + [0] * (length - len(ids))
    return values, mask


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


def _build_scoring_batch(
    traj: dict[str, torch.Tensor],
    max_tokens: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    input_ids = _as_tensor(traj["input_ids"])
    attention_mask = _as_tensor(traj["attention_mask"]).bool()
    prefix_ids = _as_tensor(traj["teacher_answer_prefix_ids"])
    prefix_mask = _as_tensor(traj["teacher_answer_prefix_mask"]).bool()
    answer_ids = _as_tensor(traj["teacher_answer_ids"])
    answer_mask = _as_tensor(traj["teacher_answer_mask"]).bool()
    context_mask = _as_tensor(traj.get("teacher_context_mask", traj["attention_mask"])).bool()

    scoring_ids: list[torch.Tensor] = []
    scoring_loss_masks: list[torch.Tensor] = []
    scoring_lengths: list[torch.Tensor] = []
    scoring_original_lengths: list[torch.Tensor] = []
    scoring_dropped_tokens: list[torch.Tensor] = []
    scoring_truncated: list[torch.Tensor] = []
    for idx in range(input_ids.shape[0]):
        base_limit = int(attention_mask[idx].sum().item())
        base = input_ids[idx, :base_limit][context_mask[idx, :base_limit]]
        prefix = prefix_ids[idx][prefix_mask[idx]]
        answer = answer_ids[idx][answer_mask[idx]]
        original_len = base.numel() + prefix.numel() + answer.numel()
        dropped_tokens = 0
        if max_tokens is not None and max_tokens > 0 and original_len > max_tokens:
            answer_budget = max(max_tokens - prefix.numel(), 0)
            if answer.numel() > answer_budget:
                dropped_tokens += answer.numel() - answer_budget
                answer = answer[:answer_budget]
            base_budget = max(max_tokens - prefix.numel() - answer.numel(), 0)
            if base.numel() > base_budget:
                dropped_tokens += base.numel() - base_budget
                base = base[-base_budget:] if base_budget > 0 else base[:0]
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
        scoring_lengths.append(torch.tensor(seq.numel(), dtype=torch.float32, device=seq.device))
        scoring_original_lengths.append(
            torch.tensor(original_len, dtype=torch.float32, device=seq.device)
        )
        scoring_dropped_tokens.append(
            torch.tensor(dropped_tokens, dtype=torch.float32, device=seq.device)
        )
        scoring_truncated.append(
            torch.tensor(float(dropped_tokens > 0), dtype=torch.float32, device=seq.device)
        )

    attention_rows = [
        torch.ones(row.numel(), dtype=torch.bool, device=row.device)
        for row in scoring_ids
    ]
    scoring_batch = {
        "input_ids": _pad_2d(scoring_ids, 0, torch.int32),
        "loss_mask": _pad_2d(scoring_loss_masks, 0, torch.int32),
        "attention_mask": _pad_2d(attention_rows, False, torch.bool),
    }
    scoring_stats = {
        "length": torch.stack(scoring_lengths).detach().cpu(),
        "original_length": torch.stack(scoring_original_lengths).detach().cpu(),
        "dropped_tokens": torch.stack(scoring_dropped_tokens).detach().cpu(),
        "truncated": torch.stack(scoring_truncated).detach().cpu(),
    }
    return scoring_batch, scoring_stats


def _scoring_max_tokens(trainer) -> int | None:
    actor_cfg = getattr(getattr(trainer, "config", None), "actor", None)
    mb_spec = getattr(actor_cfg, "mb_spec", None)
    value = getattr(mb_spec, "max_tokens_per_mb", None)
    if value is None:
        value = getattr(getattr(trainer, "config", None), "gconfig", None)
        value = getattr(value, "max_tokens", None)
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def teacher_answer_reward_postprocess(
    trainer,
    rollout_batch: list[dict[str, Any]],
    global_step: int,
) -> None:
    """Replace placeholder rewards with teacher-answer continuation likelihood."""
    del global_step
    format_bonus = _env_float("TEACHER_ANSWER_FORMAT_BONUS", 0.0)
    length_penalty = _env_float("TEACHER_ANSWER_LENGTH_PENALTY", 0.0)
    max_new_tokens = max(float(getattr(trainer.config.gconfig, "max_new_tokens", 1) or 1), 1.0)
    scoring_outputs = [
        _build_scoring_batch(traj, max_tokens=_scoring_max_tokens(trainer))
        for traj in rollout_batch
    ]
    scoring_batches = [batch for batch, _ in scoring_outputs]
    scoring_stats = [stats for _, stats in scoring_outputs]
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
    all_scoring_lengths: list[torch.Tensor] = []
    all_scoring_original_lengths: list[torch.Tensor] = []
    all_scoring_dropped_tokens: list[torch.Tensor] = []
    all_scoring_truncated: list[torch.Tensor] = []
    for traj, scoring_batch, scoring_stat, logp in zip(
        rollout_batch, scoring_batches, scoring_stats, scoring_logps
    ):
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
        all_scoring_lengths.append(scoring_stat["length"].detach().float().cpu())
        all_scoring_original_lengths.append(
            scoring_stat["original_length"].detach().float().cpu()
        )
        all_scoring_dropped_tokens.append(
            scoring_stat["dropped_tokens"].detach().float().cpu()
        )
        all_scoring_truncated.append(scoring_stat["truncated"].detach().float().cpu())

    rewards_cat = torch.cat(all_rewards)
    stats_tracker.denominator(
        teacher_answer_n_seqs=torch.ones_like(rewards_cat, dtype=torch.bool)
    )
    stats_tracker.stat(
        teacher_answer_logp=rewards_cat,
        teacher_answer_reward=torch.cat(all_adjusted_rewards),
        teacher_answer_len=torch.cat(all_lengths),
        teacher_context_len=torch.cat(all_context_lengths),
        teacher_added_prefix_len=torch.cat(all_prefix_lengths),
        teacher_format_found=torch.cat(all_format_found),
        teacher_optimized_len=torch.cat(all_optimized_lengths),
        teacher_scoring_len=torch.cat(all_scoring_lengths),
        teacher_scoring_original_len=torch.cat(all_scoring_original_lengths),
        teacher_scoring_dropped_tokens=torch.cat(all_scoring_dropped_tokens),
        teacher_scoring_truncated=torch.cat(all_scoring_truncated),
        denominator="teacher_answer_n_seqs",
    )
