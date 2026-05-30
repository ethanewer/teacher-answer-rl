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


def _config_value(trainer, name: str, default: Any) -> Any:
    dataset_cfg = getattr(getattr(trainer, "config", None), "train_dataset", None)
    kwargs = dict(getattr(dataset_cfg, "dataset_kwargs", {}) or {})
    return kwargs.get(name, os.environ.get(name.upper(), default))


def _normalize_teacher_rows(
    reward_rows: list[torch.Tensor],
    has_answer_rows: list[torch.Tensor],
    *,
    group_size: int,
    scale_by_std: bool,
) -> list[torch.Tensor]:
    if group_size <= 0:
        raise ValueError(f"teacher reward norm group_size must be positive, got {group_size}")
    refs: list[tuple[torch.Tensor, int]] = []
    values: list[torch.Tensor] = []
    for rewards, has_answer in zip(reward_rows, has_answer_rows):
        answer_indices = torch.nonzero(has_answer.to(rewards.device), as_tuple=False).flatten()
        for row_idx in answer_indices.tolist():
            refs.append((rewards, int(row_idx)))
            values.append(rewards[int(row_idx)])
    normalized_chunks: list[torch.Tensor] = []
    for start in range(0, len(values), group_size):
        chunk = values[start : start + group_size]
        if not chunk:
            continue
        vals = torch.stack(chunk)
        if vals.numel() < 2:
            normalized = torch.zeros_like(vals)
        else:
            centered = vals - vals.mean()
            if scale_by_std:
                normalized = centered / vals.std(unbiased=False).clamp(min=1e-6)
            else:
                normalized = centered
        normalized_chunks.append(normalized)
    if not normalized_chunks:
        return [row.detach().float().cpu() for row in reward_rows]
    normalized_values = torch.cat(normalized_chunks)
    for (rewards, row_idx), value in zip(refs, normalized_values):
        rewards[row_idx] = value.to(rewards.device)
    return [row.detach().float().cpu() for row in reward_rows]


def _teacher_reward_norm_scale_by_std(
    mode: str,
    *,
    global_step: int = 0,
    switch_step: int | None = None,
) -> bool | None:
    mode = mode.lower().replace("-", "_")
    if mode in {"", "none", "null", "false"}:
        return None
    if mode in {"group", "grpo", "group_std", "group_stddev"}:
        return True
    if mode in {
        "group_mean",
        "group_mean_only",
        "group_meanonly",
        "mean_only",
        "meanonly",
        "grpo_mean",
        "grpo_mean_only",
        "grpo_meanonly",
    }:
        return False
    if mode in {
        "group_then_mean",
        "group_then_mean_only",
        "group_then_meanonly",
        "group_to_mean",
        "group_to_mean_only",
        "group_to_meanonly",
        "group_std_then_mean",
        "group_std_then_mean_only",
        "group_std_then_meanonly",
    }:
        if switch_step is None:
            switch_step = _env_int("TEACHER_ANSWER_REWARD_NORM_SWITCH_STEP", 40)
        return global_step < switch_step
    raise ValueError(f"unsupported teacher_answer_reward_norm: {mode}")


def _teacher_reward_norm_is_scheduled(mode: str) -> bool:
    return mode.lower().replace("-", "_") in {
        "group_then_mean",
        "group_then_mean_only",
        "group_then_meanonly",
        "group_to_mean",
        "group_to_mean_only",
        "group_to_meanonly",
        "group_std_then_mean",
        "group_std_then_mean_only",
        "group_std_then_meanonly",
    }


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _apply_teacher_reward_norm(
    trainer,
    reward_rows: list[torch.Tensor],
    has_answer_rows: list[torch.Tensor],
    *,
    global_step: int,
) -> list[torch.Tensor] | None:
    reward_norm_mode = str(_config_value(trainer, "teacher_answer_reward_norm", ""))
    switch_step = None
    if _teacher_reward_norm_is_scheduled(reward_norm_mode):
        switch_step = int(
            _config_value(
                trainer,
                "teacher_answer_reward_norm_switch_step",
                _env_int("TEACHER_ANSWER_REWARD_NORM_SWITCH_STEP", 40),
            )
        )
    scale_by_std = _teacher_reward_norm_scale_by_std(
        reward_norm_mode,
        global_step=global_step,
        switch_step=switch_step,
    )
    if scale_by_std is None:
        return None
    group_size = int(
        _config_value(
            trainer,
            "teacher_answer_reward_norm_group_size",
            getattr(trainer.config.gconfig, "n_samples", 1),
        )
    )
    return _normalize_teacher_rows(
        reward_rows,
        has_answer_rows,
        group_size=group_size,
        scale_by_std=scale_by_std,
    )


def _teacher_reward_clip_value(trainer) -> float | None:
    value = _config_value(
        trainer,
        "teacher_answer_reward_norm_clip",
        _config_value(
            trainer,
            "teacher_answer_reward_clip",
            os.environ.get("TEACHER_ANSWER_REWARD_NORM_CLIP"),
        ),
    )
    if value is None or str(value).strip() == "":
        return None
    value = float(value)
    return value if value > 0.0 else None


def _apply_teacher_reward_clip(
    trainer,
    reward_rows: list[torch.Tensor],
    has_answer_rows: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], float] | None:
    clip_value = _teacher_reward_clip_value(trainer)
    if clip_value is None:
        return None
    clipped_flags: list[torch.Tensor] = []
    for rewards, has_answer in zip(reward_rows, has_answer_rows):
        mask = has_answer.to(device=rewards.device, dtype=torch.bool)
        flags = torch.zeros_like(rewards, dtype=torch.float32)
        if mask.any():
            original = rewards[mask]
            clipped = original.clamp(min=-clip_value, max=clip_value)
            rewards[mask] = clipped
            flags[mask] = (clipped != original).float()
        clipped_flags.append(flags.detach().float().cpu())
    return (
        [row.detach().float().cpu() for row in reward_rows],
        clipped_flags,
        clip_value,
    )


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
    answer_score_mask = _as_tensor(
        traj.get("teacher_answer_score_mask", traj["teacher_answer_mask"])
    ).bool()
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
        score_mask = answer_score_mask[idx][answer_mask[idx]]
        if answer.numel() == 0 or score_mask.sum().item() == 0:
            seq = base
            mask = torch.zeros(base.numel(), dtype=torch.int32, device=base.device)
            scoring_ids.append(seq)
            scoring_loss_masks.append(mask)
            scoring_lengths.append(
                torch.tensor(seq.numel(), dtype=torch.float32, device=base.device)
            )
            scoring_original_lengths.append(
                torch.tensor(base.numel(), dtype=torch.float32, device=base.device)
            )
            scoring_dropped_tokens.append(
                torch.tensor(0, dtype=torch.float32, device=base.device)
            )
            scoring_truncated.append(
                torch.tensor(0.0, dtype=torch.float32, device=base.device)
            )
            continue
        original_len = base.numel() + prefix.numel() + answer.numel()
        dropped_tokens = 0
        if max_tokens is not None and max_tokens > 0 and original_len > max_tokens:
            answer_budget = max(max_tokens - prefix.numel(), 0)
            if answer.numel() > answer_budget:
                dropped_tokens += answer.numel() - answer_budget
                answer = answer[:answer_budget]
                score_mask = score_mask[:answer_budget]
            base_budget = max(max_tokens - prefix.numel() - answer.numel(), 0)
            if base.numel() > base_budget:
                dropped_tokens += base.numel() - base_budget
                base = base[-base_budget:] if base_budget > 0 else base[:0]
        seq = torch.cat([base, prefix, answer], dim=0)
        mask = torch.cat(
            [
                torch.zeros(base.numel() + prefix.numel(), dtype=torch.int32, device=seq.device),
                score_mask.to(dtype=torch.int32, device=seq.device),
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
    format_bonus = float(
        _config_value(
            trainer,
            "teacher_answer_format_bonus",
            os.environ.get("TEACHER_ANSWER_FORMAT_BONUS", 0.0),
        )
    )
    length_penalty = float(
        _config_value(
            trainer,
            "teacher_answer_length_penalty",
            os.environ.get("TEACHER_ANSWER_LENGTH_PENALTY", 0.0),
        )
    )
    logp_reward_weight = float(_config_value(trainer, "teacher_answer_logp_reward_weight", 1.0))
    use_logp_reward = (
        logp_reward_weight != 0.0
        or format_bonus != 0.0
        or length_penalty != 0.0
    )

    if not use_logp_reward:
        traj_reward_rows: list[torch.Tensor] = []
        traj_has_answer_rows: list[torch.Tensor] = []
        all_adjusted_rewards: list[torch.Tensor] = []
        all_context_lengths: list[torch.Tensor] = []
        all_prefix_lengths: list[torch.Tensor] = []
        all_optimized_lengths: list[torch.Tensor] = []
        all_lengths: list[torch.Tensor] = []
        for traj in rollout_batch:
            rewards = _as_tensor(traj["rewards"]).float()
            traj["rewards"] = rewards.to(dtype=torch.float32)
            answer_mask = _as_tensor(traj["teacher_answer_mask"]).bool()
            answer_score_mask = _as_tensor(
                traj.get("teacher_answer_score_mask", traj["teacher_answer_mask"])
            ).bool()
            has_answer = (answer_score_mask & answer_mask).sum(dim=-1) > 0
            context_mask = _as_tensor(
                traj.get("teacher_context_mask", traj["attention_mask"])
            )
            prefix_mask = _as_tensor(traj["teacher_answer_prefix_mask"])
            loss_mask = _as_tensor(traj["loss_mask"]).to(rewards.device).float()
            traj_reward_rows.append(traj["rewards"])
            traj_has_answer_rows.append(has_answer.detach())
            all_adjusted_rewards.append(traj["rewards"].detach().float().cpu())
            all_context_lengths.append(context_mask.float().sum(dim=-1).detach().cpu())
            all_prefix_lengths.append(prefix_mask.float().sum(dim=-1).detach().cpu())
            all_optimized_lengths.append(loss_mask.sum(dim=-1).detach().cpu())
            all_lengths.append(answer_score_mask.float().sum(dim=-1).detach().cpu())

        normalized_reward_rows = _apply_teacher_reward_norm(
            trainer,
            traj_reward_rows,
            traj_has_answer_rows,
            global_step=global_step,
        )
        clipped_reward_rows = _apply_teacher_reward_clip(
            trainer,
            traj_reward_rows,
            traj_has_answer_rows,
        )
        clipped_flags = None
        clip_value = None
        if clipped_reward_rows is not None:
            normalized_reward_rows, clipped_flags, clip_value = clipped_reward_rows

        rewards_cat = torch.cat(all_adjusted_rewards)
        stats_tracker.denominator(
            teacher_answer_n_seqs=torch.ones_like(rewards_cat, dtype=torch.bool)
        )
        stat_kwargs = dict(
            teacher_answer_logp=torch.zeros_like(rewards_cat),
            teacher_answer_logp_reward_weight=torch.zeros_like(rewards_cat),
            teacher_answer_reward=rewards_cat,
            teacher_answer_len=torch.cat(all_lengths),
            teacher_context_len=torch.cat(all_context_lengths),
            teacher_added_prefix_len=torch.cat(all_prefix_lengths),
            teacher_format_found=torch.zeros_like(rewards_cat),
            teacher_optimized_len=torch.cat(all_optimized_lengths),
            teacher_scoring_len=torch.zeros_like(rewards_cat),
            teacher_scoring_original_len=torch.zeros_like(rewards_cat),
            teacher_scoring_dropped_tokens=torch.zeros_like(rewards_cat),
            teacher_scoring_truncated=torch.zeros_like(rewards_cat),
        )
        if normalized_reward_rows is not None:
            stat_kwargs["teacher_answer_reward_normed"] = torch.cat(normalized_reward_rows)
        if clipped_flags is not None and clip_value is not None:
            stat_kwargs["teacher_answer_reward_clipped"] = torch.cat(clipped_flags)
            stat_kwargs["teacher_answer_reward_clip"] = torch.full_like(
                rewards_cat,
                float(clip_value),
            )
        stats_tracker.stat(**stat_kwargs, denominator="teacher_answer_n_seqs")
        return

    scorer_name = str(_config_value(trainer, "teacher_answer_scorer", "actor")).lower()
    scorer = getattr(trainer, scorer_name, None) if scorer_name != "actor" else trainer.actor
    if scorer is None:
        scorer = trainer.actor
    max_new_tokens = max(float(getattr(trainer.config.gconfig, "max_new_tokens", 1) or 1), 1.0)
    scoring_outputs = [
        _build_scoring_batch(traj, max_tokens=_scoring_max_tokens(trainer))
        for traj in rollout_batch
    ]
    scoring_batches = [batch for batch, _ in scoring_outputs]
    scoring_stats = [stats for _, stats in scoring_outputs]
    scoring_logps = scorer.compute_logp(scoring_batches)
    if scoring_logps is None:
        raise RuntimeError(f"{scorer_name}.compute_logp returned None for teacher-answer scoring")

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
    traj_reward_rows: list[torch.Tensor] = []
    traj_has_answer_rows: list[torch.Tensor] = []
    for traj, scoring_batch, scoring_stat, logp in zip(
        rollout_batch, scoring_batches, scoring_stats, scoring_logps
    ):
        logp = _as_tensor(logp)
        answer_mask = torch.roll(
            scoring_batch["loss_mask"].to(logp.device).float(),
            shifts=-1,
            dims=-1,
        )
        has_answer = (scoring_batch["loss_mask"].to(logp.device).sum(dim=-1) > 0)
        lengths = answer_mask.sum(dim=-1).clamp(min=1.0)
        rewards = (logp * answer_mask).sum(dim=-1) / lengths
        context_mask = _as_tensor(traj.get("teacher_context_mask", traj["attention_mask"]))
        prefix_mask = _as_tensor(traj["teacher_answer_prefix_mask"])
        loss_mask = _as_tensor(traj["loss_mask"]).to(rewards.device).float()
        prefix_lengths = prefix_mask.float().sum(dim=-1).to(rewards.device)
        format_found = (prefix_lengths <= 0).float()
        optimized_lengths = loss_mask.sum(dim=-1)
        base_rewards = _as_tensor(traj["rewards"]).to(rewards.device).float()
        adjusted_rewards = base_rewards + torch.where(
            has_answer,
            logp_reward_weight * rewards
            + format_bonus * format_found
            - length_penalty * (optimized_lengths / max_new_tokens),
            torch.zeros_like(rewards),
        )
        traj["rewards"] = adjusted_rewards.to(dtype=torch.float32)
        traj_reward_rows.append(traj["rewards"])
        traj_has_answer_rows.append(has_answer.detach())
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

    normalized_reward_rows = _apply_teacher_reward_norm(
        trainer,
        traj_reward_rows,
        traj_has_answer_rows,
        global_step=global_step,
    )
    clipped_reward_rows = _apply_teacher_reward_clip(
        trainer,
        traj_reward_rows,
        traj_has_answer_rows,
    )
    clipped_flags = None
    clip_value = None
    if clipped_reward_rows is not None:
        normalized_reward_rows, clipped_flags, clip_value = clipped_reward_rows

    rewards_cat = torch.cat(all_rewards)
    stats_tracker.denominator(
        teacher_answer_n_seqs=torch.ones_like(rewards_cat, dtype=torch.bool)
    )
    stat_kwargs = dict(
        teacher_answer_logp=rewards_cat,
        teacher_answer_logp_reward_weight=torch.full_like(
            rewards_cat,
            float(logp_reward_weight),
        ),
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
    )
    if normalized_reward_rows is not None:
        stat_kwargs["teacher_answer_reward_normed"] = torch.cat(normalized_reward_rows)
    if clipped_flags is not None and clip_value is not None:
        stat_kwargs["teacher_answer_reward_clipped"] = torch.cat(clipped_flags)
        stat_kwargs["teacher_answer_reward_clip"] = torch.full_like(
            rewards_cat,
            float(clip_value),
        )
    stats_tracker.stat(**stat_kwargs, denominator="teacher_answer_n_seqs")
