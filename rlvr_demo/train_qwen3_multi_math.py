"""AReaL GRPO training entry point for mixed GSM8K/MATH Qwen3 experiments."""

from __future__ import annotations

import sys
from typing import Any

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer

from rlvr_demo.model_paths import localize_model_paths
from rlvr_demo.multi_math_data import get_multi_math_dataset


def _dataset_kwargs(dataset_config, seed: int) -> dict[str, Any]:
    kwargs = dict(getattr(dataset_config, "dataset_kwargs", {}) or {})
    kwargs.setdefault("seed", seed)
    return kwargs


def _reward_fn(config: GRPOConfig) -> str:
    kwargs = dict(getattr(config.train_dataset, "dataset_kwargs", {}) or {})
    return str(
        kwargs.get("reward_fn")
        or "rlvr_demo.multi_math_reward.qwen3_multi_math_reward_fn"
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    localize_model_paths(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_multi_math_dataset(
        path=config.train_dataset.path,
        split=config.train_dataset.split,
        tokenizer=tokenizer,
        max_length=config.train_dataset.max_length,
        **_dataset_kwargs(config.train_dataset, config.seed),
    )

    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_multi_math_dataset(
            path=config.valid_dataset.path,
            split=config.valid_dataset.split,
            tokenizer=tokenizer,
            max_length=config.valid_dataset.max_length,
            **_dataset_kwargs(config.valid_dataset, config.seed),
        )

    workflow_kwargs = dict(
        reward_fn=_reward_fn(config),
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=True,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig

    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
