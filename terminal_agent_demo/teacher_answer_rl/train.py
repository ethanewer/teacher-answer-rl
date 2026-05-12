"""AReaL teacher-answer RL entry point for Terminus tool-call payloads."""

from __future__ import annotations

import sys
from typing import Any

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer

from terminal_agent_demo.model_paths import localize_model_paths
from terminal_agent_demo.terminal_agent_data import get_terminal_teacher_answer_rl_dataset


def _dataset_kwargs(dataset_config, seed: int) -> dict[str, Any]:
    kwargs = dict(getattr(dataset_config, "dataset_kwargs", {}) or {})
    kwargs.setdefault("seed", seed)
    return kwargs


def _load_dataset(dataset_config, tokenizer, seed: int):
    return get_terminal_teacher_answer_rl_dataset(
        path=dataset_config.path,
        split=dataset_config.split,
        tokenizer=tokenizer,
        max_length=dataset_config.max_length,
        **_dataset_kwargs(dataset_config, seed),
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    localize_model_paths(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = _load_dataset(config.train_dataset, tokenizer, config.seed)
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = _load_dataset(config.valid_dataset, tokenizer, config.seed)

    train_dataset_kwargs = dict(getattr(config.train_dataset, "dataset_kwargs", {}) or {})
    enable_thinking = bool(train_dataset_kwargs.get("enable_thinking", True))

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=enable_thinking,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig

    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow="terminal_agent_demo.teacher_answer_rl.workflow.TerminalToolTeacherAnswerRLWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="terminal_agent_demo.teacher_answer_rl.workflow.TerminalToolTeacherAnswerRLWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
            rollout_postprocess_fn=(
                "terminal_agent_demo.teacher_answer_rl.workflow.teacher_answer_reward_postprocess"
            ),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
