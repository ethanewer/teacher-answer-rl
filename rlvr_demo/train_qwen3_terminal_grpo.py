"""AReaL GRPO entry point for Terminus-format terminal tasks."""

from __future__ import annotations

import os
import sys
from typing import Any

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger

from rlvr_demo.model_paths import localize_model_paths
from rlvr_demo.terminal_task_grpo import (
    TerminalTaskGRPOConfig,
    get_terminal_synthetic_task_dataset,
)


def _dataset_kwargs(dataset_config, seed: int) -> dict[str, Any]:
    kwargs = dict(getattr(dataset_config, "dataset_kwargs", {}) or {})
    kwargs.setdefault("seed", seed)
    return kwargs


def _load_dataset(dataset_config, seed: int):
    return get_terminal_synthetic_task_dataset(
        path=dataset_config.path,
        split=dataset_config.split,
        **_dataset_kwargs(dataset_config, seed),
    )


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, TerminalTaskGRPOConfig)
    localize_model_paths(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = _load_dataset(config.train_dataset, config.seed)
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = _load_dataset(config.valid_dataset, config.seed)

    common_workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        n_trajs=config.n_trajs,
        max_turns=config.max_turns,
        max_tokens_per_trajectory=config.max_tokens_per_trajectory,
        max_workers=config.max_workers,
        observation_max_chars=config.observation_max_chars,
        turn_discount=config.turn_discount,
        task_timeouts=config.task_timeouts,
        filter_uniform_reward=config.filter_uniform_reward,
        encourage_completion_reward=config.encourage_completion_reward,
        terminal_task_service_url=config.terminal_task_service_url,
        terminal_task_service_url_file=config.terminal_task_service_url_file,
        dump_dir=os.path.join(StatsLogger.get_log_path(config.stats_logger), "generated"),
    )
    if config.agent_harness == "default":
        workflow = "rlvr_demo.default_agent_terminal_grpo.DefaultAgentTerminalGRPOWorkflow"
        workflow_kwargs = {
            **common_workflow_kwargs,
            "tool_call_parser": config.tool_call_parser,
            "reasoning_parser": config.reasoning_parser,
            "chat_template_type": config.chat_template_type,
            "export_style": config.export_style,
        }
    elif config.agent_harness == "terminus":
        workflow = "rlvr_demo.terminal_task_grpo.TerminusTerminalGRPOWorkflow"
        workflow_kwargs = {
            **common_workflow_kwargs,
            "enable_thinking": config.enable_thinking,
        }
    else:
        raise ValueError(
            f"Unsupported terminal GRPO agent_harness={config.agent_harness!r}; "
            "expected 'terminus' or 'default'"
        )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig
    eval_workflow_kwargs["n_trajs"] = 1

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
