"""AReaL teacher-answer RL entry point for Terminus tool-call payloads."""

from __future__ import annotations

import os
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


def _bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    workflow_mode = str(
        train_dataset_kwargs.get("workflow_mode", os.environ.get("TA_RL_WORKFLOW_MODE", "prefix"))
    )

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=enable_thinking,
        teacher_answer_score_mode=str(
            train_dataset_kwargs.get(
                "teacher_answer_score_mode",
                os.environ.get("TA_RL_TEACHER_ANSWER_SCORE_MODE", "all"),
            )
        ),
    )
    workflow_path = (
        "terminal_agent_demo.teacher_answer_rl.workflow.TerminalToolTeacherAnswerRLWorkflow"
    )
    if workflow_mode in {"full_turn", "full-turn", "full"}:
        workflow_path = (
            "terminal_agent_demo.teacher_answer_rl.workflow."
            "TerminalToolFullTurnTeacherAnswerRLWorkflow"
        )
        workflow_kwargs.update(
            syntax_reward_weight=float(
                train_dataset_kwargs.get(
                    "syntax_reward_weight",
                    os.environ.get("TA_RL_SYNTAX_REWARD_WEIGHT", 0.1),
                )
            ),
            valid_syntax_reward=float(
                train_dataset_kwargs.get(
                    "valid_syntax_reward",
                    os.environ.get("TA_RL_VALID_SYNTAX_REWARD", 0.0),
                )
            ),
            invalid_syntax_reward=float(
                train_dataset_kwargs.get(
                    "invalid_syntax_reward",
                    os.environ.get("TA_RL_INVALID_SYNTAX_REWARD", -1.0),
                )
            ),
            emit_syntax_view=_bool_arg(
                train_dataset_kwargs.get(
                    "emit_syntax_view",
                    os.environ.get("TA_RL_EMIT_SYNTAX_VIEW", "1"),
                )
            ),
            emit_valid_syntax_view=_bool_arg(
                train_dataset_kwargs.get(
                    "emit_valid_syntax_view",
                    os.environ.get("TA_RL_EMIT_VALID_SYNTAX_VIEW", "0"),
                )
            ),
            teacher_loss_span=str(
                train_dataset_kwargs.get(
                    "teacher_loss_span",
                    os.environ.get("TA_RL_TEACHER_LOSS_SPAN", "thinking"),
                )
            ),
            syntax_reward_on_teacher_row=_bool_arg(
                train_dataset_kwargs.get(
                    "syntax_reward_on_teacher_row",
                    os.environ.get("TA_RL_SYNTAX_REWARD_ON_TEACHER_ROW", "0"),
                )
            ),
            supervise_teacher_answer=_bool_arg(
                train_dataset_kwargs.get(
                    "supervise_teacher_answer",
                    os.environ.get("TA_RL_SUPERVISE_TEACHER_ANSWER", "0"),
                )
            ),
            teacher_answer_supervised_weight=float(
                train_dataset_kwargs.get(
                    "teacher_answer_supervised_weight",
                    os.environ.get("TA_RL_TEACHER_ANSWER_SUPERVISED_WEIGHT", 0.0),
                )
            ),
            teacher_answer_supervised_score_mode=str(
                train_dataset_kwargs.get(
                    "teacher_answer_supervised_score_mode",
                    os.environ.get(
                        "TA_RL_TEACHER_ANSWER_SUPERVISED_SCORE_MODE",
                        train_dataset_kwargs.get("teacher_answer_score_mode", "all"),
                    ),
                )
            ),
            teacher_answer_supervised_max_prefix_tokens=(
                train_dataset_kwargs.get(
                    "teacher_answer_supervised_max_prefix_tokens",
                    os.environ.get("TA_RL_TEACHER_ANSWER_SUPERVISED_MAX_PREFIX_TOKENS"),
                )
            ),
            teacher_reward_requires_valid_syntax=_bool_arg(
                train_dataset_kwargs.get(
                    "teacher_reward_requires_valid_syntax",
                    os.environ.get("TA_RL_TEACHER_REWARD_REQUIRES_VALID_SYNTAX", "0"),
                )
            ),
            drop_invalid_teacher_row_loss=_bool_arg(
                train_dataset_kwargs.get(
                    "drop_invalid_teacher_row_loss",
                    os.environ.get("TA_RL_DROP_INVALID_TEACHER_ROW_LOSS", "0"),
                )
            ),
            syntax_thinking_length_penalty_weight=float(
                train_dataset_kwargs.get(
                    "syntax_thinking_length_penalty_weight",
                    os.environ.get(
                        "TA_RL_SYNTAX_THINKING_LENGTH_PENALTY_WEIGHT",
                        0.0,
                    ),
                )
            ),
            syntax_output_length_penalty_weight=float(
                train_dataset_kwargs.get(
                    "syntax_output_length_penalty_weight",
                    os.environ.get(
                        "TA_RL_SYNTAX_OUTPUT_LENGTH_PENALTY_WEIGHT",
                        0.0,
                    ),
                )
            ),
            syntax_length_penalty_requires_invalid=_bool_arg(
                train_dataset_kwargs.get(
                    "syntax_length_penalty_requires_invalid",
                    os.environ.get(
                        "TA_RL_SYNTAX_LENGTH_PENALTY_REQUIRES_INVALID",
                        "0",
                    ),
                )
            ),
            teacher_command_match_reward_weight=float(
                train_dataset_kwargs.get(
                    "teacher_command_match_reward_weight",
                    os.environ.get("TA_RL_TEACHER_COMMAND_MATCH_REWARD_WEIGHT", 0.0),
                )
            ),
            teacher_command_presence_reward_weight=float(
                train_dataset_kwargs.get(
                    "teacher_command_presence_reward_weight",
                    os.environ.get(
                        "TA_RL_TEACHER_COMMAND_PRESENCE_REWARD_WEIGHT",
                        0.0,
                    ),
                )
            ),
            teacher_command_newline_reward_weight=float(
                train_dataset_kwargs.get(
                    "teacher_command_newline_reward_weight",
                    os.environ.get(
                        "TA_RL_TEACHER_COMMAND_NEWLINE_REWARD_WEIGHT",
                        0.0,
                    ),
                )
            ),
            teacher_completion_match_reward_weight=float(
                train_dataset_kwargs.get(
                    "teacher_completion_match_reward_weight",
                    os.environ.get(
                        "TA_RL_TEACHER_COMPLETION_MATCH_REWARD_WEIGHT",
                        0.0,
                    ),
                )
            ),
            teacher_completion_requires_valid_syntax=_bool_arg(
                train_dataset_kwargs.get(
                    "teacher_completion_requires_valid_syntax",
                    os.environ.get(
                        "TA_RL_TEACHER_COMPLETION_REQUIRES_VALID_SYNTAX",
                        "0",
                    ),
                )
            ),
            teacher_empty_completion_reward_weight=float(
                train_dataset_kwargs.get(
                    "teacher_empty_completion_reward_weight",
                    os.environ.get(
                        "TA_RL_TEACHER_EMPTY_COMPLETION_REWARD_WEIGHT",
                        0.0,
                    ),
                )
            ),
            teacher_command_count_reward_weight=float(
                train_dataset_kwargs.get(
                    "teacher_command_count_reward_weight",
                    os.environ.get(
                        "TA_RL_TEACHER_COMMAND_COUNT_REWARD_WEIGHT",
                        0.0,
                    ),
                )
            ),
            teacher_repeated_command_penalty_weight=float(
                train_dataset_kwargs.get(
                    "teacher_repeated_command_penalty_weight",
                    os.environ.get(
                        "TA_RL_TEACHER_REPEATED_COMMAND_PENALTY_WEIGHT",
                        0.0,
                    ),
                )
            ),
            tool_call_scaffold_reward_weight=float(
                train_dataset_kwargs.get(
                    "tool_call_scaffold_reward_weight",
                    os.environ.get(
                        "TA_RL_TOOL_CALL_SCAFFOLD_REWARD_WEIGHT",
                        0.0,
                    ),
                )
            ),
            invalid_teacher_loss_span=str(
                train_dataset_kwargs.get(
                    "invalid_teacher_loss_span",
                    os.environ.get(
                        "TA_RL_INVALID_TEACHER_LOSS_SPAN",
                        "default",
                    ),
                )
            ),
        )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig
    eval_workflow = workflow_path if valid_dataset is not None else None
    if eval_workflow is None:
        eval_workflow_kwargs = None

    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow=workflow_path,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
            rollout_postprocess_fn=(
                "terminal_agent_demo.teacher_answer_rl.workflow.teacher_answer_reward_postprocess"
            ),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
