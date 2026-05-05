"""Verify Qwen3 chat-template handling for terminal-agent trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from areal.utils.hf_utils import load_hf_tokenizer
from rlvr_demo.terminal_agent_data import (
    TERMINAL_CORPUS,
    TERMINAL_CORPUS_CONFIG,
    _tokenize_sft_turn,
    _terminal_turns,
    get_terminal_teacher_answer_rl_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--dataset", default=TERMINAL_CORPUS)
    parser.add_argument("--dataset-config", default=TERMINAL_CORPUS_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-length", type=int, default=40960)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--turn-offset", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = load_hf_tokenizer(args.model)
    turns = _terminal_turns(
        path=args.dataset,
        name=args.dataset_config,
        split=args.split,
        seed=args.seed,
        limit_rows=128,
        strip_prior_assistant_thinking=True,
        shuffle_rows=False,
    )
    if len(turns) <= args.turn_offset:
        raise RuntimeError(f"Need at least {args.turn_offset + 1} turns, got {len(turns)}")

    turn = turns[args.turn_offset]
    prompt_text = tokenizer.apply_chat_template(
        turn["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    full_text = tokenizer.apply_chat_template(
        [*turn["messages"], {"role": "assistant", "content": turn["assistant"]}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    tokenized = _tokenize_sft_turn(
        turn,
        tokenizer=tokenizer,
        max_length=args.max_length,
        enable_thinking=True,
    )
    if tokenized is None:
        raise RuntimeError("Selected turn was filtered by max_length")
    prompt_len = len(tokenized["input_ids"]) - sum(tokenized["loss_mask"])
    target_text = tokenizer.decode(tokenized["input_ids"][prompt_len:])
    teacher = get_terminal_teacher_answer_rl_dataset(
        path=args.dataset,
        name=args.dataset_config,
        split=args.split,
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
        limit_rows=128,
        split_part=None,
        shuffle_records=False,
        strip_prior_assistant_thinking=True,
    )[args.turn_offset]

    think_prompt_suffix = "<|im_start|>assistant\n<think>\n"
    assistant_prompt_suffix = "<|im_start|>assistant\n"
    prompt_injects_think_prefix = prompt_text.endswith(think_prompt_suffix)
    prompt_starts_assistant_turn = prompt_text.endswith(assistant_prompt_suffix)
    if prompt_injects_think_prefix:
        prompt_history_text = prompt_text[: -len(think_prompt_suffix)]
    elif prompt_starts_assistant_turn:
        prompt_history_text = prompt_text[: -len(assistant_prompt_suffix)]
    else:
        prompt_history_text = prompt_text
    target_starts_with_open_think = target_text.lstrip().startswith("<think>")
    report = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "turn_idx": turn["turn_idx"],
        "history_messages": len(turn["messages"]),
        "history_assistant_messages": sum(
            message["role"] == "assistant" for message in turn["messages"]
        ),
        "prompt_starts_assistant_turn": prompt_starts_assistant_turn,
        "prompt_injects_think_prefix": prompt_injects_think_prefix,
        "assistant_generation_prompt_ok": prompt_starts_assistant_turn
        or prompt_injects_think_prefix,
        "prior_history_has_think_block": "<think>" in prompt_history_text
        or "</think>" in prompt_history_text,
        "full_text_has_duplicate_think_prefix": "<think>\n<think>" in full_text,
        "sft_loss_prompt_tokens": prompt_len,
        "sft_loss_target_tokens": sum(tokenized["loss_mask"]),
        "sft_target_starts_with_open_think": target_starts_with_open_think,
        "sft_target_contains_close_think": "</think>" in target_text,
        "sft_target_contains_commands": '"commands"' in target_text,
        "current_thinking_is_trainable": target_starts_with_open_think
        or prompt_injects_think_prefix,
        "teacher_student_prefix_contains_commands": '"commands"' in teacher["student_prefix"],
        "teacher_answer_starts_with_commands": teacher["teacher_answer"].lstrip().startswith(
            '"commands"'
        ),
        "teacher_answer_contains_task_complete": '"task_complete"' in teacher["teacher_answer"],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    required_true = [
        "assistant_generation_prompt_ok",
        "sft_target_contains_close_think",
        "sft_target_contains_commands",
        "current_thinking_is_trainable",
        "teacher_answer_starts_with_commands",
        "teacher_answer_contains_task_complete",
    ]
    required_false = [
        "prior_history_has_think_block",
        "full_text_has_duplicate_think_prefix",
        "teacher_student_prefix_contains_commands",
    ]
    failures = [
        key for key in required_true if not report[key]
    ] + [
        key for key in required_false if report[key]
    ]
    if failures:
        raise SystemExit(f"Chat-template check failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
