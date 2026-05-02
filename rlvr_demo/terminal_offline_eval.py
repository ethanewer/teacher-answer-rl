"""Offline non-Docker evaluation for Terminus-format terminal-agent checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvr_demo.terminal_agent_data import (
    TERMINAL_CORPUS,
    TERMINAL_CORPUS_CONFIG,
    _partition_turns,
    _terminal_turns,
    split_terminus_teacher_answer,
)


_SPACE_RE = re.compile(r"[ \t]+")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _normalize_keystrokes(value: Any) -> str:
    text = "" if not isinstance(value, str) else value
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(_SPACE_RE.sub(" ", line).rstrip() for line in text.split("\n")).strip()


def _commands_valid(obj: dict[str, Any] | None) -> bool:
    if not isinstance(obj, dict):
        return False
    commands = obj.get("commands")
    if not isinstance(commands, list):
        return False
    for command in commands:
        if not isinstance(command, dict):
            return False
        if not isinstance(command.get("keystrokes"), str):
            return False
        duration = command.get("duration", 1.0)
        if not isinstance(duration, int | float):
            return False
    return True


def _command_sequence(obj: dict[str, Any] | None) -> list[str]:
    if not _commands_valid(obj):
        return []
    return [_normalize_keystrokes(cmd.get("keystrokes")) for cmd in obj["commands"]]


def _task_complete_valid(obj: dict[str, Any] | None) -> bool:
    if not isinstance(obj, dict):
        return False
    return "task_complete" not in obj or isinstance(obj.get("task_complete"), bool)


def _task_complete_value(obj: dict[str, Any] | None) -> bool | None:
    if not _task_complete_valid(obj):
        return None
    return bool(obj.get("task_complete", False))


def _load_examples(args: argparse.Namespace, tokenizer) -> list[dict[str, Any]]:
    turns = _terminal_turns(
        path=args.dataset,
        name=args.dataset_config,
        split=args.split,
        seed=args.seed,
        limit_rows=args.limit_rows,
        strip_prior_assistant_thinking=True,
        shuffle_rows=False,
    )
    if args.skip_turns:
        turns = turns[args.skip_turns :]
    teacher_turns = []
    for turn in turns:
        try:
            student_prefix, teacher_answer = split_terminus_teacher_answer(
                str(turn["assistant"])
            )
        except Exception:
            continue
        row = dict(turn)
        row["student_prefix"] = student_prefix
        row["teacher_answer"] = teacher_answer
        teacher_turns.append(row)

    selected = _partition_turns(
        teacher_turns,
        split_part="validation",
        holdout_size=args.holdout_size,
        seed=args.seed,
        shuffle_records=False,
    )
    if args.shuffle:
        random.Random(args.seed).shuffle(selected)

    examples: list[dict[str, Any]] = []
    for turn in selected:
        prompt_ids = tokenizer.apply_chat_template(
            turn["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [*turn["messages"], {"role": "assistant", "content": turn["assistant"]}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        if len(full_ids) > args.max_length:
            continue
        target_obj = _extract_json_object(str(turn["assistant"]))
        examples.append(
            {
                "messages": turn["messages"],
                "target": turn["assistant"],
                "target_obj": target_obj,
                "prompt_tokens": len(prompt_ids),
                "target_tokens": max(0, len(full_ids) - len(prompt_ids)),
                "source_id": turn["source_id"],
                "turn_idx": turn["turn_idx"],
                "task": turn["task"],
            }
        )
        if len(examples) >= args.limit:
            break
    return examples


def _generate(model, tokenizer, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    ).to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.greedy:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, device=input_ids.device),
            **generation_kwargs,
        )
    new_tokens = output[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _score(prediction: str, target_obj: dict[str, Any] | None) -> dict[str, Any]:
    pred_obj = _extract_json_object(prediction)
    pred_commands = _command_sequence(pred_obj)
    target_commands = _command_sequence(target_obj)
    pred_joined = "\n".join(pred_commands)
    target_joined = "\n".join(target_commands)
    pred_task_complete = _task_complete_value(pred_obj)
    target_task_complete = _task_complete_value(target_obj)
    return {
        "json_parse_valid": pred_obj is not None,
        "commands_schema_valid": _commands_valid(pred_obj),
        "task_complete_valid": _task_complete_valid(pred_obj),
        "task_complete_present": isinstance(pred_obj, dict) and "task_complete" in pred_obj,
        "command_exact_match": pred_commands == target_commands,
        "normalized_command_similarity": SequenceMatcher(
            None, pred_joined, target_joined
        ).ratio()
        if target_joined or pred_joined
        else 1.0,
        "task_complete_accuracy": (
            pred_task_complete == target_task_complete
            if pred_task_complete is not None and target_task_complete is not None
            else False
        ),
        "pred_command_count": len(pred_commands),
        "target_command_count": len(target_commands),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=TERMINAL_CORPUS)
    parser.add_argument("--dataset-config", default=TERMINAL_CORPUS_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--holdout-size", type=int, default=512)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--skip-turns", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=40960)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions-output", type=Path)
    args = parser.parse_args()

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = _load_examples(args, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    scored_rows = []
    for idx, example in enumerate(examples):
        prediction = _generate(model, tokenizer, example["messages"], args)
        scores = _score(prediction, example["target_obj"])
        scored_rows.append({**example, "prediction": prediction, "scores": scores})
        if (idx + 1) % 8 == 0:
            print(f"evaluated {idx + 1}/{len(examples)}", flush=True)

    n = max(len(scored_rows), 1)
    aggregate = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split_part": "validation",
        "num_examples": len(scored_rows),
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "greedy": args.greedy,
        "temperature": args.temperature if not args.greedy else None,
        "top_p": args.top_p if not args.greedy else None,
        "top_k": args.top_k if not args.greedy else None,
        "json_parse_valid_rate": sum(r["scores"]["json_parse_valid"] for r in scored_rows) / n,
        "commands_schema_valid_rate": sum(
            r["scores"]["commands_schema_valid"] for r in scored_rows
        )
        / n,
        "task_complete_valid_rate": sum(
            r["scores"]["task_complete_valid"] for r in scored_rows
        )
        / n,
        "task_complete_present_rate": sum(
            r["scores"]["task_complete_present"] for r in scored_rows
        )
        / n,
        "normalized_command_sequence_similarity": sum(
            r["scores"]["normalized_command_similarity"] for r in scored_rows
        )
        / n,
        "command_exact_match_rate": sum(
            r["scores"]["command_exact_match"] for r in scored_rows
        )
        / n,
        "task_complete_prediction_accuracy": sum(
            r["scores"]["task_complete_accuracy"] for r in scored_rows
        )
        / n,
        "elapsed_sec": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", encoding="utf-8") as handle:
            for row in scored_rows:
                serializable = dict(row)
                serializable.pop("target_obj", None)
                handle.write(json.dumps(serializable, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
