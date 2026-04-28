"""Batched Transformers evaluation for Qwen3 GSM8K checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from rlvr_demo.model_paths import resolve_hf_snapshot
from rlvr_demo.qwen3_gsm8k_data import get_qwen3_gsm8k_dataset
from rlvr_demo.qwen3_gsm8k_reward import (
    extract_final_answer,
    has_report_format,
    numeric_equal,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument("--output", required=True, help="Metrics JSON output path.")
    parser.add_argument("--predictions", default=None, help="Optional JSONL predictions output.")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def _completion_reward(completion: str, answer: str) -> tuple[float, bool, bool, str | None]:
    final_answer = extract_final_answer(completion)
    correct = numeric_equal(final_answer, answer)
    format_ok = has_report_format(completion)
    reward = (1.0 if correct else 0.0) + (0.1 if format_ok else 0.0)
    return reward, correct, format_ok, final_answer


def _write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    model_path = resolve_hf_snapshot(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_qwen3_gsm8k_dataset(
        path=args.dataset,
        split=args.split,
        tokenizer=tokenizer,
        max_length=args.max_prompt_length,
        limit=args.limit,
        seed=args.seed,
        shuffle_limit=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype(args.dtype),
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()

    prediction_path = Path(args.predictions) if args.predictions else None
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    start = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(dataset), args.batch_size):
            batch = dataset.select(range(offset, min(offset + args.batch_size, len(dataset))))
            prompt_texts = [
                tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                for item in batch
            ]
            encoded = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            ).to(args.device)
            outputs = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completions = tokenizer.batch_decode(
                outputs[:, encoded["input_ids"].shape[1] :],
                skip_special_tokens=False,
            )
            for item, completion in zip(batch, completions, strict=True):
                reward, correct, format_ok, final_answer = _completion_reward(
                    completion,
                    str(item["answer"]),
                )
                rows.append(
                    {
                        "question": item["question"],
                        "answer": item["answer"],
                        "prediction": final_answer,
                        "correct": correct,
                        "format_ok": format_ok,
                        "reward": reward,
                        "completion": completion,
                    }
                )

            print(
                f"evaluated {min(offset + args.batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )

    elapsed = time.monotonic() - start
    n = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    formatted = sum(1 for row in rows if row["format_ok"])
    metrics = {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "format_rate": formatted / n if n else 0.0,
        "mean_reward": sum(float(row["reward"]) for row in rows) / n if n else 0.0,
        "elapsed_sec": elapsed,
        "model_path": model_path,
        "dataset_path": args.dataset,
        "dataset_split": args.split,
        "dataset_limit": args.limit,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }
    _write_metrics(args.output, metrics)

    if prediction_path is not None:
        with prediction_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
