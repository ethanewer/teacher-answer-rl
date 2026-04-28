"""Batched Transformers evaluation for mixed GSM8K/MATH checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from rlvr_demo.model_paths import resolve_hf_snapshot
from rlvr_demo.multi_math_data import get_named_eval_dataset
from rlvr_demo.multi_math_reward import extract_report_answer, has_report_format
from areal.reward import get_math_verify_worker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k_test", "math_test_l12", "math_test_l3", "math_test_l45"],
    )
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def _verify(completion: str, answer: str) -> tuple[float, bool, bool, str]:
    prediction = extract_report_answer(completion)
    worker = get_math_verify_worker()
    correct = bool(worker.verify(prediction, answer) or worker.verify(completion, answer))
    format_ok = has_report_format(completion)
    reward = (1.0 if correct else 0.0) + (0.1 if format_ok else 0.0)
    return reward, correct, format_ok, prediction


def _evaluate_benchmark(
    model,
    tokenizer,
    model_path: str,
    benchmark: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = get_named_eval_dataset(
        name=benchmark,
        tokenizer=tokenizer,
        max_length=args.max_prompt_length,
        limit=args.limit,
        seed=args.seed,
    )
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
                reward, correct, format_ok, prediction = _verify(completion, str(item["answer"]))
                rows.append(
                    {
                        "benchmark": benchmark,
                        "question": item["question"],
                        "answer": item["answer"],
                        "prediction": prediction,
                        "correct": correct,
                        "format_ok": format_ok,
                        "reward": reward,
                        "source": item.get("source", ""),
                        "level": item.get("level", ""),
                        "type": item.get("type", ""),
                        "completion": completion,
                    }
                )
            print(
                f"{benchmark}: evaluated {min(offset + args.batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )

    elapsed = time.monotonic() - start
    n = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    formatted = sum(1 for row in rows if row["format_ok"])
    metrics = {
        "benchmark": benchmark,
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "format_rate": formatted / n if n else 0.0,
        "mean_reward": sum(float(row["reward"]) for row in rows) / n if n else 0.0,
        "elapsed_sec": elapsed,
        "model_path": model_path,
        "dataset_limit": args.limit,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }
    return metrics, rows


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_hf_snapshot(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype(args.dtype),
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model.eval()

    all_metrics: list[dict[str, Any]] = []
    for benchmark in args.benchmarks:
        metrics, rows = _evaluate_benchmark(model, tokenizer, model_path, benchmark, args)
        all_metrics.append(metrics)
        (out_dir / f"{benchmark}_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.write_predictions:
            with (out_dir / f"{benchmark}_predictions.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, indent=2, sort_keys=True))

    (out_dir / "summary.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
