"""Generate mixed GSM8K/MATH SFT data with DeepSeek-V4-Pro thinking mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx

from areal.reward import get_math_verify_worker
from rlvr_demo.multi_math_data import (
    TEST_SOURCES,
    TRAIN_SOURCES,
    balanced_train_validation_hashes,
    build_multi_math_prompt,
    extract_boxed_answer,
    load_heldout_hashes,
    load_math_records,
    question_hash,
    record_bucket,
)
from rlvr_demo.multi_math_reward import extract_report_answer


SYSTEM_PROMPT = """You are generating supervised fine-tuning data for a small Qwen3 math reasoning model.
Solve the math problem carefully with concise but complete reasoning.
Your answer must use exactly this visible format:
<think>
step-by-step reasoning
</think>
Final answer: <answer only>
For exact math answers, use a simplified exact expression or LaTeX. Do not include units, markdown, or explanatory text after the final answer."""

FINAL_ANSWER_RE = re.compile(r"Final answer:\s*(?P<answer>.+?)\s*$", re.IGNORECASE | re.DOTALL)


def _read_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in records:
        hsh = question_hash(str(row["question"]))
        if hsh in seen:
            continue
        seen.add(hsh)
        unique.append(row)
    return unique


def _load_balanced_items(records_per_bucket: int, seed: int) -> list[dict[str, Any]]:
    heldout = load_heldout_hashes(TEST_SOURCES, seed)
    shared_validation = balanced_train_validation_hashes(seed)
    records = _dedupe(load_math_records(TRAIN_SOURCES, seed))
    records = [row for row in records if question_hash(str(row["question"])) not in heldout]
    records = [
        row for row in records if question_hash(str(row["question"])) not in shared_validation
    ]

    buckets: dict[str, list[dict[str, Any]]] = {
        "gsm8k": [],
        "math_l12": [],
        "math_l3": [],
        "math_l45": [],
    }
    for row in records:
        bucket = record_bucket(row)
        if bucket in buckets:
            buckets[bucket].append(row)

    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for bucket, rows in buckets.items():
        rng.shuffle(rows)
        selected = rows[: min(records_per_bucket, len(rows))]
        for row in selected:
            question = str(row["question"])
            items.append(
                {
                    "bucket": bucket,
                    "source": row.get("source", ""),
                    "level": row.get("level", ""),
                    "type": row.get("type", ""),
                    "question": question,
                    "answer": str(row["answer"]),
                    "prompt": build_multi_math_prompt(question),
                }
            )
    rng.shuffle(items)
    return items


def _already_done(path: Path, retry_failed: bool) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not retry_failed or row.get("status") == "ok":
                done.add(str(row["question"]))
    return done


def _strip_special(text: str) -> str:
    return re.sub(r"<\|[^>]+?\|>", "", text).strip()


def _extract_teacher_answer(text: str) -> str:
    cleaned = _strip_special(text)
    match = FINAL_ANSWER_RE.search(cleaned)
    if match is not None:
        return match.group("answer").strip()
    boxed = extract_boxed_answer(cleaned)
    if boxed and boxed != cleaned:
        return boxed
    for line in reversed(cleaned.splitlines()):
        candidate = line.strip().strip("$")
        if candidate:
            return candidate
    return cleaned


def _completion_from_response(message: dict[str, Any]) -> tuple[str, str]:
    reasoning = str(message.get("reasoning_content") or "").strip()
    content = str(message.get("content") or "").strip()
    combined = "\n".join(part for part in [reasoning, content] if part).strip()
    prediction = _extract_teacher_answer(content or reasoning)
    if "<think>" in content and "</think>" in content:
        completion = _strip_special(content)
    else:
        thought = reasoning or content
        completion = f"<think>\n{thought.strip()}\n</think>\nFinal answer: {prediction}"
    if "Final answer:" not in completion:
        completion = f"{completion.rstrip()}\nFinal answer: {prediction}"
    return completion, extract_report_answer(completion) or prediction or combined


async def _call_teacher(
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    item: dict[str, Any],
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["prompt"]},
    ]
    body: dict[str, Any] = {
        "model": "deepseek-v4-pro",
        "messages": messages,
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
        "max_tokens": max_tokens,
    }
    worker = get_math_verify_worker()
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=body,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"{response.status_code}: {response.text[:500]}")
            response.raise_for_status()
            payload = response.json()
            message = payload["choices"][0]["message"]
            completion, prediction = _completion_from_response(message)
            correct = bool(worker.verify(prediction, str(item["answer"])))
            if not correct and attempt < max_retries:
                body["messages"] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{item['prompt']}\n\nRe-solve from scratch and verify the final answer. "
                            "The final line must be exactly `Final answer: <answer only>`."
                        ),
                    },
                ]
                await asyncio.sleep(min(60.0, 2**attempt + random.random()))
                continue
            return {
                **item,
                "completion": completion,
                "teacher_content": message.get("content", ""),
                "teacher_reasoning": message.get("reasoning_content", ""),
                "teacher_prediction": prediction,
                "teacher_correct": correct,
                "status": "ok" if correct else "wrong",
                "model": "deepseek-v4-pro",
                "reasoning_effort": "high",
                "usage": payload.get("usage", {}),
            }
        except Exception as exc:
            last_error = str(exc)
            await asyncio.sleep(min(60.0, 2**attempt + random.random()))
    return {**item, "status": "error", "error": last_error}


async def _generate(args: argparse.Namespace) -> None:
    _read_env(args.env)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(f"DEEPSEEK_API_KEY not found in environment or {args.env}")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _already_done(args.output, retry_failed=args.retry_failed)
    items = [
        item
        for item in _load_balanced_items(args.records_per_bucket, args.seed)
        if item["question"] not in done
    ]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        print(f"No remaining items to generate; existing file is {args.output}")
        return

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    counters = {"ok": 0, "wrong": 0, "error": 0}
    started = time.time()

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:

        async def worker(index: int, item: dict[str, Any]) -> None:
            del index
            async with sem:
                row = await _call_teacher(
                    client,
                    api_key=api_key,
                    base_url=base_url,
                    item=item,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
                async with lock:
                    counters[row["status"]] = counters.get(row["status"], 0) + 1
                    with args.output.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    completed = sum(counters.values())
                    if completed % args.log_every == 0 or completed == len(items):
                        elapsed = time.time() - started
                        print(
                            f"{completed}/{len(items)} generated "
                            f"ok={counters.get('ok', 0)} wrong={counters.get('wrong', 0)} "
                            f"error={counters.get('error', 0)} elapsed={elapsed:.1f}s",
                            flush=True,
                        )

        await asyncio.gather(*(worker(i, item) for i, item in enumerate(items)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rlvr_demo/data/deepseek_v4_pro_multi_math_balanced_sft.jsonl"),
    )
    parser.add_argument("--env", type=Path, default=Path("../.env"))
    parser.add_argument("--records-per-bucket", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=32)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry rows that already have wrong/error status in the output JSONL.",
    )
    args = parser.parse_args()
    asyncio.run(_generate(args))


if __name__ == "__main__":
    main()
