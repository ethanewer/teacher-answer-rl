"""Generate matched GSM8K SFT data with DeepSeek-V4-Pro thinking mode."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx

from rlvr_demo.qwen3_gsm8k_reward import extract_final_answer, numeric_equal
from rlvr_demo.sft_data import matched_rollout_items


SYSTEM_PROMPT = """You are generating supervised fine-tuning data for a small Qwen3 math reasoning model.
Solve the grade-school math problem carefully.
Your final visible answer must be exactly:
Final answer: <pure number only>
Do not include units, commas, currency symbols, markdown, or extra explanation in the final visible answer."""


def _read_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


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


def _completion_from_response(message: dict[str, Any], gold_answer: str) -> tuple[str, str]:
    reasoning = str(message.get("reasoning_content") or "").strip()
    content = str(message.get("content") or "").strip()
    answer = extract_final_answer(content) or extract_final_answer(reasoning)
    if answer is None:
        answer = gold_answer
    completion = f"<think>\n{reasoning or content}\n</think>\nFinal answer: {answer}"
    return completion, answer


async def _call_teacher(
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    item: dict[str, str],
    max_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["prompt"]},
    ]
    body = {
        "model": "deepseek-v4-pro",
        "messages": messages,
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
        "max_tokens": max_tokens,
    }
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
            completion, prediction = _completion_from_response(message, item["answer"])
            correct = numeric_equal(prediction, item["answer"])
            if not correct and attempt < max_retries:
                body["messages"] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{item['prompt']}\n\nRe-solve from scratch and check the arithmetic. "
                            "The final visible answer must still be exactly `Final answer: <pure number only>`."
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
        for item in matched_rollout_items(args.rollout_dir, limit=args.limit)
        if item["question"] not in done
    ]
    if not items:
        print(f"No remaining items to generate; existing file is {args.output}")
        return

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    counters = {"ok": 0, "wrong": 0, "error": 0}
    started = time.time()

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:

        async def worker(index: int, item: dict[str, str]) -> None:
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
                            f"error={counters.get('error', 0)} elapsed={elapsed:.1f}s"
                        )

        await asyncio.gather(*(worker(i, item) for i, item in enumerate(items)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path(
            "/NHNHOME/areal_runs/qwen3-gsm8k-rlvr/logs/ewer/"
            "qwen3-06b-gsm8k-grpo-b200-fast10/trial0/rollout"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rlvr_demo/data/deepseek_v4_pro_gsm8k_sft_matched.jsonl"),
    )
    parser.add_argument("--env", type=Path, default=Path("../.env"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=16)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry rows that already have wrong/error status in the output JSONL.",
    )
    args = parser.parse_args()
    asyncio.run(_generate(args))


if __name__ == "__main__":
    main()
