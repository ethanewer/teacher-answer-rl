"""Extract verifier-correct rollout completions into an SFT JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_end|>\n<|im_start|>assistant\n"
QUESTION_RE = re.compile(r"Problem:\n(?P<question>.*?)\n\nIMPORTANT:", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True, help="AReaL rollout log directory.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--max-per-question", type=int, default=2)
    parser.add_argument("--max-tail-version", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _normalize(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip()).casefold()


def _question_hash(question: str) -> str:
    return hashlib.sha256(_normalize(question).encode("utf-8")).hexdigest()


def _extract_user_prompt(prompt: str) -> str:
    start = prompt.find(USER_START)
    end = prompt.rfind(ASSISTANT_START)
    if start >= 0 and end > start:
        return prompt[start + len(USER_START) : end]
    return prompt


def _extract_question(user_prompt: str) -> str:
    match = QUESTION_RE.search(user_prompt)
    if match is not None:
        return match.group("question").strip()
    return user_prompt.strip()


def _iter_rollouts(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    rows = _iter_rollouts(Path(args.rollout_dir))
    rng.shuffle(rows)

    selected_by_question: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        reward = float(row.get("reward") or 0.0)
        if reward < args.min_reward:
            continue
        tail_version = int(row.get("tail_version") or 0)
        if args.max_tail_version is not None and tail_version > args.max_tail_version:
            continue
        user_prompt = _extract_user_prompt(str(row.get("prompt") or ""))
        question = _extract_question(user_prompt)
        completion = str(row.get("completion") or "").strip()
        if not question or not completion:
            continue
        hsh = _question_hash(question)
        bucket = selected_by_question.setdefault(hsh, [])
        if len(bucket) >= args.max_per_question:
            continue
        bucket.append(
            {
                "bucket": "rollout_rft",
                "source": "areal_rollout",
                "level": "rollout_rft",
                "type": "RFT",
                "question": question,
                "prompt": user_prompt,
                "completion": completion,
                "teacher_correct": True,
                "status": "ok",
                "reward": reward,
                "tail_version": tail_version,
                "task_id": row.get("task_id"),
                "sample_idx": row.get("sample_idx"),
            }
        )

    selected = [item for bucket in selected_by_question.values() for item in bucket]
    selected.sort(key=lambda item: (str(item["question"]), int(item.get("sample_idx") or 0)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_rollouts": len(rows),
                "selected": len(selected),
                "unique_questions": len(selected_by_question),
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
