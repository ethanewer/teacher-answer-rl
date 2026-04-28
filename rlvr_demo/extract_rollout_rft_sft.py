"""Extract verifier-correct rollout completions into an SFT JSONL file."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from rlvr_demo.multi_math_data import load_math_records, question_hash


USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_end|>\n<|im_start|>assistant\n"
QUESTION_RE = re.compile(r"Problem:\n(?P<question>.*?)\n\nIMPORTANT:", re.DOTALL)

ALLOWED_SOURCE_PRESETS: dict[str, list[dict[str, Any]]] = {
    "aime_hardmath_pre2024": [
        {"name": "aime_historical", "split": "train", "max_year": 2021},
        {"name": "aime_recent", "split": "train", "min_year": 2022, "max_year": 2023},
        {"name": "math", "split": "train", "levels": ["Level 3", "Level 4", "Level 5"]},
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", required=True, help="AReaL rollout log directory.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--max-per-question", type=int, default=2)
    parser.add_argument("--max-tail-version", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--allowed-source-preset",
        choices=["none", *sorted(ALLOWED_SOURCE_PRESETS)],
        default="none",
        help="Only keep rollouts whose prompt exactly matches an allowed training source preset.",
    )
    parser.add_argument(
        "--allowed-question-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL with a question field; only matching rollout prompts are kept.",
    )
    parser.add_argument(
        "--fail-on-disallowed",
        action="store_true",
        help="Exit nonzero if any reward-passing rollout prompt is outside the allowed set.",
    )
    return parser.parse_args()


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


def _jsonl_question_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question") or "").strip()
            if question:
                hashes.add(question_hash(question))
    return hashes


def _allowed_hashes(args: argparse.Namespace) -> set[str] | None:
    allowed: set[str] | None = None
    if args.allowed_source_preset != "none":
        records = load_math_records(ALLOWED_SOURCE_PRESETS[args.allowed_source_preset], args.seed)
        allowed = {question_hash(str(row["question"])) for row in records}
    if args.allowed_question_jsonl is not None:
        jsonl_hashes = _jsonl_question_hashes(args.allowed_question_jsonl)
        allowed = jsonl_hashes if allowed is None else allowed & jsonl_hashes
    return allowed


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    rows = _iter_rollouts(Path(args.rollout_dir))
    rng.shuffle(rows)
    allowed = _allowed_hashes(args)

    selected_by_question: dict[str, list[dict[str, Any]]] = {}
    skipped_not_allowed = 0
    disallowed_examples: list[dict[str, Any]] = []
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
        hsh = question_hash(question)
        if allowed is not None and hsh not in allowed:
            skipped_not_allowed += 1
            if len(disallowed_examples) < 5:
                disallowed_examples.append(
                    {
                        "task_id": row.get("task_id"),
                        "sample_idx": row.get("sample_idx"),
                        "tail_version": tail_version,
                        "question": question[:240],
                    }
                )
            continue
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

    if args.fail_on_disallowed and skipped_not_allowed:
        print(
            json.dumps(
                {
                    "error": "disallowed_reward_passing_rollouts",
                    "skipped_not_allowed": skipped_not_allowed,
                    "examples": disallowed_examples,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_rollouts": len(rows),
                "allowed_source_preset": args.allowed_source_preset,
                "allowed_questions": len(allowed) if allowed is not None else None,
                "skipped_not_allowed": skipped_not_allowed,
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
