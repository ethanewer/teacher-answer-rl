"""SFT dataset helpers for the Qwen3 GSM8K comparison."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from rlvr_demo.qwen3_gsm8k_data import build_report_prompt, extract_gold_answer

_QUESTION_RE = re.compile(r"Problem:\n(?P<question>.*?)\n\nIMPORTANT:", re.DOTALL)


def extract_question_from_prompt(prompt: str) -> str:
    match = _QUESTION_RE.search(prompt)
    if match is None:
        raise ValueError("Could not extract GSM8K question from rollout prompt")
    return match.group("question").strip()


def collect_rollout_questions(rollout_dir: str | Path) -> list[str]:
    """Collect unique GSM8K questions from AReaL train rollout dumps in order."""
    rollout_path = Path(rollout_dir)
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout directory does not exist: {rollout_path}")

    questions: list[str] = []
    seen: set[str] = set()
    step_dirs = sorted(
        [path for path in rollout_path.iterdir() if path.is_dir()],
        key=lambda path: int(path.name),
    )
    for step_dir in step_dirs:
        files = sorted(step_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
        for path in files:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    question = extract_question_from_prompt(json.loads(line)["prompt"])
                    if question in seen:
                        continue
                    seen.add(question)
                    questions.append(question)
                    break
    return questions


def load_gsm8k_answers(path: str = "openai/gsm8k") -> dict[str, str]:
    dataset = load_dataset(path=path, name="main", split="train")
    return {
        str(row["question"]).strip(): extract_gold_answer(str(row["answer"]))
        for row in dataset
    }


def matched_rollout_items(
    rollout_dir: str | Path,
    path: str = "openai/gsm8k",
    limit: int | None = None,
) -> list[dict[str, str]]:
    answers = load_gsm8k_answers(path)
    items: list[dict[str, str]] = []
    for question in collect_rollout_questions(rollout_dir):
        if question not in answers:
            raise KeyError(f"Question from rollout dump not found in GSM8K: {question}")
        items.append(
            {
                "question": question,
                "answer": answers[question],
                "prompt": build_report_prompt(question),
            }
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def _tokenize_sft_row(row: dict[str, Any], tokenizer, max_length: int | None):
    messages = [{"role": "user", "content": row["prompt"]}]
    assistant = {"role": "assistant", "content": row["completion"]}
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [*messages, assistant],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        full_text = tokenizer.apply_chat_template(
            [*messages, assistant],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("SFT completion produced no trainable assistant tokens")
    if max_length is not None and len(full_ids) > max_length:
        return None
    loss_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return {"input_ids": full_ids, "loss_mask": loss_mask}


def get_deepseek_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    limit: int | None = None,
    **_: Any,
) -> Dataset:
    """Load DeepSeek-generated GSM8K SFT JSONL and tokenize for AReaL SFT."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split for local SFT JSONL: {split}")

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            tokenized = _tokenize_sft_row(row, tokenizer, max_length)
            if tokenized is None:
                continue
            records.append(tokenized)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No usable SFT records found in {path}")
    return Dataset.from_list(records)
