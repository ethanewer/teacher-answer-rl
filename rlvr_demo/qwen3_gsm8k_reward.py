"""Reward function for the Qwen3 GSM8K RLVR demo."""

from __future__ import annotations

import re
from fractions import Fraction

from areal.utils import logging

logger = logging.getLogger("Qwen3GSM8KReward")

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
_FINAL_ANSWER_RE = re.compile(
    r"Final answer:\s*(?:\\boxed\{\s*)?"
    r"(?P<answer>[+-]?(?:\d[\d,]*|\d)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)"
    r"\s*(?:\})?\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*Final answer:\s*"
    r"[+-]?(?:\d[\d,]*|\d)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_NUMBER_RE = re.compile(
    r"[+-]?(?:\d[\d,]*|\d)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?"
)


def _clean_text(text: str) -> str:
    text = _SPECIAL_TOKEN_RE.sub("", text)
    return text.strip()


def _canonical_number(text: str) -> Fraction | None:
    text = text.strip().replace(",", "")
    text = text.strip("$")
    if not text:
        return None
    try:
        return Fraction(text)
    except Exception:
        return None


def extract_final_answer(completion: str) -> str | None:
    """Extract the report-style final numeric answer from a completion."""
    completion = _clean_text(completion)
    match = _FINAL_ANSWER_RE.search(completion)
    if match is not None:
        return match.group("answer")

    # Fallback for evaluation diagnostics: use the last numeric string.
    matches = list(_NUMBER_RE.finditer(completion))
    if not matches:
        return None
    return matches[-1].group(0)


def has_report_format(completion: str) -> bool:
    return bool(_FORMAT_RE.match(_clean_text(completion)))


def numeric_equal(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    pred_num = _canonical_number(prediction)
    gold_num = _canonical_number(gold)
    if pred_num is None or gold_num is None:
        return False
    return pred_num == gold_num


def qwen3_gsm8k_reward_fn(
    prompt,
    completions,
    prompt_ids,
    completion_ids,
    answer,
    **kwargs,
) -> float:
    """Dominant exact-answer reward with a small report-format bonus."""
    del prompt, prompt_ids, completion_ids, kwargs
    try:
        completion = str(completions)
        final_answer = extract_final_answer(completion)
        correct = numeric_equal(final_answer, str(answer))
        format_ok = has_report_format(completion)
        return (1.0 if correct else 0.0) + (0.1 if format_ok else 0.0)
    except Exception:
        logger.warning("Exception in qwen3_gsm8k_reward_fn", exc_info=True)
        return 0.0

