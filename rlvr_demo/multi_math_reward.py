"""Reward function for mixed GSM8K/MATH RLVR experiments."""

from __future__ import annotations

import re

from areal.reward import get_math_verify_worker
from areal.utils import logging

logger = logging.getLogger("Qwen3MultiMathReward")

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
_FINAL_ANSWER_RE = re.compile(r"Final answer:\s*(?P<answer>.+?)\s*$", re.IGNORECASE | re.DOTALL)
_FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*Final answer:\s*.+?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _clean_text(text: str) -> str:
    return _SPECIAL_TOKEN_RE.sub("", text).strip()


def extract_report_answer(completion: str) -> str:
    completion = _clean_text(completion)
    match = _FINAL_ANSWER_RE.search(completion)
    if match is None:
        return completion
    return match.group("answer").strip()


def has_report_format(completion: str) -> bool:
    return bool(_FORMAT_RE.match(_clean_text(completion)))


def qwen3_multi_math_reward_fn(
    prompt,
    completions,
    prompt_ids,
    completion_ids,
    answer,
    **kwargs,
) -> float:
    del prompt, prompt_ids, completion_ids, kwargs
    try:
        completion = str(completions)
        worker = get_math_verify_worker()
        score = worker.verify(extract_report_answer(completion), str(answer))
        if score <= 0.0:
            score = worker.verify(completion, str(answer))
        return float(score) + (0.1 if has_report_format(completion) else 0.0)
    except Exception:
        logger.warning("Exception in qwen3_multi_math_reward_fn", exc_info=True)
        return 0.0
