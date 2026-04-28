"""Dataset and prompt helpers for the Qwen3 GSM8K RLVR demo."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset


REPORT_PROMPT_TEMPLATE = """Please solve the math problem step by step. Use <think> tags to show your reasoning process, then provide the final numerical answer.

Problem:
{question}

IMPORTANT: Your final answer must be a pure number only.

Format:
<think>
Your step-by-step reasoning here...
</think>
Final answer: [pure number only]"""


def extract_gold_answer(answer: str) -> str:
    """Extract the GSM8K final numeric answer after the #### marker."""
    if "####" in answer:
        return answer.rsplit("####", 1)[1].strip()
    return answer.strip()


def build_report_prompt(question: str) -> str:
    return REPORT_PROMPT_TEMPLATE.format(question=question.strip())


def get_qwen3_gsm8k_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    limit: int | None = None,
    seed: int = 1,
    shuffle_limit: bool = False,
    **_: Any,
) -> Dataset:
    """Load GSM8K in the report prompt format expected by the demo workflow."""
    dataset = load_dataset(path=path, name="main", split=split)

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when set, got {limit}")
        if shuffle_limit:
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(min(limit, len(dataset))))

    def process(sample: dict[str, Any]) -> dict[str, Any]:
        question = sample["question"]
        messages = [{"role": "user", "content": build_report_prompt(question)}]
        return {
            "messages": messages,
            "answer": extract_gold_answer(sample["answer"]),
            "question": question,
        }

    keep = {"messages", "answer", "question"}
    remove_columns = [col for col in dataset.column_names if col not in keep]
    dataset = dataset.map(process, remove_columns=remove_columns)

    if max_length is not None:

        def filter_length(sample: dict[str, Any]) -> bool:
            input_ids = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            return len(input_ids) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset

