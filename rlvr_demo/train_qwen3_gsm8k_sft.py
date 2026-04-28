"""AReaL SFT training entry point for matched DeepSeek GSM8K data."""

from __future__ import annotations

import sys
from typing import Any

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.utils.hf_utils import load_hf_tokenizer

from rlvr_demo.model_paths import localize_model_paths
from rlvr_demo.sft_data import get_deepseek_gsm8k_sft_dataset


def _dataset_kwargs(dataset_config) -> dict[str, Any]:
    return dict(getattr(dataset_config, "dataset_kwargs", {}) or {})


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, SFTConfig)
    localize_model_paths(config)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_deepseek_gsm8k_sft_dataset(
        path=config.train_dataset.path,
        split=config.train_dataset.split,
        tokenizer=tokenizer,
        max_length=config.train_dataset.max_length,
        **_dataset_kwargs(config.train_dataset),
    )

    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_deepseek_gsm8k_sft_dataset(
            path=config.valid_dataset.path,
            split=config.valid_dataset.split,
            tokenizer=tokenizer,
            max_length=config.valid_dataset.max_length,
            **_dataset_kwargs(config.valid_dataset),
        )

    with SFTTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
