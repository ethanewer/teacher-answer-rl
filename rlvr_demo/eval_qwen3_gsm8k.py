"""Standalone AReaL/SGLang evaluation for the Qwen3 GSM8K RLVR demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, SGLangConfig, load_expr_config
from areal.engine import RemoteSGLangEngine
from areal.infra import LocalScheduler, RayScheduler, SlurmScheduler
from areal.utils import logging, seeding
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger

from rlvr_demo.model_paths import localize_model_paths
from rlvr_demo.qwen3_gsm8k_data import get_qwen3_gsm8k_dataset

logger = logging.getLogger("Qwen3GSM8KEval")


def _make_scheduler(config):
    if config.scheduler.type == "local":
        return LocalScheduler(exp_config=config)
    if config.scheduler.type == "ray":
        return RayScheduler(exp_config=config)
    if config.scheduler.type == "slurm":
        return SlurmScheduler(exp_config=config)
    raise ValueError(f"Unknown scheduler type: {config.scheduler.type}")


def _flatten_rewards(results: list[dict | None]) -> list[float]:
    rewards: list[float] = []
    for result in results:
        if result is None or "rewards" not in result:
            continue
        value = result["rewards"]
        if isinstance(value, torch.Tensor):
            rewards.extend(float(x) for x in value.flatten().cpu().tolist())
        else:
            rewards.append(float(value))
    return rewards


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, GRPOConfig)
    localize_model_paths(config)
    logging.setup_file_logging(f"{config.cluster.fileroot}/eval.log")
    seeding.set_random_seed(config.seed, key="qwen3-gsm8k-eval")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    valid_kwargs = dict(getattr(config.valid_dataset, "dataset_kwargs", {}) or {})
    valid_kwargs.setdefault("seed", config.seed)
    valid_dataset = get_qwen3_gsm8k_dataset(
        path=config.valid_dataset.path,
        split=config.valid_dataset.split,
        tokenizer=tokenizer,
        max_length=config.valid_dataset.max_length,
        **valid_kwargs,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=0,
        world_size=1,
        dataset_config=config.valid_dataset,
    )

    rollout_alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    if rollout_alloc.backend != "sglang":
        raise ValueError("This eval entry point is intentionally SGLang-only.")

    scheduler = _make_scheduler(config)
    server_args = SGLangConfig.build_args(
        sglang_config=config.sglang,
        tp_size=rollout_alloc.parallel.tp_size,
        base_gpu_id=0,
    )
    eval_rollout = RemoteSGLangEngine.as_controller(config.rollout, scheduler)

    workflow_kwargs = dict(
        reward_fn="rlvr_demo.qwen3_gsm8k_reward.qwen3_gsm8k_reward_fn",
        gconfig=config.eval_gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=True,
    )

    try:
        eval_rollout.initialize(role="eval-rollout", server_args=server_args)
        count = 0
        for data in valid_dataloader:
            for item in data:
                eval_rollout.submit(
                    item,
                    workflow="areal.workflow.rlvr.RLVRWorkflow",
                    workflow_kwargs=workflow_kwargs,
                    group_size=config.eval_gconfig.n_samples,
                    is_eval=True,
                )
                count += 1

        results = eval_rollout.wait(count, timeout=None)
        rewards = _flatten_rewards(results)
        n = len(rewards)
        correct = sum(1 for reward in rewards if reward >= 1.0)
        formatted = sum(1 for reward in rewards if reward > 0.0 and abs(reward % 1.0) > 0.05)
        metrics = {
            "n": n,
            "accuracy": correct / n if n else 0.0,
            "format_rate": formatted / n if n else 0.0,
            "mean_reward": sum(rewards) / n if n else 0.0,
            "model_path": config.sglang.model_path,
            "dataset_path": config.valid_dataset.path,
            "dataset_limit": valid_kwargs.get("limit"),
            "temperature": config.eval_gconfig.temperature,
            "top_p": config.eval_gconfig.top_p,
            "top_k": config.eval_gconfig.top_k,
            "max_new_tokens": config.eval_gconfig.max_new_tokens,
        }
        logger.info("Evaluation metrics: %s", metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))

        out_dir = Path(StatsLogger.get_log_path(config.stats_logger))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        eval_rollout.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
