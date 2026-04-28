"""Model path helpers for AReaL Megatron recipes."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


_SNAPSHOT_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.tiktoken",
    "tokenizer*",
    "vocab*",
    "merges.txt",
)


def resolve_hf_snapshot(model_path: str) -> str:
    """Return a local HF snapshot path when given a repo id."""
    if Path(model_path).exists():
        return model_path
    return snapshot_download(
        repo_id=model_path,
        allow_patterns=list(_SNAPSHOT_ALLOW_PATTERNS),
    )


def localize_model_paths(config) -> str:
    """Mutate an AReaL config so Megatron and SGLang share a local snapshot."""
    local_path = resolve_hf_snapshot(config.actor.path)
    config.actor.path = local_path
    config.tokenizer_path = local_path
    if getattr(config, "rollout", None) is not None:
        config.rollout.tokenizer_path = local_path
    if getattr(config, "sglang", None) is not None:
        config.sglang.model_path = local_path
    if getattr(config, "ref", None) is not None:
        config.ref.path = local_path
    return local_path
