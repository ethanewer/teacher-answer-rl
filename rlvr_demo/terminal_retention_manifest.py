"""Write deterministic retained-target manifests for stripped terminal SFT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rlvr_demo.terminal_agent_data import (
    TERMINAL_CORPUS,
    _coerce_messages,
    is_valid_stripped_terminus_assistant,
    stable_hash,
)


def _source_id(row: dict[str, Any], messages: list[dict[str, str]]) -> str:
    task = str(row.get("task", ""))
    episode = str(row.get("episode", ""))
    return stable_hash(f"{task}\n{episode}\n{json.dumps(messages, sort_keys=True)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=TERMINAL_CORPUS)
    parser.add_argument("--config", default="skill_based_medium")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, name=args.config, split=args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "filter": "strip_think_valid_terminus_json_v1",
        "generated_unix_time": time.time(),
        "rows_seen": 0,
        "rows_retained": 0,
        "assistant_turns_seen": 0,
        "assistant_turns_retained": 0,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "metadata", **summary}, sort_keys=True) + "\n")
        for row_idx, row in enumerate(dataset):
            raw = dict(row)
            messages = _coerce_messages(raw.get("conversations") or [])
            assistant_indices: list[int] = []
            assistant_count = 0
            for msg_idx, msg in enumerate(messages):
                if msg["role"] != "assistant":
                    continue
                assistant_count += 1
                if is_valid_stripped_terminus_assistant(msg["content"]):
                    assistant_indices.append(msg_idx)

            summary["rows_seen"] += 1
            summary["assistant_turns_seen"] += assistant_count
            summary["assistant_turns_retained"] += len(assistant_indices)
            if not assistant_indices:
                continue
            summary["rows_retained"] += 1
            handle.write(
                json.dumps(
                    {
                        "type": "row",
                        "row_idx": row_idx,
                        "source_id": _source_id(raw, messages),
                        "assistant_turns_seen": assistant_count,
                        "retained_assistant_indices": assistant_indices,
                        "retained_assistant_count": len(assistant_indices),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if (row_idx + 1) % 25000 == 0:
                print(f"scanned {row_idx + 1} rows", flush=True)

    summary["row_retention_rate"] = (
        summary["rows_retained"] / summary["rows_seen"] if summary["rows_seen"] else 0.0
    )
    summary["assistant_turn_retention_rate"] = (
        summary["assistant_turns_retained"] / summary["assistant_turns_seen"]
        if summary["assistant_turns_seen"]
        else 0.0
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
