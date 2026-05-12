"""Build GRPO task manifests matched to converted SFT trajectories.

The real terminal-agent SFT/teacher-answer-RL recipes train on converted
Nemotron-Terminal-Corpus rows. GRPO uses executable task directories from
Nemotron-Terminal-Synthetic-Tasks. The stable join key between the two datasets
is the synthetic task directory name, which appears as ``source_task`` in the
converted trajectory JSONL and as the basename of each synthetic task path.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SFT_JSONL = Path(
    "/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/"
    "skill_based_medium.even_original.terminus_tool.jsonl"
)
DEFAULT_SYNTHETIC_MANIFEST = Path(
    "/wbl-fast/usrs/ee/teacher-answer-rl/terminal_synthetic_tasks/medium/manifest.csv"
)
DEFAULT_OUTPUT = Path(
    "/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data/"
    "skill_based_medium.even_original.synthetic_tasks_manifest.csv"
)


def _read_sft_sources(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    sources: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for converted_idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            source_task = row.get("source_task")
            if not isinstance(source_task, str) or not source_task:
                continue
            counts[source_task] += 1
            sources.setdefault(
                source_task,
                {
                    "source_task": source_task,
                    "first_converted_row": converted_idx,
                    "source_row_index": row.get("source_row_index"),
                    "source_trial_name": row.get("source_trial_name"),
                    "source_model": row.get("source_model"),
                    "source_agent": row.get("source_agent"),
                },
            )
    if not sources:
        raise ValueError(f"No source_task values found in {path}")
    return sources, counts


def _read_synthetic_manifest(path: Path) -> dict[str, dict[str, str]]:
    by_task: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_path = row.get("path")
            if not task_path:
                continue
            source_task = Path(task_path).name
            by_task[source_task] = dict(row)
    if not by_task:
        raise ValueError(f"No synthetic tasks found in {path}")
    return by_task


def build_manifest(
    *,
    sft_jsonl: Path,
    synthetic_manifest: Path,
    output: Path,
    summary_output: Path,
) -> None:
    sft_sources, sft_counts = _read_sft_sources(sft_jsonl)
    synthetic_by_task = _read_synthetic_manifest(synthetic_manifest)
    matched_tasks = sorted(set(sft_sources).intersection(synthetic_by_task))
    missing_tasks = sorted(set(sft_sources).difference(synthetic_by_task))
    if not matched_tasks:
        raise ValueError(
            "No overlap between SFT source_task values and synthetic task manifest "
            f"basenames: {sft_jsonl} vs {synthetic_manifest}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_name",
        "path",
        "source_task",
        "sft_trajectory_count",
        "first_converted_row",
        "source_row_index",
        "source_trial_name",
        "source_model",
        "source_agent",
        "synthetic_manifest_task_name",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source_task in matched_tasks:
            sft_meta = sft_sources[source_task]
            synthetic = synthetic_by_task[source_task]
            writer.writerow(
                {
                    "task_name": source_task,
                    "path": synthetic["path"],
                    "source_task": source_task,
                    "sft_trajectory_count": sft_counts[source_task],
                    "first_converted_row": sft_meta.get("first_converted_row"),
                    "source_row_index": sft_meta.get("source_row_index"),
                    "source_trial_name": sft_meta.get("source_trial_name"),
                    "source_model": sft_meta.get("source_model"),
                    "source_agent": sft_meta.get("source_agent"),
                    "synthetic_manifest_task_name": synthetic.get("task_name"),
                }
            )

    summary = {
        "sft_jsonl": str(sft_jsonl),
        "synthetic_dataset": "nvidia/Nemotron-Terminal-Synthetic-Tasks",
        "synthetic_manifest": str(synthetic_manifest),
        "output": str(output),
        "sft_unique_source_tasks": len(sft_sources),
        "synthetic_manifest_tasks": len(synthetic_by_task),
        "matched_tasks": len(matched_tasks),
        "missing_sft_source_tasks": len(missing_tasks),
        "duplicate_sft_trajectories_for_matched_tasks": sum(
            count - 1 for task, count in sft_counts.items() if task in matched_tasks and count > 1
        ),
        "first_matched_tasks": matched_tasks[:20],
        "first_missing_sft_source_tasks": missing_tasks[:20],
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-jsonl", type=Path, default=DEFAULT_SFT_JSONL)
    parser.add_argument("--synthetic-manifest", type=Path, default=DEFAULT_SYNTHETIC_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    build_manifest(
        sft_jsonl=args.sft_jsonl,
        synthetic_manifest=args.synthetic_manifest,
        output=args.output,
        summary_output=summary_output,
    )


if __name__ == "__main__":
    main()
