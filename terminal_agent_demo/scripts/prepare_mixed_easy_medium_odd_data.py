"""Build 50/50 easy plus medium-odd training artifacts.

The TA-RL recipes consume converted teacher trajectories, while the GRPO
recipes consume executable synthetic-task manifests. This script creates both
views with deterministic sampling and preserves the teacher-reference cache for
fast TA-RL startup.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/data")
EASY_JSONL = DATA_ROOT / "skill_based_easy.terminus_tool.jsonl"
MEDIUM_JSONL = DATA_ROOT / "skill_based_medium.odd_original.terminus_tool.jsonl"
EASY_REFS = DATA_ROOT / "skill_based_easy.terminus_tool.jsonl.teacher_refs.v2.json"
MEDIUM_REFS = DATA_ROOT / "skill_based_medium.odd_original.terminus_tool.jsonl.teacher_refs.v2.json"
EASY_MANIFEST = Path("/wbl-fast/usrs/ee/teacher-answer-rl/terminal_synthetic_tasks/easy/manifest.csv")
MEDIUM_MANIFEST = DATA_ROOT / "skill_based_medium.odd_original.synthetic_tasks_manifest.csv"
OUTPUT_JSONL = DATA_ROOT / "skill_based_mixed_easy50_medium_odd50.terminus_tool.jsonl"
OUTPUT_MANIFEST = DATA_ROOT / "skill_based_mixed_easy50_medium_odd50.synthetic_tasks_manifest.csv"


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def _sample_indices(population_size: int, sample_size: int, *, seed: int) -> set[int]:
    if sample_size > population_size:
        raise ValueError(f"Cannot sample {sample_size} rows from {population_size} rows")
    rng = random.Random(seed)
    return set(rng.sample(range(population_size), sample_size))


def _write_jsonl_mix(
    *,
    easy_jsonl: Path,
    medium_jsonl: Path,
    output_jsonl: Path,
    seed: int,
) -> tuple[list[int], dict[tuple[str, int], int]]:
    easy_rows = _count_lines(easy_jsonl)
    medium_rows = _count_lines(medium_jsonl)
    target_each = min(easy_rows, medium_rows)
    easy_indices = set(range(target_each))
    medium_indices = _sample_indices(medium_rows, target_each, seed=seed)

    offsets: list[int] = []
    row_map: dict[tuple[str, int], int] = {}
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with easy_jsonl.open("rb") as easy_handle, medium_jsonl.open("rb") as medium_handle, output_jsonl.open("wb") as out:
        new_idx = 0
        easy_iter = (
            ("easy", idx, line)
            for idx, line in enumerate(easy_handle)
            if idx in easy_indices
        )
        medium_iter = (
            ("medium_odd", idx, line)
            for idx, line in enumerate(medium_handle)
            if idx in medium_indices
        )
        for easy_item, medium_item in zip(easy_iter, medium_iter, strict=True):
            for source, old_idx, line in (easy_item, medium_item):
                offsets.append(out.tell())
                row_map[(source, old_idx)] = new_idx
                out.write(line)
                new_idx += 1

    return offsets, row_map


def _load_refs(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        refs = json.load(handle)
    if refs.get("version") != 2:
        raise ValueError(f"Expected v2 teacher refs in {path}")
    if refs.get("teacher_turn_filter") not in (None, "all"):
        raise ValueError(f"Expected all-turn teacher refs in {path}")
    return refs


def _write_teacher_refs(
    *,
    output_jsonl: Path,
    output_refs: Path,
    offsets: list[int],
    row_map: dict[tuple[str, int], int],
    easy_refs_path: Path,
    medium_refs_path: Path,
) -> dict[str, int]:
    output_refs_list: list[list[int]] = []
    source_ref_counts: dict[str, int] = {}
    for source, refs_path in (("easy", easy_refs_path), ("medium_odd", medium_refs_path)):
        refs = _load_refs(refs_path)
        source_offset_to_idx = {int(offset): idx for idx, offset in enumerate(refs["offsets"])}
        kept = 0
        for old_offset, assistant_idx in refs["refs"]:
            old_idx = source_offset_to_idx.get(int(old_offset))
            if old_idx is None:
                continue
            new_idx = row_map.get((source, old_idx))
            if new_idx is None:
                continue
            output_refs_list.append([offsets[new_idx], assistant_idx])
            kept += 1
        source_ref_counts[source] = kept

    output_refs_list.sort()
    stat = output_jsonl.stat()
    cache = {
        "version": 2,
        "path": str(output_jsonl),
        "teacher_turn_filter": "all",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "offsets": offsets,
        "refs": output_refs_list,
    }
    output_refs.write_text(json.dumps(cache, separators=(",", ":")) + "\n", encoding="utf-8")
    source_ref_counts["total"] = len(output_refs_list)
    return source_ref_counts


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_manifest_mix(
    *,
    easy_manifest: Path,
    medium_manifest: Path,
    output_manifest: Path,
    seed: int,
) -> dict[str, int]:
    easy_rows = _read_csv(easy_manifest)
    medium_rows = _read_csv(medium_manifest)
    target_each = min(len(easy_rows), len(medium_rows))
    medium_indices = sorted(_sample_indices(len(medium_rows), target_each, seed=seed))
    medium_sample = [medium_rows[idx] for idx in medium_indices]

    fieldnames = [
        "task_name",
        "path",
        "difficulty",
        "source_task",
        "sft_trajectory_count",
        "first_converted_row",
        "source_row_index",
        "source_trial_name",
        "source_model",
        "source_agent",
        "synthetic_manifest_task_name",
    ]
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for easy, medium in zip(easy_rows[:target_each], medium_sample, strict=True):
            writer.writerow(
                {
                    "task_name": easy["task_name"],
                    "path": easy["path"],
                    "difficulty": "easy",
                    "source_task": easy["task_name"],
                    "sft_trajectory_count": "",
                    "first_converted_row": "",
                    "source_row_index": "",
                    "source_trial_name": "",
                    "source_model": "",
                    "source_agent": "",
                    "synthetic_manifest_task_name": easy["task_name"],
                }
            )
            row = {name: medium.get(name, "") for name in fieldnames}
            row["difficulty"] = "medium_odd"
            writer.writerow(row)

    return {"easy_tasks": target_each, "medium_odd_tasks": target_each, "total_tasks": target_each * 2}


def build(args: argparse.Namespace) -> None:
    offsets, row_map = _write_jsonl_mix(
        easy_jsonl=args.easy_jsonl,
        medium_jsonl=args.medium_jsonl,
        output_jsonl=args.output_jsonl,
        seed=args.seed,
    )
    ref_counts = _write_teacher_refs(
        output_jsonl=args.output_jsonl,
        output_refs=args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".teacher_refs.v2.json"),
        offsets=offsets,
        row_map=row_map,
        easy_refs_path=args.easy_refs,
        medium_refs_path=args.medium_refs,
    )
    manifest_counts = _write_manifest_mix(
        easy_manifest=args.easy_manifest,
        medium_manifest=args.medium_manifest,
        output_manifest=args.output_manifest,
        seed=args.seed,
    )
    source_counts = {"easy_rows": 0, "medium_odd_rows": 0}
    for source, _old_idx in row_map:
        if source == "easy":
            source_counts["easy_rows"] += 1
        elif source == "medium_odd":
            source_counts["medium_odd_rows"] += 1

    stat = args.output_jsonl.stat()
    summary = {
        "seed": args.seed,
        "easy_jsonl": str(args.easy_jsonl),
        "medium_odd_jsonl": str(args.medium_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "output_manifest": str(args.output_manifest),
        "output_teacher_refs": str(args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".teacher_refs.v2.json")),
        "jsonl_size": stat.st_size,
        "jsonl_mtime_ns": stat.st_mtime_ns,
        **source_counts,
        **manifest_counts,
        "teacher_refs": ref_counts,
    }
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--easy-jsonl", type=Path, default=EASY_JSONL)
    parser.add_argument("--medium-jsonl", type=Path, default=MEDIUM_JSONL)
    parser.add_argument("--easy-refs", type=Path, default=EASY_REFS)
    parser.add_argument("--medium-refs", type=Path, default=MEDIUM_REFS)
    parser.add_argument("--easy-manifest", type=Path, default=EASY_MANIFEST)
    parser.add_argument("--medium-manifest", type=Path, default=MEDIUM_MANIFEST)
    parser.add_argument("--output-jsonl", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outputs = [
        args.output_jsonl,
        args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".teacher_refs.v2.json"),
        args.output_jsonl.with_suffix(".summary.json"),
        args.output_manifest,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs without --force:\n{joined}")
    for path in existing:
        path.unlink()
    build(args)


if __name__ == "__main__":
    main()
