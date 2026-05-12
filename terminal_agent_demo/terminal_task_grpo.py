"""GRPO workflow for Terminus-style terminal tasks.

The workflow samples the same JSON command protocol used by Terminus-2:

```
<think>...</think>
{
  "analysis": "...",
  "plan": "...",
  "commands": [{"keystrokes": "ls -la \n", "duration": 0.1}],
  "task_complete": false
}
```

Each sampled assistant turn is executed in a Terminal-Bench task environment.
The final reward is the task verifier pass ratio, propagated back across turns.
"""

from __future__ import annotations

import asyncio
import copy
import csv
import datetime as _datetime
import fcntl
import json
import os
import random
import shutil
import stat
import textwrap
import tomllib
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from datasets import Dataset
from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GRPOConfig, GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal.utils.perf_tracer import atrace_scope, atrace_session_phase, session_context


TERMINUS_SYSTEM_PROMPT = """You are Terminus-2, a terminal automation agent.
You must solve the user's Linux task by emitting exactly one JSON command payload per turn.

Output format:
<think>
Brief private reasoning about the terminal state and next action.
</think>

{
  "analysis": "short state analysis",
  "plan": "short next-step plan",
  "commands": [
    {
      "keystrokes": "shell command or keystrokes, ending with \\n when it should execute",
      "duration": 0.1
    }
  ],
  "task_complete": false
}

Use commands to inspect and modify the environment. Set task_complete to true only
when the task is finished. Do not use markdown fences around the JSON.
"""


DEFAULT_TBENCH_TASK_CACHE = Path(
    "/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminal-agent-demo/"
    "materialized_tbench_tasks"
)


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _link_or_copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _link_or_copy_tree(item, target)
        elif item.is_file():
            _link_or_copy_file(item, target)


def _yaml_block(value: str, indent: int = 2) -> str:
    return "|\n" + textwrap.indent(value.rstrip() + "\n", " " * indent)


def _task_difficulty(task_dir: Path) -> str:
    for part in task_dir.parts:
        if part in {"easy", "medium", "hard"}:
            return part
    return "medium"


def _task_category(task_dir: Path, tags: list[str]) -> str:
    for tag in tags:
        if tag != "datagen-flash":
            return tag
    if len(task_dir.parts) >= 2:
        return task_dir.parent.name
    return "terminal"


def _write_task_yaml(task_dir: Path, out_path: Path, task_toml: dict[str, Any]) -> None:
    instruction = _read_instruction(task_dir)
    metadata = task_toml.get("metadata") if isinstance(task_toml.get("metadata"), dict) else {}
    tags = [str(tag) for tag in metadata.get("tags", [])] if isinstance(metadata.get("tags"), list) else []
    verifier = task_toml.get("verifier") if isinstance(task_toml.get("verifier"), dict) else {}
    agent = task_toml.get("agent") if isinstance(task_toml.get("agent"), dict) else {}
    max_agent_timeout = int(float(agent.get("timeout_sec", 900.0)))
    max_test_timeout = int(float(verifier.get("timeout_sec", 900.0)))
    category = _task_category(task_dir, tags)
    difficulty = _task_difficulty(task_dir)
    tag_lines = "\n".join(f"  - {tag}" for tag in tags) if tags else "  - synthetic"
    out_path.write_text(
        "\n".join(
            [
                f"instruction: {_yaml_block(instruction)}",
                f"difficulty: {difficulty}",
                f"category: {category}",
                "tags:",
                tag_lines,
                "parser_name: pytest",
                f"max_agent_timeout_sec: {max_agent_timeout}",
                f"max_test_timeout_sec: {max_test_timeout}",
                "run_tests_in_same_shell: false",
                "disable_asciinema: true",
                f"estimated_duration_sec: {max_agent_timeout}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_docker_compose(out_path: Path) -> None:
    out_path.write_text(
        """services:
  client:
    build:
      context: .
      dockerfile: Dockerfile
    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    working_dir: /app
    command: tail -f /dev/null
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
      - ${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}
""",
        encoding="utf-8",
    )


def ensure_terminal_bench_task_layout(task_dir: Path) -> Path:
    """Return a Terminal-Bench-compatible task directory.

    Nemotron-Terminal-Synthetic-Tasks stores executable tasks as
    ``task.toml`` + ``environment/Dockerfile`` + ``tests/test.sh``. The
    Terminal-Bench runner used by GRPO expects ``task.yaml``, a top-level
    Dockerfile/docker-compose pair, ``run-tests.sh``, and ``tests/``. This
    helper materializes a compatible hardlink/copy cache lazily per task.
    """
    task_dir = task_dir.resolve()
    if (task_dir / "task.yaml").exists():
        return task_dir
    task_toml_path = task_dir / "task.toml"
    dockerfile = task_dir / "environment" / "Dockerfile"
    tests_dir = task_dir / "tests"
    run_tests = tests_dir / "test.sh"
    if not (task_toml_path.exists() and dockerfile.exists() and tests_dir.exists() and run_tests.exists()):
        return task_dir

    cache_root = Path(os.environ.get("TERMINAL_AGENT_TBENCH_TASK_CACHE", str(DEFAULT_TBENCH_TASK_CACHE)))
    task_hash = uuid.uuid5(uuid.NAMESPACE_URL, str(task_dir)).hex[:12]
    materialized = cache_root / f"{task_dir.name}-{task_hash}"
    marker = materialized / ".terminal_bench_layout_ready"
    if marker.exists():
        return materialized

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / f"{materialized.name}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return materialized
        materialized.mkdir(parents=True, exist_ok=True)
        task_toml = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
        _link_or_copy_file(task_dir / "instruction.md", materialized / "instruction.md")
        _link_or_copy_file(dockerfile, materialized / "Dockerfile")
        if (task_dir / "environment" / "files").exists():
            _link_or_copy_tree(task_dir / "environment" / "files", materialized / "files")
        _link_or_copy_tree(tests_dir, materialized / "tests")
        _link_or_copy_file(run_tests, materialized / "run-tests.sh")
        mode = (materialized / "run-tests.sh").stat().st_mode
        (materialized / "run-tests.sh").chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        _write_docker_compose(materialized / "docker-compose.yaml")
        _write_task_yaml(task_dir, materialized / "task.yaml", task_toml)
        marker.write_text(str(task_dir) + "\n", encoding="utf-8")
    return materialized


@dataclass
class TerminalTaskTimeouts:
    reset_env: float = 1800.0
    reset_agent: float = 120.0
    agent_step: float = 300.0
    command: float = 180.0
    verifier: float = 1200.0
    cleanup: float | None = None


@dataclass
class TerminalTaskGRPOConfig(GRPOConfig):
    n_trajs: int = field(default=1)
    max_turns: int = field(default=25)
    max_workers: int = field(default=16)
    max_tokens_per_trajectory: int = field(default=32768)
    observation_max_chars: int = field(default=8000)
    turn_discount: float = field(default=0.9)
    task_timeouts: TerminalTaskTimeouts = field(default_factory=TerminalTaskTimeouts)
    filter_uniform_reward: bool = field(default=False)
    encourage_completion_reward: bool = field(default=False)


class TerminusPayloadError(ValueError):
    """Raised when an assistant response is not a Terminus JSON payload."""


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise TerminusPayloadError("no JSON object found")
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise TerminusPayloadError(str(exc)) from exc
    if not isinstance(obj, dict):
        raise TerminusPayloadError("top-level JSON value is not an object")
    return obj


def parse_terminus_payload(text: str) -> tuple[list[dict[str, Any]], bool]:
    payload = _extract_json_object(text)
    commands = payload.get("commands")
    if commands is None:
        raise TerminusPayloadError("payload has no commands field")
    if not isinstance(commands, list):
        raise TerminusPayloadError("commands field is not a list")

    parsed: list[dict[str, Any]] = []
    for idx, raw in enumerate(commands):
        if not isinstance(raw, dict):
            raise TerminusPayloadError(f"commands[{idx}] is not an object")
        keystrokes = raw.get("keystrokes")
        if not isinstance(keystrokes, str):
            raise TerminusPayloadError(f"commands[{idx}].keystrokes is not a string")
        duration_raw = raw.get("duration", 0.1)
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError) as exc:
            raise TerminusPayloadError(f"commands[{idx}].duration is invalid") from exc
        parsed.append({"keystrokes": keystrokes, "duration": max(duration, 0.0)})

    return parsed, bool(payload.get("task_complete", False))


def _read_instruction(task_dir: Path) -> str:
    for name in ("instruction.md", "task.md", "README.md"):
        path = task_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace").strip()
    raise FileNotFoundError(f"No instruction.md/task.md/README.md in {task_dir}")


def _task_dirs_from_manifest(manifest_path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = Path(str(row.get("path", ""))).expanduser()
            if not path.is_absolute():
                path = (manifest_path.parent / path).resolve()
            name = str(row.get("task_name") or path.name)
            rows.append((name, path))
    return rows


def _discover_task_dirs(root: Path) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for instruction in root.rglob("instruction.md"):
        task_dir = instruction.parent
        if (task_dir / "environment").exists() and (task_dir / "tests").exists():
            tasks.append((task_dir.name, task_dir))
    return sorted(set(tasks), key=lambda item: item[0])


def get_terminal_synthetic_task_dataset(
    path: str,
    split: str = "train",
    seed: int = 1,
    limit: int | None = None,
    split_part: str | None = None,
    holdout_size: int = 128,
    shuffle_records: bool = True,
    **_: Any,
) -> Dataset:
    """Load local Nemotron Terminal synthetic task directories for GRPO.

    ``path`` may be either a manifest CSV produced by
    ``terminal_experiment prepare-synthetic-tasks`` or a directory containing
    task subdirectories.
    """
    del split
    root = Path(path).expanduser().resolve()
    if root.is_file():
        task_dirs = _task_dirs_from_manifest(root)
    else:
        task_dirs = _discover_task_dirs(root)

    if not task_dirs:
        raise ValueError(f"No synthetic terminal task directories found under {root}")

    if split_part is not None:
        if split_part not in {"train", "validation"}:
            raise ValueError("split_part must be 'train' or 'validation'")
        names = [name for name, _ in task_dirs]
        random.Random(seed).shuffle(names)
        holdout = set(names[: min(holdout_size, len(names))])
        if split_part == "validation":
            task_dirs = [item for item in task_dirs if item[0] in holdout]
        else:
            task_dirs = [item for item in task_dirs if item[0] not in holdout]

    if shuffle_records:
        random.Random(seed).shuffle(task_dirs)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when set, got {limit}")
        task_dirs = task_dirs[:limit]

    records = []
    for task_name, task_dir in task_dirs:
        records.append(
            {
                "task_name": task_name,
                "task_path": str(task_dir.resolve()),
                "instruction": _read_instruction(task_dir),
            }
        )
    return Dataset.from_list(records)


class TerminusTerminalTaskRunner:
    def __init__(
        self,
        output_path: str,
        max_turns: int,
        max_tokens_per_turn: int,
        temperature: float,
        top_p: float,
        observation_max_chars: int,
        task_timeouts: TerminalTaskTimeouts,
        encourage_completion_reward: bool,
        executor: ThreadPoolExecutor,
    ):
        self.output_path = output_path
        self.max_turns = max_turns
        self.max_tokens_per_turn = max_tokens_per_turn
        self.temperature = temperature
        self.top_p = top_p
        self.observation_max_chars = observation_max_chars
        self.task_timeouts = task_timeouts
        self.encourage_completion_reward = encourage_completion_reward
        self.executor = executor
        self.terminal: Terminal | None = None
        self.trial_handler: TrialHandler | None = None
        self.parser = None
        self.task_name = ""
        self.traj_i = 0

    async def run_in_executor(self, fn, *args, timeout: float | None = None, **kwargs):
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))
        if timeout is not None:
            return await asyncio.wait_for(task, timeout=timeout)
        return await task

    def _reset_env(self, data: dict[str, Any], uid: str) -> None:
        output_path = Path(self.output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        task_dir = ensure_terminal_bench_task_layout(Path(str(data["task_path"])))
        self.task_name = str(data["task_name"])
        self.trial_handler = TrialHandler(
            trial_name=f"{self.task_name}.{uid}.terminus-grpo",
            input_path=task_dir,
            output_path=output_path,
        )
        self.parser = ParserFactory.get_parser(self.trial_handler.task.parser_name)
        self.terminal = Terminal(
            client_container_name=self.trial_handler.client_container_name,
            client_image_name=self.trial_handler.client_image_name,
            docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
            docker_image_name_prefix=self.trial_handler.docker_image_name_prefix,
            sessions_logs_path=self.trial_handler.trial_paths.sessions_path,
            agent_logs_path=self.trial_handler.trial_paths.agent_logging_dir,
            commands_path=self.trial_handler.trial_paths.commands_path,
            no_rebuild=True,
            cleanup=False,
        )
        self.terminal.start()
        session = self.terminal.create_session("agent", is_active_stream=False)
        session.get_incremental_output()

    def _execute_commands(self, commands: Iterable[dict[str, Any]]) -> str:
        if self.terminal is None:
            raise RuntimeError("terminal is not initialized")
        session = self.terminal.get_session("agent")
        observations = []
        for command in commands:
            keystrokes = str(command["keystrokes"])
            duration = float(command.get("duration", 0.1))
            is_executing = keystrokes.endswith(("\n", "\r"))
            session.send_keys(
                [keystrokes],
                block=is_executing,
                min_timeout_sec=duration if not is_executing else 0.0,
                max_timeout_sec=self.task_timeouts.command,
            )
            observations.append(session.get_incremental_output())
        text = "\n\n".join(observations).strip()
        if len(text) > self.observation_max_chars:
            text = text[-self.observation_max_chars :]
        return text

    def _evaluate_completion_sync(self) -> float:
        if self.trial_handler is None or self.terminal is None or self.parser is None:
            raise RuntimeError("terminal environment is not initialized")

        paths = [self.trial_handler.task_paths.run_tests_path]
        if self.trial_handler.task_paths.test_dir.exists():
            paths.append(self.trial_handler.task_paths.test_dir)
        self.terminal.copy_to_container(
            paths=paths,
            container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
        )
        test_session = self.terminal.create_session(
            "tests",
            is_active_stream=False,
            as_configured_user=False,
        )
        test_script_path = str(DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh")
        try:
            test_session.send_keys(
                [f"bash {test_script_path}", "Enter"],
                block=True,
                max_timeout_sec=min(
                    self.task_timeouts.verifier,
                    4 * self.trial_handler.task.max_test_timeout_sec,
                ),
            )
            test_output = test_session.capture_pane(capture_entire=True)
            parser_results = self.parser.parse(test_output)
            pass_ratio = (
                sum(
                    1
                    for status in parser_results.values()
                    if status == UnitTestStatus.PASSED
                )
                / len(parser_results)
                if parser_results
                else 0.0
            )
        except Exception:
            pass_ratio = 0.0
        if self.encourage_completion_reward and pass_ratio == 1.0:
            pass_ratio += 1.0
        return float(pass_ratio)

    def _close_env(self) -> None:
        if self.terminal is not None:
            self.terminal.stop()
            self.terminal = None

    @session_context()
    async def run_agent(
        self,
        data: dict[str, Any],
        client: ArealOpenAI,
        uid: str,
        traj_i: int,
    ) -> float | None:
        self.traj_i = traj_i
        task_name = str(data.get("task_name"))
        messages = [
            {"role": "system", "content": TERMINUS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current date: {_datetime.date.today().isoformat()}\n"
                    f"We are in the root directory /app.\n\n"
                    f"Task name: {task_name}\n"
                    f"Task instruction:\n{data['instruction']}"
                ),
            },
        ]
        try:
            async with atrace_scope(
                f"reset_env:{task_name},traj:{traj_i}",
                args={"uid": uid, "timeout": self.task_timeouts.reset_env},
            ):
                await self.run_in_executor(
                    self._reset_env,
                    data,
                    uid,
                    timeout=self.task_timeouts.reset_env,
                )

            reward: float | None = 0.0
            for turn in range(self.max_turns):
                response = await client.chat.completions.create(
                    messages=messages,
                    max_completion_tokens=self.max_tokens_per_turn,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                content = response.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": content})
                try:
                    commands, task_complete = parse_terminus_payload(content)
                except TerminusPayloadError as exc:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Invalid Terminus JSON payload: "
                                f"{exc}. Emit a valid payload with commands."
                            ),
                        }
                    )
                    continue

                if commands:
                    observation = await self.run_in_executor(
                        self._execute_commands,
                        commands,
                        timeout=self.task_timeouts.command * max(len(commands), 1) + 10,
                    )
                else:
                    observation = "No commands were executed."
                messages.append(
                    {
                        "role": "user",
                        "content": f"Terminal observation after turn {turn + 1}:\n{observation}",
                    }
                )
                if task_complete:
                    break

            async with atrace_session_phase(
                "reward",
                start_payload={"task_name": task_name, "traj_i": traj_i},
            ):
                reward = await self.run_in_executor(
                    self._evaluate_completion_sync,
                    timeout=self.task_timeouts.verifier,
                )
            client.set_last_reward(float(reward))
            return float(reward)
        except TimeoutError:
            return None
        except Exception as exc:
            print(f"Terminus GRPO task {task_name} failed: {exc}")
            return None
        finally:
            try:
                await self.run_in_executor(
                    self._close_env,
                    timeout=self.task_timeouts.cleanup,
                )
            except Exception as exc:
                print(f"Terminus GRPO cleanup failed for {task_name}: {exc}")


class TerminusTerminalGRPOWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
        max_turns: int = 25,
        max_tokens_per_trajectory: int = 32768,
        max_workers: int = 16,
        observation_max_chars: int = 8000,
        turn_discount: float = 0.9,
        task_timeouts: TerminalTaskTimeouts | None = None,
        filter_uniform_reward: bool = False,
        encourage_completion_reward: bool = False,
    ):
        # AReaL's trainer uses config.gconfig.n_samples as the GRPO group size.
        # Keep that shared config intact and use a private one-sample generation
        # config inside each grouped rollout worker.
        self.gconfig = gconfig.new(n_samples=1) if hasattr(gconfig, "new") else copy.copy(gconfig)
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir or "terminal_grpo_generated"
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
        self.rollout_stat_scope = rollout_stat_scope
        self.n_trajs = n_trajs
        self.max_turns = max_turns
        self.max_tokens_per_trajectory = max_tokens_per_trajectory
        self.max_workers = max_workers
        self.observation_max_chars = observation_max_chars
        self.turn_discount = turn_discount
        self.task_timeouts = task_timeouts or TerminalTaskTimeouts()
        self.filter_uniform_reward = filter_uniform_reward
        self.encourage_completion_reward = encourage_completion_reward
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                reasoning_parser="qwen3",
                engine_max_tokens=self.max_tokens_per_trajectory,
                chat_template_type="hf",
            )
            for _ in range(self.n_trajs)
        ]
        uids = [uuid.uuid4().hex[:8] for _ in range(self.n_trajs)]
        rewards = await asyncio.gather(
            *[
                TerminusTerminalTaskRunner(
                    output_path=os.path.join(self.dump_dir, "TerminusTerminalTaskRunner"),
                    max_turns=self.max_turns,
                    max_tokens_per_turn=self.gconfig.max_new_tokens,
                    temperature=self.gconfig.temperature,
                    top_p=self.gconfig.top_p,
                    observation_max_chars=self.observation_max_chars,
                    task_timeouts=self.task_timeouts,
                    encourage_completion_reward=self.encourage_completion_reward,
                    executor=self.executor,
                ).run_agent(data=data, client=clients[i], uid=uids[i], traj_i=i)
                for i in range(self.n_trajs)
            ]
        )

        if self.filter_uniform_reward:
            valid_rewards = [reward for reward in rewards if reward is not None]
            if not valid_rewards or all(reward == valid_rewards[0] for reward in valid_rewards):
                return None

        completions_with_reward = {}
        for idx, (reward, client) in enumerate(zip(rewards, clients)):
            if reward is None:
                continue
            stats_tracker.get(workflow_context.stat_scope()).scalar(reward=float(reward))
            client.apply_reward_discount(turn_discount=self.turn_discount)
            completions_with_reward.update(client.export_interactions(style="individual"))

        stats_tracker.get(workflow_context.stat_scope()).scalar(
            num_full_passes=sum(1 for reward in rewards if reward == 1.0)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            num_trajectories_failed=sum(1 for reward in rewards if reward is None)
        )
        return completions_with_reward or None


__all__ = [
    "TerminalTaskGRPOConfig",
    "TerminalTaskTimeouts",
    "TerminusTerminalGRPOWorkflow",
    "get_terminal_synthetic_task_dataset",
    "parse_terminus_payload",
]
