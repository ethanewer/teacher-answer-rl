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
import csv
import datetime as _datetime
import json
import os
import random
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import httpx
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
    trajectory_timeout: float | None = field(default=None)
    filter_uniform_reward: bool = field(default=False)
    encourage_completion_reward: bool = field(default=False)
    terminal_task_service_url: str | None = field(default=None)
    terminal_task_service_url_file: str | None = field(default=None)
    enable_thinking: bool = field(default=False)
    agent_harness: str = field(default="terminus")
    tool_call_parser: str = field(default="qwen3_xml")
    reasoning_parser: str = field(default="qwen3")
    chat_template_type: str = field(default="hf")
    export_style: str = field(default="individual")
    teacher_answer_reward: bool = field(default=False)
    teacher_answer_model: str = field(default="deepseek-v4-pro")
    teacher_answer_base_url: str | None = field(default=None)
    teacher_answer_api_key_env: str = field(default="DEEPSEEK_API_KEY")
    teacher_answer_max_tokens: int = field(default=1024)
    teacher_answer_temperature: float = field(default=0.0)
    teacher_answer_top_p: float = field(default=1.0)
    teacher_answer_timeout: float = field(default=120.0)
    teacher_answer_max_retries: int = field(default=3)
    teacher_answer_concurrency: int = field(default=32)
    teacher_answer_reward_weight: float = field(default=1.0)
    verifier_reward_weight: float = field(default=1.0)


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


def _is_terminal_task_dir(path: Path) -> bool:
    return (
        (path / "instruction.md").is_file()
        and (path / "environment").is_dir()
        and (path / "tests").is_dir()
    )


def _discover_task_dirs(root: Path) -> list[tuple[str, Path]]:
    """Find extracted Nemotron terminal task directories.

    The synthetic-task release is distributed as tar archives with directory
    trees, not a normal HF `datasets` split. Task directories sit a few levels
    below the extraction root and can contain many payload files; this pruned
    traversal avoids recursively walking each task's `environment/`.
    """
    root = root.resolve()
    tasks: list[tuple[str, Path]] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    skip_names = {".git", "__pycache__", "environment", "solution", "tests"}

    while stack:
        path, depth = stack.pop()
        if _is_terminal_task_dir(path):
            tasks.append((path.relative_to(root).as_posix().replace("/", "__"), path))
            continue
        if depth >= 4 or not path.is_dir():
            continue
        try:
            children = sorted(
                child
                for child in path.iterdir()
                if child.is_dir() and child.name not in skip_names
            )
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in reversed(children))

    return sorted(set(tasks), key=lambda item: item[0])


def get_terminal_synthetic_task_dataset(
    path: str,
    split: str = "train",
    seed: int = 1,
    limit: int | None = None,
    split_part: str | None = None,
    holdout_size: int = 128,
    shuffle_records: bool = True,
    load_instructions: bool = False,
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
                "instruction": _read_instruction(task_dir) if load_instructions else "",
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
        enable_thinking: bool,
        executor: ThreadPoolExecutor,
        task_service_url: str | None = None,
        task_service_url_file: str | None = None,
    ):
        self.output_path = output_path
        self.max_turns = max_turns
        self.max_tokens_per_turn = max_tokens_per_turn
        self.temperature = temperature
        self.top_p = top_p
        self.observation_max_chars = observation_max_chars
        self.task_timeouts = task_timeouts
        self.encourage_completion_reward = encourage_completion_reward
        self.enable_thinking = enable_thinking
        self.executor = executor
        self.task_service_url = task_service_url.rstrip("/") if task_service_url else None
        self.task_service_url_file = task_service_url_file
        self.remote_session_id: str | None = None
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
        task_dir = Path(str(data["task_path"])).resolve()
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

    def _resolved_task_service_url(self) -> str | None:
        if self.task_service_url:
            return self.task_service_url
        if not self.task_service_url_file:
            return None
        path = Path(self.task_service_url_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Terminal task service URL file does not exist: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Terminal task service URL file is empty: {path}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            url = text
        else:
            url = str(data.get("url") or data.get("base_url") or "").strip()
        if not url:
            raise ValueError(f"No url/base_url in terminal task service file: {path}")
        self.task_service_url = url.rstrip("/")
        return self.task_service_url

    async def _remote_request(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base_url = self._resolved_task_service_url()
        if base_url is None:
            raise RuntimeError("terminal task service URL is not configured")
        request_timeout = httpx.Timeout(
            timeout + 30.0 if timeout is not None else None,
            connect=30.0,
        )
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            response = await client.request(
                method,
                f"{base_url}{path}",
                json=json_payload,
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"terminal task service returned non-object JSON: {data!r}")
        return data

    async def _remote_reset_env(self, data: dict[str, Any], uid: str) -> None:
        payload = {
            "output_path": self.output_path,
            "task_name": str(data["task_name"]),
            "task_path": str(data["task_path"]),
            "uid": uid,
            "observation_max_chars": self.observation_max_chars,
            "task_timeouts": asdict(self.task_timeouts),
            "encourage_completion_reward": self.encourage_completion_reward,
        }
        response = await self._remote_request(
            "POST",
            "/v1/sessions",
            timeout=self.task_timeouts.reset_env,
            json_payload=payload,
        )
        self.remote_session_id = str(response["session_id"])

    async def _remote_execute_commands(self, commands: Iterable[dict[str, Any]]) -> str:
        if self.remote_session_id is None:
            raise RuntimeError("remote terminal session is not initialized")
        response = await self._remote_request(
            "POST",
            f"/v1/sessions/{self.remote_session_id}/commands",
            timeout=self.task_timeouts.command,
            json_payload={"commands": list(commands)},
        )
        return str(response.get("observation", ""))

    async def _remote_evaluate_completion(self) -> float:
        if self.remote_session_id is None:
            raise RuntimeError("remote terminal session is not initialized")
        response = await self._remote_request(
            "POST",
            f"/v1/sessions/{self.remote_session_id}/reward",
            timeout=self.task_timeouts.verifier,
            json_payload={},
        )
        return float(response["reward"])

    async def _remote_close_env(self) -> None:
        if self.remote_session_id is None:
            return
        try:
            await self._remote_request(
                "DELETE",
                f"/v1/sessions/{self.remote_session_id}",
                timeout=self.task_timeouts.cleanup,
            )
        finally:
            self.remote_session_id = None

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
        instruction = str(data.get("instruction") or "").strip()
        if not instruction:
            instruction = _read_instruction(Path(str(data["task_path"])).resolve())
        use_remote = self._resolved_task_service_url() is not None
        messages = [
            {"role": "system", "content": TERMINUS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current date: {_datetime.date.today().isoformat()}\n"
                    f"We are in the root directory /app.\n\n"
                    f"Task name: {task_name}\n"
                    f"Task instruction:\n{instruction}"
                ),
            },
        ]
        try:
            async with atrace_scope(
                f"reset_env:{task_name},traj:{traj_i}",
                args={"uid": uid, "timeout": self.task_timeouts.reset_env},
            ):
                if use_remote:
                    await asyncio.wait_for(
                        self._remote_reset_env(data, uid),
                        timeout=self.task_timeouts.reset_env + 30,
                    )
                else:
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
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking
                        }
                    },
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
                    if use_remote:
                        observation = await asyncio.wait_for(
                            self._remote_execute_commands(commands),
                            timeout=self.task_timeouts.command * max(len(commands), 1)
                            + 30,
                        )
                    else:
                        observation = await self.run_in_executor(
                            self._execute_commands,
                            commands,
                            timeout=self.task_timeouts.command * max(len(commands), 1)
                            + 10,
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
                if use_remote:
                    reward = await asyncio.wait_for(
                        self._remote_evaluate_completion(),
                        timeout=self.task_timeouts.verifier + 30,
                    )
                else:
                    reward = await self.run_in_executor(
                        self._evaluate_completion_sync,
                        timeout=self.task_timeouts.verifier,
                    )
            client.set_last_reward(float(reward))
            print(
                f"Terminus GRPO task {task_name} traj {traj_i} reward={float(reward):.4f}",
                flush=True,
            )
            return float(reward)
        except TimeoutError:
            return None
        except Exception as exc:
            print(f"Terminus GRPO task {task_name} failed: {exc}")
            return None
        finally:
            try:
                if use_remote:
                    await self._remote_close_env()
                else:
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
        terminal_task_service_url: str | None = None,
        terminal_task_service_url_file: str | None = None,
        enable_thinking: bool = False,
    ):
        self.gconfig = gconfig
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
        self.terminal_task_service_url = terminal_task_service_url
        self.terminal_task_service_url_file = terminal_task_service_url_file
        self.enable_thinking = enable_thinking
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
                    enable_thinking=self.enable_thinking,
                    executor=self.executor,
                    task_service_url=self.terminal_task_service_url,
                    task_service_url_file=self.terminal_task_service_url_file,
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
