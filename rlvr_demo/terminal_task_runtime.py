"""Lightweight Terminal-Bench runtime used by the remote task service."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import tomllib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal


_ADAPTER_LOCKS: dict[str, threading.Lock] = {}
_ADAPTER_LOCKS_GUARD = threading.Lock()


@dataclass
class TerminalTaskTimeouts:
    reset_env: float = 1800.0
    reset_agent: float = 120.0
    agent_step: float = 300.0
    command: float = 180.0
    verifier: float = 1200.0
    cleanup: float | None = None


def _read_instruction(task_dir: Path) -> str:
    for name in ("instruction.md", "task.md", "README.md"):
        path = task_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace").strip()
    raise FileNotFoundError(f"No instruction.md/task.md/README.md in {task_dir}")


def _as_seconds(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _task_difficulty(task_dir: Path) -> str:
    parts = set(task_dir.parts)
    for difficulty in ("easy", "medium", "hard"):
        if difficulty in parts:
            return difficulty
    return "unknown"


def _adapter_lock(key: str) -> threading.Lock:
    with _ADAPTER_LOCKS_GUARD:
        lock = _ADAPTER_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _ADAPTER_LOCKS[key] = lock
        return lock


def _prepare_terminal_bench_task(task_dir: Path) -> Path:
    if (
        (task_dir / "task.yaml").is_file()
        and (task_dir / "docker-compose.yaml").is_file()
        and (task_dir / "run-tests.sh").is_file()
    ):
        return task_dir

    task_toml_path = task_dir / "task.toml"
    dockerfile = task_dir / "environment" / "Dockerfile"
    tests_dir = task_dir / "tests"
    if not task_toml_path.is_file() or not dockerfile.is_file() or not tests_dir.is_dir():
        return task_dir

    digest = hashlib.sha1(str(task_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    default_adapter_root = f"/tmp/terminal-task-service-adapters-{os.getuid()}"
    adapter_root = Path(os.environ.get("TERMINAL_TASK_ADAPTER_ROOT", default_adapter_root))
    adapter_dir = adapter_root / f"{task_dir.name}-{digest}"
    with _adapter_lock(str(adapter_dir)):
        return _prepare_terminal_bench_task_locked(task_dir, adapter_dir)


def _prepare_terminal_bench_task_locked(task_dir: Path, adapter_dir: Path) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)

    task_toml_path = task_dir / "task.toml"
    tests_dir = task_dir / "tests"
    raw_config = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    metadata = raw_config.get("metadata") if isinstance(raw_config, dict) else {}
    agent = raw_config.get("agent") if isinstance(raw_config, dict) else {}
    verifier = raw_config.get("verifier") if isinstance(raw_config, dict) else {}
    tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
    tags = [str(tag) for tag in tags] if isinstance(tags, list) else []

    task_yaml = {
        "instruction": _read_instruction(task_dir),
        "author_name": "nvidia/Nemotron-Terminal-Synthetic-Tasks",
        "author_email": "unknown",
        "difficulty": _task_difficulty(task_dir),
        "category": tags[-1] if tags else "software_engineering",
        "tags": tags,
        "parser_name": "pytest",
        "max_agent_timeout_sec": _as_seconds(
            agent.get("timeout_sec") if isinstance(agent, dict) else None,
            900.0,
        ),
        "max_test_timeout_sec": _as_seconds(
            verifier.get("timeout_sec") if isinstance(verifier, dict) else None,
            900.0,
        ),
    }
    task_yaml_tmp = adapter_dir / "task.yaml.tmp"
    task_yaml_tmp.write_text(
        yaml.safe_dump(task_yaml, sort_keys=False, width=4096),
        encoding="utf-8",
    )
    task_yaml_tmp.replace(adapter_dir / "task.yaml")

    compose = f"""services:
  client:
    build:
      context: {str((adapter_dir / "build-context").resolve())}
      dockerfile: Dockerfile
    image: ${{T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}}
    container_name: ${{T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}}
    network_mode: bridge
    command: [ "sh", "-c", "sleep infinity" ]
    environment:
      - TEST_DIR=${{T_BENCH_TEST_DIR}}
    volumes:
      - ${{T_BENCH_TASK_LOGS_PATH}}:${{T_BENCH_CONTAINER_LOGS_PATH}}
      - ${{T_BENCH_TASK_AGENT_LOGS_PATH}}:${{T_BENCH_CONTAINER_AGENT_LOGS_PATH}}
"""
    (adapter_dir / "docker-compose.yaml").write_text(compose, encoding="utf-8")

    build_context = adapter_dir / "build-context"
    shutil.copytree(task_dir / "environment", build_context, dirs_exist_ok=True)
    (build_context / "files").mkdir(exist_ok=True)
    dockerfile_path = build_context / "Dockerfile"
    dockerfile_text = dockerfile_path.read_text(encoding="utf-8", errors="replace")
    if "terminal-task-service: ensure Terminal-Bench session tools" not in dockerfile_text:
        dockerfile_path.write_text(
            dockerfile_text.rstrip()
            + """

# terminal-task-service: ensure Terminal-Bench session tools are available.
RUN set -eux; \
    if ! command -v tmux >/dev/null 2>&1 || ! command -v asciinema >/dev/null 2>&1; then \
      if command -v apt-get >/dev/null 2>&1; then \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tmux asciinema && rm -rf /var/lib/apt/lists/*; \
      elif command -v apk >/dev/null 2>&1; then \
        apk add --no-cache tmux asciinema; \
      elif command -v dnf >/dev/null 2>&1; then \
        dnf install -y tmux asciinema && dnf clean all; \
      elif command -v yum >/dev/null 2>&1; then \
        yum install -y tmux asciinema && yum clean all; \
      else \
        echo "tmux and asciinema are required by Terminal-Bench but no supported package manager was found" >&2; exit 1; \
      fi; \
    fi
"""
            + "\n",
            encoding="utf-8",
        )

    source_run_tests = tests_dir / "test.sh"
    target_run_tests = adapter_dir / "run-tests.sh"
    if source_run_tests.exists():
        shutil.copy2(source_run_tests, target_run_tests)
    else:
        target_run_tests.write_text(
            "#!/bin/bash\nset -e\npython3 -m pytest /tests -rA\n",
            encoding="utf-8",
        )
    target_run_tests.chmod(0o755)
    shutil.copytree(tests_dir, adapter_dir / "tests", dirs_exist_ok=True)
    (adapter_dir / "solution.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (adapter_dir / "solution.sh").chmod(0o755)
    return adapter_dir


class TerminusLocalTerminalTaskRunner:
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
        del max_turns, max_tokens_per_turn, temperature, top_p, executor
        self.output_path = output_path
        self.observation_max_chars = observation_max_chars
        self.task_timeouts = task_timeouts
        self.encourage_completion_reward = encourage_completion_reward
        self.terminal: Terminal | None = None
        self.trial_handler: TrialHandler | None = None
        self.parser = None
        self.task_name = ""

    def _reset_env(self, data: dict[str, Any], uid: str) -> None:
        output_path = Path(self.output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        task_dir = _prepare_terminal_bench_task(Path(str(data["task_path"])).resolve())
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
            no_rebuild=False,
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
