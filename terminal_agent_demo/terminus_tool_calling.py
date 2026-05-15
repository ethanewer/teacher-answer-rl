"""Terminus-style terminal harness using OpenAI tool calls.

This module keeps the Terminus-2 action payload, but moves it from visible JSON
assistant text into a single tool call:

``execute_commands({analysis, plan, commands, task_complete})``.

The important invariant for Qwen reasoning models is that the task prompt is the
only real user message. Terminal observations are appended as tool messages, so
the chat template does not treat every observation as a new user query and does
not strip earlier assistant thinking blocks.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import os
import random
import re
import shlex
import subprocess
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

try:
    if os.environ.get("TERMINUS_TOOL_SKIP_AREAL_IMPORT") == "1":
        raise ImportError("skipping AReaL imports for Harbor eval")
    from areal.api.workflow_api import RolloutWorkflow
except Exception:  # pragma: no cover - Harbor evals do not require AReaL.
    class RolloutWorkflow:  # type: ignore[no-redef]
        pass


TERMINUS_TOOL_SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by calling the `execute_commands` tool with batches of shell commands.

Call `execute_commands` exactly once each turn with arguments matching this structure:

{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    },
    {
      "keystrokes": "cd project\\n",
      "duration": 0.1
    }
  ],
  "task_complete": true
}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- You must end every command with a newline (\\n) or it will not execute.
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by calling `execute_commands` with {"keystrokes": "", "duration": 10.0} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- Avoid interactive pagers and editors unless the task explicitly requires them. Prefer noninteractive commands such as `git --no-pager ...`, `cat`, `sed`, `head`, and `tail`.
- Do not put the action JSON in visible assistant text; put all action fields in the `execute_commands` tool call arguments
- Tool arguments must be valid JSON - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action
"""


NONINTERACTIVE_TERMINAL_ENV = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-F -X",
}


def _terminal_env_export_command() -> str:
    exports = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in NONINTERACTIVE_TERMINAL_ENV.items()
    )
    return f"export {exports}\n"


TIMEOUT_PROMPT_TEMPLATE = """Previous command:
{command}

The previous command timed out after {timeout_sec} seconds

It is possible that the command is not yet finished executing. If that is the case, then do nothing. It is also possible that you have entered an interactive shell and should continue sending keystrokes as normal.

Here is the current state of the terminal:

{terminal_state}"""


EXECUTE_COMMANDS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_commands",
        "description": (
            "Analyze the terminal state, plan the next step, execute a batch of "
            "terminal keystrokes, and optionally mark the task complete. The "
            "arguments intentionally match the Terminus-2 JSON response shape."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Short analysis of current state and remaining work.",
                },
                "plan": {
                    "type": "string",
                    "description": "Short plan for the next commands.",
                },
                "commands": {
                    "type": "array",
                    "description": "Terminal keystrokes to send in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keystrokes": {
                                "type": "string",
                                "description": (
                                    "Exact keystrokes to send to the terminal. Append \\n when the command should execute."
                                ),
                            },
                            "duration": {
                                "type": "number",
                                "description": (
                                    "Seconds to wait after sending these keystrokes. Defaults to 1.0 if omitted."
                                ),
                            },
                        },
                        "required": ["keystrokes"],
                        "additionalProperties": False,
                    },
                },
                "task_complete": {
                    "type": "boolean",
                    "description": "Whether the task is complete and ready for grading.",
                },
            },
            "required": ["analysis", "plan", "commands"],
            "additionalProperties": False,
        },
    },
}


DEFAULT_MODEL_INFO = {
    "max_input_tokens": 32768,
    "max_output_tokens": 8192,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}


class TerminusToolPayloadError(ValueError):
    """Raised when execute_commands arguments do not match the Terminus shape."""


class ContextLengthExceededError(RuntimeError):
    """Raised when the serving backend cannot fit another generation."""


@dataclass(frozen=True)
class ParsedCommand:
    keystrokes: str
    duration: float


@dataclass(frozen=True)
class ParsedPayload:
    analysis: str
    plan: str
    commands: list[ParsedCommand]
    task_complete: bool


def read_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def build_initial_messages(
    instruction: str,
    *,
    terminal_state: str = "",
    task_name: str | None = None,
) -> list[dict[str, Any]]:
    state = terminal_state.strip() or "Current Terminal Screen:\n$"
    user_content = (
        f"Task Description:\n{instruction.strip()}\n\n"
        f"Current terminal state:\n{state}"
    )
    return [
        {"role": "system", "content": TERMINUS_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    starts = [idx for idx, char in enumerate(text) if char == "{"]
    if not starts:
        raise TerminusToolPayloadError("no JSON object found")
    decoder = json.JSONDecoder()
    first_dict: dict[str, Any] | None = None
    last_error: Exception | None = None
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(obj, dict):
            continue
        if first_dict is None:
            first_dict = obj
        if "commands" in obj or {"analysis", "plan"}.intersection(obj):
            return obj
    if first_dict is not None:
        return first_dict
    raise TerminusToolPayloadError(str(last_error or "no JSON object found"))


def parse_terminus_json_payload(text: str) -> ParsedPayload:
    return parse_execute_commands_arguments(_extract_json_object(text))


def parse_execute_commands_arguments(raw: str | dict[str, Any]) -> ParsedPayload:
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TerminusToolPayloadError(f"invalid tool JSON: {exc}") from exc
    else:
        payload = raw
    if not isinstance(payload, dict):
        raise TerminusToolPayloadError("tool arguments are not an object")

    if "commands" not in payload and "keystrokes" in payload:
        payload = {
            "analysis": "",
            "plan": "",
            "commands": [payload],
            "task_complete": False,
        }

    analysis = payload.get("analysis")
    plan = payload.get("plan")
    commands = payload.get("commands")
    if analysis is None:
        analysis = ""
    elif not isinstance(analysis, str):
        analysis = str(analysis)
    if plan is None:
        plan = ""
    elif not isinstance(plan, str):
        plan = str(plan)
    if isinstance(commands, dict):
        commands = [commands]
    if not isinstance(commands, list):
        raise TerminusToolPayloadError("commands is not a list")

    parsed_commands: list[ParsedCommand] = []
    for idx, item in enumerate(commands):
        if not isinstance(item, dict):
            raise TerminusToolPayloadError(f"commands[{idx}] is not an object")
        keystrokes = item.get("keystrokes")
        if not isinstance(keystrokes, str):
            raise TerminusToolPayloadError(f"commands[{idx}].keystrokes is not a string")
        try:
            duration = float(item.get("duration", 1.0))
        except (TypeError, ValueError) as exc:
            raise TerminusToolPayloadError(f"commands[{idx}].duration is invalid") from exc
        parsed_commands.append(ParsedCommand(keystrokes=keystrokes, duration=min(max(duration, 0.0), 60.0)))

    return ParsedPayload(
        analysis=analysis,
        plan=plan,
        commands=parsed_commands,
        task_complete=bool(payload.get("task_complete", False)),
    )


def payload_to_arguments(payload: ParsedPayload | dict[str, Any]) -> str:
    if isinstance(payload, ParsedPayload):
        obj = {
            "analysis": payload.analysis,
            "plan": payload.plan,
            "commands": [{"keystrokes": command.keystrokes, "duration": command.duration} for command in payload.commands],
            "task_complete": payload.task_complete,
        }
    else:
        obj = payload
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


_TOOL_CALL_XML_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", flags=re.DOTALL)


def _synthetic_execute_commands_tool_call(payload: ParsedPayload, *, call_id: str | None = None) -> dict[str, Any]:
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": "execute_commands",
            "arguments": payload_to_arguments(payload),
        },
    }


def _tool_call_from_assistant_content(content: str | None) -> dict[str, Any] | None:
    if not isinstance(content, str) or not content.strip():
        return None

    for match in _TOOL_CALL_XML_RE.finditer(content):
        raw = match.group(1).strip()
        try:
            call_payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(call_payload, dict):
            continue
        name = call_payload.get("name")
        arguments = call_payload.get("arguments", {})
        if name != "execute_commands":
            continue
        try:
            parsed = parse_execute_commands_arguments(arguments)
        except TerminusToolPayloadError:
            continue
        return _synthetic_execute_commands_tool_call(parsed)

    try:
        parsed = parse_terminus_json_payload(content)
    except TerminusToolPayloadError:
        return None
    return _synthetic_execute_commands_tool_call(parsed)


def _first_tool_call(message: Any) -> Any | None:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls is None and isinstance(message, dict):
        tool_calls = message.get("tool_calls")
    if tool_calls:
        return tool_calls[0]

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    return _tool_call_from_assistant_content(content)


def _message_content_without_tool_call(content: str | None) -> str | None:
    if content is None:
        return None
    cleaned = _TOOL_CALL_XML_RE.sub("", content).strip()
    return cleaned or None


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
    return str(getattr(tool_call, "id", None) or f"call_{uuid.uuid4().hex[:24]}")


def _tool_call_name(tool_call: Any) -> str:
    function = tool_call.get("function") if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(getattr(function, "name", "") or "")


def _tool_call_arguments(tool_call: Any) -> str:
    function = tool_call.get("function") if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
    if isinstance(function, dict):
        return str(function.get("arguments") or "{}")
    return str(getattr(function, "arguments", "{}") or "{}")


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        data = dict(message)
    elif hasattr(message, "model_dump"):
        data = message.model_dump(exclude_none=True)
    else:
        data = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            data["tool_calls"] = [tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else tc for tc in tool_calls]
    data.setdefault("role", "assistant")
    return data


_CONTEXT_LENGTH_RE = re.compile(
    r"You passed (\d+) input tokens and requested (\d+) output tokens.*context length is only (\d+) tokens",
    flags=re.DOTALL,
)


def _context_length_retry_body(body: dict[str, Any], response_text: str) -> dict[str, Any] | None:
    match = _CONTEXT_LENGTH_RE.search(response_text)
    if match is None:
        return None
    input_tokens, requested_tokens, context_tokens = (int(group) for group in match.groups())
    remaining = context_tokens - input_tokens
    if remaining <= 0 or remaining >= requested_tokens:
        return None
    reserve = int(os.environ.get("TERMINUS_TOOL_CONTEXT_RETRY_RESERVE", "1536"))
    retry_body = dict(body)
    retry_body["max_tokens"] = max(1, remaining - max(reserve, 0) - 8)
    return retry_body


def _is_context_length_error(response_text: str) -> bool:
    return _CONTEXT_LENGTH_RE.search(response_text) is not None


def _assistant_recovery_message(message: Any, error: str) -> dict[str, Any]:
    data = _message_to_dict(message)
    content = data.get("content")
    if isinstance(content, str) and content.strip():
        content = content.strip() + "\n\n"
    else:
        content = ""
    recovery = {
        "role": "assistant",
        "content": (
            content
            + "Previous response had parsing errors:\n"
            + f"ERROR: {error}\n\n"
            + "Please fix these issues and provide a proper tool call."
        ),
    }
    if data.get("reasoning_content"):
        recovery["reasoning_content"] = data["reasoning_content"]
    return recovery


def _synthetic_recovery_tool_turn(error: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Represent recovery feedback as a paired assistant/tool turn.

    A real user message would make Qwen thinking templates treat the error as
    the latest query and strip earlier reasoning. An unpaired tool message is
    invalid for several OpenAI-compatible servers. A synthetic empty
    execute_commands call keeps the history append-only and preserves normal
    assistant/tool alternation without executing shell commands.
    """
    call_id = f"call_recovery_{uuid.uuid4().hex[:16]}"
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "execute_commands",
                    "arguments": payload_to_arguments(
                        {
                            "analysis": "The previous response could not be used.",
                            "plan": "Retry with a valid execute_commands tool call.",
                            "commands": [],
                            "task_complete": False,
                        }
                    ),
                },
            }
        ],
    }
    tool = tool_response_message(
        call_id,
        "Previous response had parsing errors:\n"
        f"ERROR: {error}\n\n"
        "Please fix these issues and provide a proper execute_commands tool call.",
    )
    return assistant, tool


def _assistant_tool_message_from_parsed_payload(
    message: Any,
    tool_call: Any,
    payload: ParsedPayload,
) -> dict[str, Any]:
    data = _message_to_dict(message)
    data["role"] = "assistant"
    data["content"] = _message_content_without_tool_call(data.get("content"))
    data["tool_calls"] = [
        {
            "id": _tool_call_id(tool_call),
            "type": "function",
            "function": {
                "name": "execute_commands",
                "arguments": payload_to_arguments(payload),
            },
        }
    ]
    return data


def tool_response_message(
    tool_call_id: str,
    content: str,
    *,
    name: str = "execute_commands",
) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


def limit_output_length(output: str, max_bytes: int = 10000) -> str:
    if len(output.encode("utf-8")) <= max_bytes:
        return output

    portion_size = max_bytes // 2
    output_bytes = output.encode("utf-8")
    first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")
    last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")
    omitted_bytes = (
        len(output_bytes)
        - len(first_portion.encode("utf-8"))
        - len(last_portion.encode("utf-8"))
    )

    return (
        f"{first_portion}\n[... output limited to {max_bytes} bytes; "
        f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
    )


def completion_confirmation_message(terminal_output: str) -> str:
    return (
        f"Current terminal state:\n{terminal_output}\n\n"
        "Are you sure you want to mark the task as complete? "
        "This will trigger your solution to be graded and you won't be able to "
        "make any further corrections. If so, call `execute_commands` again "
        'with "task_complete": true.'
    )


def _strip_new_terminal_prefix(text: str) -> str:
    stripped = text.strip()
    for prefix in ("New Terminal Output:", "Current Terminal Screen:", "Current terminal state:"):
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].lstrip()
    return stripped


def _split_terminus_initial_prompt(content: str) -> tuple[str, str]:
    task_marker = "Task Description:"
    state_marker = "Current terminal state:"
    if task_marker in content and state_marker in content:
        before_state, state = content.split(state_marker, 1)
        instruction = before_state.split(task_marker, 1)[1].strip()
        return instruction, state.strip()
    return content.strip(), "Current Terminal Screen:\n$"


def _assistant_thinking_prefix(content: str) -> str:
    match = re.match(r"\s*(<think>.*?</think>)", content, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def assistant_tool_message_from_terminus_json(
    content: str,
    *,
    tool_call_id: str,
) -> dict[str, Any]:
    payload = parse_terminus_json_payload(content)
    return {
        "role": "assistant",
        "content": _assistant_thinking_prefix(content),
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "execute_commands",
                    "arguments": payload_to_arguments(payload),
                },
            }
        ],
    }


def convert_terminus2_conversation(
    conversations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert user/assistant Terminus-2 JSON turns to tool-call turns.

    The output always has exactly one role=user task message. Later terminal
    observations become role=tool messages bound to the preceding assistant
    `execute_commands` call.
    """
    if not conversations:
        raise ValueError("conversation is empty")
    first = conversations[0]
    if first.get("role") != "user":
        raise ValueError("first conversation message is not role=user")

    instruction, terminal_state = _split_terminus_initial_prompt(str(first.get("content", "")))
    messages = build_initial_messages(instruction, terminal_state=terminal_state)
    pending_tool_call_id: str | None = None

    for idx, msg in enumerate(conversations[1:], start=1):
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "assistant":
            pending_tool_call_id = f"call_{idx:04d}_{uuid.uuid4().hex[:8]}"
            messages.append(
                assistant_tool_message_from_terminus_json(
                    content,
                    tool_call_id=pending_tool_call_id,
                )
            )
        elif role == "user":
            if pending_tool_call_id is None:
                raise ValueError(f"user observation at index {idx} has no prior tool call")
            messages.append(
                tool_response_message(
                    pending_tool_call_id,
                    _strip_new_terminal_prefix(content),
                )
            )
            pending_tool_call_id = None
        else:
            raise ValueError(f"unsupported role at index {idx}: {role!r}")
    return messages


class _DeepSeekMessage(SimpleNamespace):
    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        raw_tool_calls = getattr(self, "tool_calls", None)
        tool_calls = None
        if raw_tool_calls:
            tool_calls = [
                call.model_dump(exclude_none=exclude_none) if hasattr(call, "model_dump") else call
                for call in raw_tool_calls
            ]
        data = {
            "role": "assistant",
            "content": getattr(self, "content", None),
            "tool_calls": tool_calls,
        }
        if getattr(self, "reasoning_content", None):
            data["reasoning_content"] = self.reasoning_content
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


class _DeepSeekCompletions:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float,
        thinking: bool,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.thinking = thinking
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def create(self, **kwargs: Any) -> Any:
        body = {
            "model": self.model,
            "messages": kwargs["messages"],
            "tools": kwargs.get("tools"),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 1.0),
            "max_tokens": kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or 2048,
        }
        if self.thinking:
            body["reasoning_effort"] = "high"
            body["thinking"] = {"type": "enabled"}
        body = {k: v for k, v in body.items() if v is not None}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
            )
        if response.status_code >= 400:
            raise RuntimeError(f"DeepSeek API error {response.status_code}: {response.text[:2000]}")
        payload = response.json()
        usage = payload.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        raw_message = payload["choices"][0]["message"]
        tool_calls = []
        for raw_call in raw_message.get("tool_calls") or []:
            raw_function = raw_call.get("function") or {}
            call_id = raw_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"

            class _ToolCall(SimpleNamespace):
                def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
                    data = {
                        "id": self.id,
                        "type": self.type,
                        "function": {
                            "name": self.function.name,
                            "arguments": self.function.arguments,
                        },
                    }
                    if exclude_none:
                        data = {k: v for k, v in data.items() if v is not None}
                    return data

            tool_calls.append(
                _ToolCall(
                    id=call_id,
                    type=raw_call.get("type", "function"),
                    function=SimpleNamespace(
                        name=raw_function.get("name"),
                        arguments=raw_function.get("arguments") or "{}",
                    ),
                )
            )
        message = _DeepSeekMessage(
            role="assistant",
            content=raw_message.get("content"),
            reasoning_content=raw_message.get("reasoning_content"),
            tool_calls=tool_calls or None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DeepSeekRewardShim:
    def __init__(
        self,
        *,
        model: str = "deepseek-v4-pro",
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 240.0,
        thinking: bool = False,
    ) -> None:
        self.chat = SimpleNamespace(
            completions=_DeepSeekCompletions(
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                thinking=thinking,
            )
        )
        self.last_reward: float | None = None

    def set_last_reward(self, reward: float) -> None:
        self.last_reward = reward


@dataclass
class _CliExecResult:
    exit_code: int
    output: bytes


class _CliDockerContainer:
    """Small docker-py-compatible wrapper that uses the setgid docker CLI."""

    def __init__(self, container_name: str) -> None:
        self.name = container_name
        result = subprocess.run(
            ["docker", "inspect", container_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = json.loads(result.stdout.decode("utf-8"))
        self.attrs = payload[0] if payload else {}

    def exec_run(self, cmd: str | Sequence[str], user: str = "") -> _CliExecResult:
        if isinstance(cmd, str):
            exec_args = ["sh", "-lc", cmd]
        else:
            exec_args = [str(part) for part in cmd]
        full_cmd = ["docker", "exec"]
        if user:
            full_cmd.extend(["-u", user])
        full_cmd.append(self.name)
        full_cmd.extend(exec_args)
        result = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return _CliExecResult(exit_code=result.returncode, output=result.stdout)

    def put_archive(self, container_dir: str, data: bytes) -> None:
        result = subprocess.run(
            ["docker", "cp", "-", f"{self.name}:{container_dir}"],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))


class _CliDockerComposeManager:
    """Subset of Terminal-Bench's DockerComposeManager implemented with docker CLI."""

    CONTAINER_SESSION_LOGS_PATH = "/logs"
    CONTAINER_AGENT_LOGS_PATH = "/agent-logs"
    CONTAINER_TEST_DIR = Path("/tests")

    def __init__(
        self,
        *,
        client_container_name: str,
        client_image_name: str,
        docker_compose_path: Path,
        docker_image_name_prefix: str | None = None,
        no_rebuild: bool = False,
        cleanup: bool = False,
        sessions_logs_path: Path | None = None,
        agent_logs_path: Path | None = None,
    ) -> None:
        from terminal_bench.terminal.docker_compose_manager import DockerComposeEnvVars

        self._client_container_name = client_container_name
        self._client_image_name = client_image_name
        self._docker_compose_path = docker_compose_path
        self._no_rebuild = no_rebuild
        self._cleanup = cleanup
        self._client_container: _CliDockerContainer | None = None
        self.env = DockerComposeEnvVars(
            task_docker_client_image_name=client_image_name,
            task_docker_client_container_name=client_container_name,
            task_docker_name_prefix=docker_image_name_prefix,
            container_logs_path=self.CONTAINER_SESSION_LOGS_PATH,
            container_agent_logs_path=self.CONTAINER_AGENT_LOGS_PATH,
            test_dir=str(self.CONTAINER_TEST_DIR),
            task_logs_path=str(sessions_logs_path.absolute()) if sessions_logs_path else None,
            task_agent_logs_path=str(agent_logs_path.absolute()) if agent_logs_path else None,
        ).to_env_dict(include_os_env=True)

    def _compose_command(self, command: Sequence[str]) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            self._client_container_name,
            "-f",
            str(self._docker_compose_path.resolve().absolute()),
            *command,
        ]

    def _run_compose(self, command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            self._compose_command(command),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if check and result.returncode != 0:
            raise RuntimeError(result.stdout)
        return result

    def start(self) -> _CliDockerContainer:
        if not self._no_rebuild:
            self._run_compose(["build"])
        self._run_compose(["up", "-d"])
        try:
            self._client_container = _CliDockerContainer(self._client_container_name)
        except Exception:
            result = self._run_compose(["ps", "-q", "main"])
            container_id = result.stdout.strip().splitlines()[-1]
            self._client_container = _CliDockerContainer(container_id)
        return self._client_container

    def stop(self) -> None:
        try:
            self._run_compose(["down"], check=False)
            if self._cleanup:
                self._run_compose(["down", "--rmi", "all", "--volumes"], check=False)
        finally:
            self._client_container = None

    def copy_to_client_container(
        self,
        paths: list[Path] | Path,
        container_dir: str | None = None,
        container_filename: str | None = None,
    ) -> None:
        from terminal_bench.terminal.docker_compose_manager import DockerComposeManager

        if self._client_container is None:
            raise RuntimeError("client container is not started")
        DockerComposeManager.copy_to_container(
            container=self._client_container,  # type: ignore[arg-type]
            paths=paths,
            container_dir=container_dir,
            container_filename=container_filename,
        )


class _CliTerminal:
    """Terminal-Bench Terminal equivalent that avoids docker-py's socket access."""

    def __init__(
        self,
        *,
        client_container_name: str,
        client_image_name: str,
        docker_compose_path: Path,
        docker_image_name_prefix: str | None = None,
        sessions_logs_path: Path | None = None,
        agent_logs_path: Path | None = None,
        commands_path: Path | None = None,
        no_rebuild: bool = False,
        cleanup: bool = False,
        disable_recording: bool = True,
    ) -> None:
        self._commands_path = commands_path
        self._disable_recording = disable_recording
        self._sessions: dict[str, Any] = {}
        self._compose_manager = _CliDockerComposeManager(
            client_container_name=client_container_name,
            client_image_name=client_image_name,
            docker_compose_path=docker_compose_path,
            docker_image_name_prefix=docker_image_name_prefix,
            no_rebuild=no_rebuild,
            cleanup=cleanup,
            sessions_logs_path=sessions_logs_path,
            agent_logs_path=agent_logs_path,
        )
        self.container: _CliDockerContainer | None = None

    def start(self) -> None:
        self.container = self._compose_manager.start()

    def stop(self) -> None:
        for session in self._sessions.values():
            try:
                session.stop()
            except Exception:
                pass
        self._compose_manager.stop()
        self._sessions.clear()
        self.container = None

    def create_session(
        self,
        session_name: str,
        is_active_stream: bool = False,
        as_configured_user: bool = True,
    ) -> Any:
        del is_active_stream
        from terminal_bench.terminal.tmux_session import TmuxSession

        if self.container is None:
            raise RuntimeError("container is not started")
        user = self.container.attrs.get("Config", {}).get("User", "") if as_configured_user else "root"
        session = TmuxSession(
            session_name=session_name,
            container=self.container,  # type: ignore[arg-type]
            commands_path=self._commands_path,
            disable_recording=self._disable_recording,
            user=user,
        )
        self._sessions[session_name] = session
        session.start()
        session.send_keys(
            [_terminal_env_export_command()],
            block=True,
            min_timeout_sec=0.1,
            max_timeout_sec=5.0,
        )
        session.get_incremental_output()
        return session

    def get_session(self, session_name: str) -> Any:
        return self._sessions[session_name]

    def copy_to_container(
        self,
        paths: list[Path] | Path,
        container_dir: str | None = None,
        container_filename: str | None = None,
    ) -> None:
        self._compose_manager.copy_to_client_container(
            paths=paths,
            container_dir=container_dir,
            container_filename=container_filename,
        )


@dataclass
class TerminalTaskTimeouts:
    reset_env: float = 1800.0
    reset_agent: float = 120.0
    agent_step: float = 300.0
    command: float = 180.0
    verifier: float = 1200.0
    cleanup: float | None = None


def get_terminal_synthetic_task_dataset(*args: Any, **kwargs: Any) -> Any:
    from terminal_agent_demo.terminal_task_grpo import get_terminal_synthetic_task_dataset as _loader

    return _loader(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "TerminusToolTerminalGRPOConfig":
        from terminal_agent_demo.terminal_task_grpo import TerminalTaskGRPOConfig

        class _Config(TerminalTaskGRPOConfig):  # type: ignore[misc,valid-type]
            pass

        _Config.__name__ = "TerminusToolTerminalGRPOConfig"
        globals()[name] = _Config
        return _Config
    raise AttributeError(name)


class TerminusToolTerminalTaskRunner:
    """Terminal-Bench task runner for the AReaL GRPO workflow."""

    def __init__(
        self,
        *,
        output_path: str,
        max_turns: int = 25,
        max_tokens_per_turn: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        observation_max_chars: int = 8000,
        task_timeouts: TerminalTaskTimeouts | None = None,
        encourage_completion_reward: bool = False,
        executor: Any = None,
    ) -> None:
        self.output_path = output_path
        self.max_turns = max_turns
        self.max_tokens_per_turn = max_tokens_per_turn
        self.temperature = temperature
        self.top_p = top_p
        self.observation_max_chars = observation_max_chars
        self.task_timeouts = task_timeouts or TerminalTaskTimeouts()
        self.encourage_completion_reward = encourage_completion_reward
        self.executor = executor
        self.terminal: Any = None
        self.trial_handler: Any = None
        self.parser: Any = None
        self.task_name = ""
        self.traj_i = 0
        self._pending_completion = False

    async def run_in_executor(self, fn: Any, *args: Any, timeout: float | None = None, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))
        if timeout is not None:
            return await asyncio.wait_for(task, timeout=timeout)
        return await task

    def _reset_env(self, data: dict[str, Any], uid: str) -> None:
        from terminal_bench.handlers.trial_handler import TrialHandler
        from terminal_bench.parsers.parser_factory import ParserFactory

        output_path = Path(self.output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        from terminal_agent_demo.terminal_task_grpo import ensure_terminal_bench_task_layout

        task_dir = ensure_terminal_bench_task_layout(Path(str(data["task_path"])))
        self.task_name = str(data["task_name"])
        self.trial_handler = TrialHandler(
            trial_name=f"{self.task_name}.{uid}.terminus-tool-grpo",
            input_path=task_dir,
            output_path=output_path,
        )
        self.parser = ParserFactory.get_parser(self.trial_handler.task.parser_name)
        self.terminal = _CliTerminal(
            client_container_name=self.trial_handler.client_container_name,
            client_image_name=self.trial_handler.client_image_name,
            docker_compose_path=self.trial_handler.task_paths.docker_compose_path,
            docker_image_name_prefix=self.trial_handler.docker_image_name_prefix,
            sessions_logs_path=self.trial_handler.trial_paths.sessions_path,
            agent_logs_path=self.trial_handler.trial_paths.agent_logging_dir,
            commands_path=self.trial_handler.trial_paths.commands_path,
            no_rebuild=False,
            cleanup=False,
            disable_recording=True,
        )
        self.terminal.start()
        session = self.terminal.create_session("agent", is_active_stream=False)
        session.get_incremental_output()

    def _execute_commands(self, commands: Iterable[dict[str, Any]]) -> str:
        if self.terminal is None:
            raise RuntimeError("terminal is not initialized")
        session = self.terminal.get_session("agent")
        for command in commands:
            keystrokes = str(command["keystrokes"])
            duration = min(max(float(command.get("duration", 1.0)), 0.0), 60.0)
            try:
                session.send_keys(
                    [keystrokes],
                    block=False,
                    min_timeout_sec=duration,
                    max_timeout_sec=self.task_timeouts.command,
                )
            except TimeoutError:
                return TIMEOUT_PROMPT_TEMPLATE.format(
                    timeout_sec=duration,
                    command=keystrokes,
                    terminal_state=limit_output_length(session.get_incremental_output()),
                )
        return limit_output_length(session.get_incremental_output())

    def _evaluate_completion_sync(self) -> float:
        from terminal_bench.parsers.base_parser import UnitTestStatus

        if self.trial_handler is None or self.terminal is None or self.parser is None:
            raise RuntimeError("terminal environment is not initialized")

        paths = [self.trial_handler.task_paths.run_tests_path]
        if self.trial_handler.task_paths.test_dir.exists():
            paths.append(self.trial_handler.task_paths.test_dir)
        self.terminal.copy_to_container(
            paths=paths,
            container_dir=str(_CliDockerComposeManager.CONTAINER_TEST_DIR),
        )
        test_session = self.terminal.create_session(
            "tests",
            is_active_stream=False,
            as_configured_user=False,
        )
        test_script_path = str(_CliDockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh")
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
                sum(1 for status in parser_results.values() if status == UnitTestStatus.PASSED) / len(parser_results)
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

    @staticmethod
    def _commands_for_base_runner(payload: ParsedPayload) -> list[dict[str, Any]]:
        return [{"keystrokes": command.keystrokes, "duration": command.duration} for command in payload.commands]

    async def _handle_tool_call(
        self,
        messages: list[dict[str, Any]],
        tool_call: Any,
        payload: ParsedPayload,
    ) -> bool:
        call_id = _tool_call_id(tool_call)
        observation = await self.run_in_executor(
            self._execute_commands,
            self._commands_for_base_runner(payload),
            timeout=self.task_timeouts.command * max(len(payload.commands), 1) + 10,
        )
        if payload.task_complete:
            if self._pending_completion:
                messages.append(tool_response_message(call_id, observation))
                return True
            self._pending_completion = True
            observation = completion_confirmation_message(observation)
        else:
            self._pending_completion = False
        messages.append(tool_response_message(call_id, observation))
        return False

    async def run_agent(
        self,
        data: dict[str, Any],
        client: Any,
        uid: str,
        traj_i: int,
    ) -> float | None:
        from areal.utils.perf_tracer import atrace_scope, atrace_session_phase, session_context

        @session_context()
        async def _run() -> float | None:
            self.traj_i = traj_i
            self._pending_completion = False
            task_name = str(data.get("task_name"))
            messages = build_initial_messages(
                str(data["instruction"]),
                task_name=task_name,
            )
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

                for _turn in range(self.max_turns):
                    response = await client.chat.completions.create(
                        messages=messages,
                        tools=[EXECUTE_COMMANDS_TOOL],
                        tool_choice={
                            "type": "function",
                            "function": {"name": "execute_commands"},
                        },
                        max_completion_tokens=self.max_tokens_per_turn,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                    )
                    message = response.choices[0].message
                    tool_call = _first_tool_call(message)
                    if tool_call is None:
                        messages.extend(
                            _synthetic_recovery_tool_turn(
                                "No execute_commands tool call was produced."
                            )
                        )
                        continue
                    name = _tool_call_name(tool_call)
                    if name != "execute_commands":
                        messages.extend(_synthetic_recovery_tool_turn(f"Unknown tool: {name}"))
                        continue
                    try:
                        payload = parse_execute_commands_arguments(_tool_call_arguments(tool_call))
                    except TerminusToolPayloadError as exc:
                        messages.extend(_synthetic_recovery_tool_turn(str(exc)))
                        continue
                    messages.append(_assistant_tool_message_from_parsed_payload(message, tool_call, payload))
                    if await self._handle_tool_call(messages, tool_call, payload):
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
                print(f"Terminus tool-call GRPO task {task_name} failed: {exc}")
                return None
            finally:
                try:
                    await self.run_in_executor(
                        self._close_env,
                        timeout=self.task_timeouts.cleanup,
                    )
                except Exception as exc:
                    print(f"Terminus tool-call cleanup failed for {task_name}: {exc}")

        return await _run()


class TerminusToolTerminalGRPOWorkflow(RolloutWorkflow):
    """AReaL rollout workflow for Terminus tool-calling terminal tasks."""

    def __init__(
        self,
        gconfig: Any,
        tokenizer: Any,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
        max_turns: int = 25,
        max_tokens_per_trajectory: int = 32768,
        max_workers: int = 16,
        observation_max_chars: int = 8000,
        turn_discount: float = 0.9,
        task_timeouts: Any | None = None,
        filter_uniform_reward: bool = False,
        encourage_completion_reward: bool = False,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        # AReaL's trainer uses config.gconfig.n_samples as the GRPO group size.
        # Keep that shared config intact and use a private one-sample generation
        # config inside each grouped rollout worker.
        self.gconfig = gconfig.new(n_samples=1) if hasattr(gconfig, "new") else copy.copy(gconfig)
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir or "terminus_tool_grpo_generated"
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

    async def arun_episode(self, engine: Any, data: dict[str, Any]):
        from areal import workflow_context
        from areal.experimental.openai import ArealOpenAI
        from areal.utils import stats_tracker

        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                tool_call_parser="qwen3",
                reasoning_parser="qwen3",
                engine_max_tokens=self.max_tokens_per_trajectory,
                chat_template_type="hf",
            )
            for _ in range(self.n_trajs)
        ]
        uids = [uuid.uuid4().hex[:8] for _ in range(self.n_trajs)]
        rewards = await asyncio.gather(
            *[
                TerminusToolTerminalTaskRunner(
                    output_path=os.path.join(self.dump_dir, "TerminusToolTerminalTaskRunner"),
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

        completions_with_reward: dict[str, Any] = {}
        for reward, client in zip(rewards, clients):
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


try:
    from harbor.agents.base import BaseAgent as HarborBaseAgent
    from harbor.agents.terminus_2.tmux_session import TmuxSession as HarborTmuxSession
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.trial.paths import EnvironmentPaths
except Exception:  # pragma: no cover
    HarborBaseAgent = object  # type: ignore[assignment]
    HarborTmuxSession = None  # type: ignore[assignment]
    BaseEnvironment = object  # type: ignore[assignment]
    AgentContext = object  # type: ignore[assignment]
    EnvironmentPaths = None  # type: ignore[assignment]


class TerminusToolCallingAgent(HarborBaseAgent):  # type: ignore[misc]
    """Harbor/Terminal-Bench agent using the execute_commands tool."""

    SUPPORTS_ATIF = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_turns: int | None = None,
        max_tokens: int = 8192,
        top_p: float | None = None,
        top_k: int | None = None,
        model_info: dict[str, Any] | None = None,
        record_terminal_session: bool = True,
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        llm_kwargs: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if model_name is None:
            raise ValueError("model_name is required")
        self._model_name = model_name
        self._api_base = (api_base or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self._temperature = temperature
        self._max_turns = max_turns if max_turns is not None else 1000000
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._model_info = model_info or DEFAULT_MODEL_INFO
        self._record_terminal_session = record_terminal_session
        self._tmux_pane_width = tmux_pane_width
        self._tmux_pane_height = tmux_pane_height
        self._llm_kwargs = dict(llm_kwargs or {})
        self._extra_env = dict(NONINTERACTIVE_TERMINAL_ENV)
        self._extra_env.update(extra_env or {})
        self._session: Any = None
        self._pending_completion = False
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._messages: list[dict[str, Any]] = []
        self._api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "EMPTY"

    @staticmethod
    def name() -> str:
        return "terminus-tool-calling"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: Any) -> None:
        if HarborTmuxSession is None or EnvironmentPaths is None:
            raise RuntimeError("Harbor is not available")
        if self._record_terminal_session:
            local_recording_path = environment.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            local_recording_path = None
            remote_recording_path = None
        self._session = HarborTmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_tool_calling.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        await self._session.start()
        await self._session.send_keys(
            _terminal_env_export_command(),
            block=True,
            min_timeout_sec=0.1,
        )
        await self._session.get_incremental_output()

    async def _call_llm(self, messages: list[dict[str, Any]], logging_path: Path | None) -> dict[str, Any]:
        body = {
            "model": self._model_name,
            "messages": messages,
            "tools": [EXECUTE_COMMANDS_TOOL],
            "tool_choice": {
                "type": "function",
                "function": {"name": "execute_commands"},
            },
            "max_tokens": self._max_tokens,
            **self._llm_kwargs,
        }
        if self._temperature is not None:
            body.setdefault("temperature", self._temperature)
        if self._top_p is not None:
            body.setdefault("top_p", self._top_p)
        if self._top_k is not None:
            body.setdefault("top_k", self._top_k)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        current_body = body

        async def post_once(request_body: dict[str, Any], path: Path | None) -> httpx.Response:
            async with httpx.AsyncClient(timeout=300.0) as client:
                result = await client.post(
                    f"{self._api_base}/chat/completions",
                    headers=headers,
                    json=request_body,
                )
            if path is not None:
                path.write_text(result.text, encoding="utf-8")
            return result

        response = await post_once(current_body, logging_path)
        context_retry_body = _context_length_retry_body(current_body, response.text) if response.status_code >= 400 else None
        if context_retry_body is not None:
            current_body = context_retry_body
            response = await post_once(
                current_body,
                logging_path.with_suffix(".context_retry.json") if logging_path is not None else None,
            )
        context_retry_body = _context_length_retry_body(current_body, response.text) if response.status_code >= 400 else None
        if context_retry_body is not None:
            current_body = context_retry_body
            response = await post_once(
                current_body,
                logging_path.with_suffix(".context_retry2.json") if logging_path is not None else None,
            )
        if response.status_code >= 400 and "tool_choice" in current_body:
            retry_body = dict(current_body)
            retry_body.pop("tool_choice", None)
            response = await post_once(
                retry_body,
                logging_path.with_suffix(".retry.json") if logging_path is not None else None,
            )
            context_retry_body = _context_length_retry_body(retry_body, response.text) if response.status_code >= 400 else None
            if context_retry_body is not None:
                response = await post_once(
                    context_retry_body,
                    logging_path.with_suffix(".retry_context.json") if logging_path is not None else None,
                )
                context_retry_body = _context_length_retry_body(context_retry_body, response.text) if response.status_code >= 400 else None
                if context_retry_body is not None:
                    response = await post_once(
                        context_retry_body,
                        logging_path.with_suffix(".retry_context2.json") if logging_path is not None else None,
                    )
        if response.status_code >= 400 and _is_context_length_error(response.text):
            raise ContextLengthExceededError(response.text[:1000])
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self._completion_tokens += int(usage.get("completion_tokens") or 0)
        return payload["choices"][0]["message"]

    async def _execute_commands(self, payload: ParsedPayload) -> str:
        if self._session is None:
            raise RuntimeError("tmux session is not initialized")
        for command in payload.commands:
            try:
                await self._session.send_keys(
                    command.keystrokes,
                    block=False,
                    min_timeout_sec=command.duration,
                )
            except TimeoutError:
                return TIMEOUT_PROMPT_TEMPLATE.format(
                    timeout_sec=command.duration,
                    command=command.keystrokes,
                    terminal_state=limit_output_length(await self._session.get_incremental_output()),
                )
        return limit_output_length(await self._session.get_incremental_output())

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        if self._session is None:
            raise RuntimeError("setup() must be called before run()")
        initial_state = await self._session.get_incremental_output()
        messages = build_initial_messages(instruction, terminal_state=initial_state)
        self._messages = messages
        stop_reason = "max_turns"

        for turn in range(self._max_turns):
            response_path = self.logs_dir / f"turn-{turn:03d}-response.json"
            try:
                message = await self._call_llm(messages, response_path)
            except ContextLengthExceededError:
                stop_reason = "context_length_exceeded"
                break
            tool_call = _first_tool_call(message)
            if tool_call is None:
                messages.extend(
                    _synthetic_recovery_tool_turn(
                        "No execute_commands tool call was produced."
                    )
                )
                continue
            call_id = _tool_call_id(tool_call)
            name = _tool_call_name(tool_call)
            if name != "execute_commands":
                messages.extend(_synthetic_recovery_tool_turn(f"Unknown tool: {name}"))
                continue
            try:
                payload = parse_execute_commands_arguments(_tool_call_arguments(tool_call))
            except TerminusToolPayloadError as exc:
                messages.extend(_synthetic_recovery_tool_turn(str(exc)))
                continue
            messages.append(_assistant_tool_message_from_parsed_payload(message, tool_call, payload))
            observation = await self._execute_commands(payload)
            if payload.task_complete:
                if self._pending_completion:
                    messages.append(tool_response_message(call_id, observation))
                    stop_reason = "task_complete"
                    break
                self._pending_completion = True
                observation = completion_confirmation_message(observation)
            else:
                self._pending_completion = False
            messages.append(tool_response_message(call_id, observation))

        context.n_input_tokens = self._prompt_tokens
        context.n_output_tokens = self._completion_tokens
        context.n_cache_tokens = 0
        context.cost_usd = 0.0
        context.metadata = {"message_count": len(messages), "agent": self.name(), "stop_reason": stop_reason}


def _find_arrow_files(cache_root: Path, config: str) -> list[Path]:
    base = cache_root / "datasets" / "nvidia___nemotron-terminal-corpus" / config
    arrow_files = sorted(base.glob("*/**/nemotron-terminal-corpus-train-*.arrow"))
    if arrow_files:
        return arrow_files
    return sorted(cache_root.glob(f"**/{config}/**/nemotron-terminal-corpus-train-*.arrow"))


def _iter_corpus_rows(cache_root: Path, config: str, limit: int | None) -> Iterable[dict[str, Any]]:
    from datasets import Dataset

    yielded = 0
    for arrow_path in _find_arrow_files(cache_root, config):
        dataset = Dataset.from_file(str(arrow_path))
        for row in dataset:
            yield dict(row)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def convert_nemotron_corpus(args: argparse.Namespace) -> None:
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "nvidia/Nemotron-Terminal-Corpus",
        "config": args.config,
        "row_index_parity": args.row_index_parity,
        "source_rows_seen": 0,
        "source_rows_skipped_by_parity": 0,
        "converted": 0,
        "failed": 0,
        "failures": [],
    }
    with output.open("w", encoding="utf-8") as handle:
        for row_idx, row in enumerate(_iter_corpus_rows(args.cache_root, args.config, args.limit)):
            summary["source_rows_seen"] += 1
            if args.row_index_parity == "even" and row_idx % 2 != 0:
                summary["source_rows_skipped_by_parity"] += 1
                continue
            if args.row_index_parity == "odd" and row_idx % 2 == 0:
                summary["source_rows_skipped_by_parity"] += 1
                continue
            try:
                messages = convert_terminus2_conversation(row["conversations"])
                out = {
                    "messages": messages,
                    "tools": [EXECUTE_COMMANDS_TOOL],
                    "source_dataset": "nvidia/Nemotron-Terminal-Corpus",
                    "source_config": args.config,
                    "source_row_index": row_idx,
                    "source_task": row.get("task"),
                    "source_trial_name": row.get("trial_name"),
                    "source_model": row.get("model"),
                    "source_agent": row.get("agent"),
                }
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                summary["converted"] += 1
            except Exception as exc:
                summary["failed"] += 1
                if len(summary["failures"]) < 20:
                    summary["failures"].append({"row": row_idx, "error": str(exc)})
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def inspect_converted(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.n:
                break
    parts = ["# Terminus Tool-Calling Conversion Inspection\n"]
    for idx, row in enumerate(rows):
        parts.append(f"\n## Row {idx}\n")
        parts.append(f"- source_task: `{row.get('source_task')}`\n")
        messages = row["messages"]
        parts.append(f"- roles: `{[m['role'] for m in messages]}`\n")
        assistant = next((m for m in messages if m["role"] == "assistant"), None)
        tool = next((m for m in messages if m["role"] == "tool"), None)
        parts.append("\nFirst assistant tool call arguments:\n\n```json\n")
        if assistant and assistant.get("tool_calls"):
            args_text = assistant["tool_calls"][0]["function"]["arguments"]
            parts.append(json.dumps(json.loads(args_text), indent=2, ensure_ascii=False)[:4000])
        parts.append("\n```\n")
        if tool:
            parts.append("\nFirst tool response excerpt:\n\n```text\n")
            parts.append(str(tool.get("content", ""))[:4000])
            parts.append("\n```\n")
    args.output.write_text("".join(parts), encoding="utf-8")
    print(args.output)


def check_qwen_template(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    model_path = args.model
    if args.local_files_only and "/" in model_path and not Path(model_path).exists():
        candidate = args.cache_dir / "hub" / ("models--" + model_path.replace("/", "--")) / "refs" / "main"
        if candidate.exists():
            revision = candidate.read_text(encoding="utf-8").strip()
            snapshot = candidate.parent.parent / "snapshots" / revision
            if snapshot.exists():
                model_path = str(snapshot)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=str(args.cache_dir),
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    messages = [
        *build_initial_messages("inspect the repo"),
        {
            "role": "assistant",
            "content": "<think>\nfirst turn reasoning\n</think>",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "execute_commands",
                        "arguments": payload_to_arguments(
                            {
                                "analysis": "Need inspect files first.",
                                "plan": "Run ls.",
                                "commands": [{"keystrokes": "ls\n", "duration": 0.1}],
                                "task_complete": False,
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "execute_commands",
            "content": "README.md\nsrc/",
        },
        {
            "role": "assistant",
            "content": "<think>\nsecond turn reasoning\n</think>",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "execute_commands",
                        "arguments": payload_to_arguments(
                            {
                                "analysis": "Repo has README and src.",
                                "plan": "Finish.",
                                "commands": [],
                                "task_complete": True,
                            }
                        ),
                    },
                }
            ],
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=[EXECUTE_COMMANDS_TOOL],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    result = {
        "model": args.model,
        "resolved_model_path": model_path,
        "roles": [m["role"] for m in messages],
        "user_message_count": sum(1 for m in messages if m["role"] == "user"),
        "contains_first_think": "first turn reasoning" in rendered,
        "contains_second_think": "second turn reasoning" in rendered,
        "contains_tool_response": "<tool_response>" in rendered,
        "rendered_output": str(args.output) if args.output else None,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["contains_first_think"] or not result["contains_second_think"]:
        raise SystemExit("Qwen chat template stripped assistant thinking")


async def deepseek_synthetic_smoke(args: argparse.Namespace) -> None:
    if get_terminal_synthetic_task_dataset is None:
        raise RuntimeError("terminal_task_grpo dataset loader is not available")
    read_env_file(args.env)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(f"DEEPSEEK_API_KEY not found in environment or {args.env}")
    dataset = get_terminal_synthetic_task_dataset(
        path=str(args.manifest),
        split="train",
        seed=args.seed,
        limit=args.limit,
        shuffle_records=args.shuffle,
    )
    from concurrent.futures import ThreadPoolExecutor

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for idx, row in enumerate(dataset):
            runner = TerminusToolTerminalTaskRunner(
                output_path=str(args.output_dir),
                max_turns=args.max_turns,
                max_tokens_per_turn=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                observation_max_chars=args.observation_max_chars,
                task_timeouts=TerminalTaskTimeouts(
                    reset_env=args.reset_timeout,
                    command=args.command_timeout,
                    verifier=args.verifier_timeout,
                    cleanup=args.cleanup_timeout,
                ),
                encourage_completion_reward=False,
                executor=executor,
            )
            client = DeepSeekRewardShim(
                model=args.model,
                api_key=api_key,
                base_url=args.base_url,
                timeout=args.api_timeout,
                thinking=args.thinking,
            )
            reward = await runner.run_agent(
                data=dict(row),
                client=client,
                uid=f"deepseek-smoke-{idx}-{uuid.uuid4().hex[:6]}",
                traj_i=idx,
            )
            result = {
                "task_name": row.get("task_name"),
                "reward": reward,
                "passed": reward == 1.0,
                "model": args.model,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if reward == 1.0 and args.stop_after_pass:
                break

    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not any(item["passed"] for item in results):
        raise SystemExit("No smoke task verifier passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    convert = sub.add_parser("convert-corpus")
    convert.add_argument("--cache-root", type=Path, default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache"))
    convert.add_argument("--config", default="skill_based_medium")
    convert.add_argument("--limit", type=int)
    convert.add_argument("--row-index-parity", choices=("all", "even", "odd"), default="all")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--summary-output", type=Path, required=True)
    convert.set_defaults(func=convert_nemotron_corpus)

    inspect = sub.add_parser("inspect-converted")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("-n", type=int, default=3)
    inspect.set_defaults(func=inspect_converted)

    template = sub.add_parser("check-qwen-template")
    template.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    template.add_argument("--cache-dir", type=Path, default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/hf_cache"))
    template.add_argument("--local-files-only", action="store_true")
    template.add_argument("--output", type=Path)
    template.set_defaults(func=check_qwen_template)

    smoke = sub.add_parser("deepseek-synthetic-smoke")
    smoke.add_argument("--env", type=Path, default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/.env"))
    smoke.add_argument("--model", default="deepseek-v4-pro")
    smoke.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    smoke.add_argument(
        "--manifest",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/terminal_synthetic_tasks/easy/manifest.csv"),
    )
    smoke.add_argument("--limit", type=int, default=4)
    smoke.add_argument("--seed", type=int, default=7)
    smoke.add_argument("--shuffle", action="store_true")
    smoke.add_argument("--max-workers", type=int, default=1)
    smoke.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/deepseek_smoke"),
    )
    smoke.add_argument(
        "--results-output",
        type=Path,
        default=Path("/wbl-fast/usrs/ee/teacher-answer-rl/areal_runs/terminus-tool-calling/deepseek_smoke/results.json"),
    )
    smoke.add_argument("--max-turns", type=int, default=12)
    smoke.add_argument("--max-tokens", type=int, default=4096)
    smoke.add_argument("--temperature", type=float, default=0.2)
    smoke.add_argument("--top-p", type=float, default=0.8)
    smoke.add_argument("--observation-max-chars", type=int, default=6000)
    smoke.add_argument("--api-timeout", type=float, default=240.0)
    smoke.add_argument("--reset-timeout", type=float, default=1200.0)
    smoke.add_argument("--command-timeout", type=float, default=120.0)
    smoke.add_argument("--verifier-timeout", type=float, default=600.0)
    smoke.add_argument("--cleanup-timeout", type=float)
    smoke.add_argument("--thinking", action="store_true")
    smoke.add_argument("--stop-after-pass", action="store_true")
    smoke.set_defaults(func=lambda args: asyncio.run(deepseek_synthetic_smoke(args)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


__all__ = [
    "EXECUTE_COMMANDS_TOOL",
    "TERMINUS_TOOL_SYSTEM_PROMPT",
    "TerminusToolCallingAgent",
    "TerminusToolTerminalGRPOConfig",
    "TerminusToolTerminalGRPOWorkflow",
    "build_initial_messages",
    "convert_terminus2_conversation",
    "parse_execute_commands_arguments",
    "payload_to_arguments",
]
