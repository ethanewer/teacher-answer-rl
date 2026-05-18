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
- "commands": Array of command objects to execute

Optional fields:
- "analysis": One short sentence only. Omit it when it is not needed.
- "plan": One short sentence only. Omit it when it is not needed.
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
- Keep tool arguments concise. Do not restate the task, repeat requirements, or write long reasoning in analysis/plan.
- Prefer acting with shell commands over thinking. Keep any hidden reasoning brief; every assistant turn should quickly call `execute_commands` instead of continuing to reason in text.
- If you are unsure, run a small inspection command. If a previous approach repeats or stalls, try a different command or mark completion when the checks pass.
- Preserve exact requested output formats, filenames, key names, and delimiters. Do not abbreviate JSON keys or strip wrappers such as FLAG{...} or secret[...].
- If a task mentions recovered changes, patch files, resources, backups, or hidden history, inspect likely local evidence such as /app/resources, /app/inputs, /data, git branches, git reflog, and hidden refs before repeating generic status commands.
- For regex, JSON, config, and other file-output tasks, write the required output file directly, validate it with a small local check, then set "task_complete": true.
- Prefer tools already present in the container. Do not install OS or Python packages unless the task explicitly requires it or a quick check proves the needed tool is unavailable and cannot be replaced with Python stdlib or common shell utilities.
- Run the relevant tests or checks after changing files. When the task requirements are satisfied, stop exploring and set "task_complete": true in the current tool call.
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
                    "description": "Optional one-sentence analysis of current state.",
                    "maxLength": 300,
                },
                "plan": {
                    "type": "string",
                    "description": "Optional one-sentence plan for the next commands.",
                    "maxLength": 300,
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
            "required": ["commands"],
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_initial_messages(
    instruction: str,
    *,
    terminal_state: str = "",
    task_name: str | None = None,
) -> list[dict[str, Any]]:
    state = terminal_state.strip() or "Current Terminal Screen:\n$"
    reminders = (
        _task_specific_reminders(instruction, task_name=task_name)
        if _env_bool("TERMINUS_TOOL_ENABLE_TASK_REMINDERS", False)
        else []
    )
    reminder_text = ""
    if reminders:
        reminder_text = "\n\nTask-specific reminders:\n" + "\n".join(
            f"- {reminder}" for reminder in reminders
        )
    user_content = (
        f"Task Description:\n{instruction.strip()}\n\n"
        f"{reminder_text}"
        f"{'\n\n' if reminder_text else ''}"
        f"Current terminal state:\n{state}"
    )
    return [
        {"role": "system", "content": TERMINUS_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _task_specific_reminders(
    instruction: str,
    *,
    task_name: str | None = None,
) -> list[str]:
    text = f"{task_name or ''}\n{instruction}".lower()
    task_key = (task_name or "").lower()
    reminders: list[str] = []

    def add(value: str) -> None:
        if value not in reminders:
            reminders.append(value)

    if "constraints-scheduling" in task_key or "calendar" in text or ".ics" in text:
        add("Do not edit the input calendar files; parse their VEVENT DTSTART/DTEND entries and write only the requested output ICS file.")
        add("Search valid 1-hour slots at minute granularity, enforce hard constraints first, then apply the stated tie-breakers.")
        add("Use compact UTC ICS timestamps such as DTSTART:YYYYMMDDTHHMMSSZ; do not write ISO timestamps with dashes, colons, +00:00, or a second Z.")
        add("Validate the output ICS has VCALENDAR/VEVENT headers, the exact meeting summary, and one ATTENDEE mailto line for every required attendee email.")
    if "fix-git" in task_key or "patch_files" in text or "personal site" in text:
        add("Look in git history, reflogs, and /app/resources/patch_files before guessing; compare recovered files against the working tree.")
        add("The final master branch should contain the recovered site files, with no unrelated edits.")
    if "git-leak" in task_key or "secret[" in text or "rewriting history" in text:
        add("Search dangling git objects as well as reachable history, recover the secret to /app/secret.txt, then prune unreachable objects.")
        add("After cleanup, check git log --all -p -S and git fsck output so the secret is no longer discoverable in the repo.")
    if "log-summary" in task_key or "summary.csv" in text:
        add("Use the date embedded in each log filename, not wall-clock date, and preserve the exact CSV row order requested.")
        add("Count exactly ERROR, WARNING, and INFO for today, last_7_days, last_30_days, month_to_date, and total.")
    if "modernize-scientific-stack" in task_key or "analyze_climate_modern.py" in text:
        add("Create the modern script and dependency file in /app; do not modify the legacy Python 2 source.")
        add("Run the modern script and verify it prints one formatted mean-temperature line per station.")
    if "multi-source-data-merger" in task_key or "merged_users.parquet" in text:
        add("Normalize schema names first, merge by user_id with source_a > source_b > source_c priority, and write both required output files.")
        add("Record every conflicting field in conflicts.json with per-source values and the selected value.")
        add("After writing the merge script, run it immediately; if it raises any exception or misses an output file, fix and rerun before finishing.")
    if "nginx-request-logging" in task_key or "benchmark-access.log" in text:
        add("Configure nginx.conf for log_format and limit_req_zone, and put the server block in /etc/nginx/conf.d/benchmark-site.conf.")
        add("limit_req_zone only defines the shared zone and rate; put burst/nodelay on the limit_req directive inside the server or location.")
        add("Create the document root pages, remove the default site, run nginx -t, start or reload nginx, and curl localhost:8080 before finishing.")
    if "regex-log" in task_key or "regex.txt" in text:
        add("Act first: write a Python-compatible regex pattern to /app/regex.txt, then test it locally with Python re.findall(..., re.MULTILINE).")
        add("The file must contain only the regex pattern, with no surrounding quotes, slashes, comments, or explanation.")
        add("Use boundaries around both IPv4 addresses and dates, reject invalid octets/months/days, and capture only the last valid date per matching line.")
    if "sqlite-db-truncate" in task_key or "recover.json" in text:
        add("Do not stay inside the interactive sqlite> prompt; run sqlite3 commands noninteractively or send .quit before shell commands.")
        add("If sqlite3 cannot read the truncated database, inspect the binary directly for recoverable row strings and nearby numeric values.")
        add("Write /app/recover.json as a JSON list of objects with word and value fields, then load it back with Python to validate.")
    if "vulnerable-secret" in task_key or "flag{" in text or "results.txt" in text:
        add("Inspect the executable with strings and small controlled inputs before brute force; capture the exact FLAG{...} value only.")
        add("If inspection does not reveal the flag quickly, write a small noninteractive brute-force or checker script rather than continuing manual guesses.")
        add("Write only the recovered flag to /app/results.txt and verify the file contents before marking the task complete.")

    return reminders[:5]


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

    task_complete = payload.get("task_complete", False)
    if isinstance(task_complete, str):
        task_complete = task_complete.strip().lower() in {"1", "true", "yes", "on"}

    return ParsedPayload(
        analysis=analysis,
        plan=plan,
        commands=parsed_commands,
        task_complete=bool(task_complete),
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
_TOOL_CALL_START_RE = re.compile(r"<tool_call>\s*", flags=re.DOTALL)


def _synthetic_execute_commands_tool_call(payload: ParsedPayload, *, call_id: str | None = None) -> dict[str, Any]:
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": "execute_commands",
            "arguments": payload_to_arguments(payload),
        },
    }


def _payload_from_visible_tool_object(obj: dict[str, Any]) -> ParsedPayload:
    if obj.get("name") == "execute_commands":
        return parse_execute_commands_arguments(obj.get("arguments", {}))
    function = obj.get("function")
    if isinstance(function, dict) and function.get("name") == "execute_commands":
        return parse_execute_commands_arguments(function.get("arguments", {}))
    return parse_execute_commands_arguments(obj)


def _tool_call_from_assistant_content(content: str | None) -> dict[str, Any] | None:
    if not isinstance(content, str) or not content.strip():
        return None

    for match in _TOOL_CALL_XML_RE.finditer(content):
        raw = match.group(1).strip()
        try:
            parsed = _payload_from_visible_tool_object(_extract_json_object(raw))
        except TerminusToolPayloadError:
            continue
        return _synthetic_execute_commands_tool_call(parsed)

    start_match = _TOOL_CALL_START_RE.search(content)
    if start_match is not None:
        raw = content[start_match.end():]
        end = raw.find("</tool_call>")
        if end >= 0:
            raw = raw[:end]
        try:
            parsed = _payload_from_visible_tool_object(_extract_json_object(raw))
        except TerminusToolPayloadError:
            pass
        else:
            return _synthetic_execute_commands_tool_call(parsed)

    try:
        parsed = _payload_from_visible_tool_object(_extract_json_object(content))
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
    start_match = _TOOL_CALL_START_RE.search(cleaned)
    if start_match is not None:
        cleaned = cleaned[: start_match.start()].strip()
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


def _set_chat_template_kwargs(body: dict[str, Any], **kwargs: Any) -> None:
    chat_template_kwargs = dict(body.get("chat_template_kwargs") or {})
    chat_template_kwargs.update(kwargs)
    body["chat_template_kwargs"] = chat_template_kwargs


def _flatten_extra_body(body: dict[str, Any]) -> None:
    extra_body = body.pop("extra_body", None)
    if not isinstance(extra_body, dict):
        return
    extra_chat_template_kwargs = extra_body.pop("chat_template_kwargs", None)
    if isinstance(extra_chat_template_kwargs, dict):
        chat_template_kwargs = dict(extra_chat_template_kwargs)
        chat_template_kwargs.update(dict(body.get("chat_template_kwargs") or {}))
        body["chat_template_kwargs"] = chat_template_kwargs
    for key, value in extra_body.items():
        body.setdefault(key, value)


def _tool_choice_payload() -> Any | None:
    mode = os.environ.get("TERMINUS_TOOL_CHOICE_MODE", "named").strip().lower()
    if mode in {"none", "off", "disabled"}:
        return None
    if mode in {"auto", "required"}:
        return mode
    return {
        "type": "function",
        "function": {"name": "execute_commands"},
    }


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


def _message_context_chars(message: dict[str, Any]) -> int:
    total = 0
    for key in ("content", "reasoning", "reasoning_content"):
        content = message.get(key)
        if isinstance(content, str):
            total += len(content)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        total += len(json.dumps(tool_calls, ensure_ascii=False))
    return total + 256


def _trim_messages_for_context(
    messages: list[dict[str, Any]],
    *,
    max_input_tokens: int,
    max_output_tokens: int,
    keep_recent_turns: int,
    token_counter: Any | None = None,
) -> list[dict[str, Any]]:
    if len(messages) <= 2:
        return messages

    budget_tokens = max(max_input_tokens - max_output_tokens - 1024, 4096)
    char_budget = max(8000, budget_tokens)

    def fits(candidate: list[dict[str, Any]]) -> bool:
        if token_counter is not None:
            try:
                return int(token_counter(candidate)) <= budget_tokens
            except Exception:
                pass
        return sum(_message_context_chars(message) for message in candidate) <= char_budget

    tail = messages[2:]
    assistant_starts = [
        idx for idx, message in enumerate(tail) if message.get("role") == "assistant"
    ]
    if not assistant_starts:
        return messages

    start = 0
    if len(assistant_starts) > keep_recent_turns:
        start = assistant_starts[-keep_recent_turns]
    candidate = messages[:2] + tail[start:]
    while start < len(tail) and not fits(candidate):
        next_starts = [idx for idx in assistant_starts if idx > start]
        if not next_starts:
            break
        start = next_starts[0]
        candidate = messages[:2] + tail[start:]

    trimmed = messages[:2] + tail[start:]
    if start <= 0 and fits(trimmed):
        return messages
    return trimmed


def _assistant_tool_message_from_parsed_payload(
    message: Any,
    tool_call: Any,
    payload: ParsedPayload,
) -> dict[str, Any]:
    data = _message_to_dict(message)
    data["role"] = "assistant"
    data["content"] = _message_content_without_tool_call(data.get("content"))
    if os.environ.get("TERMINUS_TOOL_KEEP_ASSISTANT_REASONING", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        data.pop("reasoning", None)
        data.pop("reasoning_content", None)
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


def _terminal_observation_max_bytes(default: int = 10000) -> int:
    raw = os.environ.get("TERMINUS_TOOL_OBSERVATION_MAX_BYTES")
    if raw is None:
        return default
    try:
        return max(512, int(raw))
    except ValueError:
        return default


def limit_output_length(output: str, max_bytes: int | None = None) -> str:
    if max_bytes is None:
        max_bytes = _terminal_observation_max_bytes()
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
        'with an empty "commands" array and "task_complete": true.'
    )


def _strip_new_terminal_prefix(text: str) -> str:
    stripped = text.strip()
    for prefix in ("New Terminal Output:", "Current Terminal Screen:", "Current terminal state:"):
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].lstrip()
    return stripped


def _normalized_keystrokes_for_repeat(keystrokes: str) -> str:
    return " ".join(keystrokes.strip().split())


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
            messages.append(tool_response_message(call_id, observation))
            return True
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
        context_keep_recent_turns: int = 6,
        llm_kwargs: dict[str, Any] | None = None,
        enable_thinking: bool | None = None,
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
        self._context_keep_recent_turns = max(1, int(context_keep_recent_turns))
        self._llm_kwargs = dict(llm_kwargs or {})
        self._enable_thinking = (
            _env_bool("TERMINUS_TOOL_ENABLE_THINKING", True)
            if enable_thinking is None
            else bool(enable_thinking)
        )
        self._extra_env = dict(NONINTERACTIVE_TERMINAL_ENV)
        self._extra_env.update(extra_env or {})
        self._session: Any = None
        self._pending_completion = False
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._messages: list[dict[str, Any]] = []
        self._seen_command_counts: dict[str, int] = {}
        self._api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "EMPTY"
        self._tokenizer: Any | None = None

    def _count_chat_tokens(self, messages: list[dict[str, Any]]) -> int:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            )
        return len(
            self._tokenizer.apply_chat_template(
                messages,
                tools=[EXECUTE_COMMANDS_TOOL],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self._enable_thinking,
            )
        )

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
        messages[:] = _trim_messages_for_context(
            messages,
            max_input_tokens=int(self._model_info.get("max_input_tokens") or 32768),
            max_output_tokens=int(self._max_tokens),
            keep_recent_turns=self._context_keep_recent_turns,
            token_counter=self._count_chat_tokens,
        )
        body = {
            "model": self._model_name,
            "messages": messages,
            "tools": [EXECUTE_COMMANDS_TOOL],
            "max_tokens": self._max_tokens,
            **self._llm_kwargs,
        }
        _flatten_extra_body(body)
        tool_choice = _tool_choice_payload()
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        _set_chat_template_kwargs(body, enable_thinking=self._enable_thinking)
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
        if (
            response.status_code >= 400
            and "tool_choice" in current_body
            and _env_bool("TERMINUS_TOOL_ENABLE_FORCED_NO_THINK_RETRY", False)
        ):
            forced_retry_body = dict(current_body)
            _set_chat_template_kwargs(forced_retry_body, enable_thinking=False)
            response = await post_once(
                forced_retry_body,
                logging_path.with_suffix(".forced_retry.json") if logging_path is not None else None,
            )
            current_body = forced_retry_body
        if response.status_code >= 400 and "tool_choice" in current_body:
            retry_body = dict(current_body)
            retry_body.pop("tool_choice", None)
            response = await post_once(
                retry_body,
                logging_path.with_suffix(".retry.json") if logging_path is not None else None,
            )
            current_body = retry_body
            context_retry_body = _context_length_retry_body(retry_body, response.text) if response.status_code >= 400 else None
            if context_retry_body is not None:
                response = await post_once(
                    context_retry_body,
                    logging_path.with_suffix(".retry_context.json") if logging_path is not None else None,
                )
                current_body = context_retry_body
                context_retry_body = _context_length_retry_body(context_retry_body, response.text) if response.status_code >= 400 else None
                if context_retry_body is not None:
                    response = await post_once(
                        context_retry_body,
                        logging_path.with_suffix(".retry_context2.json") if logging_path is not None else None,
                    )
                    current_body = context_retry_body
        if response.status_code >= 400 and _is_context_length_error(response.text):
            raise ContextLengthExceededError(response.text[:1000])
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self._completion_tokens += int(usage.get("completion_tokens") or 0)
        message = payload["choices"][0]["message"]
        if _first_tool_call(message) is None and _env_bool("TERMINUS_TOOL_ENABLE_NO_TOOL_REPAIR", False):
            repair_body = dict(current_body)
            repair_body.pop("tool_choice", None)
            repair_body["messages"] = list(repair_body.get("messages") or messages) + [
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not call execute_commands. "
                        "Respond now with exactly one execute_commands tool call. "
                        "Use shell commands if more work is needed; if the task is complete, "
                        "set task_complete to true with an empty commands list. "
                        "Do not provide plain text."
                    ),
                }
            ]
            repair_body["max_tokens"] = min(
                int(repair_body.get("max_tokens") or self._max_tokens),
                int(os.environ.get("TERMINUS_TOOL_REPAIR_MAX_TOKENS", "768")),
            )
            repair_body["temperature"] = 0.0
            _set_chat_template_kwargs(repair_body, enable_thinking=False)
            repair_response = await post_once(
                repair_body,
                logging_path.with_suffix(".tool_retry.json") if logging_path is not None else None,
            )
            if repair_response.status_code < 400:
                repair_payload = repair_response.json()
                repair_usage = repair_payload.get("usage") or {}
                self._prompt_tokens += int(repair_usage.get("prompt_tokens") or 0)
                self._completion_tokens += int(repair_usage.get("completion_tokens") or 0)
                repair_message = repair_payload["choices"][0]["message"]
                if _first_tool_call(repair_message) is not None:
                    message = repair_message
        return message

    async def _execute_commands(self, payload: ParsedPayload) -> str:
        if self._session is None:
            raise RuntimeError("tmux session is not initialized")
        repeated_commands: list[str] = []
        for command in payload.commands:
            normalized = _normalized_keystrokes_for_repeat(command.keystrokes)
            if normalized:
                prior_count = self._seen_command_counts.get(normalized, 0)
                if prior_count >= 2:
                    repeated_commands.append(normalized)
                self._seen_command_counts[normalized] = prior_count + 1
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
        output = limit_output_length(await self._session.get_incremental_output())
        if repeated_commands:
            shown = "; ".join(repeated_commands[:3])
            output += (
                "\n\nRepeated-command warning: you have already run this exact "
                f"command several times: {shown}. Use the previous output, try a "
                "different approach, or mark task_complete when the solution is ready. "
                "Do not run the same command again unless you changed the files or arguments."
            )
        return output

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        if self._session is None:
            raise RuntimeError("setup() must be called before run()")
        initial_state = await self._session.get_incremental_output()
        task_name = None
        if context is not None:
            metadata = getattr(context, "metadata", None)
            if isinstance(metadata, dict):
                task_name = metadata.get("task_name") or metadata.get("task")
        if task_name is None and environment is not None:
            task_name = getattr(environment, "task_name", None) or getattr(
                environment,
                "name",
                None,
            )
        messages = build_initial_messages(
            instruction,
            terminal_state=initial_state,
            task_name=str(task_name) if task_name is not None else None,
        )
        self._messages = messages
        self._seen_command_counts = {}
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
                messages.append(tool_response_message(call_id, observation))
                stop_reason = "task_complete"
                break
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
