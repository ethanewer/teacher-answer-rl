"""GRPO workflow for the terminal-agent-data-gen default tool harness.

This is a self-contained adaptation of the default harness from
ethanewer/terminal-agent-data-gen. The model-facing history uses native tool
calls and tool results, so each rollout has exactly one user message: the task
prompt. This matters for reasoning-model chat templates that strip old
reasoning before the latest non-tool user query.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import time
import textwrap
import unicodedata
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import torch
from openai import AsyncOpenAI
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.perf_tracer import atrace_scope, atrace_session_phase, session_context

from rlvr_demo.teacher_answer_rl import (
    _as_tensor,
    _build_scoring_batch,
    _scoring_max_tokens,
)
from rlvr_demo.terminal_task_grpo import TerminalTaskTimeouts, _read_instruction


DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024

SYSTEM_PROMPT = """You are an expert coding assistant. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:

- read: Read file contents

- bash: Execute bash commands (ls, grep, find, etc.)

- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call
- write: Create or overwrite files

Guidelines:

- Use bash for file operations like ls, rg, find

- Use read to examine files instead of cat or sed.

- Use edit for precise changes (edits[].oldText must match exactly)

- When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls
- Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.

- Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.

- Use write only for new files or complete rewrites.

- Be concise in your responses

- Show file paths clearly when working with files

Current working directory: {cwd}"""


@dataclass
class ToolSpec:
    kind: str
    name: str
    description: str
    parameters: dict[str, Any]

    def api_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": False,
            },
        }


def _tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(
            kind="read",
            name="read",
            description=(
                "Read the contents of a file. Supports text files and images "
                "(jpg, png, gif, webp). Images are sent as attachments. For text "
                "files, output is truncated to 2000 lines or 50KB (whichever is "
                "hit first). Use offset/limit for large files. When you need the "
                "full file, continue with offset until complete."
            ),
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read (relative or absolute)",
                    },
                    "offset": {
                        "type": "number",
                        "description": "Line number to start reading from (1-indexed)",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of lines to read",
                    },
                },
            },
        ),
        ToolSpec(
            kind="bash",
            name="bash",
            description=(
                "Execute a bash command in the current working directory. Returns "
                "stdout and stderr. Output is truncated to last 2000 lines or "
                "50KB (whichever is hit first). If truncated, full output is saved "
                "to a temp file. Optionally provide a timeout in seconds."
            ),
            parameters={
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (optional, no default timeout)",
                    },
                },
            },
        ),
        ToolSpec(
            kind="edit",
            name="edit",
            description=(
                "Edit a single file using exact text replacement. Every "
                "edits[].oldText must match a unique, non-overlapping region of "
                "the original file. If two changes affect the same block or "
                "nearby lines, merge them into one edit instead of emitting "
                "overlapping edits. Do not include large unchanged regions just "
                "to connect distant changes."
            ),
            parameters={
                "type": "object",
                "required": ["path", "edits"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit (relative or absolute)",
                    },
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["oldText", "newText"],
                            "properties": {
                                "oldText": {
                                    "type": "string",
                                    "description": (
                                        "Exact text for one targeted replacement. "
                                        "It must be unique in the original file "
                                        "and must not overlap with any other "
                                        "edits[].oldText in the same call."
                                    ),
                                },
                                "newText": {
                                    "type": "string",
                                    "description": "Replacement text for this targeted edit.",
                                },
                            },
                            "additionalProperties": False,
                        },
                        "description": (
                            "One or more targeted replacements. Each edit is "
                            "matched against the original file, not incrementally. "
                            "Do not include overlapping or nested edits. If two "
                            "changes touch the same block or nearby lines, merge "
                            "them into one edit instead."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        ),
        ToolSpec(
            kind="write",
            name="write",
            description=(
                "Write content to a file. Creates the file if it doesn't exist, "
                "overwrites if it does. Automatically creates parent directories."
            ),
            parameters={
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write (relative or absolute)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
            },
        ),
    ]


TOOL_SPECS = _tool_specs()


@dataclass
class DefaultAgentTurnRecord:
    traj_i: int
    turn_index: int
    response_id: str
    prefix_messages: list[dict[str, Any]]
    assistant_message: dict[str, Any]


@dataclass
class DefaultAgentRunResult:
    reward: float
    turn_records: list[DefaultAgentTurnRecord]


def _repair_json(value: str) -> str:
    repaired: list[str] = []
    in_string = False
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    index = 0
    while index < len(value):
        char = value[index]
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue
        if char == '"':
            repaired.append(char)
            in_string = False
            index += 1
            continue
        if char == "\\":
            next_char = value[index + 1] if index + 1 < len(value) else None
            if (
                next_char == "u"
                and index + 6 <= len(value)
                and re.match(r"^[0-9a-fA-F]{4}$", value[index + 2 : index + 6])
            ):
                repaired.append(value[index : index + 6])
                index += 6
                continue
            if next_char in valid_escapes:
                repaired.append("\\" + str(next_char))
                index += 2
                continue
            repaired.append("\\\\")
            index += 1
            continue
        code = ord(char)
        if 0 <= code <= 0x1F:
            repaired.append(
                {
                    "\b": "\\b",
                    "\f": "\\f",
                    "\n": "\\n",
                    "\r": "\\r",
                    "\t": "\\t",
                }.get(char, f"\\u{code:04x}")
            )
        else:
            repaired.append(char)
        index += 1
    return "".join(repaired)


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    parsed = json.loads(_repair_json(raw))
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to an object")
    return parsed


def _tool_call_dict(tool_call: Any) -> dict[str, Any]:
    if hasattr(tool_call, "model_dump"):
        return tool_call.model_dump(exclude_none=True)
    if isinstance(tool_call, dict):
        return tool_call
    raise TypeError(f"unsupported tool call value: {type(tool_call).__name__}")


def _tool_call_name_and_args(tool_call: Any) -> tuple[str, dict[str, Any], str]:
    data = _tool_call_dict(tool_call)
    function = data.get("function") or {}
    name = str(function.get("name") or data.get("name") or "")
    call_id = str(data.get("id") or f"call_{uuid.uuid4().hex[:24]}")
    args = _parse_tool_arguments(function.get("arguments") or data.get("arguments") or {})
    return name, args, call_id


def _is_context_limit_error(exc: BaseException) -> bool:
    text = str(exc)
    return "len of prompt tokens" in text and (
        "max_total_tokens" in text or "engine_max_tokens" in text
    )


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _parse_tagged_tool_calls(content: str) -> tuple[list[dict[str, Any]], str]:
    """Parse Qwen/Nanbeige-style XML tool calls from assistant text.

    AReaL can parse this itself when the optional vLLM/sglang parser package is
    importable in the rollout process. The H200 training environment used for
    this job does not have that dependency, so the harness keeps a small local
    parser for the model's native chat-template format.
    """
    tool_calls: list[dict[str, Any]] = []

    for match in TOOL_CALL_RE.finditer(content):
        payload = match.group(1).strip()
        if not payload:
            continue
        parsed = json.loads(_repair_json(payload))
        if not isinstance(parsed, dict):
            continue
        function = parsed.get("function") or {}
        name = function.get("name") or parsed.get("name")
        if not name:
            continue
        raw_args = function.get("arguments") if function else parsed.get("arguments", {})
        args = _parse_tool_arguments(raw_args)
        call_id = str(parsed.get("id") or f"call_{uuid.uuid4().hex[:24]}")
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": str(name),
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )

    cleaned = TOOL_CALL_RE.sub("", content).rstrip()
    return tool_calls, cleaned


def _validate_tool_args(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError(f"Tool {spec.name} arguments must be an object")
    schema = spec.parameters
    required = schema.get("required") or []
    for key in required:
        if key not in args:
            raise ValueError(f"Tool {spec.name} missing required argument: {key}")
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        for key in args:
            if key not in properties:
                raise ValueError(f"Tool {spec.name} got unexpected argument: {key}")
    return dict(args)


def _b64_json(value: Any) -> str:
    return base64.b64encode(json.dumps(value, ensure_ascii=False).encode("utf-8")).decode(
        "ascii"
    )


REMOTE_READ_SCRIPT = r"""
import base64
import mimetypes
import pathlib
import sys
import json

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024
args = json.loads(base64.b64decode(PAYLOAD_B64).decode("utf-8"))

def byte_len(value):
    return len(value.encode("utf-8"))

def resolve(path):
    text = str(path)
    if text.startswith("@"):
        text = text[1:]
    candidate = pathlib.Path(text).expanduser()
    return candidate if candidate.is_absolute() else (pathlib.Path.cwd() / candidate).resolve()

def image_mime(path):
    data = path.read_bytes()[:16]
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    guessed, _ = mimetypes.guess_type(path)
    return guessed if guessed in {"image/png", "image/jpeg", "image/gif", "image/webp"} else None

path = resolve(args["path"])
if not path.exists():
    raise FileNotFoundError(f"[Errno 2] No such file or directory: '{path}'")
if path.is_dir():
    raise IsADirectoryError(f"[Errno 21] Is a directory: '{path}'")
mime = image_mime(path)
if mime:
    print(f"Read image file [{mime}]")
    print("[Current model does not support images. The image will be omitted from this request.]")
    raise SystemExit(0)

text = path.read_bytes().decode("utf-8", errors="replace")
lines = text.split("\n")
offset = args.get("offset")
limit = args.get("limit")
start = max(0, int(offset) - 1) if offset is not None else 0
if start >= len(lines):
    raise ValueError(f"Offset {offset} is beyond end of file ({len(lines)} lines total)")
if limit is not None:
    end = min(start + int(limit), len(lines))
    selected = "\n".join(lines[start:end])
else:
    end = len(lines)
    selected = "\n".join(lines[start:])

out_lines = []
out_bytes = 0
for line in selected.split("\n"):
    next_bytes = byte_len(line) + (1 if out_lines else 0)
    if len(out_lines) >= DEFAULT_MAX_LINES or out_bytes + next_bytes > DEFAULT_MAX_BYTES:
        break
    out_lines.append(line)
    out_bytes += next_bytes
output = "\n".join(out_lines)
truncated = output != selected
if truncated:
    shown_end = start + len(out_lines)
    output += f"\n\n[Showing lines {start + 1}-{shown_end} of {len(lines)}. Use offset={shown_end + 1} to continue.]"
elif limit is not None and end < len(lines):
    output += f"\n\n[{len(lines) - end} more lines in file. Use offset={end + 1} to continue.]"
print(output if output else "(empty file)")
"""


REMOTE_WRITE_SCRIPT = r"""
import base64
import pathlib
import json

args = json.loads(base64.b64decode(PAYLOAD_B64).decode("utf-8"))

def resolve(path):
    text = str(path)
    if text.startswith("@"):
        text = text[1:]
    candidate = pathlib.Path(text).expanduser()
    return candidate if candidate.is_absolute() else (pathlib.Path.cwd() / candidate).resolve()

path = resolve(args["path"])
content = str(args["content"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(content, encoding="utf-8")
print(f"Successfully wrote {len(content.encode('utf-8'))} bytes to {args['path']}")
"""


REMOTE_EDIT_SCRIPT = r"""
import base64
import difflib
import pathlib
import json

args = json.loads(base64.b64decode(PAYLOAD_B64).decode("utf-8"))

def resolve(path):
    text = str(path)
    if text.startswith("@"):
        text = text[1:]
    candidate = pathlib.Path(text).expanduser()
    return candidate if candidate.is_absolute() else (pathlib.Path.cwd() / candidate).resolve()

def detect_line_ending(content):
    crlf = content.find("\r\n")
    lf = content.find("\n")
    if lf == -1 or crlf == -1:
        return "\n"
    return "\r\n" if crlf < lf else "\n"

def normalize_to_lf(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")

def restore_line_endings(text, ending):
    return text.replace("\n", "\r\n") if ending == "\r\n" else text

def normalize_for_fuzzy_match(text):
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.translate(str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-"}))
    return normalized

def fuzzy_find(content, old):
    exact = content.find(old)
    if exact != -1:
        return True, exact, len(old), False, content
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old)
    index = fuzzy_content.find(fuzzy_old)
    if index == -1:
        return False, -1, 0, False, content
    return True, index, len(fuzzy_old), True, fuzzy_content

path = resolve(args["path"])
edits = args["edits"]
if not isinstance(edits, list) or not edits:
    raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
if not path.exists():
    raise FileNotFoundError(f"File not found: {args['path']}")

raw = path.read_text(encoding="utf-8", errors="replace")
bom = "\ufeff" if raw.startswith("\ufeff") else ""
content = raw[1:] if bom else raw
ending = detect_line_ending(content)
normalized = normalize_to_lf(content)
normalized_edits = []
for edit in edits:
    normalized_edits.append({
        "oldText": normalize_to_lf(str(edit["oldText"])),
        "newText": normalize_to_lf(str(edit["newText"])),
    })
for i, edit in enumerate(normalized_edits):
    if not edit["oldText"]:
        raise ValueError(f"edits[{i}].oldText must not be empty in {args['path']}.")

initial = [fuzzy_find(normalized, edit["oldText"]) for edit in normalized_edits]
base = normalize_for_fuzzy_match(normalized) if any(match[3] for match in initial) else normalized
matched = []
for i, edit in enumerate(normalized_edits):
    found, index, length, _, _ = fuzzy_find(base, edit["oldText"])
    if not found:
        raise ValueError(f"Could not find edits[{i}] in {args['path']}. The oldText must match exactly including whitespace.")
    occurrences = normalize_for_fuzzy_match(base).count(normalize_for_fuzzy_match(edit["oldText"]))
    if occurrences > 1:
        raise ValueError(f"Found {occurrences} occurrences of edits[{i}] in {args['path']}. Each oldText must be unique.")
    matched.append({"editIndex": i, "matchIndex": index, "matchLength": length, "newText": edit["newText"]})
matched.sort(key=lambda item: item["matchIndex"])
for i in range(1, len(matched)):
    prev = matched[i - 1]
    cur = matched[i]
    if prev["matchIndex"] + prev["matchLength"] > cur["matchIndex"]:
        raise ValueError(f"edits[{prev['editIndex']}] and edits[{cur['editIndex']}] overlap in {args['path']}.")

new_content = base
for edit in reversed(matched):
    new_content = new_content[: edit["matchIndex"]] + edit["newText"] + new_content[edit["matchIndex"] + edit["matchLength"] :]
if new_content == base:
    raise ValueError(f"No changes made to {args['path']}.")
path.write_text(bom + restore_line_endings(new_content, ending), encoding="utf-8")
diff = "\n".join(difflib.unified_diff(base.splitlines(), new_content.splitlines(), lineterm="", n=3))
print(f"Successfully replaced {len(edits)} block(s) in {args['path']}.")
if diff:
    print(diff[:12000])
"""


REMOTE_EDIT_SCRIPT = "import unicodedata\n" + REMOTE_EDIT_SCRIPT


class DefaultAgentTerminalTaskRunner:
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
        trajectory_timeout: float | None = None,
        task_service_url: str | None = None,
        task_service_url_file: str | None = None,
        return_turn_records: bool = False,
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
        self.trajectory_timeout = trajectory_timeout
        self.task_service_url = task_service_url.rstrip("/") if task_service_url else None
        self.task_service_url_file = task_service_url_file
        self.remote_session_id: str | None = None
        self.return_turn_records = return_turn_records

    def _remaining_trajectory_timeout(self, started_at: float) -> float | None:
        if self.trajectory_timeout is None:
            return None
        return max(0.0, self.trajectory_timeout - (time.monotonic() - started_at))

    async def _await_with_trajectory_timeout(self, awaitable, started_at: float):
        timeout = self._remaining_trajectory_timeout(started_at)
        if timeout is None:
            return await awaitable
        if timeout <= 0.0:
            if hasattr(awaitable, "close"):
                awaitable.close()
            elif hasattr(awaitable, "cancel"):
                awaitable.cancel()
            raise TimeoutError(
                f"trajectory exceeded {self.trajectory_timeout:.1f}s budget"
            )
        return await asyncio.wait_for(awaitable, timeout=timeout)

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
            raise RuntimeError("default-agent harness requires terminal_task_service_url")
        request_timeout = httpx.Timeout(
            timeout + 30.0 if timeout is not None else None,
            connect=30.0,
        )
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            response = await client.request(method, f"{base_url}{path}", json=json_payload)
        if response.status_code >= 400:
            detail = response.text[:2000]
            raise httpx.HTTPStatusError(
                f"{response.status_code} response from terminal task service: {detail}",
                request=response.request,
                response=response,
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
        response: dict[str, Any] | None = None
        attempts = 5
        for attempt in range(attempts):
            try:
                response = await self._remote_request(
                    "POST",
                    "/v1/sessions",
                    timeout=self.task_timeouts.reset_env,
                    json_payload=payload,
                )
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 or attempt == attempts - 1:
                    raise
                await asyncio.sleep(min(2.0**attempt, 8.0))
            except httpx.TransportError:
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(min(2.0**attempt, 8.0))
        if response is None:
            raise RuntimeError("terminal task service did not return a session")
        self.remote_session_id = str(response["session_id"])

    async def _remote_execute_keystrokes(self, keystrokes: str) -> str:
        if self.remote_session_id is None:
            raise RuntimeError("remote terminal session is not initialized")
        if not keystrokes.endswith(("\n", "\r")):
            keystrokes += "\n"
        response = await self._remote_request(
            "POST",
            f"/v1/sessions/{self.remote_session_id}/commands",
            timeout=self.task_timeouts.command + 30,
            json_payload={"commands": [{"keystrokes": keystrokes, "duration": 0.1}]},
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

    async def _remote_python(self, script: str, payload: dict[str, Any]) -> str:
        payload_b64 = _b64_json(payload)
        script_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
        token = uuid.uuid4().hex
        output_path = f"/tmp/default-agent-tool-{token}.out"
        status_path = f"/tmp/default-agent-tool-{token}.status"
        begin_marker = f"__DEFAULT_AGENT_TOOL_{token}_BEGIN__"
        end_marker = f"__DEFAULT_AGENT_TOOL_{token}_END__"
        python_command = (
            f"SCRIPT_B64={shlex.quote(script_b64)} "
            f"PAYLOAD_B64={shlex.quote(payload_b64)} "
            "\"$PYTHON_BIN\" -c "
            + shlex.quote(
                "import base64, os; "
                "PAYLOAD_B64 = os.environ['PAYLOAD_B64']; "
                "exec(base64.b64decode(os.environ['SCRIPT_B64']).decode('utf-8'))"
            )
        )
        run_command = (
            'PYTHON_BIN="$(command -v python3 || command -v python || true)"; '
            'if [ -z "$PYTHON_BIN" ]; then '
            f"echo 'python3/python not found' > {shlex.quote(output_path)}; "
            f"echo 127 > {shlex.quote(status_path)}; "
            "else "
            f"({python_command}) > {shlex.quote(output_path)} 2>&1; "
            f"echo $? > {shlex.quote(status_path)}; "
            "fi"
        )
        await self._remote_execute_keystrokes(run_command)

        collect_command = (
            f"code=$(cat {shlex.quote(status_path)} 2>/dev/null || echo 1); "
            f"printf '%s\\n' {shlex.quote(begin_marker)}; "
            f"cat {shlex.quote(output_path)} 2>/dev/null; "
            'if [ "$code" -ne 0 ]; then echo "Command exited with code $code"; fi; '
            f"printf '%s\\n' {shlex.quote(end_marker)}; "
            f"rm -f {shlex.quote(output_path)} {shlex.quote(status_path)}"
        )
        observation = await self._remote_execute_keystrokes(collect_command)
        begin = observation.rfind(begin_marker)
        if begin == -1:
            return observation
        start = begin + len(begin_marker)
        end = observation.find(end_marker, start)
        if end == -1:
            return observation[start:].strip()
        return observation[start:end].strip()

    async def _execute_bash(self, args: dict[str, Any]) -> str:
        command = str(args["command"])
        timeout = args.get("timeout")
        if timeout is not None:
            try:
                seconds = max(float(timeout), 1.0)
            except (TypeError, ValueError):
                seconds = self.task_timeouts.command
            command = f"timeout --kill-after=5s {seconds}s bash -lc {shlex.quote(command)}"
        return await self._remote_execute_keystrokes(command)

    async def _execute_tool_call(self, tool_call: Any) -> dict[str, str]:
        name, raw_args, call_id = _tool_call_name_and_args(tool_call)
        spec_by_name = {spec.name: spec for spec in TOOL_SPECS}
        spec = spec_by_name.get(name)
        if spec is None:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"Tool {name!r} not found",
            }
        try:
            args = _validate_tool_args(spec, raw_args)
            if name == "bash":
                content = await self._execute_bash(args)
            elif name == "read":
                content = await self._remote_python(REMOTE_READ_SCRIPT, args)
            elif name == "write":
                content = await self._remote_python(REMOTE_WRITE_SCRIPT, args)
            elif name == "edit":
                content = await self._remote_python(REMOTE_EDIT_SCRIPT, args)
            else:
                content = f"Tool {name!r} is not implemented"
        except Exception as exc:
            content = f"{type(exc).__name__}: {exc}"
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content[-self.observation_max_chars :],
        }

    @staticmethod
    def _assert_single_user_message(messages: list[dict[str, Any]]) -> None:
        user_messages = [message for message in messages if message.get("role") == "user"]
        if len(user_messages) != 1:
            raise RuntimeError(
                f"default-agent harness must keep exactly one user message, got {len(user_messages)}"
            )

    @staticmethod
    def _assistant_message_with_local_tool_parsing(
        response: Any,
        client: ArealOpenAI,
    ) -> dict[str, Any]:
        assistant_message = response.choices[0].message.model_dump(exclude_none=True)
        if assistant_message.get("tool_calls"):
            return assistant_message

        content = str(assistant_message.get("content") or "")
        try:
            tool_calls, cleaned_content = _parse_tagged_tool_calls(content)
        except (json.JSONDecodeError, ValueError):
            return assistant_message
        if not tool_calls:
            return assistant_message

        assistant_message["content"] = cleaned_content
        assistant_message["tool_calls"] = tool_calls

        interaction = client.get_interaction(response.id)
        if interaction is not None:
            interaction.output_message_list = [assistant_message]
        return assistant_message

    @session_context()
    async def run_agent(
        self,
        data: dict[str, Any],
        client: ArealOpenAI,
        uid: str,
        traj_i: int,
    ) -> float | DefaultAgentRunResult | None:
        task_name = str(data.get("task_name"))
        instruction = str(data.get("instruction") or "").strip()
        if not instruction:
            instruction = _read_instruction(Path(str(data["task_path"])).resolve())

        system_prompt = SYSTEM_PROMPT.replace("{cwd}", "/app")
        user_prompt = (
            "Complete the terminal task below in the current workspace.\n\n"
            f"Task ID: {task_name}\n\n"
            f"{instruction}\n\n"
            "Files from the task environment are already available in the current "
            "directory. Treat the current directory as /app. Use the available "
            "tools to inspect files, execute commands, edit code, and write files. "
            "Start by calling a tool unless the task is already complete. "
            "When the task is complete, respond without tool calls."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tools = [spec.api_definition() for spec in TOOL_SPECS]
        started_at = time.monotonic()
        turn_records: list[DefaultAgentTurnRecord] = []

        try:
            async with atrace_scope(
                f"default_agent_reset_env:{task_name},traj:{traj_i}",
                args={"uid": uid, "timeout": self.task_timeouts.reset_env},
            ):
                await self._await_with_trajectory_timeout(
                    asyncio.wait_for(
                        self._remote_reset_env(data, uid),
                        timeout=self.task_timeouts.reset_env + 30,
                    ),
                    started_at,
                )

            reward: float | None = 0.0
            for turn in range(self.max_turns):
                self._assert_single_user_message(messages)
                prefix_messages = deepcopy(messages)
                try:
                    response = await self._await_with_trajectory_timeout(
                        client.chat.completions.create(
                            messages=messages,
                            tools=tools,
                            max_completion_tokens=self.max_tokens_per_turn,
                            temperature=self.temperature,
                            top_p=self.top_p,
                        ),
                        started_at,
                    )
                except ValueError as exc:
                    if not _is_context_limit_error(exc):
                        raise
                    print(
                        f"Default-agent GRPO task {task_name} traj {traj_i} "
                        f"context exhausted at turn {turn + 1}: {exc}; evaluating",
                        flush=True,
                    )
                    break
                assistant_message = self._assistant_message_with_local_tool_parsing(
                    response=response,
                    client=client,
                )
                if self.return_turn_records:
                    turn_records.append(
                        DefaultAgentTurnRecord(
                            traj_i=traj_i,
                            turn_index=turn,
                            response_id=str(response.id),
                            prefix_messages=prefix_messages,
                            assistant_message=deepcopy(assistant_message),
                        )
                    )
                messages.append(assistant_message)
                tool_calls = assistant_message.get("tool_calls") or []
                if not tool_calls:
                    break
                names = [
                    (_tool_call_dict(tool_call).get("function") or {}).get("name", "?")
                    for tool_call in tool_calls
                ]
                print(
                    f"Default-agent GRPO task {task_name} traj {traj_i} "
                    f"turn {turn + 1} tool_calls={names}",
                    flush=True,
                )
                tool_results = await self._await_with_trajectory_timeout(
                    asyncio.gather(
                        *[
                            self._execute_tool_call(tool_call)
                            for tool_call in tool_calls
                        ]
                    ),
                    started_at,
                )
                for tool_result in tool_results:
                    messages.append(tool_result)
                if turn == self.max_turns - 1:
                    break

            self._assert_single_user_message(messages)
            async with atrace_session_phase(
                "reward",
                start_payload={"task_name": task_name, "traj_i": traj_i},
            ):
                reward = await self._await_with_trajectory_timeout(
                    asyncio.wait_for(
                        self._remote_evaluate_completion(),
                        timeout=self.task_timeouts.verifier + 30,
                    ),
                    started_at,
                )
            try:
                client.set_last_reward(float(reward))
            except RuntimeError:
                print(
                    f"Default-agent GRPO task {task_name} traj {traj_i} "
                    "has no model interaction to reward",
                    flush=True,
                )
                return None
            print(
                f"Default-agent GRPO task {task_name} traj {traj_i} reward={float(reward):.4f}",
                flush=True,
            )
            if self.return_turn_records:
                return DefaultAgentRunResult(
                    reward=float(reward),
                    turn_records=turn_records,
                )
            return float(reward)
        except TimeoutError:
            print(
                f"Default-agent GRPO task {task_name} traj {traj_i} timed out "
                f"after {time.monotonic() - started_at:.1f}s",
                flush=True,
            )
            return None
        except Exception as exc:
            print(f"Default-agent GRPO task {task_name} failed: {exc}", flush=True)
            return None
        finally:
            try:
                await self._remote_close_env()
            except Exception as exc:
                print(f"Default-agent GRPO cleanup failed for {task_name}: {exc}", flush=True)


class DefaultAgentTerminalGRPOWorkflow(RolloutWorkflow):
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
        turn_discount: float = 1.0,
        task_timeouts: TerminalTaskTimeouts | None = None,
        filter_uniform_reward: bool = False,
        encourage_completion_reward: bool = False,
        terminal_task_service_url: str | None = None,
        terminal_task_service_url_file: str | None = None,
        tool_call_parser: str = "qwen3_xml",
        reasoning_parser: str = "qwen3",
        chat_template_type: str = "concat",
        export_style: str = "concat",
        trajectory_timeout: float | None = None,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir or "default_agent_terminal_grpo_generated"
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
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.chat_template_type = chat_template_type
        self.export_style = export_style
        self.trajectory_timeout = trajectory_timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                tool_call_parser=self.tool_call_parser,
                reasoning_parser=self.reasoning_parser,
                engine_max_tokens=self.max_tokens_per_trajectory,
                chat_template_type=self.chat_template_type,
            )
            for _ in range(self.n_trajs)
        ]
        uids = [uuid.uuid4().hex[:8] for _ in range(self.n_trajs)]
        rewards = await asyncio.gather(
            *[
                DefaultAgentTerminalTaskRunner(
                    output_path=os.path.join(
                        self.dump_dir, "DefaultAgentTerminalTaskRunner"
                    ),
                    max_turns=self.max_turns,
                    max_tokens_per_turn=self.gconfig.max_new_tokens,
                    temperature=self.gconfig.temperature,
                    top_p=self.gconfig.top_p,
                    observation_max_chars=self.observation_max_chars,
                    task_timeouts=self.task_timeouts,
                    encourage_completion_reward=self.encourage_completion_reward,
                    executor=self.executor,
                    trajectory_timeout=self.trajectory_timeout,
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
        for reward, client in zip(rewards, clients):
            if reward is None:
                continue
            stats_tracker.get(workflow_context.stat_scope()).scalar(reward=float(reward))
            if self.export_style == "individual":
                client.apply_reward_discount(turn_discount=self.turn_discount)
            completions_with_reward.update(
                client.export_interactions(style=self.export_style)
            )

        stats_tracker.get(workflow_context.stat_scope()).scalar(
            num_full_passes=sum(1 for reward in rewards if reward == 1.0)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            num_trajectories_failed=sum(1 for reward in rewards if reward is None)
        )
        return completions_with_reward or None


def _external_teacher_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only OpenAI-compatible chat fields for the external teacher API."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        if role in {"system", "user"}:
            normalized.append(
                {
                    "role": role,
                    "content": str(message.get("content") or ""),
                }
            )
            continue
        if role == "assistant":
            content, reasoning_content = _split_assistant_content_for_teacher(
                str(message.get("content") or "")
            )
            raw_tool_calls = message.get("tool_calls") or []
            if raw_tool_calls and content and reasoning_content is None:
                reasoning_content = content
                content = ""
            out: dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            if reasoning_content is not None:
                out["reasoning_content"] = reasoning_content
            tool_calls = []
            for raw_tool_call in raw_tool_calls:
                try:
                    name, args, call_id = _tool_call_name_and_args(raw_tool_call)
                except Exception:
                    continue
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
            if tool_calls:
                out["tool_calls"] = tool_calls
            normalized.append(out)
            continue
        if role == "tool":
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("tool_call_id") or ""),
                    "content": str(message.get("content") or ""),
                }
            )
            continue
    return normalized


def _split_assistant_content_for_teacher(content: str) -> tuple[str, str | None]:
    """Map Qwen visible thinking text into DeepSeek's reasoning_content field.

    DeepSeek's v4 API rejects prior assistant messages that contain thinking text
    in ``content`` without a corresponding ``reasoning_content`` field.  The
    student history is therefore preserved, but represented with the field names
    expected by the teacher API.
    """
    if "</think>" in content:
        reasoning, normal = content.split("</think>", maxsplit=1)
        reasoning = reasoning.split("<think>", maxsplit=1)[-1].strip("\n")
        return normal.lstrip("\n"), reasoning
    if "<think>" in content:
        match = re.search(r"<think>(.*?)(?:</think>|$)", content, flags=re.DOTALL)
        reasoning = match.group(1).strip("\n") if match else ""
        normal = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL)
        return normal.lstrip("\n"), reasoning
    return content, None


def _format_qwen_tool_call_target(tool_calls: list[Any]) -> str:
    targets: list[str] = []
    for raw_tool_call in tool_calls:
        name, args, _ = _tool_call_name_and_args(raw_tool_call)
        if not name:
            continue
        targets.append(
            '<tool_call>\n'
            f"{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n"
            "</tool_call>"
        )
    return "\n".join(targets)


def _teacher_tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = list(message.get("tool_calls") or [])
    if tool_calls:
        return [_tool_call_dict(tool_call) for tool_call in tool_calls]

    content = str(message.get("content") or "")
    if not content:
        return []
    try:
        parsed_tool_calls, _ = _parse_tagged_tool_calls(content)
    except Exception:
        return []
    return parsed_tool_calls


def _find_token_subsequence(tokens: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(tokens):
        return None
    for start in range(len(tokens) - len(pattern) + 1):
        if tokens[start : start + len(pattern)] == pattern:
            return start
    return None


def _output_tokens_without_stop(interaction: Any) -> list[int]:
    response = interaction.model_response
    try:
        return list(response.output_tokens_without_stop)
    except Exception:
        return list(response.output_tokens)


def _student_reasoning_output_len(
    tokenizer: PreTrainedTokenizerFast,
    output_tokens: list[int],
    assistant_message: dict[str, Any],
) -> int:
    tool_tag = tokenizer.encode("<tool_call>", add_special_tokens=False)
    start = _find_token_subsequence(output_tokens, tool_tag)
    if start is not None:
        return start

    output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)
    match = TOOL_CALL_RE.search(output_text)
    if match is not None:
        prefix_ids = tokenizer.encode(
            output_text[: match.start()],
            add_special_tokens=False,
        )
        return min(len(prefix_ids), len(output_tokens))

    if assistant_message.get("tool_calls"):
        content_ids = tokenizer.encode(
            str(assistant_message.get("content") or ""),
            add_special_tokens=False,
        )
        if output_tokens[: len(content_ids)] == content_ids:
            return len(content_ids)

    return len(output_tokens)


def _metadata_tensor(ids: list[int], dtype: torch.dtype) -> torch.Tensor:
    values = ids if ids else [0]
    return torch.tensor(values, dtype=dtype).unsqueeze(0)


def _build_teacher_turn_tensor(
    *,
    tokenizer: PreTrainedTokenizerFast,
    interaction: Any,
    turn_record: DefaultAgentTurnRecord,
    verifier_reward: float,
    teacher_target: str,
) -> dict[str, torch.Tensor] | None:
    response = interaction.model_response
    if response is None:
        return None
    output_tokens = _output_tokens_without_stop(interaction)
    reasoning_len = _student_reasoning_output_len(
        tokenizer=tokenizer,
        output_tokens=output_tokens,
        assistant_message=turn_record.assistant_message,
    )
    reasoning_len = max(0, min(reasoning_len, len(output_tokens)))
    if reasoning_len <= 0:
        return None

    seq = list(response.input_tokens) + output_tokens[:reasoning_len]
    output_logprobs = list(response.output_logprobs)[:reasoning_len]
    output_versions = list(response.output_versions)[:reasoning_len]
    logprobs = [0.0] * len(response.input_tokens) + output_logprobs
    versions = [-1] * len(response.input_tokens) + output_versions
    loss_mask = [0] * len(response.input_tokens) + [1] * reasoning_len
    answer_ids = tokenizer.encode(teacher_target, add_special_tokens=False)
    answer_mask = [1] * len(answer_ids) if answer_ids else []

    return {
        "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
        "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
        "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
        "rewards": torch.tensor([float(verifier_reward)], dtype=torch.float32),
        "verifier_rewards": torch.tensor([float(verifier_reward)], dtype=torch.float32),
        "teacher_answer_prefix_ids": _metadata_tensor([], torch.int32),
        "teacher_answer_prefix_mask": _metadata_tensor([], torch.bool),
        "teacher_answer_ids": _metadata_tensor(answer_ids, torch.int32),
        "teacher_answer_mask": _metadata_tensor(answer_mask, torch.bool),
        "teacher_context_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
        "teacher_answer_turn_index": torch.tensor(
            [int(turn_record.turn_index)],
            dtype=torch.int32,
        ),
        "teacher_answer_traj_index": torch.tensor(
            [int(turn_record.traj_i)],
            dtype=torch.int32,
        ),
    }


class DefaultAgentTerminalTeacherAnswerRLWorkflow(RolloutWorkflow):
    """Student rollouts plus per-turn online teacher-answer rewards.

    Student trajectories are generated with the same default-agent harness as
    GRPO.  After rollout, DeepSeek is asked for the next tool call from each
    student prefix.  The training sample for that turn is truncated to the
    student's current reasoning prefix, and the teacher tool call is scored in
    the postprocess hook.
    """

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
        turn_discount: float = 1.0,
        task_timeouts: TerminalTaskTimeouts | None = None,
        filter_uniform_reward: bool = False,
        encourage_completion_reward: bool = False,
        terminal_task_service_url: str | None = None,
        terminal_task_service_url_file: str | None = None,
        tool_call_parser: str = "qwen3_xml",
        reasoning_parser: str = "qwen3",
        chat_template_type: str = "concat",
        export_style: str = "concat",
        trajectory_timeout: float | None = None,
        teacher_answer_model: str = "deepseek-v4-pro",
        teacher_answer_base_url: str | None = None,
        teacher_answer_api_key_env: str = "DEEPSEEK_API_KEY",
        teacher_answer_max_tokens: int = 1024,
        teacher_answer_temperature: float = 0.0,
        teacher_answer_top_p: float = 1.0,
        teacher_answer_timeout: float = 120.0,
        teacher_answer_max_retries: int = 3,
        teacher_answer_concurrency: int = 32,
    ):
        del rollout_stat_scope, turn_discount, export_style
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir or "default_agent_terminal_teacher_answer_rl_generated"
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
        self.n_trajs = n_trajs
        self.max_turns = max_turns
        self.max_tokens_per_trajectory = max_tokens_per_trajectory
        self.max_workers = max_workers
        self.observation_max_chars = observation_max_chars
        self.task_timeouts = task_timeouts or TerminalTaskTimeouts()
        self.filter_uniform_reward = filter_uniform_reward
        self.encourage_completion_reward = encourage_completion_reward
        self.terminal_task_service_url = terminal_task_service_url
        self.terminal_task_service_url_file = terminal_task_service_url_file
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.chat_template_type = chat_template_type
        self.trajectory_timeout = trajectory_timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.teacher_answer_model = teacher_answer_model
        self.teacher_answer_base_url = teacher_answer_base_url
        self.teacher_answer_api_key_env = teacher_answer_api_key_env
        self.teacher_answer_max_tokens = teacher_answer_max_tokens
        self.teacher_answer_temperature = teacher_answer_temperature
        self.teacher_answer_top_p = teacher_answer_top_p
        self.teacher_answer_timeout = teacher_answer_timeout
        self.teacher_answer_max_retries = teacher_answer_max_retries
        self.teacher_answer_semaphore = asyncio.Semaphore(
            max(1, teacher_answer_concurrency)
        )
        self._teacher_client: AsyncOpenAI | None = None

    def _get_teacher_client(self) -> AsyncOpenAI:
        if self._teacher_client is not None:
            return self._teacher_client
        api_key = os.environ.get(self.teacher_answer_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.teacher_answer_api_key_env} is not set for DeepSeek teacher calls"
            )
        base_url = (
            self.teacher_answer_base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("DEEPSEEK_API_BASE")
            or "https://api.deepseek.com"
        )
        self._teacher_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.teacher_answer_timeout,
            max_retries=max(0, int(self.teacher_answer_max_retries)),
        )
        return self._teacher_client

    async def _teacher_tool_target(
        self,
        turn_record: DefaultAgentTurnRecord,
    ) -> str:
        messages = _external_teacher_messages(turn_record.prefix_messages)
        async with self.teacher_answer_semaphore:
            response = await self._get_teacher_client().chat.completions.create(
                model=self.teacher_answer_model,
                messages=messages,
                tools=[spec.api_definition() for spec in TOOL_SPECS],
                tool_choice="auto",
                max_tokens=self.teacher_answer_max_tokens,
                temperature=self.teacher_answer_temperature,
                top_p=self.teacher_answer_top_p,
            )
        message = response.choices[0].message.model_dump(exclude_none=True)
        return _format_qwen_tool_call_target(_teacher_tool_calls_from_message(message))

    async def _teacher_targets(
        self,
        turn_records: list[DefaultAgentTurnRecord],
    ) -> list[str]:
        results = await asyncio.gather(
            *[self._teacher_tool_target(record) for record in turn_records],
            return_exceptions=True,
        )
        targets: list[str] = []
        for record, result in zip(turn_records, results):
            if isinstance(result, Exception):
                print(
                    "DeepSeek teacher target failed for "
                    f"traj {record.traj_i} turn {record.turn_index}: {result}",
                    flush=True,
                )
                targets.append("")
            else:
                targets.append(str(result))
        return targets

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                tool_call_parser=self.tool_call_parser,
                reasoning_parser=self.reasoning_parser,
                engine_max_tokens=self.max_tokens_per_trajectory,
                chat_template_type=self.chat_template_type,
            )
            for _ in range(self.n_trajs)
        ]
        uids = [uuid.uuid4().hex[:8] for _ in range(self.n_trajs)]
        raw_results = await asyncio.gather(
            *[
                DefaultAgentTerminalTaskRunner(
                    output_path=os.path.join(
                        self.dump_dir, "DefaultAgentTerminalTaskRunner"
                    ),
                    max_turns=self.max_turns,
                    max_tokens_per_turn=self.gconfig.max_new_tokens,
                    temperature=self.gconfig.temperature,
                    top_p=self.gconfig.top_p,
                    observation_max_chars=self.observation_max_chars,
                    task_timeouts=self.task_timeouts,
                    encourage_completion_reward=self.encourage_completion_reward,
                    executor=self.executor,
                    trajectory_timeout=self.trajectory_timeout,
                    task_service_url=self.terminal_task_service_url,
                    task_service_url_file=self.terminal_task_service_url_file,
                    return_turn_records=True,
                ).run_agent(data=data, client=clients[i], uid=uids[i], traj_i=i)
                for i in range(self.n_trajs)
            ]
        )

        results: list[DefaultAgentRunResult | None] = []
        rewards: list[float | None] = []
        for result in raw_results:
            if isinstance(result, DefaultAgentRunResult):
                results.append(result)
                rewards.append(result.reward)
            elif isinstance(result, (float, int)):
                results.append(None)
                rewards.append(float(result))
            else:
                results.append(None)
                rewards.append(None)

        if self.filter_uniform_reward:
            valid_rewards = [reward for reward in rewards if reward is not None]
            if not valid_rewards or all(reward == valid_rewards[0] for reward in valid_rewards):
                return None

        turn_records: list[DefaultAgentTurnRecord] = []
        record_rewards: list[float] = []
        for result in results:
            if result is None:
                continue
            for turn_record in result.turn_records:
                turn_records.append(turn_record)
                record_rewards.append(float(result.reward))

        if not turn_records:
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                num_trajectories_failed=sum(1 for reward in rewards if reward is None)
            )
            return None

        teacher_targets = await self._teacher_targets(turn_records)
        rows: list[dict[str, torch.Tensor]] = []
        for turn_record, verifier_reward, teacher_target in zip(
            turn_records,
            record_rewards,
            teacher_targets,
        ):
            interaction = clients[turn_record.traj_i].get_interaction(
                turn_record.response_id
            )
            if interaction is None:
                continue
            row = _build_teacher_turn_tensor(
                tokenizer=self.tokenizer,
                interaction=interaction,
                turn_record=turn_record,
                verifier_reward=verifier_reward,
                teacher_target=teacher_target,
            )
            if row is not None:
                rows.append(row)

        valid_rewards = [reward for reward in rewards if reward is not None]
        for reward in valid_rewards:
            stats_tracker.get(workflow_context.stat_scope()).scalar(reward=float(reward))
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            num_full_passes=sum(1 for reward in rewards if reward == 1.0)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            num_trajectories_failed=sum(1 for reward in rewards if reward is None)
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            teacher_answer_turns=len(rows),
            teacher_answer_targets=sum(1 for target in teacher_targets if target),
        )
        return concat_padded_tensors(rows) if rows else None


def _config_float(config: Any, key: str, env_key: str, default: float) -> float:
    value = getattr(config, key, None)
    if value is None:
        value = os.environ.get(env_key)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def default_agent_teacher_answer_reward_postprocess(
    trainer,
    rollout_batch: list[dict[str, Any]],
    global_step: int,
) -> None:
    """Combine verifier rewards with per-turn teacher-answer likelihood rewards."""
    del global_step
    teacher_weight = _config_float(
        trainer.config,
        "teacher_answer_reward_weight",
        "TEACHER_ANSWER_REWARD_WEIGHT",
        1.0,
    )
    verifier_weight = _config_float(
        trainer.config,
        "verifier_reward_weight",
        "VERIFIER_REWARD_WEIGHT",
        1.0,
    )

    scoring_outputs = [
        _build_scoring_batch(traj, max_tokens=_scoring_max_tokens(trainer))
        for traj in rollout_batch
    ]
    scoring_batches = [batch for batch, _ in scoring_outputs]
    scoring_stats = [stats for _, stats in scoring_outputs]
    scoring_logps = trainer.actor.compute_logp(scoring_batches)
    if scoring_logps is None:
        raise RuntimeError("actor.compute_logp returned None for teacher-answer scoring")

    all_teacher_logp: list[torch.Tensor] = []
    all_combined: list[torch.Tensor] = []
    all_verifier: list[torch.Tensor] = []
    all_target_available: list[torch.Tensor] = []
    all_target_len: list[torch.Tensor] = []
    all_scoring_len: list[torch.Tensor] = []
    for traj, scoring_batch, scoring_stat, logp in zip(
        rollout_batch,
        scoring_batches,
        scoring_stats,
        scoring_logps,
    ):
        logp = _as_tensor(logp)
        answer_mask = torch.roll(
            scoring_batch["loss_mask"].to(logp.device).float(),
            shifts=-1,
            dims=-1,
        )
        target_len = answer_mask.sum(dim=-1)
        target_available = (target_len > 0).float()
        teacher_logp = (logp * answer_mask).sum(dim=-1) / target_len.clamp(min=1.0)
        teacher_logp = teacher_logp * target_available
        verifier_rewards = _as_tensor(
            traj.get("verifier_rewards", traj["rewards"])
        ).to(logp.device).float()
        combined_rewards = verifier_weight * verifier_rewards + teacher_weight * teacher_logp
        traj["rewards"] = combined_rewards.to(dtype=torch.float32)

        all_teacher_logp.append(teacher_logp.detach().float().cpu())
        all_combined.append(combined_rewards.detach().float().cpu())
        all_verifier.append(verifier_rewards.detach().float().cpu())
        all_target_available.append(target_available.detach().float().cpu())
        all_target_len.append(target_len.detach().float().cpu())
        all_scoring_len.append(scoring_stat["length"].detach().float().cpu())

        for key in (
            "teacher_answer_prefix_ids",
            "teacher_answer_prefix_mask",
            "teacher_answer_ids",
            "teacher_answer_mask",
            "teacher_context_mask",
            "teacher_answer_turn_index",
            "teacher_answer_traj_index",
            "verifier_rewards",
        ):
            traj.pop(key, None)

    teacher_logp_cat = torch.cat(all_teacher_logp)
    combined_cat = torch.cat(all_combined)
    verifier_cat = torch.cat(all_verifier)
    target_available_cat = torch.cat(all_target_available)
    target_len_cat = torch.cat(all_target_len)
    scoring_len_cat = torch.cat(all_scoring_len)
    stats_tracker.denominator(
        default_teacher_answer_n_turns=torch.ones_like(
            combined_cat,
            dtype=torch.bool,
        ),
        default_teacher_answer_n_targets=target_available_cat.bool(),
    )
    stats_tracker.stat(
        default_teacher_answer_logp=teacher_logp_cat,
        default_teacher_answer_reward=combined_cat,
        default_teacher_answer_verifier_reward=verifier_cat,
        default_teacher_answer_target_available=target_available_cat,
        default_teacher_answer_target_len=target_len_cat,
        default_teacher_answer_scoring_len=scoring_len_cat,
        denominator="default_teacher_answer_n_turns",
    )
    stats_tracker.stat(
        default_teacher_answer_target_logp=teacher_logp_cat,
        default_teacher_answer_target_len_only=target_len_cat,
        denominator="default_teacher_answer_n_targets",
    )


__all__ = [
    "DefaultAgentTerminalGRPOWorkflow",
    "DefaultAgentTerminalTeacherAnswerRLWorkflow",
    "DefaultAgentTerminalTaskRunner",
    "default_agent_teacher_answer_reward_postprocess",
    "SYSTEM_PROMPT",
    "TOOL_SPECS",
]
