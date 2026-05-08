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
import textwrap
import unicodedata
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal.utils.perf_tracer import atrace_scope, atrace_session_phase, session_context

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
        self.executor = executor
        self.task_service_url = task_service_url.rstrip("/") if task_service_url else None
        self.task_service_url_file = task_service_url_file
        self.remote_session_id: str | None = None

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
    ) -> float | None:
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

        try:
            async with atrace_scope(
                f"default_agent_reset_env:{task_name},traj:{traj_i}",
                args={"uid": uid, "timeout": self.task_timeouts.reset_env},
            ):
                await asyncio.wait_for(
                    self._remote_reset_env(data, uid),
                    timeout=self.task_timeouts.reset_env + 30,
                )

            reward: float | None = 0.0
            for turn in range(self.max_turns):
                self._assert_single_user_message(messages)
                response = await client.chat.completions.create(
                    messages=messages,
                    tools=tools,
                    max_completion_tokens=self.max_tokens_per_turn,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                assistant_message = self._assistant_message_with_local_tool_parsing(
                    response=response,
                    client=client,
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
                for tool_result in await asyncio.gather(
                    *[self._execute_tool_call(tool_call) for tool_call in tool_calls]
                ):
                    messages.append(tool_result)
                if turn == self.max_turns - 1:
                    break

            self._assert_single_user_message(messages)
            async with atrace_session_phase(
                "reward",
                start_payload={"task_name": task_name, "traj_i": traj_i},
            ):
                reward = await asyncio.wait_for(
                    self._remote_evaluate_completion(),
                    timeout=self.task_timeouts.verifier + 30,
                )
            client.set_last_reward(float(reward))
            print(
                f"Default-agent GRPO task {task_name} traj {traj_i} reward={float(reward):.4f}",
                flush=True,
            )
            return float(reward)
        except TimeoutError:
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


__all__ = [
    "DefaultAgentTerminalGRPOWorkflow",
    "DefaultAgentTerminalTaskRunner",
    "SYSTEM_PROMPT",
    "TOOL_SPECS",
]
