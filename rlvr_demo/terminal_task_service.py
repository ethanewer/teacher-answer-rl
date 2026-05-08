"""HTTP service for running Terminal-Bench task environments on CPU nodes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rlvr_demo.terminal_task_runtime import (
    TerminalTaskTimeouts,
    TerminusLocalTerminalTaskRunner,
)


class CreateSessionRequest(BaseModel):
    output_path: str
    task_name: str
    task_path: str
    uid: str
    observation_max_chars: int = 8000
    task_timeouts: dict[str, Any] = Field(default_factory=dict)
    encourage_completion_reward: bool = False


class CommandRequest(BaseModel):
    commands: list[dict[str, Any]]


class SessionState:
    def __init__(self, runner: TerminusLocalTerminalTaskRunner, data: dict[str, Any]):
        self.runner = runner
        self.data = data
        self.lock = asyncio.Lock()


def _coerce_timeouts(raw: dict[str, Any]) -> TerminalTaskTimeouts:
    allowed = TerminalTaskTimeouts.__dataclass_fields__
    return TerminalTaskTimeouts(
        **{key: value for key, value in raw.items() if key in allowed}
    )


def _http_error(exc: BaseException) -> HTTPException:
    if isinstance(exc, subprocess.CalledProcessError):
        details = [f"CalledProcessError: command={exc.cmd!r} exit={exc.returncode}"]
        if exc.stdout:
            details.append(f"stdout={str(exc.stdout)[:1200]}")
        if exc.stderr:
            details.append(f"stderr={str(exc.stderr)[:1200]}")
        return HTTPException(status_code=500, detail="; ".join(details))
    return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


def _service_url(host: str, port: int) -> str:
    advertised_host = os.environ.get("TERMINAL_TASK_SERVICE_ADVERTISE_HOST")
    if advertised_host is None:
        advertised_host = socket.getfqdn() or socket.gethostname()
    if host not in {"0.0.0.0", "::"} and advertised_host in {"", "localhost"}:
        advertised_host = host
    return f"http://{advertised_host}:{port}"


def create_app(
    *,
    max_workers: int,
    max_sessions: int,
    max_starts: int | None,
    host: str,
    port: int,
    ready_file: str | None = None,
    output_root: str | None = None,
) -> FastAPI:
    executor = ThreadPoolExecutor(max_workers=max_workers)
    session_slots = asyncio.Semaphore(max_sessions)
    start_slots = asyncio.Semaphore(max_starts or max_sessions)
    sessions: dict[str, SessionState] = {}

    async def run_blocking(
        fn: Callable[..., Any],
        *args: Any,
        timeout: float | None,
        **kwargs: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(
            executor,
            lambda: fn(*args, **kwargs),
        )
        if timeout is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout)

    async def close_session(session_id: str) -> None:
        state = sessions.pop(session_id, None)
        if state is None:
            return
        try:
            async with state.lock:
                await run_blocking(
                    state.runner._close_env,
                    timeout=state.runner.task_timeouts.cleanup,
                )
        finally:
            session_slots.release()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        if ready_file is not None:
            path = Path(ready_file).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "url": _service_url(host, port),
                        "host": socket.gethostname(),
                        "port": port,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        try:
            yield
        finally:
            close_tasks: list[Awaitable[None]] = [
                close_session(session_id) for session_id in list(sessions)
            ]
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Terminus Terminal Task Service", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "sessions": len(sessions),
            "max_sessions": max_sessions,
            "max_starts": max_starts or max_sessions,
        }

    @app.post("/v1/sessions")
    async def create_session(request: CreateSessionRequest) -> dict[str, str]:
        await session_slots.acquire()
        timeouts = _coerce_timeouts(request.task_timeouts)
        output_path = request.output_path
        if output_root is not None:
            output_path = str(Path(output_root).expanduser())
        runner = TerminusLocalTerminalTaskRunner(
            output_path=output_path,
            max_turns=0,
            max_tokens_per_turn=0,
            temperature=0.0,
            top_p=1.0,
            observation_max_chars=request.observation_max_chars,
            task_timeouts=timeouts,
            encourage_completion_reward=request.encourage_completion_reward,
            executor=executor,
        )
        data = {
            "task_name": request.task_name,
            "task_path": request.task_path,
            "instruction": "",
        }
        session_id = uuid.uuid4().hex
        state = SessionState(runner=runner, data=data)
        sessions[session_id] = state
        try:
            async with start_slots, state.lock:
                await run_blocking(
                    runner._reset_env,
                    data,
                    request.uid,
                    timeout=timeouts.reset_env,
                )
        except BaseException as exc:
            sessions.pop(session_id, None)
            try:
                await run_blocking(runner._close_env, timeout=timeouts.cleanup)
            except BaseException:
                pass
            session_slots.release()
            raise _http_error(exc) from exc
        return {"session_id": session_id}

    @app.post("/v1/sessions/{session_id}/commands")
    async def execute_commands(
        session_id: str,
        request: CommandRequest,
    ) -> dict[str, str]:
        state = sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown session")
        try:
            async with state.lock:
                observation = await run_blocking(
                    state.runner._execute_commands,
                    request.commands,
                    timeout=state.runner.task_timeouts.command
                    * max(len(request.commands), 1)
                    + 10,
                )
        except BaseException as exc:
            raise _http_error(exc) from exc
        return {"observation": observation}

    @app.post("/v1/sessions/{session_id}/reward")
    async def evaluate_reward(session_id: str) -> dict[str, float]:
        state = sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="unknown session")
        try:
            async with state.lock:
                reward = await run_blocking(
                    state.runner._evaluate_completion_sync,
                    timeout=state.runner.task_timeouts.verifier,
                )
        except BaseException as exc:
            raise _http_error(exc) from exc
        return {"reward": float(reward)}

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, bool]:
        await close_session(session_id)
        return {"ok": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=39080)
    parser.add_argument("--ready-file")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--max-sessions", type=int, default=5)
    parser.add_argument(
        "--max-starts",
        type=int,
        default=None,
        help=(
            "Maximum concurrent docker-compose starts. Defaults to max-sessions; "
            "use a smaller value to avoid overloading Docker while keeping many "
            "live task sessions."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("TERMINAL_TASK_SERVICE_OUTPUT_ROOT"),
        help="Optional local scratch root for Terminal-Bench trial outputs.",
    )
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args()

    if args.ready_file:
        Path(args.ready_file).expanduser().unlink(missing_ok=True)

    import uvicorn

    uvicorn.run(
        create_app(
            max_workers=args.max_workers,
            max_sessions=args.max_sessions,
            max_starts=args.max_starts,
            host=args.host,
            port=args.port,
            ready_file=args.ready_file,
            output_root=args.output_root,
        ),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
