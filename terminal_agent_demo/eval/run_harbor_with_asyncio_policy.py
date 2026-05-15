"""Run Harbor with a subprocess-capable asyncio policy on Linux.

Some installed packages set uvloop as the process-wide policy. Harbor's Docker
environment starts docker compose with ``asyncio.create_subprocess_exec``, which
requires a Unix child watcher and fails under uvloop's policy.
"""

from __future__ import annotations

import asyncio
import sys


def _install_subprocess_policy() -> None:
    if sys.platform == "win32":
        return
    policy = asyncio.DefaultEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)
    try:
        policy.set_child_watcher(asyncio.SafeChildWatcher())
    except (AttributeError, NotImplementedError):
        pass


def _run_async(coro):
    _install_subprocess_policy()
    return asyncio.run(coro)


def main() -> None:
    _install_subprocess_policy()

    import harbor.cli.jobs as harbor_jobs
    import harbor.cli.utils as harbor_utils
    from harbor.cli.main import app

    harbor_utils.run_async = _run_async
    harbor_jobs.run_async = _run_async
    sys.exit(app())


if __name__ == "__main__":
    main()
