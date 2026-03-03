"""Shared subprocess runner with timeout, truncation, and process-group cleanup."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_CAPTURE_BYTES_DEFAULT = 1_048_576  # 1 MiB per stream


@dataclass(frozen=True)
class CommandRunResult:
    """Result payload from a subprocess execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    truncated: bool = False


def constrained_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a minimal environment to reduce accidental secret leakage.

    Keep process execution practical (PATH/HOME/TMPDIR), while avoiding
    unbounded inheritance of host environment variables.
    """
    env: dict[str, str] = {}
    passthrough = (
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SHELL",
        "USER",
        "LANG",
        "LC_ALL",
    )
    for key in passthrough:
        value = os.environ.get(key)
        if value:
            env[key] = value

    if extra:
        for key, value in extra.items():
            clean_key = str(key or "").strip()
            if not clean_key:
                continue
            env[clean_key] = str(value or "")

    return env


async def _read_limited(
    stream: asyncio.StreamReader | None,
    max_bytes: int,
) -> tuple[bytes, bool]:
    """Read stream with bounded buffering; drain overflow without storing."""
    if stream is None:
        return b"", False

    max_bytes = max(1, int(max_bytes))
    chunks: list[bytes] = []
    total = 0
    truncated = False

    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = max_bytes - total
        if remaining <= 0:
            truncated = True
            continue
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            total += remaining
            truncated = True
        else:
            chunks.append(chunk)
            total += len(chunk)

    return b"".join(chunks), truncated


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    """Terminate process and children best-effort."""
    if process.returncode is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
            return
        except Exception:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            await process.wait()
        except Exception:
            pass
        return

    # Windows fallback
    try:
        process.kill()
    except Exception:
        pass
    try:
        await process.wait()
    except Exception:
        pass


async def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 30,
    env: dict[str, str] | None = None,
    max_capture_bytes: int = MAX_CAPTURE_BYTES_DEFAULT,
) -> CommandRunResult:
    """Execute argv via subprocess with bounded output and timeout handling."""
    if not argv:
        raise ValueError("argv must not be empty")

    start = time.monotonic()
    process: asyncio.subprocess.Process | None = None
    timed_out = False
    stdout_bytes = b""
    stderr_bytes = b""
    truncated = False

    kwargs: dict[str, Any] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": str(cwd) if cwd is not None else None,
        "env": env,
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True

    try:
        process = await asyncio.create_subprocess_exec(*argv, **kwargs)
        read_task = asyncio.gather(
            _read_limited(process.stdout, max_capture_bytes),
            _read_limited(process.stderr, max_capture_bytes),
        )
        (stdout_pair, stderr_pair) = await asyncio.wait_for(
            read_task,
            timeout=max(1, int(timeout_seconds)),
        )
        stdout_bytes, stdout_trunc = stdout_pair
        stderr_bytes, stderr_trunc = stderr_pair
        truncated = stdout_trunc or stderr_trunc
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except TimeoutError:
        timed_out = True
        if process is not None:
            await _terminate_process_tree(process)
    except asyncio.CancelledError:
        if process is not None:
            await _terminate_process_tree(process)
        raise

    duration_ms = int((time.monotonic() - start) * 1000)
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    if timed_out:
        return CommandRunResult(
            exit_code=-1,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_ms=duration_ms,
            timed_out=True,
            truncated=truncated,
        )

    exit_code = process.returncode if process is not None else -1
    return CommandRunResult(
        exit_code=int(exit_code),
        stdout=stdout_text,
        stderr=stderr_text,
        duration_ms=duration_ms,
        timed_out=False,
        truncated=truncated,
    )
