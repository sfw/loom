"""Shared bounded threadpool helpers for blocking local IO work."""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

_DEFAULT_MAX_WORKERS = max(4, min(32, (os.cpu_count() or 4) * 2))
_BLOCKING_IO_EXECUTOR = ThreadPoolExecutor(
    max_workers=_DEFAULT_MAX_WORKERS,
    thread_name_prefix="loom-io",
)


async def run_blocking_io(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking callable on Loom's bounded IO executor."""
    loop = asyncio.get_running_loop()
    call = partial(fn, *args, **kwargs)
    return await loop.run_in_executor(_BLOCKING_IO_EXECUTOR, call)
