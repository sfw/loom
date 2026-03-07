"""Cache and inflight dedupe helpers for semantic compactor."""

from __future__ import annotations

import asyncio
from collections.abc import Hashable


async def lookup_cache_or_inflight(
    *,
    cache: dict[Hashable, str],
    inflight: dict[Hashable, asyncio.Future[str]],
    lock: asyncio.Lock,
    key: Hashable,
) -> tuple[str | None, asyncio.Future[str] | None, bool]:
    async with lock:
        cached = cache.get(key)
        if cached is not None:
            return cached, None, False

        waiting = inflight.get(key)
        if waiting is None:
            waiting = asyncio.get_running_loop().create_future()
            inflight[key] = waiting
            return None, waiting, True
        return None, waiting, False


async def store_cache_and_release_inflight(
    *,
    cache: dict[Hashable, str],
    inflight: dict[Hashable, asyncio.Future[str]],
    lock: asyncio.Lock,
    key: Hashable,
    value: str,
) -> None:
    async with lock:
        cache[key] = value
        waiting = inflight.pop(key, None)
        if waiting is not None and not waiting.done():
            waiting.set_result(value)
