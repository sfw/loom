"""Cache/inflight dedupe parity tests for semantic compactor."""

from __future__ import annotations

import asyncio
import json

import pytest

from loom.engine.semantic_compactor import SemanticCompactor
from loom.engine.semantic_compactor import cache as compactor_cache
from loom.models.base import ModelConnectionError, ModelResponse


class _NeverCompletesModel:
    name = "never-completes-model"
    tier = 1
    roles = ["compactor"]
    configured_temperature = 1.0

    def __init__(self) -> None:
        self.calls = 0
        self._gate = asyncio.Event()

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        self.calls += 1
        await self._gate.wait()
        return ModelResponse(text=json.dumps({"compressed_text": "unused"}))

    async def health_check(self) -> bool:
        return True


class _RaiseAfterGateModel:
    name = "raise-after-gate-model"
    tier = 1
    roles = ["compactor"]
    configured_temperature = 1.0

    def __init__(self) -> None:
        self.calls = 0
        self._gate = asyncio.Event()

    def release(self) -> None:
        self._gate.set()

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        self.calls += 1
        await self._gate.wait()
        raise ModelConnectionError(
            (
                "Model server returned HTTP 400: "
                '{"error":{"message":"invalid temperature: only 1 is allowed '
                'for this model"}}'
            ),
        )

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_cache_helper_owner_join_and_release_flow() -> None:
    key = ("abc", 200)
    cache: dict[tuple[str, int], str] = {}
    inflight: dict[tuple[str, int], asyncio.Future[str]] = {}
    lock = asyncio.Lock()

    cached_1, waiting_1, owner_1 = await compactor_cache.lookup_cache_or_inflight(
        cache=cache,
        inflight=inflight,
        lock=lock,
        key=key,
    )
    assert cached_1 is None
    assert waiting_1 is not None
    assert owner_1 is True

    cached_2, waiting_2, owner_2 = await compactor_cache.lookup_cache_or_inflight(
        cache=cache,
        inflight=inflight,
        lock=lock,
        key=key,
    )
    assert cached_2 is None
    assert waiting_2 is waiting_1
    assert owner_2 is False

    await compactor_cache.store_cache_and_release_inflight(
        cache=cache,
        inflight=inflight,
        lock=lock,
        key=key,
        value="compacted value",
    )

    assert await waiting_2 == "compacted value"
    assert key in cache
    assert key not in inflight


@pytest.mark.asyncio
async def test_semantic_compactor_inflight_cleanup_on_owner_cancellation() -> None:
    model = _NeverCompletesModel()
    compactor = SemanticCompactor(model=model)  # type: ignore[arg-type]
    source = "x" * 3600

    owner_task = asyncio.create_task(
        compactor.compact(source, max_chars=300, label="owner-cancel"),
    )
    await asyncio.sleep(0)
    join_task = asyncio.create_task(
        compactor.compact(source, max_chars=300, label="owner-cancel-join"),
    )
    await asyncio.sleep(0)

    owner_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner_task

    joined = await asyncio.wait_for(join_task, timeout=1.0)
    assert joined == source

    # Inflight entry must be cleaned up so subsequent calls cannot hang.
    again = await asyncio.wait_for(
        compactor.compact(source, max_chars=300, label="after-cancel"),
        timeout=1.0,
    )
    assert again == source
    assert model.calls == 1


@pytest.mark.asyncio
async def test_semantic_compactor_inflight_cleanup_on_owner_model_error() -> None:
    model = _RaiseAfterGateModel()
    compactor = SemanticCompactor(model=model)  # type: ignore[arg-type]
    source = "x " * 1800
    expected = SemanticCompactor._normalize_whitespace(source)

    owner_task = asyncio.create_task(
        compactor.compact(source, max_chars=220, label="owner-error"),
    )
    await asyncio.sleep(0)
    join_task = asyncio.create_task(
        compactor.compact(source, max_chars=220, label="owner-error-join"),
    )
    await asyncio.sleep(0)

    model.release()
    owner, joined = await asyncio.gather(owner_task, join_task)
    assert owner == expected
    assert joined == expected
    assert model.calls == 1
