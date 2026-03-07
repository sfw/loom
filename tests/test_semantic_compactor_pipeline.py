"""Pipeline orchestration parity tests for semantic compactor."""

from __future__ import annotations

import pytest

from loom.engine.semantic_compactor import pipeline as compactor_pipeline


class _DummyModel:
    pass


@pytest.mark.asyncio
async def test_compact_with_model_preserves_round_reduction_flow() -> None:
    model = _DummyModel()
    strict_flags: list[bool] = []
    chunked_calls = 0

    async def _chunked_compaction(**kwargs):
        nonlocal chunked_calls
        del kwargs
        chunked_calls += 1
        return "b" * 500, 2

    async def _compact_once(**kwargs):
        strict_flags.append(bool(kwargs["strict"]))
        if len(strict_flags) == 1:
            return "c" * 300, 1
        return "c" * 300, 1

    compacted, retry_count = await compactor_pipeline.compact_with_model(
        model=model,  # type: ignore[arg-type]
        text="a" * 1000,
        max_chars=200,
        label="rounds",
        max_chunk_chars=800,
        max_reduction_rounds=4,
        compact_once=_compact_once,
        chunked_compaction=_chunked_compaction,
    )

    assert chunked_calls == 1
    assert strict_flags == [False, True]
    assert compacted == "c" * 300
    assert retry_count == 4


@pytest.mark.asyncio
async def test_chunked_compaction_preserves_premerge_and_map_stages() -> None:
    model = _DummyModel()
    calls: list[tuple[str, int, bool, int]] = []

    def _split_text(_: str, __: int) -> list[str]:
        return [
            "A" * 600,
            "B" * 600,
            "C" * 600,
            "D" * 100,
        ]

    async def _compact_once(**kwargs):
        label = str(kwargs["label"])
        max_chars = int(kwargs["max_chars"])
        strict = bool(kwargs["strict"])
        text = str(kwargs["text"])
        calls.append((label, max_chars, strict, len(text)))

        if label.endswith("pre-merge"):
            if len(text) > 500:
                return "P" * 700, 1
            return "Q" * 100, 1
        return "M" * 400, 2

    compacted, retry_count = await compactor_pipeline.chunked_compaction(
        model=model,  # type: ignore[arg-type]
        text="ignored source",
        max_chars=500,
        label="chunk-flow",
        max_chunk_chars=700,
        max_chunks_per_round=2,
        min_compact_target_chars=140,
        split_text=_split_text,
        compact_once=_compact_once,
    )

    assert all(strict is False for _, _, strict, _ in calls)
    assert [label for label, _, _, _ in calls] == [
        "chunk-flow pre-merge",
        "chunk-flow pre-merge",
        "chunk-flow chunk 1/2",
    ]
    assert compacted == ("M" * 400) + "\n" + ("Q" * 100)
    assert retry_count == 4
