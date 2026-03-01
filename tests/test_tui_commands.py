"""Unit tests for TUI command palette provider behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from loom.tui.commands import _COMMANDS, LoomCommands


@pytest.mark.asyncio
async def test_discover_returns_sorted_full_command_list() -> None:
    app = SimpleNamespace(
        run_action=MagicMock(),
        iter_dynamic_process_palette_entries=lambda: [
            ("Run zeta-process…", "process_run_prompt:zeta-process", "dynamic"),
            ("Run alpha-process…", "process_run_prompt:alpha-process", "dynamic"),
        ],
    )
    screen = SimpleNamespace(app=app, focused=None)
    provider = LoomCommands(screen)

    hits = [hit async for hit in provider.discover()]
    labels = [str(hit.text) for hit in hits]

    assert labels == sorted(labels, key=str.casefold)

    expected = [label for label, _, _ in _COMMANDS] + [
        "Run zeta-process…",
        "Run alpha-process…",
    ]
    assert sorted(labels, key=str.casefold) == sorted(expected, key=str.casefold)


@pytest.mark.asyncio
async def test_discover_without_dynamic_entries_uses_sorted_builtins() -> None:
    app = SimpleNamespace(run_action=MagicMock())
    screen = SimpleNamespace(app=app, focused=None)
    provider = LoomCommands(screen)

    hits = [hit async for hit in provider.discover()]
    labels = [str(hit.text) for hit in hits]
    expected = [label for label, _, _ in _COMMANDS]

    assert labels == sorted(expected, key=str.casefold)
