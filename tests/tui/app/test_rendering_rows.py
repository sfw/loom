from __future__ import annotations

from loom.tui.app import LoomApp, rendering
from loom.tui.app.constants import _plain_text


def test_one_line_compaction_and_truncation() -> None:
    assert rendering.one_line("a  b\n c", max_len=None, plain_text=_plain_text) == "a b c"
    assert rendering.one_line("abcdefgh", max_len=5, plain_text=_plain_text) == "abcd…"
    assert LoomApp._one_line("x   y") == "x y"


def test_aggregate_phase_state() -> None:
    assert rendering.aggregate_phase_state(["pending"]) == "pending"
    assert rendering.aggregate_phase_state(["completed", "pending"]) == "in_progress"
    assert rendering.aggregate_phase_state(["failed", "completed"]) == "failed"
    assert LoomApp._aggregate_phase_state(["completed", "skipped"]) == "completed"
