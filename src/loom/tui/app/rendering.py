"""Rendering helpers for process and chat rows."""

from __future__ import annotations

from collections.abc import Callable


def one_line(
    text: object | None,
    *,
    max_len: int | None = 180,
    plain_text: Callable[[object | None], str],
) -> str:
    """Normalize whitespace and cap a string for concise progress rows."""
    if text is None:
        return ""
    compact = " ".join(plain_text(text).split())
    if max_len is None or max_len <= 0:
        return compact
    if len(compact) <= max_len:
        return compact
    return f"{compact[:max_len - 1].rstrip()}…"


def aggregate_phase_state(statuses: list[str]) -> str:
    """Aggregate multiple subtask states into one phase-level state."""
    normalized = [
        str(item).strip()
        for item in statuses
        if str(item).strip() in {"pending", "in_progress", "completed", "failed", "skipped"}
    ]
    if not normalized:
        return "pending"
    if "failed" in normalized:
        return "failed"
    if "in_progress" in normalized:
        return "in_progress"
    if "pending" in normalized:
        if any(item in {"completed", "skipped"} for item in normalized):
            return "in_progress"
        return "pending"
    if all(item == "skipped" for item in normalized):
        return "skipped"
    return "completed"
