"""Unified token estimation.

Single source of truth for the ~4 chars/token heuristic used
throughout the codebase.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars/token). Always returns >= 1."""
    if not text:
        return 1
    return max(1, len(text) // 4)
