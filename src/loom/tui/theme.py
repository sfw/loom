"""Loom Dark theme — Tokyo Night-inspired palette for the TUI."""

from __future__ import annotations

from textual.theme import Theme

LOOM_DARK = Theme(
    name="loom-dark",
    primary="#7dcfff",
    secondary="#bb9af7",
    accent="#ff9e64",
    warning="#e0af68",
    error="#f7768e",
    success="#9ece6a",
    foreground="#c0caf5",
    background="#1a1b26",
    surface="#1e2030",
    panel="#24283b",
    dark=True,
)

# Semantic color constants for Rich markup in widgets.
USER_MSG = "#73daca"
TOOL_OK = "#9ece6a"
TOOL_ERR = "#f7768e"
DIM = "#565f89"
ACCENT_CYAN = "#7dcfff"
ACCENT_LAVENDER = "#bb9af7"
ACCENT_AMBER = "#e0af68"
