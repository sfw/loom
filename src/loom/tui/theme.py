"""Loom Dark theme — Tokyo Night-inspired palette for the TUI."""

from __future__ import annotations

from rich.theme import Theme as RichTheme
from textual.theme import Theme as TextualTheme

LOOM_DARK = TextualTheme(
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

# Rich markdown style overrides used by cowork chat rendering.
LOOM_MARKDOWN_RICH_THEME = RichTheme(
    {
        "markdown.h1": "bold underline #7dcfff",
        "markdown.h2": "underline #7dcfff",
        "markdown.h3": "bold #73daca",
        "markdown.h4": "italic #bb9af7",
        "markdown.h5": "italic #c0caf5",
        "markdown.h6": "dim #9aa5ce",
        "markdown.block_quote": "#9aa5ce",
        "markdown.link": "#7dcfff",
        "markdown.link_url": "underline #7dcfff",
        "markdown.hr": "#565f89",
        "markdown.item.number": "#7dcfff",
        "markdown.code": "bold #7dcfff on #1f2335",
        "markdown.code_block": "#7dcfff on #1f2335",
        "markdown.table.header": "bold #7dcfff",
        "markdown.table.border": "#565f89",
    }
)

# Semantic color constants for Rich markup in widgets.
USER_MSG = "#73daca"
TOOL_OK = "#9ece6a"
TOOL_ERR = "#f7768e"
DIM = "#565f89"
ACCENT_CYAN = "#7dcfff"
ACCENT_LAVENDER = "#bb9af7"
ACCENT_AMBER = "#e0af68"
