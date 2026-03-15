"""Constants and pure helpers for the Loom TUI app package."""

from __future__ import annotations

from rich.text import Text

from .models import SlashCommandSpec


def _plain_text(value: object | None) -> str:
    """Coerce rich/plain values to a user-facing plain string."""
    if value is None:
        return ""
    if isinstance(value, Text):
        return value.plain
    return str(value)


def _escape_markup_text(value: object | None) -> str:
    """Escape Rich markup control chars in dynamic text."""
    return _plain_text(value).replace("[", "\\[")


_SLASH_COMMANDS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec(
        canonical="/quit",
        aliases=("/exit", "/q"),
        description="exit Loom",
    ),
    SlashCommandSpec(canonical="/clear", description="clear chat log"),
    SlashCommandSpec(canonical="/help", description="show command help"),
    SlashCommandSpec(canonical="/setup", description="run setup wizard"),
    SlashCommandSpec(canonical="/model", description="show active model details"),
    SlashCommandSpec(canonical="/models", description="show configured models details"),
    SlashCommandSpec(
        canonical="/mcp",
        usage=(
            "[manage|list|show <alias>|test <alias>|add <alias> ...|"
            "edit <alias> ...|enable <alias>|disable <alias>|remove <alias>]"
        ),
        description="inspect/manage MCP server config",
    ),
    SlashCommandSpec(
        canonical="/auth",
        description="open auth manager (use CLI for scriptable auth operations)",
    ),
    SlashCommandSpec(
        canonical="/tool",
        usage="<tool-name> [key=value ... | json-object-args]",
        description="run a tool directly with key=value or JSON arguments",
    ),
    SlashCommandSpec(canonical="/tools", description="list available tools"),
    SlashCommandSpec(canonical="/tokens", description="show session token usage"),
    SlashCommandSpec(
        canonical="/config",
        usage=(
            "[list|search <query>|show <path>|set <path> <value> "
            "[--scope runtime|persist|both]|reset <path> "
            "[--scope runtime|persist|both]]"
        ),
        description="inspect/set runtime and persisted config values",
    ),
    SlashCommandSpec(
        canonical="/telemetry",
        usage="[status|off|active|all_typed|debug|internal_only]",
        description="show/set runtime telemetry mode (process-local)",
    ),
    SlashCommandSpec(canonical="/session", description="show current session info"),
    SlashCommandSpec(canonical="/new", description="start a new session"),
    SlashCommandSpec(canonical="/sessions", description="list recent sessions"),
    SlashCommandSpec(
        canonical="/resume",
        usage="<session-id-prefix>",
        description="resume old cowork session by session ID prefix",
    ),
    SlashCommandSpec(
        canonical="/history",
        usage="[older]",
        description="load older chat history for current session",
    ),
    SlashCommandSpec(
        canonical="/learned",
        description="review/delete learned patterns",
    ),
    SlashCommandSpec(
        canonical="/processes",
        description="list available process definitions",
    ),
    SlashCommandSpec(
        canonical="/run",
        usage=(
            "<goal|close [run-id-prefix]|"
            "resume <run-id-prefix|current>|"
            "pause [run-id-prefix|current>|"
            "play [run-id-prefix|current>|"
            "stop [run-id-prefix|current>|"
            "inject [run-id-prefix|current] <text>|"
            "save <run-id-prefix|current> <name>>"
        ),
        description="run goal via process orchestrator (ad hoc by default)",
    ),
    SlashCommandSpec(
        canonical="/pause",
        description="pause active cowork chat execution",
    ),
    SlashCommandSpec(
        canonical="/inject",
        usage="<text>",
        description="queue cowork steering instruction for next safe boundary",
    ),
    SlashCommandSpec(
        canonical="/redirect",
        usage="<text>",
        description="immediately redirect cowork objective",
    ),
    SlashCommandSpec(
        canonical="/steer",
        usage="pause|resume|queue|clear",
        description="cowork steering controls",
    ),
    SlashCommandSpec(
        canonical="/stop",
        description="stop active cowork chat execution",
    ),
)

_SLASH_COMMAND_PRIORITY: dict[str, int] = {
    "/new": 10,
    "/sessions": 11,
    "/resume": 12,
    "/history": 13,
    "/session": 14,
    "/run": 20,
    "/pause": 21,
    "/inject": 22,
    "/redirect": 23,
    "/steer": 24,
    "/stop": 22,
    "/processes": 25,
    "/mcp": 30,
    "/auth": 31,
    "/tools": 32,
    "/tool": 33,
    "/model": 40,
    "/models": 41,
    "/config": 42,
    "/tokens": 42,
    "/telemetry": 43,
    "/setup": 43,
    "/learned": 44,
    "/help": 45,
    "/clear": 46,
    "/quit": 47,
}

_MUTATING_TOOL_FALLBACK = frozenset({"document_write", "humanize_writing"})
_WORKSPACE_SCAN_EXCLUDE_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
    }
)

_PROCESS_STATUS_ICON = {
    "queued": "\u25cb",
    "running": "\u25c9",
    "paused": "\u23f8",
    "cancel_requested": "\u25c9",
    "cancel_failed": "!",
    "force_closed": "\u25a1",
    "completed": "\u2713",
    "failed": "\u2717",
    "cancelled": "\u25a0",
}
_PROCESS_STATUS_LABEL = {
    "queued": "Queued",
    "running": "Running",
    "paused": "Paused",
    "cancel_requested": "Cancel Requested",
    "cancel_failed": "Cancel Failed",
    "force_closed": "Force Closed",
    "completed": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
}

_MAX_CONCURRENT_PROCESS_RUNS = 4
_MAX_PERSISTED_PROCESS_RUNS = 12
_MAX_PERSISTED_PROCESS_ACTIVITY = 300
_MAX_PERSISTED_PROCESS_RESULTS = 120
_INFO_WRAP_WIDTH = 108
_RUN_GOAL_FILE_CONTENT_MAX_CHARS = 32_000
_MAX_INPUT_HISTORY = 500
_DEFAULT_CHAT_RESUME_PAGE_SIZE = 250
_DEFAULT_CHAT_RESUME_MAX_RENDERED_ROWS = 1200
_DEFAULT_TUI_REALTIME_REFRESH_ENABLED = True
_DEFAULT_TUI_WORKSPACE_WATCH_BACKEND = "poll"
_DEFAULT_TUI_WORKSPACE_POLL_INTERVAL_MS = 1000
_DEFAULT_TUI_WORKSPACE_REFRESH_DEBOUNCE_MS = 250
_DEFAULT_TUI_WORKSPACE_REFRESH_MAX_WAIT_MS = 1500
_DEFAULT_TUI_WORKSPACE_SCAN_MAX_ENTRIES = 20_000
_DEFAULT_TUI_CHAT_STREAM_FLUSH_INTERVAL_MS = 120
_DEFAULT_TUI_FILES_PANEL_MAX_ROWS = 2000
_DEFAULT_TUI_DELEGATE_PROGRESS_MAX_LINES = 150
_DEFAULT_TUI_RUN_LAUNCH_HEARTBEAT_INTERVAL_MS = 6000
_DEFAULT_TUI_RUN_LAUNCH_TIMEOUT_SECONDS = 300
_DEFAULT_TUI_RUN_CLOSE_MODAL_TIMEOUT_SECONDS = 45
_DEFAULT_TUI_RUN_CANCEL_WAIT_TIMEOUT_SECONDS = 10
_DEFAULT_TUI_RUN_PROGRESS_REFRESH_INTERVAL_MS = 200
_DEFAULT_TUI_RUN_PREFLIGHT_ASYNC_ENABLED = True
_DEFAULT_TUI_CHAT_STOP_COOPERATIVE_WAIT_SECONDS = 0.35
_DEFAULT_TUI_CHAT_STOP_SETTLE_TIMEOUT_SECONDS = 2.0
_PROCESS_COMMAND_INDEX_REFRESH_INTERVAL_SECONDS = 2.0
_EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS = 0.5
_EVENT_LOOP_LAG_WARN_THRESHOLD_SECONDS = 0.05

_PROCESS_RUN_LAUNCH_STAGES: tuple[tuple[str, str], ...] = (
    ("accepted", "Accepted"),
    ("resolving_process", "Resolving process"),
    ("provisioning_workspace", "Provisioning workspace"),
    ("auth_preflight", "Auth preflight"),
    ("queueing_delegate", "Queueing delegate"),
    ("running", "Running"),
)
_PROCESS_RUN_LAUNCH_STAGE_INDEX = {
    stage: idx for idx, (stage, _label) in enumerate(_PROCESS_RUN_LAUNCH_STAGES)
}
_PROCESS_RUN_LAUNCH_STAGE_LABEL = {
    stage: label for stage, label in _PROCESS_RUN_LAUNCH_STAGES
}
_PROCESS_RUN_HEARTBEAT_STAGES: frozenset[str] = frozenset(
    {
        "resolving_process",
        "provisioning_workspace",
        "auth_preflight",
        "queueing_delegate",
        "running",
    }
)

__all__ = [
    name
    for name in globals()
    if name.startswith("_") and not name.startswith("__")
]
