"""Registry of operator-visible runtime config entries."""

from __future__ import annotations

from typing import Any

from loom.events.verbosity import DEFAULT_TELEMETRY_MODE, normalize_telemetry_mode

from .schema import (
    APPLICATION_LIVE,
    APPLICATION_NEXT_CALL,
    APPLICATION_NEXT_RUN,
    ConfigRuntimeEntry,
    ParsedConfigValue,
)


def _parse_int(
    raw_value: object,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> ParsedConfigValue:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as e:
        raise ValueError("Value must be an integer.") from e
    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    if value == 0 and default == 0 and minimum in {0, None}:
        pass
    return ParsedConfigValue(value=value, display_value=str(value))


def _parse_enum(raw_value: object, *, choices: tuple[str, ...]) -> ParsedConfigValue:
    text = str(raw_value or "").strip().lower()
    if text not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"Value must be one of: {allowed}.")
    return ParsedConfigValue(value=text, display_value=text)


def _parse_telemetry_mode(raw_value: object) -> ParsedConfigValue:
    resolution = normalize_telemetry_mode(raw_value, default=DEFAULT_TELEMETRY_MODE)
    return ParsedConfigValue(
        value=resolution.mode,
        display_value=resolution.mode,
        warning_code=str(resolution.warning_code or "").strip(),
    )


CONFIG_RUNTIME_ENTRIES: tuple[ConfigRuntimeEntry, ...] = (
    ConfigRuntimeEntry(
        path="execution.delegate_task_timeout_seconds",
        section="execution",
        field="delegate_task_timeout_seconds",
        kind="int",
        description="Timeout for delegated orchestration calls (/run, delegate_task).",
        default=3600,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_LIVE,
        minimum=1,
        parser=lambda raw: _parse_int(raw, default=3600, minimum=1),
        aliases=("delegate_timeout", "run_timeout"),
        search_terms=("delegate", "run", "timeout", "orchestration"),
    ),
    ConfigRuntimeEntry(
        path="execution.ask_user_timeout_seconds",
        section="execution",
        field="ask_user_timeout_seconds",
        kind="int",
        description="Timeout for ask_user responses when timeout policy is enabled.",
        default=0,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_RUN,
        minimum=0,
        parser=lambda raw: _parse_int(raw, default=0, minimum=0),
        aliases=("ask_timeout",),
        search_terms=("ask_user", "questions", "timeout"),
    ),
    ConfigRuntimeEntry(
        path="execution.ask_user_policy",
        section="execution",
        field="ask_user_policy",
        kind="enum",
        description="Runtime policy for unanswered ask_user prompts.",
        default="block",
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_RUN,
        enum_values=("block", "timeout_default", "fail_closed"),
        parser=lambda raw: _parse_enum(
            raw,
            choices=("block", "timeout_default", "fail_closed"),
        ),
        aliases=("ask_policy",),
        search_terms=("ask_user", "questions", "policy"),
    ),
    ConfigRuntimeEntry(
        path="execution.agent_tools_max_timeout_seconds",
        section="execution",
        field="agent_tools_max_timeout_seconds",
        kind="int",
        description="Max allowed timeout for coding-agent tool calls.",
        default=1800,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_CALL,
        minimum=30,
        parser=lambda raw: _parse_int(raw, default=1800, minimum=30),
        aliases=("agent_timeout",),
        search_terms=("agent", "codex", "claude", "opencode", "timeout"),
    ),
    ConfigRuntimeEntry(
        path="execution.cowork_tool_exposure_mode",
        section="execution",
        field="cowork_tool_exposure_mode",
        kind="enum",
        description="Cowork tool exposure mode.",
        default="adaptive",
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_CALL,
        enum_values=("full", "adaptive", "hybrid"),
        parser=lambda raw: _parse_enum(raw, choices=("full", "adaptive", "hybrid")),
        aliases=("tool_exposure",),
        search_terms=("cowork", "tool", "exposure"),
    ),
    ConfigRuntimeEntry(
        path="telemetry.mode",
        section="telemetry",
        field="mode",
        kind="enum",
        description="Runtime telemetry verbosity mode.",
        default=DEFAULT_TELEMETRY_MODE,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_LIVE,
        enum_values=("off", "active", "all_typed", "debug", "internal_only"),
        parser=_parse_telemetry_mode,
        aliases=("telemetry",),
        search_terms=("telemetry", "events", "verbosity"),
    ),
    ConfigRuntimeEntry(
        path="tui.run_launch_timeout_seconds",
        section="tui",
        field="run_launch_timeout_seconds",
        kind="int",
        description="Timeout for /run preflight launch stages before delegate execution starts.",
        default=300,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_RUN,
        minimum=5,
        maximum=600,
        parser=lambda raw: _parse_int(raw, default=300, minimum=5, maximum=600),
        aliases=("launch_timeout",),
        search_terms=("tui", "run", "launch", "timeout"),
    ),
    ConfigRuntimeEntry(
        path="tui.run_close_modal_timeout_seconds",
        section="tui",
        field="run_close_modal_timeout_seconds",
        kind="int",
        description=(
            "Max time a process-run close confirmation modal waits for "
            "input before auto-dismiss."
        ),
        default=45,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_CALL,
        minimum=5,
        maximum=300,
        parser=lambda raw: _parse_int(raw, default=45, minimum=5, maximum=300),
        aliases=("close_modal_timeout",),
        search_terms=("tui", "run", "close", "modal", "timeout"),
    ),
    ConfigRuntimeEntry(
        path="tui.run_cancel_wait_timeout_seconds",
        section="tui",
        field="run_cancel_wait_timeout_seconds",
        kind="int",
        description="Max wait after cancel request before offering force-close on active run tabs.",
        default=10,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_CALL,
        minimum=1,
        maximum=120,
        parser=lambda raw: _parse_int(raw, default=10, minimum=1, maximum=120),
        aliases=("cancel_wait_timeout",),
        search_terms=("tui", "run", "cancel", "timeout"),
    ),
    ConfigRuntimeEntry(
        path="tui.chat_stream_flush_interval_ms",
        section="tui",
        field="chat_stream_flush_interval_ms",
        kind="int",
        description="Sparse flush cadence for buffered streaming chat chunks.",
        default=120,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_CALL,
        minimum=10,
        maximum=2000,
        parser=lambda raw: _parse_int(raw, default=120, minimum=10, maximum=2000),
        aliases=("stream_flush_ms",),
        search_terms=("tui", "chat", "stream", "flush"),
    ),
    ConfigRuntimeEntry(
        path="tui.delegate_progress_max_lines",
        section="tui",
        field="delegate_progress_max_lines",
        kind="int",
        description="Maximum retained lines per collapsed delegate-progress section.",
        default=150,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_CALL,
        minimum=10,
        maximum=5000,
        parser=lambda raw: _parse_int(raw, default=150, minimum=10, maximum=5000),
        aliases=("delegate_progress_lines",),
        search_terms=("tui", "delegate", "progress", "lines"),
    ),
    ConfigRuntimeEntry(
        path="tui.run_progress_refresh_interval_ms",
        section="tui",
        field="run_progress_refresh_interval_ms",
        kind="int",
        description=(
            "Minimum cadence for /run progress/UI refreshes under "
            "high-frequency event streams."
        ),
        default=200,
        supports_runtime=True,
        supports_persist=True,
        application_class=APPLICATION_NEXT_CALL,
        minimum=10,
        maximum=5000,
        parser=lambda raw: _parse_int(raw, default=200, minimum=10, maximum=5000),
        aliases=("run_refresh_ms",),
        search_terms=("tui", "run", "progress", "refresh"),
    ),
)

_PATH_MAP = {entry.path: entry for entry in CONFIG_RUNTIME_ENTRIES}
_ALIAS_MAP = {
    alias: entry
    for entry in CONFIG_RUNTIME_ENTRIES
    for alias in entry.aliases
}


def list_entries() -> list[ConfigRuntimeEntry]:
    """Return operator-visible config entries in deterministic order."""
    return list(CONFIG_RUNTIME_ENTRIES)


def find_entry(path_or_alias: str) -> ConfigRuntimeEntry | None:
    """Resolve a config entry by canonical path or alias."""
    clean = str(path_or_alias or "").strip().lower()
    if not clean:
        return None
    return _PATH_MAP.get(clean) or _ALIAS_MAP.get(clean)


def search_entries(query: str) -> list[ConfigRuntimeEntry]:
    """Return entries matching a free-form query."""
    clean = str(query or "").strip().lower()
    if not clean:
        return list_entries()
    matches: list[tuple[int, ConfigRuntimeEntry]] = []
    for entry in CONFIG_RUNTIME_ENTRIES:
        haystacks = (
            entry.path,
            entry.description,
            *entry.aliases,
            *entry.search_terms,
        )
        score = 0
        for haystack in haystacks:
            text = str(haystack or "").lower()
            if clean == text:
                score = max(score, 100)
            elif text.startswith(clean):
                score = max(score, 75)
            elif clean in text:
                score = max(score, 50)
        if score:
            matches.append((score, entry))
    matches.sort(key=lambda item: (-item[0], item[1].path))
    return [entry for _score, entry in matches]


def entries_by_section() -> dict[str, list[ConfigRuntimeEntry]]:
    """Return entries grouped by top-level section."""
    grouped: dict[str, list[ConfigRuntimeEntry]] = {}
    for entry in CONFIG_RUNTIME_ENTRIES:
        grouped.setdefault(entry.section, []).append(entry)
    return grouped


def allowed_values_text(entry: ConfigRuntimeEntry) -> str:
    """Render a human-readable allowed-values summary for one entry."""
    if entry.kind == "enum":
        return ", ".join(entry.enum_values)
    if entry.kind == "bool":
        return "true, false"
    if entry.kind == "int":
        if entry.minimum is not None and entry.maximum is not None:
            return f"{entry.minimum}..{entry.maximum}"
        if entry.minimum is not None:
            return f">= {entry.minimum}"
    return ""


def display_value(value: Any, *, redact: bool = False) -> str:
    """Render one config value for user-facing output."""
    if redact and value not in {None, ""}:
        return "(redacted)"
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return text if text else "(empty)"
