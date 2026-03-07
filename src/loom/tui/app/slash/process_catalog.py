"""Process-backed slash command catalog helpers."""

from __future__ import annotations

import asyncio
import logging
import time

from loom.tui.widgets import ChatLog
from loom.utils.latency import log_latency_event

from ..constants import (
    _PROCESS_COMMAND_INDEX_REFRESH_INTERVAL_SECONDS,
    _SLASH_COMMANDS,
)

logger = logging.getLogger("loom.tui.app.core")


def reserved_slash_command_names() -> set[str]:
    """Return reserved slash command names (without leading slash)."""
    reserved: set[str] = set()
    for spec in _SLASH_COMMANDS:
        reserved.add(spec.canonical.lstrip("/").lower())
        for alias in spec.aliases:
            reserved.add(alias.lstrip("/").lower())
    return reserved


def is_reserved_process_name(self, name: str) -> bool:
    """Return True when process name collides with built-in slash command."""
    return name.strip().lower() in self._reserved_slash_command_names()


def compute_process_command_index(
    self,
) -> tuple[list[dict[str, str]], dict[str, str], list[str]]:
    """Compute selectable process catalog and dynamic command map."""
    started = time.monotonic()
    loader = self._create_process_loader()
    available = loader.list_available()

    selectable: list[dict[str, str]] = []
    command_map: dict[str, str] = {}
    blocked: set[str] = set()
    reserved = self._reserved_slash_command_names()

    for proc in available:
        name = str(proc.get("name", "")).strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered in reserved:
            blocked.add(name)
            continue
        selectable.append(proc)
        command_map[f"/{lowered}"] = name

    log_latency_event(
        logger,
        event="process_command_index_refresh",
        duration_seconds=time.monotonic() - started,
        fields={"available": len(available), "selectable": len(selectable)},
    )
    return selectable, command_map, sorted(blocked, key=str.lower)


def refresh_process_command_index(
    self,
    *,
    chat: ChatLog | None = None,
    notify_conflicts: bool = False,
    background: bool = False,
    force: bool = False,
) -> None:
    """Refresh process catalog and dynamic command map."""
    if background and not self.is_running:
        background = False

    if background:
        now = time.monotonic()
        if self._process_command_index_refresh_inflight:
            return
        if (
            not force
            and self._cached_process_catalog
            and (now - self._process_command_index_last_refresh_at)
            < _PROCESS_COMMAND_INDEX_REFRESH_INTERVAL_SECONDS
        ):
            return
        self._process_command_index_refresh_inflight = True

        async def _refresh_in_background() -> None:
            try:
                selectable, command_map, blocked = await asyncio.to_thread(
                    self._compute_process_command_index,
                )
            except Exception:
                logger.exception("Failed background process command index refresh")
                return
            try:
                self._cached_process_catalog = selectable
                self._process_command_map = command_map
                self._blocked_process_commands = blocked
                self._process_command_index_last_refresh_at = time.monotonic()
                if notify_conflicts and chat and self._blocked_process_commands:
                    blocked_cmds = ", ".join(
                        f"/{name}" for name in self._blocked_process_commands
                    )
                    chat.add_info(
                        "[bold #f7768e]Process command name collision:[/] "
                        f"{blocked_cmds}\n"
                        "[dim]These process names collide with built-in slash commands "
                        "and were skipped in TUI.[/dim]",
                    )
            finally:
                self._process_command_index_refresh_inflight = False

        self.run_worker(
            _refresh_in_background(),
            group="process-command-index-refresh",
            exclusive=False,
        )
        return

    try:
        selectable, command_map, blocked = self._compute_process_command_index()
    except Exception:
        logger.exception("Failed process command index refresh")
        return

    self._cached_process_catalog = selectable
    self._process_command_map = command_map
    self._blocked_process_commands = blocked
    self._process_command_index_last_refresh_at = time.monotonic()

    if notify_conflicts and chat and self._blocked_process_commands:
        blocked_cmds = ", ".join(f"/{name}" for name in self._blocked_process_commands)
        chat.add_info(
            "[bold #f7768e]Process command name collision:[/] "
            f"{blocked_cmds}\n"
            "[dim]These process names collide with built-in slash commands "
            "and were skipped in TUI.[/dim]",
        )
