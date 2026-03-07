"""Delegate progress streaming helpers for cowork chat."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from loom.tui.widgets import ChatLog, EventPanel


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def ensure_delegate_progress_widget(
    self,
    *,
    tool_call_id: str,
    title: str,
    status: str = "running",
    elapsed_ms: int = 0,
    lines: list[str] | None = None,
) -> bool:
    key = str(tool_call_id or "").strip()
    if not key:
        return False
    chat = self.query_one("#chat-log", ChatLog)
    chat.add_delegate_progress_section(
        key,
        title=title,
        max_lines=self._tui_delegate_progress_max_lines(),
        status=status,
        elapsed_ms=elapsed_ms,
        lines=lines,
    )
    normalized_lines = [
        " ".join(str(item or "").split())
        for item in (lines or [])
        if " ".join(str(item or "").split())
    ]
    max_lines_cap = self._tui_delegate_progress_max_lines()
    if len(normalized_lines) > max_lines_cap:
        normalized_lines = normalized_lines[-max_lines_cap:]
    stream = self._active_delegate_streams.get(key, {})
    if not isinstance(stream, dict):
        stream = {}
    stream.update({
        "tool_call_id": key,
        "title": str(title or "Delegated progress").strip() or "Delegated progress",
        "status": str(status or "running").strip().lower() or "running",
        "elapsed_ms": max(0, int(elapsed_ms)),
        "lines": normalized_lines,
        "started_at": float(stream.get("started_at", time.monotonic()) or time.monotonic()),
        "finalized": str(status or "").strip().lower() in {"completed", "failed"},
    })
    self._active_delegate_streams[key] = stream
    self._sync_activity_indicator()
    return True


def append_delegate_progress_widget_line(self, tool_call_id: str, line: str) -> bool:
    key = str(tool_call_id or "").strip()
    if not key:
        return False
    stream = self._active_delegate_streams.get(key)
    if not isinstance(stream, dict):
        return False
    if bool(stream.get("finalized", False)):
        return False
    chat = self.query_one("#chat-log", ChatLog)
    accepted = chat.append_delegate_progress_line(key, line)
    if not accepted:
        return False
    compact = " ".join(str(line or "").split())
    if not compact:
        return False
    lines = stream.get("lines")
    if not isinstance(lines, list):
        lines = []
    lines.append(compact)
    max_lines_cap = self._tui_delegate_progress_max_lines()
    if len(lines) > max_lines_cap:
        del lines[:-max_lines_cap]
    stream["lines"] = lines
    self._active_delegate_streams[key] = stream
    return True


async def start_delegate_progress_stream(
    self,
    *,
    tool_call_id: str,
    caller_tool_name: str,
    persist: bool = True,
) -> None:
    key = str(tool_call_id or "").strip()
    if not key:
        return
    existing = self._active_delegate_streams.get(key)
    if isinstance(existing, dict):
        # Late callback events can arrive after tool completion; never
        # reopen/reset a finalized section.
        if bool(existing.get("finalized", False)):
            return
        return
    title = self._delegate_progress_title(caller_tool_name)
    if not self._ensure_delegate_progress_widget(
        tool_call_id=key,
        title=title,
        status="running",
        elapsed_ms=0,
        lines=[],
    ):
        return
    if not persist:
        return
    await self._append_chat_replay_event(
        "delegate_progress_started",
        {
            "tool_call_id": key,
            "title": title,
            "status": "running",
        },
    )


async def finalize_delegate_progress_stream(
    self,
    *,
    tool_call_id: str,
    success: bool,
    elapsed_ms: int = 0,
    persist: bool = True,
) -> None:
    key = str(tool_call_id or "").strip()
    if not key:
        return
    stream = self._active_delegate_streams.get(key)
    if not isinstance(stream, dict):
        return
    status = "completed" if success else "failed"
    normalized_elapsed = max(0, int(elapsed_ms))
    if normalized_elapsed <= 0:
        started_at = float(stream.get("started_at", 0.0) or 0.0)
        if started_at > 0:
            normalized_elapsed = max(
                1,
                int((time.monotonic() - started_at) * 1000),
            )
    title = str(stream.get("title", "Delegated progress") or "Delegated progress")
    lines = stream.get("lines")
    if not isinstance(lines, list):
        lines = []
    chat = self.query_one("#chat-log", ChatLog)
    chat.finalize_delegate_progress_section(
        key,
        success=bool(success),
        elapsed_ms=normalized_elapsed,
    )
    stream.update({
        "status": status,
        "elapsed_ms": normalized_elapsed,
        "finalized": True,
        "lines": lines,
        "title": title,
    })
    self._active_delegate_streams[key] = stream
    self._sync_activity_indicator()
    if not persist:
        return
    await self._append_chat_replay_event(
        "delegate_progress_finalized",
        {
            "tool_call_id": key,
            "title": title,
            "status": status,
            "elapsed_ms": normalized_elapsed,
            "lines": list(lines),
        },
    )


async def on_cowork_delegate_progress_event(self, payload: dict[str, Any]) -> None:
    """Handle delegate_task incremental progress for cowork chat streams."""
    if not isinstance(payload, dict):
        return
    tool_call_id = str(payload.get("tool_call_id", "") or "").strip()
    if not tool_call_id:
        return
    existing = self._active_delegate_streams.get(tool_call_id)
    if isinstance(existing, dict) and bool(existing.get("finalized", False)):
        return

    caller_tool_name = str(
        payload.get("caller_tool_name", payload.get("tool_name", "delegate_task"))
        or "delegate_task",
    ).strip() or "delegate_task"
    await self._start_delegate_progress_stream(
        tool_call_id=tool_call_id,
        caller_tool_name=caller_tool_name,
        persist=True,
    )

    self._update_sidebar_tasks(payload)

    event_type = str(payload.get("event_type", "") or "").strip()
    event_data = payload.get("event_data", {})
    if not isinstance(event_data, dict):
        event_data = {}
    if event_type == "tool_call_started":
        nested_tool = str(event_data.get("tool", "") or "").strip()
        if nested_tool:
            self._mark_cowork_tool_inflight(nested_tool)
    if event_type == "tool_call_completed":
        nested_tool = str(event_data.get("tool", "") or "").strip()
        if nested_tool:
            self._clear_cowork_tool_inflight(nested_tool)
        if nested_tool and self._is_mutating_tool(nested_tool):
            self._request_workspace_refresh(f"delegate:{nested_tool}")
        self._ingest_files_panel_from_paths(
            event_data.get("files_changed_paths", event_data.get("files_changed", [])),
            operation_hint="modify",
        )
    if event_type in {
        "subtask_completed",
        "subtask_failed",
        "task_completed",
        "task_failed",
    }:
        self._request_workspace_refresh(f"delegate:{event_type}")
    if event_type == "token_streamed":
        # Token bursts are noisy; keep cowork progress sections high-signal.
        return

    message = self._format_process_progress_event(
        payload,
        context="cowork_delegate",
    )
    if message and self._append_delegate_progress_widget_line(tool_call_id, message):
        if event_type != "token_streamed":
            try:
                events_panel = self.query_one("#events-panel", EventPanel)
                events_panel.add_event(
                    _now_str(),
                    "delegate",
                    message[:140],
                )
            except Exception:
                pass
