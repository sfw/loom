"""Chat history replay helpers."""

from __future__ import annotations

import logging
import time
from typing import Any

from loom.cowork.approval import ToolApprover
from loom.tui.widgets import ChatLog
from loom.tui.widgets.tool_call import tool_args_preview, tool_output_preview

logger = logging.getLogger(__name__)
_CHAT_RENDER_SHIFT_STEP_MAX = 50
_TRANSCRIPT_NOISY_TOOL_NAMES = frozenset(
    {
        "glob_find",
        "read_file",
        "ripgrep_search",
        "search_files",
        "web_fetch",
        "web_fetch_html",
        "web_search",
    }
)


def _cowork_session_cls():
    # Resolve from concrete sibling module so monkeypatches on loom.tui.app facade
    # continue propagating to internal session construction sites.
    from .. import core as app_core

    return app_core.CoworkSession


def _chat_render_shift_step(max_rows: int) -> int:
    """Return the row overrun tolerated before the visible window advances."""
    if max_rows <= 1:
        return 1
    return max(1, min(_CHAT_RENDER_SHIFT_STEP_MAX, max_rows // 8))


def _is_transcript_noisy_tool(event: dict[str, Any]) -> bool:
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        return False
    tool_name = str(payload.get("tool_name", "") or "").strip()
    return tool_name in _TRANSCRIPT_NOISY_TOOL_NAMES


def _summarize_transcript_noisy_tools(events: list[dict[str, Any]]) -> dict[str, Any]:
    lines: list[str] = [f"Collapsed {len(events)} lookup tool calls."]
    shown = 0
    for event in events:
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        tool_name = str(payload.get("tool_name", "") or "").strip() or "tool"
        args = payload.get("args", {})
        if not isinstance(args, dict):
            args = {}
        preview = tool_args_preview(tool_name, args)
        detail = f"{tool_name} {preview}".strip()
        lines.append(f"- {detail}")
        shown += 1
        if shown >= 5:
            break
    remaining = len(events) - shown
    if remaining > 0:
        lines.append(f"... {remaining} more")
    return {
        "event_type": "info",
        "payload": {
            "text": "\n".join(lines),
            "markup": False,
        },
    }


def _merge_thinking_events(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    chunks: list[str] = []
    for event in events:
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        text = str(payload.get("text", "") or "").strip()
        if text:
            chunks.append(text)
    if not chunks:
        return None
    return {
        "event_type": "assistant_thinking",
        "payload": {
            "text": "\n\n".join(chunks),
            "streaming": False,
        },
    }


def build_chat_render_events(
    events: list[dict[str, Any]],
    *,
    transcript_mode: bool,
    show_thinking: bool,
) -> list[dict[str, Any]]:
    """Collapse transcript-only noise and optionally include thinking rows."""
    if not transcript_mode:
        return events

    rendered: list[dict[str, Any]] = []
    index = 0
    while index < len(events):
        event = events[index]
        event_type = str(event.get("event_type", "") or "").strip()

        if event_type == "assistant_thinking":
            grouped: list[dict[str, Any]] = []
            while index < len(events):
                candidate = events[index]
                if str(candidate.get("event_type", "") or "").strip() != "assistant_thinking":
                    break
                grouped.append(candidate)
                index += 1
            if show_thinking:
                merged = _merge_thinking_events(grouped)
                if merged is not None:
                    rendered.append(merged)
            continue

        if event_type == "tool_call_started" and _is_transcript_noisy_tool(event):
            index += 1
            continue

        if event_type == "tool_call_completed" and _is_transcript_noisy_tool(event):
            grouped = [event]
            index += 1
            while index < len(events):
                candidate = events[index]
                candidate_type = str(candidate.get("event_type", "") or "").strip()
                if candidate_type == "tool_call_started" and _is_transcript_noisy_tool(candidate):
                    index += 1
                    continue
                if candidate_type == "content_indicator":
                    index += 1
                    continue
                if candidate_type == "tool_call_completed" and _is_transcript_noisy_tool(candidate):
                    grouped.append(candidate)
                    index += 1
                    continue
                break
            if len(grouped) >= 2:
                rendered.append(_summarize_transcript_noisy_tools(grouped))
            else:
                rendered.extend(grouped)
            continue

        rendered.append(event)
        index += 1

    return rendered


def chat_transcript_notice(*, transcript_mode: bool, show_thinking: bool) -> str:
    """Build a transcript-mode status line."""
    if not transcript_mode:
        return ""
    thinking_state = "shown" if show_thinking else "hidden"
    return (
        "[dim]Transcript mode active: search/jump enabled; "
        f"lookup bursts collapsed; thinking {thinking_state}.[/dim]"
    )


def _event_search_text(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type", "") or "").strip()
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    if event_type in {
        "user_message",
        "assistant_text",
        "assistant_thinking",
        "turn_interrupted",
        "info",
    }:
        return str(payload.get("text", payload.get("message", "")) or "")
    if event_type in {"tool_call_started", "tool_call_completed"}:
        tool_name = str(payload.get("tool_name", "") or "").strip()
        args = payload.get("args", {})
        if not isinstance(args, dict):
            args = {}
        parts = [tool_name, tool_args_preview(tool_name, args)]
        if event_type == "tool_call_completed":
            parts.append(tool_output_preview(tool_name, str(payload.get("output", "") or "")))
            parts.append(str(payload.get("error", "") or ""))
        return " ".join(part for part in parts if str(part).strip())
    if event_type.startswith("steer_"):
        return str(payload.get("text", payload.get("message", "")) or "")
    return ""


def search_chat_replay_events(
    replay_events: list[dict[str, Any]],
    *,
    query: str,
    include_thinking: bool = True,
) -> list[int]:
    """Return replay-event indices whose plain text matches the query."""
    needle = str(query or "").strip().lower()
    if not needle:
        return []
    matches: list[int] = []
    for index, event in enumerate(replay_events):
        event_type = str(event.get("event_type", "") or "").strip()
        if event_type == "assistant_thinking" and not include_thinking:
            continue
        haystack = _event_search_text(event).lower()
        if haystack and needle in haystack:
            matches.append(index)
    return matches


def chat_search_notice(*, query: str, match_count: int, current_match: int) -> str:
    """Build the transcript search status line."""
    clean_query = str(query or "").strip()
    if not clean_query:
        return ""
    safe_query = clean_query.replace("[", "\\[")
    if match_count <= 0:
        return f"[dim]Search: '{safe_query}' — no matches.[/dim]"
    return f"[dim]Search: '{safe_query}' — match {current_match}/{match_count}.[/dim]"


def focus_chat_event_index(
    *,
    total_rows: int,
    max_rows: int,
    index: int,
) -> int:
    """Return a render-window start that keeps the target row in view."""
    total = max(0, int(total_rows))
    limit = max(1, int(max_rows))
    if total <= limit:
        return 0
    target = min(max(0, int(index)), total - 1)
    preferred = max(0, target - max(1, limit // 3))
    return min(preferred, max(0, total - limit))


def update_chat_render_window(
    *,
    total_rows: int,
    max_rows: int,
    current_start: int,
    follow_latest: bool,
    mode: str,
) -> tuple[bool, int, bool, int, int]:
    """Update the visible transcript window without discarding loaded rows."""
    total = max(0, int(total_rows))
    limit = max(1, int(max_rows))
    start = max(0, int(current_start))
    visible_limit = max(0, total - limit)

    if total <= limit:
        hidden_newer = 0
        hidden_older = 0
        changed = start != 0 or not follow_latest
        return changed, 0, True, hidden_older, hidden_newer

    shift_step = _chat_render_shift_step(limit)
    next_follow_latest = bool(follow_latest)
    rerender = False

    if mode == "hydrate":
        start = visible_limit
        next_follow_latest = True
        rerender = True
    elif mode == "prepend_older":
        start = min(start, visible_limit)
        next_follow_latest = False
        rerender = True
    elif mode == "focus":
        start = min(start, visible_limit)
        next_follow_latest = False
        rerender = True
    else:
        if next_follow_latest and total - start > limit + shift_step:
            start = visible_limit
            rerender = True
        else:
            start = min(start, visible_limit)

    if next_follow_latest and total - start <= limit + shift_step:
        end = total
    else:
        end = min(total, start + limit)
    hidden_older = start
    hidden_newer = max(0, total - end)
    changed = rerender or start != current_start or next_follow_latest != follow_latest
    return changed, start, next_follow_latest, hidden_older, hidden_newer


def visible_chat_replay_events(
    *,
    replay_events: list[dict[str, Any]],
    start: int,
    max_rows: int,
    follow_latest: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return the visible transcript slice plus hidden row counts."""
    total = len(replay_events)
    limit = max(1, int(max_rows))
    if total <= limit:
        return replay_events, 0, 0

    clamped_start = min(max(0, int(start)), max(0, total - limit))
    shift_step = _chat_render_shift_step(limit)
    if follow_latest and total - clamped_start <= limit + shift_step:
        end = total
    else:
        end = min(total, clamped_start + limit)
    return (
        replay_events[clamped_start:end],
        clamped_start,
        max(0, total - end),
    )


def chat_window_notice(*, hidden_older: int, hidden_newer: int) -> str:
    """Build the transcript window status line."""
    parts: list[str] = []
    if hidden_older > 0:
        parts.append(f"{hidden_older} older row(s) hidden")
    if hidden_newer > 0:
        parts.append(f"{hidden_newer} newer row(s) hidden")
    if not parts:
        return ""
    return "[dim]Transcript window truncated: " + "; ".join(parts) + ".[/dim]"

def render_chat_event(self, event: dict, *, source: str = "unknown") -> bool:
    """Render one normalized replay event into the chat widget."""
    event_type = str(event.get("event_type", "") or "").strip()
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    chat = self.query_one("#chat-log", ChatLog)

    try:
        if event_type == "user_message":
            chat.add_user_message(str(payload.get("text", "") or ""))
            return True
        if event_type == "assistant_text":
            chat.add_model_text(
                str(payload.get("text", "") or ""),
                markup=self._coerce_bool(payload.get("markup", False)),
            )
            return True
        if event_type == "assistant_thinking":
            if getattr(self, "_chat_transcript_mode", False) and getattr(
                self,
                "_chat_transcript_show_thinking",
                False,
            ):
                chat.add_thinking_text(str(payload.get("text", "") or ""))
            return True
        if event_type in {"tool_call_started", "tool_call_completed"}:
            args = payload.get("args", {})
            if not isinstance(args, dict):
                args = {}
            chat.add_tool_call(
                str(payload.get("tool_name", "") or ""),
                args,
                tool_call_id=str(payload.get("tool_call_id", "") or ""),
                success=(
                    None
                    if event_type == "tool_call_started"
                    else self._coerce_bool(payload.get("success", False))
                ),
                elapsed_ms=self._coerce_int(payload.get("elapsed_ms", 0), default=0),
                output=str(payload.get("output", "") or ""),
                error=str(payload.get("error", "") or ""),
            )
            return True
        if event_type == "delegate_progress_started":
            tool_call_id = str(payload.get("tool_call_id", "") or "").strip()
            title = str(payload.get("title", "Delegated progress") or "Delegated progress")
            if not tool_call_id:
                return False
            self._ensure_delegate_progress_widget(
                tool_call_id=tool_call_id,
                title=title,
                status="running",
                elapsed_ms=0,
                lines=[],
            )
            return True
        if event_type == "delegate_progress_line":
            tool_call_id = str(payload.get("tool_call_id", "") or "").strip()
            line = str(payload.get("line", "") or "")
            if not tool_call_id or not line.strip():
                return False
            title = str(payload.get("title", "Delegated progress") or "Delegated progress")
            if tool_call_id not in self._active_delegate_streams:
                self._ensure_delegate_progress_widget(
                    tool_call_id=tool_call_id,
                    title=title,
                    status="running",
                    elapsed_ms=0,
                    lines=[],
                )
            self._append_delegate_progress_widget_line(tool_call_id, line)
            return True
        if event_type == "delegate_progress_finalized":
            tool_call_id = str(payload.get("tool_call_id", "") or "").strip()
            title = str(payload.get("title", "Delegated progress") or "Delegated progress")
            status = str(payload.get("status", "completed") or "completed").strip().lower()
            if status not in {"completed", "failed"}:
                status = "completed"
            elapsed_ms = self._coerce_int(payload.get("elapsed_ms", 0), default=0)
            raw_lines = payload.get("lines", [])
            lines = (
                [str(item or "") for item in raw_lines]
                if isinstance(raw_lines, list)
                else []
            )
            if not tool_call_id:
                return False
            self._ensure_delegate_progress_widget(
                tool_call_id=tool_call_id,
                title=title,
                status=status,
                elapsed_ms=elapsed_ms,
                lines=lines,
            )
            chat.finalize_delegate_progress_section(
                tool_call_id,
                success=(status == "completed"),
                elapsed_ms=elapsed_ms,
            )
            return True
        if event_type == "content_indicator":
            from loom.content import deserialize_block

            raw_blocks = payload.get("content_blocks", [])
            if not isinstance(raw_blocks, list):
                return False
            blocks = []
            for block in raw_blocks:
                if not isinstance(block, dict):
                    continue
                try:
                    blocks.append(deserialize_block(block))
                except Exception:
                    continue
            if blocks:
                chat.add_content_indicator(blocks)
            return True
        if event_type == "turn_separator":
            chat.add_turn_separator(
                self._coerce_int(payload.get("tool_count", 0), default=0),
                self._coerce_int(payload.get("tokens", 0), default=0),
                str(payload.get("model", "") or ""),
                tokens_per_second=self._coerce_float(
                    payload.get("tokens_per_second", 0.0),
                    default=0.0,
                ),
                latency_ms=self._coerce_int(payload.get("latency_ms", 0), default=0),
                total_time_ms=self._coerce_int(payload.get("total_time_ms", 0), default=0),
                context_tokens=self._coerce_int(payload.get("context_tokens", 0), default=0),
                context_messages=self._coerce_int(
                    payload.get("context_messages", 0),
                    default=0,
                ),
                omitted_messages=self._coerce_int(
                    payload.get("omitted_messages", 0),
                    default=0,
                ),
                recall_index_used=self._coerce_bool(
                    payload.get("recall_index_used", False),
                ),
            )
            return True
        if event_type == "turn_interrupted":
            chat.add_info(
                str(payload.get("message", "") or ""),
                markup=self._coerce_bool(payload.get("markup", True), default=True),
            )
            return True
        if event_type.startswith("steer_"):
            text = str(
                payload.get(
                    "text",
                    payload.get("message", event_type.replace("_", " ")),
                )
                or "",
            )
            if text:
                chat.add_info(
                    text,
                    markup=self._coerce_bool(payload.get("markup", True), default=True),
                )
            return True
        if event_type == "info":
            chat.add_info(
                str(payload.get("text", "") or ""),
                markup=self._coerce_bool(payload.get("markup", True), default=True),
            )
            return True
    except Exception as e:
        logger.warning(
            "chat_hydrate_row_skipped session=%s source=%s event_type=%s error_class=%s",
            self._active_session_id(),
            source,
            event_type or "<unknown>",
            e.__class__.__name__,
        )
        return False
    return True

def rerender_chat_from_replay_events(self) -> tuple[int, int]:
    """Repaint chat panel from in-memory replay buffer."""
    self._clear_chat_widgets()
    chat = self.query_one("#chat-log", ChatLog)
    visible_events, hidden_older, hidden_newer = visible_chat_replay_events(
        replay_events=self._chat_replay_events,
        start=getattr(self, "_chat_render_window_start", 0),
        max_rows=self._chat_resume_max_rendered_rows(),
        follow_latest=getattr(self, "_chat_follow_latest", True),
    )
    render_events = build_chat_render_events(
        visible_events,
        transcript_mode=getattr(self, "_chat_transcript_mode", False),
        show_thinking=getattr(self, "_chat_transcript_show_thinking", False),
    )
    transcript_notice = chat_transcript_notice(
        transcript_mode=getattr(self, "_chat_transcript_mode", False),
        show_thinking=getattr(self, "_chat_transcript_show_thinking", False),
    )
    notice = chat_window_notice(
        hidden_older=hidden_older,
        hidden_newer=hidden_newer,
    )
    search_notice = chat_search_notice(
        query=getattr(self, "_chat_search_query", ""),
        match_count=len(getattr(self, "_chat_search_match_positions", [])),
        current_match=max(0, int(getattr(self, "_chat_search_match_current", 0))) + 1,
    )
    self._chat_hidden_older_count = hidden_older
    self._chat_hidden_newer_count = hidden_newer
    if transcript_notice:
        chat.add_info(transcript_notice)
    if notice:
        chat.add_info(notice)
    if search_notice:
        chat.add_info(search_notice)
    rendered = 0
    skipped = 0
    for event in render_events:
        if self._render_chat_event(event, source="hydrate"):
            rendered += 1
        else:
            skipped += 1
    return rendered, skipped

async def append_chat_replay_event(
    self,
    event_type: str,
    payload: dict,
    *,
    turn_number: int | None = None,
    journal_through_turn: int | None = None,
    persist: bool = True,
    render: bool = False,
) -> None:
    """Append one UI chat event to replay state and optional journal."""
    event: dict[str, Any] = {
        "event_type": str(event_type or "").strip(),
        "payload": dict(payload or {}),
    }
    if turn_number is not None and turn_number > 0:
        event["turn_number"] = int(turn_number)

    session_id = self._active_session_id()
    if (
        persist
        and self._store is not None
        and session_id
        and self._chat_resume_use_event_journal()
    ):
        try:
            seq = await self._store.append_chat_event(
                session_id,
                event["event_type"],
                event["payload"],
                journal_through_turn=journal_through_turn,
            )
            event["seq"] = seq
            if not self._chat_history_source:
                self._chat_history_source = "journal"
            if self._chat_history_source == "journal" and self._chat_history_oldest_seq is None:
                self._chat_history_oldest_seq = seq
        except Exception as e:
            logger.warning(
                "chat_event_append_failed session=%s event_type=%s error_class=%s",
                session_id,
                event["event_type"],
                e.__class__.__name__,
            )

    self._chat_replay_events.append(event)
    rerender = self._apply_chat_render_cap(mode="append")
    if rerender:
        self._rerender_chat_from_replay_events()
    elif render:
        self._render_chat_event(event, source="live")

async def hydrate_chat_history_for_active_session(self) -> None:
    """Hydrate visible chat transcript for the active session."""
    self._reset_chat_history_state()
    self._clear_chat_widgets()

    if self._store is None:
        return
    session_id = self._active_session_id()
    if not session_id:
        return

    page_size = self._chat_resume_page_size()
    events: list[dict] = []
    parse_failures = 0
    source = ""
    start = time.monotonic()
    source_pref = "transcript"
    logger.info(
        "chat_hydrate_started session=%s source=%s page_size=%s",
        session_id,
        source_pref,
        page_size,
    )

    try:
        events = await _load_transcript_page(
            self,
            session_id,
            limit=page_size,
        )
    except Exception as e:
        logger.warning(
            "chat_hydrate_failed session=%s source=transcript error_class=%s",
            session_id,
            e.__class__.__name__,
        )
        events = []
    if events:
        source = "transcript"
        first = events[0]
        try:
            self._chat_history_oldest_seq = int(first.get("seq", 0) or 0)
        except (TypeError, ValueError):
            self._chat_history_oldest_seq = None
        turns = [
            turn
            for turn in (
                self._chat_event_cursor_turn(event)
                for event in events
            )
            if turn is not None
        ]
        self._chat_history_oldest_turn = min(turns) if turns else None
        parse_failures = sum(
            1 for row in events if bool(row.get("payload_parse_error", False))
        )

    self._chat_replay_events = events
    self._chat_history_source = source
    self._apply_chat_render_cap(mode="hydrate")
    rendered, skipped = self._rerender_chat_from_replay_events()
    elapsed_ms = max(1, int((time.monotonic() - start) * 1000))
    logger.info(
        "chat_hydrate_completed session=%s source=%s rows_rendered=%s "
        "rows_skipped=%s parse_failures=%s elapsed_ms=%s",
        session_id,
        source or "none",
        rendered,
        skipped,
        parse_failures,
        elapsed_ms,
    )

async def load_older_chat_history(self) -> bool:
    """Load one older page of chat history for the active session."""
    if self._store is None:
        return False
    session_id = self._active_session_id()
    if not session_id:
        return False
    page_size = self._chat_resume_page_size()
    start = time.monotonic()

    older: list[dict] = []
    parse_failures = 0
    source = self._chat_history_source
    logger.info(
        "chat_hydrate_started session=%s source=%s page_size=%s mode=older",
        session_id,
        source or "none",
        page_size,
    )
    if source == "transcript":
        before_seq = self._chat_history_oldest_seq
        if before_seq is None or before_seq <= 1:
            return False
        try:
            older = await _load_transcript_page(
                self,
                session_id,
                before_seq=before_seq,
                limit=page_size,
            )
        except Exception as e:
            logger.warning(
                "chat_hydrate_failed session=%s source=transcript "
                "mode=older error_class=%s",
                session_id,
                e.__class__.__name__,
            )
            return False
        if older:
            first = older[0]
            try:
                self._chat_history_oldest_seq = int(first.get("seq", 0) or 0)
            except (TypeError, ValueError):
                pass
            parse_failures = sum(
                1 for row in older if bool(row.get("payload_parse_error", False))
            )
    else:
        return False

    if not older:
        return False
    self._chat_replay_events = [*older, *self._chat_replay_events]
    self._apply_chat_render_cap(mode="prepend_older")
    rendered, skipped = self._rerender_chat_from_replay_events()
    elapsed_ms = max(1, int((time.monotonic() - start) * 1000))
    logger.info(
        "chat_hydrate_completed session=%s source=%s rows_rendered=%s "
        "rows_skipped=%s parse_failures=%s elapsed_ms=%s mode=older",
        session_id,
        source,
        rendered,
        skipped,
        parse_failures,
        elapsed_ms,
    )
    return True


def set_chat_transcript_mode(self, enabled: bool) -> bool:
    """Toggle transcript-mode rendering and rerender when it changes."""
    next_value = bool(enabled)
    current = bool(getattr(self, "_chat_transcript_mode", False))
    if current == next_value:
        return False
    self._chat_transcript_mode = next_value
    if not next_value:
        self._chat_search_query = ""
        self._chat_search_match_positions = []
        self._chat_search_match_current = -1
        self._chat_follow_latest = True
        self._chat_render_window_start = max(
            0,
            len(self._chat_replay_events) - self._chat_resume_max_rendered_rows(),
        )
    self._rerender_chat_from_replay_events()
    return True


def set_chat_transcript_show_thinking(self, enabled: bool) -> bool:
    """Toggle transcript thinking visibility and rerender when it changes."""
    next_value = bool(enabled)
    current = bool(getattr(self, "_chat_transcript_show_thinking", False))
    if current == next_value:
        return False
    self._chat_transcript_show_thinking = next_value
    if getattr(self, "_chat_transcript_mode", False):
        self._rerender_chat_from_replay_events()
    return True


def clear_chat_search(self) -> bool:
    """Clear transcript search state and rerender if necessary."""
    had_query = bool(getattr(self, "_chat_search_query", ""))
    self._chat_search_query = ""
    self._chat_search_match_positions = []
    self._chat_search_match_current = -1
    if had_query:
        self._rerender_chat_from_replay_events()
    return had_query


def search_chat_history(self, query: str) -> int:
    """Search replay history, focus the first match, and rerender."""
    clean_query = str(query or "").strip()
    self._chat_search_query = clean_query
    self._chat_search_match_positions = search_chat_replay_events(
        self._chat_replay_events,
        query=clean_query,
        include_thinking=bool(getattr(self, "_chat_transcript_show_thinking", False)),
    )
    if not self._chat_search_match_positions:
        self._chat_search_match_current = -1
        self._chat_transcript_mode = True
        self._rerender_chat_from_replay_events()
        return 0
    self._chat_search_match_current = 0
    self._chat_transcript_mode = True
    target_index = self._chat_search_match_positions[0]
    self._chat_render_window_start = focus_chat_event_index(
        total_rows=len(self._chat_replay_events),
        max_rows=self._chat_resume_max_rendered_rows(),
        index=target_index,
    )
    self._chat_follow_latest = False
    self._apply_chat_render_cap(mode="focus")
    self._rerender_chat_from_replay_events()
    return len(self._chat_search_match_positions)


def step_chat_history_search(self, direction: int) -> tuple[bool, int, int]:
    """Move to the next/previous active search match."""
    matches = list(getattr(self, "_chat_search_match_positions", []))
    if not matches:
        return False, 0, 0
    current = int(getattr(self, "_chat_search_match_current", -1))
    if current < 0:
        current = 0
    else:
        current = (current + int(direction)) % len(matches)
    self._chat_search_match_current = current
    self._chat_transcript_mode = True
    self._chat_render_window_start = focus_chat_event_index(
        total_rows=len(self._chat_replay_events),
        max_rows=self._chat_resume_max_rendered_rows(),
        index=matches[current],
    )
    self._chat_follow_latest = False
    self._apply_chat_render_cap(mode="focus")
    self._rerender_chat_from_replay_events()
    return True, current + 1, len(matches)


def jump_chat_history_latest(self) -> bool:
    """Jump transcript view back to the live tail."""
    total_rows = len(self._chat_replay_events)
    self._chat_follow_latest = True
    self._chat_render_window_start = max(
        0,
        total_rows - self._chat_resume_max_rendered_rows(),
    )
    self._apply_chat_render_cap(mode="focus")
    self._rerender_chat_from_replay_events()
    return True


async def _load_transcript_page(
    self,
    session_id: str,
    *,
    before_seq: int | None = None,
    before_turn: int | None = None,
    after_seq: int = 0,
    limit: int = 200,
) -> list[dict[str, Any]]:
    getter = getattr(self._store, "get_transcript_page", None)
    if callable(getter):
        try:
            return await getter(
                session_id,
                before_seq=before_seq,
                before_turn=before_turn,
                after_seq=after_seq,
                limit=limit,
            )
        except TypeError:
            logger.debug(
                "chat_transcript_loader_fallback session=%s reason=type_error",
                session_id,
                exc_info=True,
            )

    if before_turn is not None and before_seq is None:
        synth = getattr(self._store, "synthesize_chat_events_from_turns", None)
        if callable(synth):
            return await synth(
                session_id,
                before_turn=before_turn,
                limit=limit,
            )
        return []

    get_chat_events = getattr(self._store, "get_chat_events", None)
    if not callable(get_chat_events):
        return []
    return await get_chat_events(
        session_id,
        before_seq=before_seq,
        after_seq=(None if after_seq <= 0 else after_seq),
        limit=limit,
    )

async def new_session(self) -> None:
    """Create a fresh session, replacing the current one."""
    if self._store is None or self._model is None:
        return
    if self._session is None:
        await self._initialize_session(
            allow_auto_resume=False,
            emit_info_messages=False,
        )
        await self._enter_workspace_surface(ensure_session=False)
        if self._session is None:
            return
        sid = str(getattr(self._session, "session_id", "") or "").strip()
        if sid:
            chat = self.query_one("#chat-log", ChatLog)
            short_id = self._escape_markup(sid[:12])
            chat.add_info(f"[dim]New session: {short_id}...[/dim]")
        return

    # Persist any UI state for the old session before rotating.
    await self._persist_process_run_ui_state(is_active=False)

    system_prompt = self._build_system_prompt()
    approver = ToolApprover(prompt_callback=self._approval_callback)
    compactor_model = self._cowork_compactor_model()
    memory_index_model, memory_index_role = self._cowork_memory_indexer_model()
    session_id = await self._store.create_session(
        workspace=str(self._workspace),
        model_name=self._model.name,
        system_prompt=system_prompt,
    )
    session_cls = _cowork_session_cls()
    self._session = session_cls(
        model=self._model,
        tools=self._tools,
        compactor_model=compactor_model,
        workspace=self._workspace,
        scratch_dir=self._cowork_scratch_dir(),
        system_prompt=system_prompt,
        approver=approver,
        store=self._store,
        session_id=session_id,
        max_context_tokens=self._cowork_max_context_tokens(),
        model_retry_policy=self._model_retry_policy(),
        tool_exposure_mode=self._cowork_tool_exposure_mode(),
        enable_filetype_ingest_router=self._cowork_enable_filetype_ingest_router(),
        ingest_artifact_retention_max_age_days=self._cowork_ingest_artifact_retention_max_age_days(),
        ingest_artifact_retention_max_files_per_scope=self._cowork_ingest_artifact_retention_max_files_per_scope(),
        ingest_artifact_retention_max_bytes_per_scope=self._cowork_ingest_artifact_retention_max_bytes_per_scope(),
        delegate_progress_callback=self._on_cowork_delegate_progress_event,
        memory_index_enabled=self._cowork_memory_index_enabled(),
        memory_index_llm_extraction_enabled=self._cowork_memory_index_llm_extraction_enabled(),
        memory_index_model=memory_index_model,
        memory_index_model_role=memory_index_role,
        memory_index_role_strict=self._cowork_indexer_model_role_strict(),
        memory_index_queue_max_batches=self._cowork_memory_index_queue_max_batches(),
        memory_index_section_limit=self._cowork_memory_index_section_limit(),
        recall_index_max_chars=self._cowork_recall_index_max_chars(),
    )
    self._reset_cowork_steering_state(clear_session=True)
    self._total_tokens = 0
    self._bind_session_tools()
    self._hydrate_input_history_from_session()
    self._clear_files_panel()
    await self._hydrate_chat_history_for_active_session()
    chat = self.query_one("#chat-log", ChatLog)
    await self._restore_process_run_tabs(chat)
    self._process_close_hint_shown = bool(self._process_runs)
    info_message = f"[dim]New session: {self._escape_markup(session_id[:12])}...[/dim]"
    chat.add_info(info_message)
    await self._append_chat_replay_event(
        "info",
        {"text": info_message, "markup": True},
        render=False,
    )
    await self._enter_workspace_surface(ensure_session=False)

async def switch_to_session(self, session_id: str) -> None:
    """Resume a different session by ID."""
    if self._store is None or self._session is None or self._model is None:
        return

    system_prompt = self._build_system_prompt()
    approver = ToolApprover(prompt_callback=self._approval_callback)
    compactor_model = self._cowork_compactor_model()
    memory_index_model, memory_index_role = self._cowork_memory_indexer_model()

    # Persist outgoing session UI state before switching.
    await self._persist_process_run_ui_state(is_active=False)

    session_cls = _cowork_session_cls()
    new_session = session_cls(
        model=self._model,
        tools=self._tools,
        compactor_model=compactor_model,
        workspace=self._workspace,
        scratch_dir=self._cowork_scratch_dir(),
        system_prompt=system_prompt,
        approver=approver,
        store=self._store,
        max_context_tokens=self._cowork_max_context_tokens(),
        model_retry_policy=self._model_retry_policy(),
        tool_exposure_mode=self._cowork_tool_exposure_mode(),
        enable_filetype_ingest_router=self._cowork_enable_filetype_ingest_router(),
        ingest_artifact_retention_max_age_days=self._cowork_ingest_artifact_retention_max_age_days(),
        ingest_artifact_retention_max_files_per_scope=self._cowork_ingest_artifact_retention_max_files_per_scope(),
        ingest_artifact_retention_max_bytes_per_scope=self._cowork_ingest_artifact_retention_max_bytes_per_scope(),
        delegate_progress_callback=self._on_cowork_delegate_progress_event,
        memory_index_enabled=self._cowork_memory_index_enabled(),
        memory_index_llm_extraction_enabled=self._cowork_memory_index_llm_extraction_enabled(),
        memory_index_model=memory_index_model,
        memory_index_model_role=memory_index_role,
        memory_index_role_strict=self._cowork_indexer_model_role_strict(),
        memory_index_queue_max_batches=self._cowork_memory_index_queue_max_batches(),
        memory_index_section_limit=self._cowork_memory_index_section_limit(),
        recall_index_max_chars=self._cowork_recall_index_max_chars(),
    )
    await new_session.resume(session_id)

    self._session = new_session
    self._reset_cowork_steering_state(clear_session=True)
    self._total_tokens = new_session.total_tokens
    self._bind_session_tools()
    self._hydrate_input_history_from_session()
    self._clear_files_panel()
    await self._hydrate_chat_history_for_active_session()
    chat = self.query_one("#chat-log", ChatLog)
    await self._restore_process_run_tabs(chat)
    self._process_close_hint_shown = bool(self._process_runs)
    info_message = (
        "[bold #7dcfff]Switched Session[/bold #7dcfff]\n"
        f"  [bold]Session ID:[/] [dim]{self._escape_markup(session_id)}[/dim]\n"
        f"  [bold]Turns:[/] {new_session.session_state.turn_count}"
    )
    chat.add_info(info_message)
    await self._append_chat_replay_event(
        "info",
        {"text": info_message, "markup": True},
        render=False,
    )
