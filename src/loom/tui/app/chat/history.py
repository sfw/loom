"""Chat history replay helpers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from loom.cowork.approval import ToolApprover
from loom.tui.widgets import ChatLog

logger = logging.getLogger(__name__)


def _cowork_session_cls():
    # Resolve from concrete sibling module so monkeypatches on loom.tui.app facade
    # continue propagating to internal session construction sites.
    from .. import core as app_core

    return app_core.CoworkSession


def apply_chat_render_cap(
    *,
    replay_events: list[dict[str, Any]],
    max_rows: int,
    trim_total: int,
    history_source: str,
    oldest_seq: int | None,
    oldest_turn: int | None,
    active_session_id: str,
    event_cursor_turn: Callable[[dict[str, Any]], int | None],
    logger: logging.Logger,
) -> tuple[bool, list[dict[str, Any]], int, int | None, int | None]:
    """Trim replay buffer to configured cap and return updated state."""
    total = len(replay_events)
    if total <= max_rows:
        return False, replay_events, trim_total, oldest_seq, oldest_turn

    trimmed = total - max_rows
    replay_events = replay_events[-max_rows:]
    trim_total += trimmed
    if history_source != "legacy":
        first = replay_events[0] if replay_events else {}
        try:
            seq = int(first.get("seq", 0) or 0)
        except (TypeError, ValueError):
            seq = 0
        oldest_seq = seq if seq > 0 else oldest_seq
    else:
        turns = [
            turn
            for turn in (event_cursor_turn(event) for event in replay_events)
            if turn is not None
        ]
        oldest_turn = min(turns) if turns else oldest_turn

    logger.info(
        "chat_render_trimmed session=%s trimmed_rows=%s max_rows=%s total_trimmed=%s",
        active_session_id,
        trimmed,
        max_rows,
        trim_total,
    )
    return True, replay_events, trim_total, oldest_seq, oldest_turn

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
    if self._chat_trimmed_total > 0:
        chat.add_info(
            (
                "[dim]Transcript window truncated: "
                f"{self._chat_trimmed_total} older row(s) hidden.[/dim]"
            ),
        )
    rendered = 0
    skipped = 0
    for event in self._chat_replay_events:
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
    trimmed = self._apply_chat_render_cap()
    if trimmed:
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
    self._apply_chat_render_cap()
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
    self._apply_chat_render_cap()
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
