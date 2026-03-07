"""Session utility helpers for the TUI chat subsystem."""

from __future__ import annotations

import asyncio
from typing import Any


def active_session_id(session: Any | None) -> str:
    """Return normalized active session id."""
    if session is None:
        return ""
    return str(getattr(session, "session_id", "") or "").strip()


def chat_event_cursor_turn(event: dict[str, Any]) -> int | None:
    """Return positive turn number from a replay event, if present."""
    try:
        value = int(event.get("turn_number", 0) or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def coerce_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_bool(value: object, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


async def initialize_session(
    self,
    *,
    startup_resume: tuple[str | None, bool] | None = None,
    allow_auto_resume: bool = True,
    emit_info_messages: bool = True,
    session_cls: type[Any] | None = None,
    approver_cls: type[Any] | None = None,
) -> None:
    """Initialize tools, session, and welcome message."""
    if session_cls is None:
        from loom.cowork.session import CoworkSession

        session_cls = CoworkSession
    if approver_cls is None:
        from loom.cowork.approval import ToolApprover

        approver_cls = ToolApprover

    chat = self.query_one("#chat-log")
    self._total_tokens = 0
    self._sidebar_cowork_tasks = []
    self._active_delegate_streams = {}
    self._sync_activity_indicator()
    chat.set_stream_flush_interval_ms(self._tui_chat_stream_flush_interval_ms())

    # Build a clean registry once at session initialization.
    # Keep startup responsive by moving registry construction off the UI loop.
    await asyncio.to_thread(self._refresh_tool_registry)
    # Process command discovery can hit disk; refresh in background.
    self._refresh_process_command_index(
        chat=chat,
        notify_conflicts=True,
        background=True,
        force=True,
    )

    # Ensure persistence-dependent tools are present and tracked.
    self._ensure_persistence_tools()

    # Build system prompt
    self._apply_process_tool_policy(chat)
    system_prompt = self._build_system_prompt()

    # Build approver
    approver = approver_cls(prompt_callback=self._approval_callback)
    compactor_model = self._cowork_compactor_model()
    memory_index_model, memory_index_role = self._cowork_memory_indexer_model()

    if startup_resume is not None:
        resume_target, auto_resume = startup_resume
    elif allow_auto_resume:
        resume_target, auto_resume = await self._resolve_startup_resume_target()
    else:
        resume_target, auto_resume = None, False
    resume_info_message: str | None = None

    # Create or resume session
    if self._store is not None and resume_target:
        # Resume existing session
        self._session = session_cls(
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
        try:
            await self._session.resume(resume_target)
            self._total_tokens = self._session.total_tokens
            resume_label = (
                "Resumed latest workspace session"
                if auto_resume
                else "Resumed session"
            )
            short_id = self._escape_markup(str(resume_target)[:12])
            turns = int(getattr(self._session.session_state, "turn_count", 0) or 0)
            resume_info_message = (
                f"[dim]{self._escape_markup(resume_label)}: "
                f"{short_id}... ({turns} turns)[/dim]"
            )
        except Exception as e:
            chat.add_info(
                f"[bold #f7768e]Resume failed: {e}[/] "
                f"Starting fresh session."
            )
            self._session = None

    if self._session is not None:
        # Successfully resumed - keep it
        pass
    elif self._store is not None:
        # New persisted session
        session_id = await self._store.create_session(
            workspace=str(self._workspace),
            model_name=self._model.name,
            system_prompt=system_prompt,
        )
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
    else:
        # Ephemeral session (no database)
        self._session = session_cls(
            model=self._model,
            tools=self._tools,
            compactor_model=compactor_model,
            workspace=self._workspace,
            scratch_dir=self._cowork_scratch_dir(),
            system_prompt=system_prompt,
            approver=approver,
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
    self._hydrate_input_history_from_session()

    # Bind session-dependent tools
    self._bind_session_tools()
    await self._hydrate_chat_history_for_active_session()
    if emit_info_messages and resume_info_message:
        chat.add_info(resume_info_message)

    # Configure status bar
    status = self.query_one("#status-bar")
    status.workspace_name = self._workspace.name
    status.model_name = self._model.name
    status.process_name = self._active_process_name()

    await self._restore_process_run_tabs(chat)
    self._process_close_hint_shown = bool(self._process_runs)

    # Resume is a one-shot startup hint; subsequent reinitializations
    # should not keep trying to reopen the same prior session.
    self._resume_session = None
    self._auto_resume_workspace_on_init = False

    self.query_one("#user-input").focus()
    # Ensure command/footer bars are visible after any prior slash-hint state.
    self._set_slash_hint("")
    self._refresh_sidebar_progress_summary()
