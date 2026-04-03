"""Process-run UI state, stage tracking, and persistence helpers."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path

from textual.widgets import TabbedContent, TabPane

from loom.tui.widgets import ChatLog, EventPanel
from loom.utils.latency import log_latency_event

from ..constants import (
    _MAX_INPUT_HISTORY,
    _MAX_PERSISTED_PROCESS_ACTIVITY,
    _MAX_PERSISTED_PROCESS_RESULTS,
    _MAX_PERSISTED_PROCESS_RUNS,
    _PROCESS_RUN_HEARTBEAT_STAGES,
    _PROCESS_RUN_LAUNCH_STAGE_INDEX,
    _PROCESS_RUN_LAUNCH_STAGE_LABEL,
    _PROCESS_RUN_LAUNCH_STAGES,
    _PROCESS_STATUS_ICON,
    _PROCESS_STATUS_LABEL,
    _plain_text,
)
from ..models import ProcessRunState
from ..widgets import ProcessRunPane
from . import state as process_run_state

logger = logging.getLogger(__name__)


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _new_process_run_id(self) -> str:
    """Create a short unique run ID for display and routing."""
    while True:
        run_id = uuid.uuid4().hex[:6]
        if run_id not in self._process_runs:
            return run_id

def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as MM:SS or H:MM:SS."""
    return process_run_state.format_elapsed(seconds)

def _elapsed_seconds_for_run(self, run: ProcessRunState) -> float:
    """Return elapsed seconds for a run (live or finalized)."""
    return process_run_state.elapsed_seconds_for_run(run)

def _is_process_run_busy_status(status: str) -> bool:
    """Return True while a run is actively consuming execution resources."""
    return process_run_state.is_process_run_busy_status(status)

def _set_process_run_status(self, run: ProcessRunState, status: str) -> None:
    """Apply run status transition and keep paused-time bookkeeping consistent."""
    process_run_state.set_process_run_status(run, status)

def _append_process_run_activity(
    self, run: ProcessRunState, message: str,
) -> None:
    """Record and render one process-run activity line."""
    text = self._one_line(message, 1200)
    if not text:
        return
    log = getattr(run, "activity_log", None)
    if not isinstance(log, list):
        try:
            run.activity_log = []
            log = run.activity_log
        except Exception:
            log = None
    if isinstance(log, list):
        log.append(text)
        self._trim_process_run_activity_log(run)
    now = time.monotonic()
    run.launch_last_progress_at = now
    run.launch_last_heartbeat_at = 0.0
    run.launch_silent_warning_emitted = False
    try:
        run.pane.add_activity(text)
    except Exception:
        pass

def _trim_process_run_activity_log(self, run: ProcessRunState) -> None:
    """Bound persisted run activity log and rebase keyed line indices."""
    log = getattr(run, "activity_log", None)
    if not isinstance(log, list):
        return
    if len(log) <= _MAX_PERSISTED_PROCESS_ACTIVITY:
        return
    overflow = len(log) - _MAX_PERSISTED_PROCESS_ACTIVITY
    if overflow <= 0:
        return
    del log[:overflow]
    index_map = getattr(run, "launch_stage_activity_indices", None)
    if not isinstance(index_map, dict):
        return
    rebased: dict[str, int] = {}
    for raw_stage, raw_index in index_map.items():
        stage_id = str(raw_stage or "").strip()
        if not stage_id:
            continue
        try:
            index = int(raw_index) - overflow
        except Exception:
            continue
        if index >= 0:
            rebased[stage_id] = index
    run.launch_stage_activity_indices = rebased

def _process_run_stage_activity_key(stage: str) -> str:
    """Stable keyed widget id for one launch-stage activity line."""
    return f"launch-stage:{str(stage or '').strip() or 'unknown'}"

def _upsert_process_run_stage_activity(
    self,
    run: ProcessRunState,
    *,
    stage: str,
    text: str,
) -> None:
    """Insert/update one launch-stage activity line without duplicating rows."""
    stage_id = str(stage or "").strip()
    rendered = self._one_line(text, 1200)
    if not stage_id or not rendered:
        return
    log = getattr(run, "activity_log", None)
    if not isinstance(log, list):
        try:
            run.activity_log = []
            log = run.activity_log
        except Exception:
            log = None
    index_map = getattr(run, "launch_stage_activity_indices", None)
    if not isinstance(index_map, dict):
        try:
            run.launch_stage_activity_indices = {}
            index_map = run.launch_stage_activity_indices
        except Exception:
            index_map = None
    if isinstance(log, list) and isinstance(index_map, dict):
        try:
            idx = int(index_map.get(stage_id, -1))
        except Exception:
            idx = -1
        if 0 <= idx < len(log):
            log[idx] = rendered
        else:
            log.append(rendered)
            index_map[stage_id] = len(log) - 1
            self._trim_process_run_activity_log(run)
            idx = int(index_map.get(stage_id, -1))
            if idx < 0 and isinstance(log, list):
                for offset, line in enumerate(log):
                    if str(line) == rendered:
                        index_map[stage_id] = offset
                        break
    now = time.monotonic()
    run.launch_last_progress_at = now
    try:
        run.pane.upsert_activity(self._process_run_stage_activity_key(stage_id), rendered)
    except Exception:
        try:
            run.pane.add_activity(rendered)
        except Exception:
            pass

def _render_process_run_stage_activity_text(
    self,
    stage: str,
    *,
    dots: int,
    duration_seconds: float | None = None,
) -> str:
    """Render one launch-stage activity line with dot animation + optional elapsed."""
    label = self._process_run_launch_stage_label(str(stage or "").strip())
    dot_count = max(1, int(dots))
    text = f"{label}{'.' * dot_count}"
    if duration_seconds is None:
        return text
    elapsed = self._format_elapsed(max(0.0, float(duration_seconds)))
    return f"{text} {elapsed}"

def _start_process_run_stage_activity_line(self, run: ProcessRunState, stage: str) -> None:
    """Start (or reset) the active keyed line for stage heartbeat updates."""
    stage_id = str(stage or "").strip()
    if stage_id not in _PROCESS_RUN_HEARTBEAT_STAGES:
        return
    run.launch_stage_heartbeat_stage = stage_id
    run.launch_stage_heartbeat_dots = 1
    self._upsert_process_run_stage_activity(
        run,
        stage=stage_id,
        text=self._render_process_run_stage_activity_text(stage_id, dots=1),
    )

def _finalize_process_run_stage_activity_line(
    self,
    run: ProcessRunState,
    *,
    stage: str,
    duration_seconds: float,
) -> None:
    """Finalize a stage heartbeat line with elapsed timer."""
    stage_id = str(stage or "").strip()
    if stage_id not in _PROCESS_RUN_HEARTBEAT_STAGES:
        return
    dots = int(getattr(run, "launch_stage_heartbeat_dots", 1) or 1)
    if str(getattr(run, "launch_stage_heartbeat_stage", "")).strip() != stage_id:
        dots = max(dots, 1)
    self._upsert_process_run_stage_activity(
        run,
        stage=stage_id,
        text=self._render_process_run_stage_activity_text(
            stage_id,
            dots=dots,
            duration_seconds=duration_seconds,
        ),
    )
    if str(getattr(run, "launch_stage_heartbeat_stage", "")).strip() == stage_id:
        run.launch_stage_heartbeat_stage = ""
        run.launch_stage_heartbeat_dots = 0

def _append_process_run_result(
    self, run: ProcessRunState, text: str, *, success: bool,
) -> None:
    """Record and render one process-run final result line."""
    message = _plain_text(text).strip()
    if not message:
        message = "Process run completed." if success else "Process run failed."
    records = getattr(run, "result_log", None)
    if not isinstance(records, list):
        try:
            run.result_log = []
            records = run.result_log
        except Exception:
            records = None
    if isinstance(records, list):
        records.append({"text": message, "success": bool(success)})
        if len(records) > _MAX_PERSISTED_PROCESS_RESULTS:
            del records[:-_MAX_PERSISTED_PROCESS_RESULTS]
    try:
        run.pane.add_result(message, success=success)
    except Exception:
        pass

def _process_run_launch_stage_label(stage: str) -> str:
    """Return display label for a launch-stage identifier."""
    return process_run_state.process_run_launch_stage_label(
        stage,
        stage_labels=_PROCESS_RUN_LAUNCH_STAGE_LABEL,
    )

def _process_run_stage_rows(self, run: ProcessRunState) -> list[dict]:
    """Render launch/provisioning checklist rows while plan tasks are not ready."""
    return process_run_state.process_run_stage_rows(
        run,
        stages=_PROCESS_RUN_LAUNCH_STAGES,
        stage_index=_PROCESS_RUN_LAUNCH_STAGE_INDEX,
        one_line=self._one_line,
    )

def _process_run_stage_summary_row(self, run: ProcessRunState) -> dict | None:
    """Return a compact launch-stage summary row to prepend above task rows."""
    return process_run_state.process_run_stage_summary_row(
        run,
        stage_labels=_PROCESS_RUN_LAUNCH_STAGE_LABEL,
        launch_stage_label=self._process_run_launch_stage_label,
    )

def _refresh_process_run_progress(self, run: ProcessRunState) -> None:
    """Refresh process progress list with tasks or launch-stage rows."""
    tasks = [
        dict(row)
        for row in getattr(run, "tasks", [])
        if isinstance(row, dict)
    ]
    if not tasks:
        tasks = self._process_run_stage_rows(run)
    else:
        summary = self._process_run_stage_summary_row(run)
        if summary is not None and not any(
            str(row.get("id", "")).strip() == "stage:summary"
            for row in tasks
            if isinstance(row, dict)
        ):
            tasks = [summary, *tasks]
    try:
        run.pane.set_tasks(tasks)
    except Exception:
        pass

def _set_process_run_launch_stage(
    self,
    run: ProcessRunState,
    stage: str,
    *,
    note: str = "",
) -> None:
    """Update launch stage and keep progress/timestamps coherent."""
    clean_stage = str(stage or "").strip() or "accepted"
    now = time.monotonic()
    previous_stage = str(getattr(run, "launch_stage", "")).strip()
    previous_started = float(getattr(run, "launch_stage_started_at", 0.0) or 0.0)
    if previous_stage and previous_stage != clean_stage and previous_started > 0:
        duration = max(0.0, now - previous_started)
        log_latency_event(
            logger,
            event="run_stage_duration",
            duration_seconds=duration,
            fields={
                "run_id": str(getattr(run, "run_id", "")).strip(),
                "process": str(getattr(run, "process_name", "")).strip(),
                "stage": previous_stage,
                "next_stage": clean_stage,
                "run_stage_duration_ms": int(duration * 1000),
            },
        )
        logger.debug(
            "run_stage_transition run_id=%s from=%s to=%s duration_ms=%s",
            str(getattr(run, "run_id", "")).strip(),
            previous_stage,
            clean_stage,
            int(duration * 1000),
        )
        self._finalize_process_run_stage_activity_line(
            run,
            stage=previous_stage,
            duration_seconds=duration,
        )
    run.launch_stage = clean_stage
    run.launch_stage_started_at = now
    run.launch_last_progress_at = now
    run.launch_silent_warning_emitted = False
    if clean_stage in _PROCESS_RUN_HEARTBEAT_STAGES:
        self._start_process_run_stage_activity_line(run, clean_stage)
    else:
        run.launch_stage_heartbeat_stage = ""
        run.launch_stage_heartbeat_dots = 0
    if note:
        self._append_process_run_activity(run, note)
    self._refresh_process_run_progress(run)
    self._update_process_run_visuals(run)
    self._refresh_sidebar_progress_summary()

def _fail_process_run_launch(self, run: ProcessRunState, message: str) -> None:
    """Transition a run into failed state during launch/preflight."""
    if run.closed:
        return
    detail = self._one_line(message, 1200) or "Process run failed during launch."
    self._set_process_run_status(run, "failed")
    run.ended_at = time.monotonic()
    run.launch_error = detail
    self._log_terminal_stage_duration(run, terminal_state="failed")
    run.launch_last_progress_at = time.monotonic()
    run.launch_silent_warning_emitted = False
    self._append_process_run_activity(run, detail)
    self._append_process_run_result(run, detail, success=False)
    self._refresh_process_run_progress(run)
    self._update_process_run_visuals(run)
    self._refresh_sidebar_progress_summary()
    try:
        events_panel = self.query_one("#events-panel", EventPanel)
        events_panel.add_event(
            _now_str(),
            "process_err",
            f"{run.process_name} #{run.run_id}",
        )
    except Exception:
        pass
    try:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(f"[bold #f7768e]Process run {run.run_id} failed:[/] {detail}")
    except Exception:
        pass
    self.notify(detail, severity="error", timeout=5)

def _maybe_emit_process_run_heartbeat(self, run: ProcessRunState) -> None:
    """Emit a periodic liveness heartbeat while launch/execution is active."""
    if getattr(run, "closed", False):
        return
    if not self._is_process_run_busy_status(str(getattr(run, "status", ""))):
        return
    stage = str(getattr(run, "launch_stage", "")).strip()
    if stage not in _PROCESS_RUN_HEARTBEAT_STAGES:
        return
    now = time.monotonic()
    interval = self._tui_run_launch_heartbeat_interval_seconds()
    last_progress = float(getattr(run, "launch_last_progress_at", 0.0) or 0.0)
    last_heartbeat = float(getattr(run, "launch_last_heartbeat_at", 0.0) or 0.0)
    if last_progress and (now - last_progress) < interval:
        return
    if last_heartbeat and (now - last_heartbeat) < interval:
        return
    silent_window = max(0.0, now - last_progress) if last_progress > 0 else interval
    if not bool(getattr(run, "launch_silent_warning_emitted", False)):
        log_latency_event(
            logger,
            event="run_silent_window",
            duration_seconds=silent_window,
            fields={
                "run_id": str(getattr(run, "run_id", "")).strip(),
                "process": str(getattr(run, "process_name", "")).strip(),
                "stage": str(getattr(run, "launch_stage", "")).strip(),
                "run_silent_window_ms": int(silent_window * 1000),
            },
        )
        run.launch_silent_warning_emitted = True
    current_stage = str(getattr(run, "launch_stage_heartbeat_stage", "")).strip()
    if current_stage != stage:
        run.launch_stage_heartbeat_stage = stage
        run.launch_stage_heartbeat_dots = 1
    else:
        run.launch_stage_heartbeat_dots = int(
            getattr(run, "launch_stage_heartbeat_dots", 0) or 0,
        ) + 1
        run.launch_stage_heartbeat_dots = min(run.launch_stage_heartbeat_dots, 48)
    self._upsert_process_run_stage_activity(
        run,
        stage=stage,
        text=self._render_process_run_stage_activity_text(
            stage,
            dots=run.launch_stage_heartbeat_dots,
        ),
    )
    run.launch_last_heartbeat_at = now
    run.launch_last_progress_at = now
    run.launch_silent_warning_emitted = True

def _log_terminal_stage_duration(
    self,
    run: ProcessRunState,
    *,
    terminal_state: str,
) -> None:
    """Log final duration for the current launch stage on terminal transitions."""
    started = float(getattr(run, "launch_stage_started_at", 0.0) or 0.0)
    if started <= 0:
        return
    stage = str(getattr(run, "launch_stage", "")).strip()
    now = time.monotonic()
    duration = max(0.0, now - started)
    log_latency_event(
        logger,
        event="run_stage_duration",
        duration_seconds=duration,
        fields={
            "run_id": str(getattr(run, "run_id", "")).strip(),
            "process": str(getattr(run, "process_name", "")).strip(),
            "stage": stage,
            "terminal_state": str(terminal_state or "").strip(),
            "run_stage_duration_ms": int(duration * 1000),
        },
    )
    self._finalize_process_run_stage_activity_line(
        run,
        stage=stage,
        duration_seconds=duration,
    )
    run.launch_stage_started_at = now

def _serialize_process_run_state(self, run: ProcessRunState) -> dict:
    """Serialize one in-memory process run for session UI persistence."""
    tasks = [
        dict(row)
        for row in getattr(run, "tasks", [])
        if isinstance(row, dict)
    ][-_MAX_PERSISTED_PROCESS_ACTIVITY:]
    labels = getattr(run, "task_labels", {})
    if not isinstance(labels, dict):
        labels = {}
    phase_map = getattr(run, "subtask_phase_ids", {})
    if not isinstance(phase_map, dict):
        phase_map = {}
    activity = [
        self._one_line(line, 1200)
        for line in getattr(run, "activity_log", [])
        if self._one_line(line, 1200)
    ][-_MAX_PERSISTED_PROCESS_ACTIVITY:]
    result_log: list[dict] = []
    for item in getattr(run, "result_log", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        result_log.append({"text": text, "success": bool(item.get("success", False))})
    result_log = result_log[-_MAX_PERSISTED_PROCESS_RESULTS:]
    status = str(getattr(run, "status", "completed")).strip()
    if status not in _PROCESS_STATUS_LABEL:
        status = "completed"
    return {
        "run_id": str(getattr(run, "run_id", "")).strip(),
        "process_name": str(getattr(run, "process_name", "")).strip(),
        "goal": str(getattr(run, "goal", "")).strip(),
        "run_workspace": str(getattr(run, "run_workspace", self._workspace)),
        "status": status,
        "task_id": str(getattr(run, "task_id", "")).strip(),
        "elapsed_seconds": float(self._elapsed_seconds_for_run(run)),
        "launch_stage": str(getattr(run, "launch_stage", "accepted")).strip() or "accepted",
        "launch_error": str(getattr(run, "launch_error", "")).strip(),
        "launch_tab_created_at": float(getattr(run, "launch_tab_created_at", 0.0) or 0.0),
        "tasks": tasks,
        "task_labels": {str(k): str(v) for k, v in labels.items()},
        "subtask_phase_ids": {
            str(k): str(v)
            for k, v in phase_map.items()
            if str(k).strip() and str(v).strip()
        },
        "activity_log": activity,
        "result_log": result_log,
        "auth_profile_overrides": {
            str(k): str(v)
            for k, v in getattr(run, "auth_profile_overrides", {}).items()
            if str(k).strip() and str(v).strip()
        },
        "auth_required_resources": [
            dict(item)
            for item in getattr(run, "auth_required_resources", [])
            if isinstance(item, dict)
        ],
    }

def _sync_process_runs_into_session_state(self) -> None:
    """Mirror process-run tab state into SessionState.ui_state."""
    session = self._session
    if session is None:
        return
    state = getattr(session, "session_state", None)
    if state is None:
        return
    ui_state = getattr(state, "ui_state", None)
    if not isinstance(ui_state, dict):
        ui_state = {}
        try:
            state.ui_state = ui_state
        except Exception:
            return

    serialized_runs = [
        self._serialize_process_run_state(run)
        for run in sorted(self._process_runs.values(), key=lambda r: r.started_at)
        if not getattr(run, "closed", False)
    ][-_MAX_PERSISTED_PROCESS_RUNS:]

    active_run_id = ""
    try:
        tabs = self.query_one("#tabs", TabbedContent)
        active_tab = str(getattr(tabs, "active", "") or "")
        for run in self._process_runs.values():
            if getattr(run, "pane_id", "") == active_tab:
                active_run_id = str(getattr(run, "run_id", "")).strip()
                break
    except Exception:
        pass

    ui_state["process_tabs"] = {
        "version": 1,
        "active_run_id": active_run_id,
        "runs": serialized_runs,
    }

def _sync_input_history_into_session_state(self) -> None:
    """Mirror input history into SessionState.ui_state."""
    session = self._session
    if session is None:
        return
    state = getattr(session, "session_state", None)
    if state is None:
        return
    ui_state = getattr(state, "ui_state", None)
    if not isinstance(ui_state, dict):
        ui_state = {}
        try:
            state.ui_state = ui_state
        except Exception:
            return
    ui_state["input_history"] = {
        "version": 1,
        "items": list(self._input_history[-_MAX_INPUT_HISTORY:]),
    }

async def _persist_process_run_ui_state(
    self, *, is_active: bool | None = None,
) -> None:
    """Persist SessionState (including process-tab UI state) to storage."""
    session = self._session
    store = self._store
    if session is None or store is None:
        return
    session_id = str(getattr(session, "session_id", "") or "").strip()
    if not session_id:
        return

    self._sync_process_runs_into_session_state()
    self._sync_input_history_into_session_state()
    payload: dict = {}
    state = getattr(session, "session_state", None)
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        try:
            payload["session_state"] = to_dict()
        except Exception:
            pass
    if is_active is not None:
        payload["is_active"] = is_active
    if not payload:
        return
    try:
        ui_state = payload.get("session_state", {}).get("ui_state", {})
        await store.patch_session_state_metadata(
            session_id,
            ui_state=ui_state if isinstance(ui_state, dict) else {},
            is_active=is_active,
        )
    except Exception as e:
        logger.debug("Failed to persist process UI state: %s", e)

def _persisted_process_tabs_payload(self) -> tuple[list[dict], str]:
    """Return persisted process-tab payload from SessionState.ui_state."""
    session = self._session
    if session is None:
        return [], ""
    state = getattr(session, "session_state", None)
    if state is None:
        return [], ""
    ui_state = getattr(state, "ui_state", None)
    if not isinstance(ui_state, dict):
        return [], ""

    payload = ui_state.get("process_tabs")
    if isinstance(payload, dict):
        runs = payload.get("runs", [])
        active = str(payload.get("active_run_id", "")).strip()
        if isinstance(runs, list):
            return runs, active
    legacy_runs = ui_state.get("process_runs")
    if isinstance(legacy_runs, list):
        return legacy_runs, ""
    return [], ""

async def _drop_process_run_tabs(self) -> None:
    """Remove all process run panes from the UI and clear in-memory state."""
    if not self._process_runs:
        self._process_run_pending_inject.clear()
        return
    tabs = None
    try:
        tabs = self.query_one("#tabs", TabbedContent)
    except Exception:
        tabs = None

    for run in list(self._process_runs.values()):
        worker = getattr(run, "worker", None)
        if worker is not None and hasattr(worker, "cancel"):
            try:
                worker.cancel()
            except Exception:
                pass
        self._clear_process_run_cancel_handler(str(getattr(run, "run_id", "")))
        self._process_run_pending_inject.pop(str(getattr(run, "run_id", "")).strip(), None)
        pane_id = str(getattr(run, "pane_id", "") or "").strip()
        if tabs is not None and pane_id:
            try:
                await tabs.remove_pane(pane_id)
            except Exception:
                pass
    self._process_runs.clear()
    self._process_run_pending_inject.clear()
    if tabs is not None:
        try:
            if not tabs.active:
                tabs.active = "tab-chat"
        except Exception:
            pass
    self._refresh_sidebar_progress_summary()

async def _restore_process_run_tabs(self, chat: ChatLog | None = None) -> None:
    """Restore process run tabs for the current resumed session."""
    runs_payload, active_run_id = self._persisted_process_tabs_payload()
    await self._drop_process_run_tabs()
    if not runs_payload:
        self._refresh_sidebar_progress_summary()
        return
    try:
        tabs = self.query_one("#tabs", TabbedContent)
    except Exception:
        self._refresh_sidebar_progress_summary()
        return

    loader = None
    try:
        loader = self._create_process_loader()
    except Exception:
        loader = None

    restored = 0
    interrupted = 0
    seen_ids: set[str] = set()

    for raw in runs_payload[:_MAX_PERSISTED_PROCESS_RUNS]:
        if not isinstance(raw, dict):
            continue

        run_id = str(raw.get("run_id", "")).strip()[:12]
        if not run_id:
            run_id = self._new_process_run_id()
        while run_id in seen_ids or run_id in self._process_runs:
            run_id = self._new_process_run_id()
        seen_ids.add(run_id)

        process_name = str(raw.get("process_name", "")).strip() or "process"
        goal = str(raw.get("goal", "")).strip() or "(restored run)"
        status = str(raw.get("status", "completed")).strip()
        if status not in _PROCESS_STATUS_LABEL:
            status = "completed"
        task_id = str(raw.get("task_id", "")).strip()
        launch_stage = str(raw.get("launch_stage", "accepted")).strip() or "accepted"
        launch_error = str(raw.get("launch_error", "")).strip()
        try:
            launch_tab_created_at = float(raw.get("launch_tab_created_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            launch_tab_created_at = 0.0
        try:
            elapsed_seconds = max(0.0, float(raw.get("elapsed_seconds", 0.0)))
        except (TypeError, ValueError):
            elapsed_seconds = 0.0
        started_at = time.monotonic() - elapsed_seconds
        ended_at = (
            None
            if self._is_process_run_active_status(status)
            else time.monotonic()
        )
        paused_started_at = time.monotonic() if status == "paused" else 0.0

        tasks = [
            dict(row)
            for row in raw.get("tasks", [])
            if isinstance(row, dict)
        ]
        labels_raw = raw.get("task_labels", {})
        task_labels = (
            {str(k): str(v) for k, v in labels_raw.items()}
            if isinstance(labels_raw, dict)
            else {}
        )
        phase_map_raw = raw.get("subtask_phase_ids", {})
        subtask_phase_ids = (
            {str(k): str(v) for k, v in phase_map_raw.items()}
            if isinstance(phase_map_raw, dict)
            else {}
        )
        activity_log = [
            self._one_line(line, 1200)
            for line in raw.get("activity_log", [])
            if self._one_line(line, 1200)
        ][-_MAX_PERSISTED_PROCESS_ACTIVITY:]
        result_log: list[dict] = []
        for item in raw.get("result_log", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            result_log.append({"text": text, "success": bool(item.get("success", False))})
        result_log = result_log[-_MAX_PERSISTED_PROCESS_RESULTS:]
        auth_profile_overrides_raw = raw.get("auth_profile_overrides", {})
        auth_profile_overrides = {}
        if isinstance(auth_profile_overrides_raw, dict):
            for key, value in auth_profile_overrides_raw.items():
                selector = str(key or "").strip()
                profile_id = str(value or "").strip()
                if selector and profile_id:
                    auth_profile_overrides[selector] = profile_id
        auth_required_resources = [
            dict(item)
            for item in raw.get("auth_required_resources", [])
            if isinstance(item, dict)
        ]

        process_defn = None
        if loader is not None:
            try:
                process_defn = loader.load(process_name)
            except Exception:
                process_defn = None

        pane_id = f"tab-run-{run_id}"
        pane = ProcessRunPane(
            run_id=run_id,
            process_name=process_name,
            goal=goal,
        )
        run_workspace_raw = str(raw.get("run_workspace", "")).strip()
        run_workspace = Path(run_workspace_raw or str(self._workspace)).expanduser()
        try:
            run_workspace.resolve().relative_to(self._workspace.resolve())
        except Exception:
            run_workspace = self._workspace
        run = ProcessRunState(
            run_id=run_id,
            process_name=process_name,
            goal=goal,
            run_workspace=run_workspace,
            process_defn=process_defn,
            pane_id=pane_id,
            pane=pane,
            status=status,
            task_id=task_id,
            started_at=started_at,
            ended_at=ended_at,
            tasks=tasks,
            task_labels=task_labels,
            subtask_phase_ids=subtask_phase_ids,
            activity_log=activity_log,
            result_log=result_log,
            auth_profile_overrides=auth_profile_overrides,
            auth_required_resources=auth_required_resources,
            launch_stage=launch_stage if launch_stage in _PROCESS_RUN_LAUNCH_STAGE_INDEX else (
                "queueing_delegate" if status == "queued" else "running"
            ),
            launch_error=launch_error,
            launch_tab_created_at=launch_tab_created_at,
            paused_started_at=paused_started_at,
            paused_accumulated_seconds=0.0,
        )

        if self._is_process_run_busy_status(run.status):
            interrupted += 1
            self._set_process_run_status(run, "failed")
            run.ended_at = time.monotonic()
            note = "Run interrupted before session resume; marked failed."
            run.activity_log.append(note)
            run.result_log.append({"text": note, "success": False})

        self._process_runs[run_id] = run
        await tabs.add_pane(
            TabPane(
                self._format_process_run_tab_title(run),
                pane,
                id=pane_id,
            ),
            after="tab-events",
        )
        self._refresh_process_run_progress(run)
        self._refresh_process_run_outputs(run)
        for line in run.activity_log:
            run.pane.add_activity(line)
        for item in run.result_log:
            run.pane.add_result(
                str(item.get("text", "")),
                success=bool(item.get("success", False)),
            )
        if run.activity_log:
            run.last_progress_message = run.activity_log[-1]
            run.last_progress_at = time.monotonic()
        self._update_process_run_visuals(run)
        restored += 1

    if active_run_id and active_run_id in self._process_runs:
        tabs.active = self._process_runs[active_run_id].pane_id
    self._refresh_sidebar_progress_summary()

    if restored and chat is not None:
        info = (
            "[bold #7dcfff]Restored Process Tabs[/bold #7dcfff]\n"
            f"  [bold]Count:[/] {restored}"
        )
        if interrupted:
            info += (
                "\n"
                f"  [#f7768e]{interrupted} interrupted run(s) were marked failed. "
                "Use /run resume <run-id-prefix> to continue.[/]"
            )
        chat.add_info(info)

def _format_process_run_tab_title(self, run: ProcessRunState) -> str:
    """Build tab title with status indicator and elapsed timer."""
    icon = _PROCESS_STATUS_ICON.get(run.status, "\u25cb")
    elapsed = self._format_elapsed(self._elapsed_seconds_for_run(run))
    name = run.process_name
    if len(name) > 16:
        name = f"{name[:15]}\u2026"
    return f"{icon} {name} #{run.run_id} {elapsed}"

def _update_process_run_visuals(self, run: ProcessRunState) -> None:
    """Update pane header and tab title from current run state."""
    elapsed = self._format_elapsed(self._elapsed_seconds_for_run(run))
    working_folder = self._process_run_working_folder_label(run)
    pending_queue = self._process_run_pending_inject.get(run.run_id, [])
    pending_inject = len(pending_queue)
    pending_preview = str(pending_queue[0]).strip() if pending_queue else ""
    run.pane.set_status_header(
        status=run.status,
        elapsed=elapsed,
        task_id=run.task_id,
        working_folder=working_folder,
        pending_inject_count=pending_inject,
        pending_inject_preview=pending_preview,
    )
    try:
        tabs = self.query_one("#tabs", TabbedContent)
        tab = tabs.get_tab(run.pane_id)
        tab.label = self._format_process_run_tab_title(run)
    except Exception:
        pass
    self._sync_activity_indicator()

def _process_run_working_folder_label(self, run: ProcessRunState) -> str:
    """Return display label for the run's working folder."""
    raw_workspace = getattr(run, "run_workspace", None)
    if raw_workspace is None:
        return ""
    try:
        resolved_workspace = Path(raw_workspace).expanduser().resolve()
    except Exception:
        return str(raw_workspace)
    try:
        root = self._workspace.resolve()
    except Exception:
        return str(resolved_workspace)
    try:
        rel = resolved_workspace.relative_to(root)
    except ValueError:
        return str(resolved_workspace)
    rel_text = str(rel).strip()
    if not rel_text or rel_text in {".", "./"}:
        return "(workspace root)"
    return rel_text

def _tick_process_run_elapsed(self) -> None:
    """Periodic timer to refresh elapsed text for active process runs."""
    active = False
    for run in self._process_runs.values():
        if self._is_process_run_busy_status(run.status):
            active = True
            self._maybe_emit_process_run_heartbeat(run)
            self._update_process_run_visuals(run)
    if active:
        self._refresh_sidebar_progress_summary()
    self._sync_activity_indicator()
