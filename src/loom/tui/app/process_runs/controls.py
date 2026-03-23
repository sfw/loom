"""Process-run control, cancellation, and in-place restart helpers."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from textual.widgets import TabbedContent

from loom.tui.screens import ProcessRunCloseScreen
from loom.tui.widgets import ChatLog, EventPanel
from loom.utils.latency import log_latency_event

from ..models import ProcessRunLaunchRequest, ProcessRunState
from . import state as process_run_state

logger = logging.getLogger(__name__)


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _current_process_run(self) -> ProcessRunState | None:
    """Return run associated with currently active tab, if any."""
    try:
        tabs = self.query_one("#tabs", TabbedContent)
        active = tabs.active
    except Exception:
        return None
    if not active:
        return None
    return next(
        (r for r in self._process_runs.values() if r.pane_id == active),
        None,
    )

def _is_process_run_active_status(status: str) -> bool:
    state = str(status or "").strip().lower()
    return state in {"queued", "running", "paused", "cancel_requested"}

def _resolve_process_run_target(
    self, target: str,
) -> tuple[ProcessRunState | None, str | None]:
    """Resolve a process run by target selector or current tab."""
    token = target.strip().lstrip("#")
    if not token or token.lower() in {"current", "this"}:
        current = self._current_process_run()
        if current is not None:
            return current, None
        if len(self._process_runs) == 1:
            return next(iter(self._process_runs.values())), None
        if not self._process_runs:
            return None, "No process run tabs are open."
        return None, "Multiple runs open. Use /run close <run-id-prefix>."

    matches = [
        run
        for run in self._process_runs.values()
        if run.run_id.startswith(token)
    ]
    if not matches:
        return None, f"No run found matching '{token}'."
    if len(matches) > 1:
        return None, f"Ambiguous run prefix '{token}'."
    return matches[0], None

async def _confirm_close_process_run(self, run: ProcessRunState) -> bool:
    """Prompt before closing a process run tab."""
    waiter: asyncio.Future[bool] = asyncio.Future()
    screen = None

    def handle_result(confirmed: bool) -> None:
        if not waiter.done():
            waiter.set_result(bool(confirmed))

    running = self._is_process_run_active_status(run.status)
    screen = ProcessRunCloseScreen(
        run_label=f"{run.process_name} #{run.run_id}",
        running=running,
    )
    self.push_screen(screen, callback=handle_result)
    try:
        timeout = self._tui_run_close_modal_timeout_seconds()
        return await asyncio.wait_for(waiter, timeout=timeout)
    except TimeoutError:
        try:
            if screen is not None:
                screen.dismiss(False)
        except Exception:
            pass
        logger.warning(
            "process_close_confirm_timeout run_id=%s timeout_seconds=%s",
            run.run_id,
            int(timeout),
        )
        try:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_info(
                f"Close confirmation timed out for run [dim]{run.run_id}[/dim]. "
                "Please try ctrl + w again."
            )
        except Exception:
            pass
        return False
    except asyncio.CancelledError:
        # If the close-flow worker is cancelled, dismiss the modal so the
        # UI doesn't end up blocked behind an orphaned confirmation screen.
        try:
            if screen is not None:
                screen.dismiss(False)
        except Exception:
            pass
        raise

async def _confirm_force_close_process_run(
    self,
    run: ProcessRunState,
    *,
    timeout_seconds: float,
) -> bool:
    """Prompt to force-close a run tab after cancellation does not settle."""
    waiter: asyncio.Future[bool] = asyncio.Future()
    screen = ProcessRunCloseScreen(
        run_label=f"{run.process_name} #{run.run_id}",
        running=False,
        prompt_override=f"[bold #e0af68]Force close tab {run.process_name} #{run.run_id}?[/]",
        detail_override=(
            f"Cancellation has not settled after {int(max(1, timeout_seconds))}s. "
            "The background run may still continue."
        ),
        confirm_label="Force Close Tab",
        cancel_label="Keep Open",
        confirm_variant="warning",
    )

    def handle_result(confirmed: bool) -> None:
        if not waiter.done():
            waiter.set_result(bool(confirmed))

    self.push_screen(screen, callback=handle_result)
    try:
        timeout = self._tui_run_close_modal_timeout_seconds()
        return await asyncio.wait_for(waiter, timeout=timeout)
    except TimeoutError:
        try:
            screen.dismiss(False)
        except Exception:
            pass
        logger.warning(
            "process_force_close_confirm_timeout run_id=%s timeout_seconds=%s",
            run.run_id,
            int(timeout),
        )
        return False
    except asyncio.CancelledError:
        try:
            screen.dismiss(False)
        except Exception:
            pass
        raise

async def _confirm_stop_process_run(self, run: ProcessRunState) -> bool:
    """Prompt before issuing a terminal stop for a process run."""
    waiter: asyncio.Future[bool] = asyncio.Future()
    screen = ProcessRunCloseScreen(
        run_label=f"{run.process_name} #{run.run_id}",
        running=True,
        prompt_override=f"[bold #e0af68]Stop process {run.process_name} #{run.run_id}?[/]",
        detail_override=(
            "This cancels the process and it can't be revived. "
            "Start a new run if you need to continue."
        ),
        confirm_label="Stop Process",
        cancel_label="Keep Running",
        confirm_variant="error",
    )

    def handle_result(confirmed: bool) -> None:
        if not waiter.done():
            waiter.set_result(bool(confirmed))

    self.push_screen(screen, callback=handle_result)
    try:
        timeout = self._tui_run_close_modal_timeout_seconds()
        return await asyncio.wait_for(waiter, timeout=timeout)
    except TimeoutError:
        try:
            screen.dismiss(False)
        except Exception:
            pass
        logger.warning(
            "process_stop_confirm_timeout run_id=%s timeout_seconds=%s",
            run.run_id,
            int(timeout),
        )
        try:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_info(
                f"Stop confirmation timed out for run [dim]{run.run_id}[/dim]. "
                "Please try again."
            )
        except Exception:
            pass
        return False
    except asyncio.CancelledError:
        try:
            screen.dismiss(False)
        except Exception:
            pass
        raise

def _register_process_run_cancel_handler(self, run_id: str, payload: object) -> None:
    """Store orchestrator-backed control callbacks for one process run."""
    run = self._process_runs.get(run_id)
    if run is None or run.closed:
        return
    if not isinstance(payload, dict):
        return
    cancel_cb = payload.get("cancel")
    if callable(cancel_cb):
        self._process_run_cancel_handlers[run_id] = cancel_cb
    pause_cb = payload.get("pause")
    if callable(pause_cb):
        self._process_run_pause_handlers[run_id] = pause_cb
    play_cb = payload.get("resume")
    if callable(play_cb):
        self._process_run_play_handlers[run_id] = play_cb
    inject_cb = payload.get("inject")
    if callable(inject_cb):
        self._process_run_inject_handlers[run_id] = inject_cb
    answer_cb = payload.get("answer_question")
    if callable(answer_cb):
        self._process_run_answer_handlers[run_id] = answer_cb
    task_id = str(payload.get("task_id", "")).strip()
    if task_id:
        run.task_id = task_id
        self._update_process_run_visuals(run)
    # Flush any queued inject directives once inject callback is available.
    if callable(inject_cb) and self._process_run_pending_inject.get(run_id):
        self.run_worker(
            self._flush_pending_process_run_inject(run_id),
            name=f"process-run-inject-flush-{run_id}",
            group=f"process-run-inject-flush-{run_id}",
            exclusive=False,
        )

def _clear_process_run_cancel_handler(self, run_id: str) -> None:
    """Drop ephemeral control callback state for a run."""
    clean_id = str(run_id or "").strip()
    self._process_run_cancel_handlers.pop(clean_id, None)
    self._process_run_pause_handlers.pop(clean_id, None)
    self._process_run_play_handlers.pop(clean_id, None)
    self._process_run_inject_handlers.pop(clean_id, None)
    self._process_run_answer_handlers.pop(clean_id, None)
    self._process_run_question_locks.pop(clean_id, None)
    self._process_run_seen_questions.pop(clean_id, None)

def _normalize_process_run_status(raw_status: object | None) -> str:
    """Map task-engine status strings into process-run pane status values."""
    return process_run_state.normalize_process_run_status(raw_status)

async def _request_process_run_cancellation(self, run: ProcessRunState) -> dict:
    """Request cancellation, preferring orchestrator cancel over worker fallback."""
    run_id = str(getattr(run, "run_id", "")).strip()
    handler_error = ""
    handler = self._process_run_cancel_handlers.get(run_id)
    if callable(handler):
        try:
            response = await handler(
                # Keep settle waiting centralized in TUI close flow to avoid
                # compounding bridge wait + UI wait into a longer hang window.
                wait_timeout_seconds=0.0,
            )
            if isinstance(response, dict):
                path = str(response.get("path", "orchestrator")).strip() or "orchestrator"
                status = self._normalize_process_run_status(response.get("status"))
                if status in {"completed", "failed", "cancelled"}:
                    self._set_process_run_status(run, status)
                    if status != "running":
                        run.ended_at = run.ended_at or time.monotonic()
                return {
                    "requested": bool(response.get("requested", True)),
                    "path": path,
                    "error": str(response.get("error", "")).strip(),
                    "timeout": bool(response.get("timeout", False)),
                }
            return {
                "requested": True,
                "path": "orchestrator",
                "error": "",
                "timeout": False,
            }
        except Exception as e:
            logger.warning("Process run cancel bridge failed for %s: %s", run_id, e)
            handler_error = str(e)

    worker = getattr(run, "worker", None)
    if worker is not None and hasattr(worker, "cancel"):
        try:
            worker.cancel()
            return {
                "requested": True,
                "path": "worker_fallback",
                "error": handler_error,
                "timeout": False,
            }
        except Exception as e:
            error_text = str(e)
            if handler_error:
                error_text = f"{handler_error}; fallback failed: {error_text}"
            return {
                "requested": False,
                "path": "worker_fallback",
                "error": error_text,
                "timeout": False,
            }
    if handler_error:
        return {
            "requested": False,
            "path": "orchestrator",
            "error": handler_error,
            "timeout": False,
        }
    return {
        "requested": False,
        "path": "none",
            "error": "No cancellation path is available.",
            "timeout": False,
        }

async def _request_process_run_pause(self, run: ProcessRunState) -> dict:
    """Request cooperative pause for a process run."""
    run_id = str(getattr(run, "run_id", "")).strip()
    handler = self._process_run_pause_handlers.get(run_id)
    if not callable(handler):
        return {
            "requested": False,
            "path": "none",
            "error": "Pause is unavailable until delegate task control is ready.",
            "status": self._normalize_process_run_status(getattr(run, "status", "")),
        }
    try:
        response = await handler()
    except Exception as e:
        logger.warning("Process run pause bridge failed for %s: %s", run_id, e)
        return {
            "requested": False,
            "path": "orchestrator",
            "error": str(e),
            "status": self._normalize_process_run_status(getattr(run, "status", "")),
        }
    if isinstance(response, dict):
        return {
            "requested": bool(response.get("requested", False)),
            "path": str(response.get("path", "orchestrator")).strip() or "orchestrator",
            "error": str(response.get("error", "")).strip(),
            "status": self._normalize_process_run_status(response.get("status")),
        }
    return {
        "requested": True,
        "path": "orchestrator",
        "error": "",
        "status": "paused",
    }

async def _request_process_run_play(self, run: ProcessRunState) -> dict:
    """Request resume/play for a paused process run."""
    run_id = str(getattr(run, "run_id", "")).strip()
    handler = self._process_run_play_handlers.get(run_id)
    if not callable(handler):
        return {
            "requested": False,
            "path": "none",
            "error": "Play is unavailable until delegate task control is ready.",
            "status": self._normalize_process_run_status(getattr(run, "status", "")),
        }
    try:
        response = await handler()
    except Exception as e:
        logger.warning("Process run play bridge failed for %s: %s", run_id, e)
        return {
            "requested": False,
            "path": "orchestrator",
            "error": str(e),
            "status": self._normalize_process_run_status(getattr(run, "status", "")),
        }
    if isinstance(response, dict):
        return {
            "requested": bool(response.get("requested", False)),
            "path": str(response.get("path", "orchestrator")).strip() or "orchestrator",
            "error": str(response.get("error", "")).strip(),
            "status": self._normalize_process_run_status(response.get("status")),
        }
    return {
        "requested": True,
        "path": "orchestrator",
        "error": "",
        "status": "running",
    }

async def _request_process_run_inject(self, run: ProcessRunState, text: str) -> dict:
    """Inject one steering instruction into a running/paused process run."""
    run_id = str(getattr(run, "run_id", "")).strip()
    handler = self._process_run_inject_handlers.get(run_id)
    if not callable(handler):
        return {
            "requested": False,
            "path": "none",
            "error": "Inject is unavailable until delegate task control is ready.",
            "status": self._normalize_process_run_status(getattr(run, "status", "")),
        }
    try:
        response = await handler(instruction=text)
    except Exception as e:
        logger.warning("Process run inject bridge failed for %s: %s", run_id, e)
        return {
            "requested": False,
            "path": "orchestrator",
            "error": str(e),
            "status": self._normalize_process_run_status(getattr(run, "status", "")),
        }
    if isinstance(response, dict):
        return {
            "requested": bool(response.get("requested", False)),
            "path": str(response.get("path", "orchestrator")).strip() or "orchestrator",
            "error": str(response.get("error", "")).strip(),
            "status": self._normalize_process_run_status(response.get("status")),
        }
    return {
        "requested": True,
        "path": "orchestrator",
        "error": "",
        "status": self._normalize_process_run_status(getattr(run, "status", "")),
    }

async def _request_process_run_question_answer(
    self,
    run: ProcessRunState,
    *,
    question_id: str,
    answer_payload: dict[str, Any],
) -> dict[str, Any]:
    """Submit an ask_user answer for a process run via delegate bridge."""
    run_id = str(getattr(run, "run_id", "")).strip()
    handler = self._process_run_answer_handlers.get(run_id)
    if not callable(handler):
        return {
            "requested": False,
            "path": "none",
            "error": "Question answer bridge is unavailable.",
            "status": self._normalize_process_run_status(getattr(run, "status", "")),
            "question_id": str(question_id or "").strip(),
        }
    try:
        response = await handler(
            question_id=str(question_id or "").strip(),
            answer_payload=dict(answer_payload or {}),
        )
    except Exception as e:
        logger.warning(
            "Process run question answer bridge failed for %s: %s",
            run_id,
            e,
        )
        return {
            "requested": False,
            "path": "orchestrator",
            "error": str(e),
            "status": self._normalize_process_run_status(getattr(run, "status", "")),
            "question_id": str(question_id or "").strip(),
        }
    if isinstance(response, dict):
        return {
            "requested": bool(response.get("requested", False)),
            "path": str(response.get("path", "orchestrator")).strip() or "orchestrator",
            "error": str(response.get("error", "")).strip(),
            "status": self._normalize_process_run_status(response.get("status")),
            "question_id": str(
                response.get("question_id", question_id),
            ).strip(),
        }
    return {
        "requested": True,
        "path": "orchestrator",
        "error": "",
        "status": self._normalize_process_run_status(getattr(run, "status", "")),
        "question_id": str(question_id or "").strip(),
    }

async def _flush_pending_process_run_inject(self, run_id: str) -> None:
    """Flush queued inject instructions once the delegate control hook is ready."""
    pending = self._process_run_pending_inject.get(run_id)
    if not pending:
        return
    run = self._process_runs.get(run_id)
    if run is None or run.closed:
        self._process_run_pending_inject.pop(run_id, None)
        return
    while pending:
        text = str(pending[0] or "").strip()
        if not text:
            pending.pop(0)
            continue
        response = await self._request_process_run_inject(run, text)
        requested = bool(response.get("requested", False))
        error = str(response.get("error", "")).strip()
        if requested:
            pending.pop(0)
            self._append_process_run_activity(
                run,
                f"Applied queued inject: {self._one_line(text, 120)}",
            )
            continue
        if str(response.get("path", "")).strip() == "none":
            # Control path still unavailable; keep queue and retry later.
            break
        pending.pop(0)
        self._append_process_run_activity(
            run,
            (
                "Queued inject failed: "
                f"{error or 'unknown error'}"
            ),
        )
    if pending:
        self._process_run_pending_inject[run_id] = pending[-100:]
    else:
        self._process_run_pending_inject.pop(run_id, None)
    self._update_process_run_visuals(run)
    self._refresh_process_run_progress(run)
    self._refresh_sidebar_progress_summary()
    await self._persist_process_run_ui_state()

async def _wait_for_process_run_terminal_state(
    self,
    run_id: str,
    *,
    timeout_seconds: float,
) -> bool:
    """Wait for a run to reach terminal state after cancel is requested."""
    timeout = max(0.1, float(timeout_seconds))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = self._process_runs.get(run_id)
        if run is None or getattr(run, "closed", False):
            return True
        status = str(getattr(run, "status", "") or "").strip().lower()
        if status in {"completed", "failed", "cancelled", "cancel_failed", "force_closed"}:
            return True
        await asyncio.sleep(0.1)
    return False

async def _finalize_process_run_tab_close(
    self,
    run: ProcessRunState,
    *,
    tabs: TabbedContent,
    cancel_worker: bool = False,
) -> bool:
    """Remove run tab + state and release transient close/cancel handlers."""
    if run.closed and run.run_id not in self._process_runs:
        return False
    run.closed = True
    if cancel_worker:
        worker = getattr(run, "worker", None)
        if worker is not None and hasattr(worker, "cancel"):
            try:
                worker.cancel()
            except Exception:
                pass
    try:
        await tabs.remove_pane(run.pane_id)
    except Exception:
        pass
    self._process_runs.pop(run.run_id, None)
    self._process_run_pending_inject.pop(run.run_id, None)
    self._sync_activity_indicator()
    self._clear_process_run_cancel_handler(run.run_id)
    self._refresh_sidebar_progress_summary()
    await self._persist_process_run_ui_state()
    if not tabs.active:
        tabs.active = "tab-chat"
    return True

async def _close_process_run(self, run: ProcessRunState) -> bool:
    """Close a process run tab with cancel-first semantics for active runs."""
    if run.closed:
        return False
    chat = self.query_one("#chat-log", ChatLog)
    events_panel = self.query_one("#events-panel", EventPanel)
    tabs = self.query_one("#tabs", TabbedContent)

    was_running = self._is_process_run_active_status(run.status)
    if not was_running:
        if not await self._confirm_close_process_run(run):
            return False
    if was_running:
        run.close_after_cancel = True
        run.cancel_requested_at = time.monotonic()
        self._set_process_run_status(run, "cancel_requested")
        self._append_process_run_activity(run, "Cancellation requested...")
        self._update_process_run_visuals(run)
        self._refresh_process_run_progress(run)
        self._refresh_sidebar_progress_summary()

        cancel_started_at = time.monotonic()
        cancel_response = await self._request_process_run_cancellation(run)
        cancel_path = str(cancel_response.get("path", "unknown")).strip() or "unknown"
        cancel_error = str(cancel_response.get("error", "")).strip()
        cancel_requested = bool(cancel_response.get("requested", False))
        cancel_ack_seconds = max(0.0, time.monotonic() - cancel_started_at)
        log_latency_event(
            logger,
            event="run_cancel_ack",
            duration_seconds=cancel_ack_seconds,
            fields={
                "run_id": str(getattr(run, "run_id", "")).strip(),
                "process": str(getattr(run, "process_name", "")).strip(),
                "run_cancel_ack_ms": int(cancel_ack_seconds * 1000),
                "run_cancel_path": cancel_path,
                "run_cancel_result": "requested" if cancel_requested else "request_failed",
                "run_cancel_error": cancel_error,
            },
        )
        if cancel_error:
            self._append_process_run_activity(
                run,
                f"Cancellation request failed via {cancel_path}: {cancel_error}",
            )
            chat.add_info(
                f"[bold #f7768e]Cancel request failed[/] for run "
                f"[dim]{run.run_id}[/dim] via {cancel_path}: {cancel_error}"
            )
        else:
            self._append_process_run_activity(
                run,
                f"Cancellation requested via {cancel_path}.",
            )
            chat.add_info(
                f"Cancel requested for process run [dim]{run.run_id}[/dim] "
                f"via {cancel_path}."
            )
        events_panel.add_event(
            _now_str(),
            "process",
            f"{run.process_name} #{run.run_id} cancel requested ({cancel_path})",
        )

        wait_timeout = self._tui_run_cancel_wait_timeout_seconds()
        settled = False
        if cancel_path == "worker_fallback" and not str(getattr(run, "task_id", "")).strip():
            self._set_process_run_status(run, "cancelled")
            run.ended_at = time.monotonic()
            settled = True
        elif not bool(cancel_response.get("timeout", False)):
            settled = await self._wait_for_process_run_terminal_state(
                run.run_id,
                timeout_seconds=wait_timeout,
            )
        if settled:
            log_latency_event(
                logger,
                event="run_cancel_settled",
                duration_seconds=max(0.0, time.monotonic() - run.cancel_requested_at),
                fields={
                    "run_id": str(getattr(run, "run_id", "")).strip(),
                    "process": str(getattr(run, "process_name", "")).strip(),
                    "run_cancel_path": cancel_path,
                    "run_cancel_result": "settled",
                },
            )
            closed = await self._finalize_process_run_tab_close(run, tabs=tabs)
            if closed:
                chat.add_info(
                    f"Closed process run tab [dim]{run.run_id}[/dim] "
                    "after cancellation settled."
                )
            return closed

        self._set_process_run_status(run, "cancel_failed")
        self._append_process_run_activity(
            run,
            f"Cancellation timed out after {int(wait_timeout)}s.",
        )
        log_latency_event(
            logger,
            event="run_cancel_wait_timeout",
            duration_seconds=wait_timeout,
            fields={
                "run_id": str(getattr(run, "run_id", "")).strip(),
                "process": str(getattr(run, "process_name", "")).strip(),
                "run_cancel_path": cancel_path,
                "run_cancel_result": "timeout",
            },
        )
        self._update_process_run_visuals(run)
        self._refresh_process_run_progress(run)
        self._refresh_sidebar_progress_summary()
        events_panel.add_event(
            _now_str(),
            "process_err",
            f"{run.process_name} #{run.run_id} cancel timeout",
        )
        chat.add_info(
            f"[bold #e0af68]Cancel timed out[/] for run "
            f"[dim]{run.run_id}[/dim]. You can keep waiting or force-close the tab."
        )
        force_close = await self._confirm_force_close_process_run(
            run,
            timeout_seconds=wait_timeout,
        )
        if not force_close:
            run.close_after_cancel = False
            return False
        self._set_process_run_status(run, "force_closed")
        run.ended_at = time.monotonic()
        self._append_process_run_activity(
            run,
            "Force-close selected; tab closed while run may still be finishing.",
        )
        self._update_process_run_visuals(run)
        closed = await self._finalize_process_run_tab_close(
            run,
            tabs=tabs,
            cancel_worker=True,
        )
        log_latency_event(
            logger,
            event="run_force_close",
            duration_seconds=max(0.0, time.monotonic() - run.cancel_requested_at),
            fields={
                "run_id": str(getattr(run, "run_id", "")).strip(),
                "process": str(getattr(run, "process_name", "")).strip(),
                "run_cancel_path": cancel_path,
                "run_cancel_result": "force_closed",
            },
        )
        if closed:
            chat.add_info(
                f"[bold #e0af68]Force-closed[/] process run tab "
                f"[dim]{run.run_id}[/dim]."
            )
        return closed
    else:
        chat.add_info(f"Closed process run tab [dim]{run.run_id}[/dim].")
        return await self._finalize_process_run_tab_close(run, tabs=tabs)

async def _close_process_run_from_target(self, target: str) -> bool:
    """Resolve and close a process run from /run close target syntax."""
    chat = self.query_one("#chat-log", ChatLog)
    run, error = self._resolve_process_run_target(target)
    if run is None:
        if error:
            chat.add_info(error)
        return False
    return await self._close_process_run(run)

async def _pause_process_run(self, run: ProcessRunState) -> bool:
    """Pause an active process run without closing its tab."""
    chat = self.query_one("#chat-log", ChatLog)
    events_panel = self.query_one("#events-panel", EventPanel)
    if run.closed:
        chat.add_info(f"Run [dim]{run.run_id}[/dim] is already closed.")
        return False
    if run.status == "paused":
        chat.add_info(f"Run [dim]{run.run_id}[/dim] is already paused.")
        return True
    if run.status in {"completed", "failed", "cancelled", "force_closed"}:
        chat.add_info(f"Run [dim]{run.run_id}[/dim] is not active.")
        return False
    response = await self._request_process_run_pause(run)
    requested = bool(response.get("requested", False))
    path = str(response.get("path", "unknown")).strip() or "unknown"
    error = str(response.get("error", "")).strip()
    next_status = self._normalize_process_run_status(response.get("status"))
    if requested:
        self._set_process_run_status(run, next_status or "paused")
        self._append_process_run_activity(run, f"Pause requested via {path}.")
        chat.add_info(f"Paused process run [dim]{run.run_id}[/dim] via {path}.")
        events_panel.add_event(
            _now_str(),
            "process",
            f"{run.process_name} #{run.run_id} paused ({path})",
        )
    else:
        detail = error or "Pause request was not accepted."
        self._append_process_run_activity(run, f"Pause request failed via {path}: {detail}")
        chat.add_info(
            f"[bold #f7768e]Pause failed[/] for run "
            f"[dim]{run.run_id}[/dim] via {path}: {detail}"
        )
    self._update_process_run_visuals(run)
    self._refresh_process_run_progress(run)
    self._refresh_sidebar_progress_summary()
    await self._persist_process_run_ui_state()
    return requested

async def _play_process_run(self, run: ProcessRunState) -> bool:
    """Resume/play a paused process run without relaunching it."""
    chat = self.query_one("#chat-log", ChatLog)
    events_panel = self.query_one("#events-panel", EventPanel)
    if run.closed:
        chat.add_info(f"Run [dim]{run.run_id}[/dim] is already closed.")
        return False
    if run.status in {"completed", "failed", "cancelled", "force_closed"}:
        chat.add_info(f"Run [dim]{run.run_id}[/dim] is not active.")
        return False
    response = await self._request_process_run_play(run)
    requested = bool(response.get("requested", False))
    path = str(response.get("path", "unknown")).strip() or "unknown"
    error = str(response.get("error", "")).strip()
    next_status = self._normalize_process_run_status(response.get("status"))
    if requested:
        self._set_process_run_status(run, next_status or "running")
        self._append_process_run_activity(run, f"Play requested via {path}.")
        chat.add_info(f"Resumed process run [dim]{run.run_id}[/dim] via {path}.")
        events_panel.add_event(
            _now_str(),
            "process",
            f"{run.process_name} #{run.run_id} resumed ({path})",
        )
    elif (
        path == "none"
        and run.status == "paused"
        and getattr(run, "worker", None) is None
    ):
        # Restored paused runs have no live task-control callbacks after app restart.
        # Fall back to persisted task resume so Play works as expected.
        chat.add_info(
            f"No live play hook for run [dim]{run.run_id}[/dim]; "
            "resuming from persisted task state."
        )
        return await self._restart_process_run_in_place(run.run_id, mode="resume")
    else:
        detail = error or "Play request was not accepted."
        self._append_process_run_activity(run, f"Play request failed via {path}: {detail}")
        chat.add_info(
            f"[bold #f7768e]Play failed[/] for run "
            f"[dim]{run.run_id}[/dim] via {path}: {detail}"
        )
    self._update_process_run_visuals(run)
    self._refresh_process_run_progress(run)
    self._refresh_sidebar_progress_summary()
    await self._persist_process_run_ui_state()
    return requested

async def _inject_process_run(
    self,
    run: ProcessRunState,
    text: str,
    *,
    source: str = "slash",
    queue_if_unavailable: bool = True,
) -> bool:
    """Inject instruction into a process run, or queue until control hook is ready."""
    chat = self.query_one("#chat-log", ChatLog)
    clean = str(text or "").strip()
    if not clean:
        chat.add_info(self._render_slash_command_usage("/run inject", "<target> <text>"))
        return False
    if run.closed:
        chat.add_info(f"Run [dim]{run.run_id}[/dim] is already closed.")
        return False
    if run.status in {"completed", "failed", "cancelled", "force_closed"}:
        chat.add_info(f"Run [dim]{run.run_id}[/dim] is not active.")
        return False

    response = await self._request_process_run_inject(run, clean)
    requested = bool(response.get("requested", False))
    path = str(response.get("path", "unknown")).strip() or "unknown"
    error = str(response.get("error", "")).strip()
    if requested:
        self._append_process_run_activity(
            run,
            f"Inject ({source}) via {path}: {self._one_line(clean, 140)}",
        )
        chat.add_info(
            f"Injected into run [dim]{run.run_id}[/dim]: "
            f"{self._escape_markup(self._one_line(clean, 120))}"
        )
        self._update_process_run_visuals(run)
        self._refresh_process_run_progress(run)
        self._refresh_sidebar_progress_summary()
        await self._persist_process_run_ui_state()
        return True

    if queue_if_unavailable and path == "none":
        queue = self._process_run_pending_inject.setdefault(run.run_id, [])
        queue.append(clean)
        if len(queue) > 100:
            del queue[:-100]
        self._append_process_run_activity(
            run,
            f"Queued inject ({source}) pending task control: {self._one_line(clean, 140)}",
        )
        chat.add_info(
            f"Queued inject for run [dim]{run.run_id}[/dim]. "
            "It will apply when task control is ready."
        )
        self._update_process_run_visuals(run)
        self._refresh_process_run_progress(run)
        self._refresh_sidebar_progress_summary()
        await self._persist_process_run_ui_state()
        return True

    detail = error or "Inject request was not accepted."
    self._append_process_run_activity(run, f"Inject failed via {path}: {detail}")
    chat.add_info(
        f"[bold #f7768e]Inject failed[/] for run "
        f"[dim]{run.run_id}[/dim] via {path}: {detail}"
    )
    self._update_process_run_visuals(run)
    self._refresh_process_run_progress(run)
    self._refresh_sidebar_progress_summary()
    await self._persist_process_run_ui_state()
    return False

async def _stop_process_run(self, run: ProcessRunState, *, confirm: bool = False) -> bool:
    """Request cancellation for a process run while keeping its tab open."""
    chat = self.query_one("#chat-log", ChatLog)
    events_panel = self.query_one("#events-panel", EventPanel)
    if run.closed:
        chat.add_info(f"Run [dim]{run.run_id}[/dim] is already closed.")
        return False
    if run.status in {"completed", "failed", "cancelled", "force_closed"}:
        chat.add_info(f"Run [dim]{run.run_id}[/dim] is not active.")
        return False
    if run.status == "cancel_requested":
        chat.add_info(f"Cancellation already requested for run [dim]{run.run_id}[/dim].")
        return True
    if confirm and not await self._confirm_stop_process_run(run):
        return False

    prior_status = run.status
    self._set_process_run_status(run, "cancel_requested")
    run.cancel_requested_at = time.monotonic()
    self._append_process_run_activity(run, "Cancellation requested...")
    self._update_process_run_visuals(run)
    self._refresh_process_run_progress(run)
    self._refresh_sidebar_progress_summary()

    response = await self._request_process_run_cancellation(run)
    requested = bool(response.get("requested", False))
    path = str(response.get("path", "unknown")).strip() or "unknown"
    error = str(response.get("error", "")).strip()
    if not requested:
        self._set_process_run_status(run, prior_status)
        detail = error or "Cancel request was not accepted."
        self._append_process_run_activity(run, f"Cancel request failed via {path}: {detail}")
        chat.add_info(
            f"[bold #f7768e]Stop failed[/] for run "
            f"[dim]{run.run_id}[/dim] via {path}: {detail}"
        )
        self._update_process_run_visuals(run)
        self._refresh_process_run_progress(run)
        self._refresh_sidebar_progress_summary()
        await self._persist_process_run_ui_state()
        return False

    self._append_process_run_activity(run, f"Cancellation requested via {path}.")
    chat.add_info(
        f"Stop requested for process run [dim]{run.run_id}[/dim] via {path}."
    )
    events_panel.add_event(
        _now_str(),
        "process",
        f"{run.process_name} #{run.run_id} stop requested ({path})",
    )
    self._update_process_run_visuals(run)
    self._refresh_process_run_progress(run)
    self._refresh_sidebar_progress_summary()
    await self._persist_process_run_ui_state()
    return True

async def _pause_process_run_from_target(self, target: str) -> bool:
    """Resolve and pause a process run from /run pause target syntax."""
    chat = self.query_one("#chat-log", ChatLog)
    run, error = self._resolve_process_run_target(target)
    if run is None:
        if error:
            if error == "Multiple runs open. Use /run close <run-id-prefix>.":
                error = "Multiple runs open. Use /run pause <run-id-prefix>."
            chat.add_info(error)
        return False
    return await self._pause_process_run(run)

async def _play_process_run_from_target(self, target: str) -> bool:
    """Resolve and play/resume a process run from /run play target syntax."""
    chat = self.query_one("#chat-log", ChatLog)
    run, error = self._resolve_process_run_target(target)
    if run is None:
        if error:
            if error == "Multiple runs open. Use /run close <run-id-prefix>.":
                error = "Multiple runs open. Use /run play <run-id-prefix>."
            chat.add_info(error)
        return False
    return await self._play_process_run(run)

async def _stop_process_run_from_target(self, target: str) -> bool:
    """Resolve and stop a process run from /run stop target syntax."""
    chat = self.query_one("#chat-log", ChatLog)
    run, error = self._resolve_process_run_target(target)
    if run is None:
        if error:
            if error == "Multiple runs open. Use /run close <run-id-prefix>.":
                error = "Multiple runs open. Use /run stop <run-id-prefix>."
            chat.add_info(error)
        return False
    return await self._stop_process_run(run)

async def _inject_process_run_from_target(
    self,
    target: str,
    text: str,
    *,
    source: str = "slash",
) -> bool:
    """Resolve and inject into a process run from /run inject target syntax."""
    chat = self.query_one("#chat-log", ChatLog)
    run, error = self._resolve_process_run_target(target)
    if run is None:
        if error:
            if error == "Multiple runs open. Use /run close <run-id-prefix>.":
                error = "Multiple runs open. Use /run inject <run-id-prefix> <text>."
            chat.add_info(error)
        return False
    return await self._inject_process_run(run, text, source=source)

async def _resume_process_run_from_target(self, target: str) -> bool:
    """Resolve and resume a failed/cancelled process run from /run resume."""
    chat = self.query_one("#chat-log", ChatLog)
    run, error = self._resolve_process_run_target(target)
    if run is None:
        if error:
            if error == "Multiple runs open. Use /run close <run-id-prefix>.":
                error = "Multiple runs open. Use /run resume <run-id-prefix>."
            chat.add_info(error)
        return False

    chat.add_user_message(f"/run resume {target}")
    return await self._restart_process_run_in_place(run.run_id, mode="resume")

def _resume_seed_task_rows(run: ProcessRunState) -> tuple[list[dict], dict[str, str]]:
    """Clone prior task rows for resume; keep completed rows and reset the rest."""
    rows: list[dict] = []
    row_ids: set[str] = set()
    for item in getattr(run, "tasks", []):
        if not isinstance(item, dict):
            continue
        cloned = dict(item)
        status = str(cloned.get("status", "pending")).strip()
        if status != "completed":
            status = "pending"
        cloned["status"] = status
        subtask_id = str(cloned.get("id", "")).strip()
        if subtask_id:
            row_ids.add(subtask_id)
        rows.append(cloned)

    labels: dict[str, str] = {}
    raw_labels = getattr(run, "task_labels", {})
    if isinstance(raw_labels, dict):
        for key, value in raw_labels.items():
            subtask_id = str(key).strip()
            if not subtask_id:
                continue
            if row_ids and subtask_id not in row_ids:
                continue
            labels[subtask_id] = str(value)
    return rows, labels

async def _restart_process_run_in_place(
    self,
    run_id: str,
    *,
    mode: str = "restart",
) -> bool:
    """Restart one failed/cancelled run in the same tab."""
    run = self._process_runs.get(run_id)
    if run is None or run.closed:
        return False

    normalized_mode = str(mode or "").strip().lower()
    is_resume = normalized_mode == "resume"
    verb_denied = "resumed" if is_resume else "restarted"
    verb_ongoing = "Resuming" if is_resume else "Restarting"
    verb_done = "Resumed" if is_resume else "Restarted"
    event_action = "resumed" if is_resume else "restarted"

    chat = self.query_one("#chat-log", ChatLog)
    events_panel = self.query_one("#events-panel", EventPanel)
    run_key = str(getattr(run, "run_id", "")).strip()
    has_live_controls = any(
        callable(registry.get(run_key))
        for registry in (
            self._process_run_cancel_handlers,
            self._process_run_pause_handlers,
            self._process_run_play_handlers,
            self._process_run_inject_handlers,
        )
    )
    can_resume_persisted_paused = (
        is_resume
        and run.status == "paused"
        and getattr(run, "worker", None) is None
        and not has_live_controls
    )

    if self._is_process_run_active_status(run.status) and not can_resume_persisted_paused:
        chat.add_info(
            f"Run [dim]{run.run_id}[/dim] is already active and cannot be {verb_denied}."
        )
        return False
    if not run.task_id and is_resume:
        chat.add_info(
            f"Run [dim]{run.run_id}[/dim] has no task ID, so it cannot be resumed."
        )
        return False

    self._set_process_run_status(run, "queued")
    run.started_at = time.monotonic()
    run.ended_at = None
    run.launch_error = ""
    run.launch_stage = "queueing_delegate"
    run.launch_stage_started_at = time.monotonic()
    run.launch_last_progress_at = time.monotonic()
    run.launch_last_heartbeat_at = 0.0
    run.launch_stage_heartbeat_dots = 0
    run.launch_stage_heartbeat_stage = ""
    run.launch_stage_activity_indices = {}
    run.close_after_cancel = False
    run.cancel_requested_at = 0.0
    run.progress_ui_last_refresh_at = 0.0
    run.paused_started_at = 0.0
    run.paused_accumulated_seconds = 0.0
    self._clear_process_run_user_input_pause(run.run_id)
    self._clear_process_run_cancel_handler(run.run_id)
    self._process_run_pending_inject.pop(run.run_id, None)
    if run.task_id:
        run.tasks, run.task_labels = self._resume_seed_task_rows(run)
    else:
        run.tasks = []
        run.task_labels = {}
        run.subtask_phase_ids = {}
    row_ids = {
        str(row.get("id", "")).strip()
        for row in run.tasks
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }
    phase_map = getattr(run, "subtask_phase_ids", {})
    if isinstance(phase_map, dict):
        run.subtask_phase_ids = {
            str(subtask_id): str(phase_id)
            for subtask_id, phase_id in phase_map.items()
            if str(subtask_id).strip()
            and str(phase_id).strip()
            and (not row_ids or str(subtask_id) in row_ids)
        }
    else:
        run.subtask_phase_ids = {}
    run.last_progress_message = ""
    run.last_progress_at = 0.0
    self._update_process_run_visuals(run)
    self._refresh_process_run_progress(run)
    self._refresh_process_run_outputs(run)
    if run.task_id:
        self._append_process_run_activity(
            run,
            f"{verb_ongoing} in place from task state {run.task_id}.",
        )
        worker_coro = self._execute_process_run(run_id)
    else:
        self._append_process_run_activity(
            run,
            "Restarting in place from run goal (no prior task ID).",
        )
        launch_request = ProcessRunLaunchRequest(
            goal=str(getattr(run, "goal", "") or "").strip(),
            command_prefix="/run",
            process_defn=getattr(run, "process_defn", None),
            process_name_override=(
                ""
                if getattr(run, "process_defn", None) is not None
                else str(getattr(run, "process_name", "") or "").strip()
            ),
            is_adhoc=bool(getattr(run, "is_adhoc", False)),
            recommended_tools=list(getattr(run, "recommended_tools", []) or []),
            goal_context_overrides=dict(
                getattr(run, "goal_context_overrides", {}) or {},
            ),
            run_workspace_override=Path(getattr(run, "run_workspace", self._workspace)),
            resume_task_id="",
        )
        worker_coro = self._prepare_and_execute_process_run(run_id, launch_request)
    run.worker = self.run_worker(
        worker_coro,
        name=f"process-run-{run_id}",
        group=f"process-run-{run_id}",
        exclusive=False,
    )
    self._refresh_sidebar_progress_summary()
    chat.add_info(
        f"{verb_done} process run [dim]{run.run_id}[/dim] in place."
    )
    events_panel.add_event(
        _now_str(),
        "process_run",
        f"{run.process_name} #{run.run_id} {event_action}",
    )
    await self._persist_process_run_ui_state()
    return True
