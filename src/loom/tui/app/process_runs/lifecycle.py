"""Process-run launch and execution lifecycle helpers."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.widgets import TabbedContent, TabPane

from loom.tui.widgets import ChatLog, EventPanel
from loom.utils.latency import log_latency_event

from ..constants import _MAX_CONCURRENT_PROCESS_RUNS
from ..models import ProcessRunLaunchRequest, ProcessRunState
from ..widgets import ProcessRunPane
from . import state as process_run_state

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

logger = logging.getLogger(__name__)


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


async def start_process_run(
    self,
    goal: str,
    *,
    process_defn: ProcessDefinition | None = None,
    process_name_override: str | None = None,
    command_prefix: str = "/run",
    is_adhoc: bool = False,
    recommended_tools: list[str] | None = None,
    adhoc_synthesis_notes: list[str] | None = None,
    goal_context_overrides: dict[str, Any] | None = None,
    resume_task_id: str = "",
    run_workspace_override: Path | None = None,
    synthesis_goal: str = "",
    force_fresh: bool = False,
) -> None:
    """Create a run tab and launch process execution in a background worker."""
    submit_started = time.monotonic()
    chat = self.query_one("#chat-log", ChatLog)
    events_panel = self.query_one("#events-panel", EventPanel)

    delegate_ready, delegate_reason = self._ensure_delegate_task_ready_for_run()
    if not delegate_ready:
        detail = self._escape_markup(delegate_reason or "delegate_task unavailable.")
        chat.add_info(
            "[bold #f7768e]Process orchestration is unavailable in this "
            f"session.[/]\n[dim]{detail}[/]"
        )
        return

    if not self._tools.has("delegate_task"):
        chat.add_info(
            "[bold #f7768e]Process orchestration is unavailable in this "
            "session.[/]",
        )
        return

    active_runs = sum(
        1 for run in self._process_runs.values()
        if self._is_process_run_active_status(run.status)
    )
    if active_runs >= _MAX_CONCURRENT_PROCESS_RUNS:
        chat.add_info(
            f"[bold #f7768e]Too many active process runs "
            f"({_MAX_CONCURRENT_PROCESS_RUNS} max).[/]"
        )
        return

    selected_process = process_defn
    requested_process_name = str(process_name_override or "").strip()
    process_name = requested_process_name
    if not process_name and selected_process is not None:
        process_name = str(selected_process.name).strip()
    resolve_adhoc = bool(is_adhoc)
    if selected_process is None and not process_name and not resume_task_id:
        resolve_adhoc = True
    if not process_name:
        process_name = "adhoc-process-run" if resolve_adhoc else "process-run"

    launch_request = ProcessRunLaunchRequest(
        goal=goal,
        command_prefix=command_prefix,
        process_defn=selected_process,
        process_name_override=requested_process_name,
        is_adhoc=resolve_adhoc,
        recommended_tools=list(recommended_tools or []),
        adhoc_synthesis_notes=list(adhoc_synthesis_notes or []),
        goal_context_overrides=dict(goal_context_overrides or {}),
        resume_task_id=str(resume_task_id or "").strip(),
        run_workspace_override=(
            Path(run_workspace_override).expanduser()
            if run_workspace_override is not None
            else None
        ),
        synthesis_goal=str(synthesis_goal or "").strip(),
        force_fresh=bool(force_fresh),
    )

    run_id = self._new_process_run_id()
    pane_id = f"tab-run-{run_id}"
    pane = ProcessRunPane(
        run_id=run_id,
        process_name=process_name,
        goal=goal,
    )
    run = ProcessRunState(
        run_id=run_id,
        process_name=process_name,
        goal=goal,
        run_workspace=self._workspace,
        process_defn=selected_process,
        pane_id=pane_id,
        pane=pane,
        status="queued",
        task_id=launch_request.resume_task_id,
        started_at=time.monotonic(),
        is_adhoc=resolve_adhoc,
        recommended_tools=list(launch_request.recommended_tools),
        goal_context_overrides=dict(launch_request.goal_context_overrides),
    )
    self._process_runs[run_id] = run

    tabs = self.query_one("#tabs", TabbedContent)
    await tabs.add_pane(
        TabPane(
            self._format_process_run_tab_title(run),
            pane,
            id=pane_id,
        ),
        after="tab-events",
    )
    tabs.active = pane_id
    run.launch_tab_created_at = time.monotonic()
    self._update_process_run_visuals(run)
    self._refresh_process_run_progress(run)
    self._refresh_process_run_outputs(run)
    self._refresh_sidebar_progress_summary()
    submit_to_tab = max(0.0, run.launch_tab_created_at - submit_started)
    log_latency_event(
        logger,
        event="run_submit_to_tab",
        duration_seconds=submit_to_tab,
        fields={
            "run_id": run_id,
            "process": process_name,
            "run_submit_to_tab_ms": int(submit_to_tab * 1000),
        },
    )
    self._set_process_run_launch_stage(
        run,
        "accepted",
        note="Accepted /run request.",
    )

    chat.add_user_message(f"{command_prefix} {goal}")
    chat.add_info(
        f"Started process run [dim]{run_id}[/dim] "
        f"([bold]{process_name}[/bold])."
    )
    self._append_process_run_activity(run, "Queued process run.")
    if not self._process_close_hint_shown:
        chat.add_info(
            "[dim]Tip: close tabs with ctrl + w, /run close "
            "[run-id-prefix], or ctrl + p -> Close tab.[/]"
        )
        self._process_close_hint_shown = True
    events_panel.add_event(
        _now_str(),
        "process_run",
        f"{process_name} #{run_id}: {goal[:120]}",
    )

    if self._tui_run_preflight_async_enabled():
        run.worker = self.run_worker(
            self._prepare_and_execute_process_run(run_id, launch_request),
            name=f"process-run-{run_id}",
            group=f"process-run-{run_id}",
            exclusive=False,
        )
    else:
        self._append_process_run_activity(
            run,
            "Run launch mode: inline preflight (run_preflight_async_enabled=false).",
        )
        prepared = await self._prepare_process_run_with_timeout(run_id, launch_request)
        if prepared and not run.closed:
            run.worker = self.run_worker(
                self._execute_process_run(run_id),
                name=f"process-run-{run_id}",
                group=f"process-run-{run_id}",
                exclusive=False,
            )
        else:
            run.worker = None
    await self._persist_process_run_ui_state()


async def prepare_process_run_with_timeout(
    self,
    run_id: str,
    launch_request: ProcessRunLaunchRequest,
) -> bool:
    """Run preflight with timeout/error normalization."""
    launch_timeout = self._tui_run_launch_timeout_seconds()
    launch_task = asyncio.create_task(
        self._prepare_process_run_launch(run_id, launch_request),
    )
    launch_started_at = time.monotonic()
    try:
        while not launch_task.done():
            now = time.monotonic()
            run = self._process_runs.get(run_id)
            paused_seconds = self._process_run_user_input_paused_seconds(run_id, now=now)
            if run is not None:
                paused_seconds += process_run_state.status_paused_seconds_for_run(
                    run,
                    now=now,
                )
            elapsed = max(0.0, now - launch_started_at - paused_seconds)
            if elapsed >= launch_timeout:
                launch_task.cancel()
                try:
                    await launch_task
                except asyncio.CancelledError:
                    pass
                run = self._process_runs.get(run_id)
                if run is not None:
                    stage = self._process_run_launch_stage_label(run.launch_stage).lower()
                    self._fail_process_run_launch(
                        run,
                        (
                            f"Run launch timed out after {int(launch_timeout)}s "
                            f"during {stage}."
                        ),
                    )
                return False
            await asyncio.sleep(min(0.05, max(0.01, launch_timeout - elapsed)))
        return bool(await launch_task)
    except TimeoutError:
        run = self._process_runs.get(run_id)
        if run is not None:
            stage = self._process_run_launch_stage_label(run.launch_stage).lower()
            self._fail_process_run_launch(
                run,
                (
                    f"Run launch timed out after {int(launch_timeout)}s "
                    f"during {stage}."
                ),
        )
        return False
    except asyncio.CancelledError:
        launch_task.cancel()
        raise
    except Exception as e:  # pragma: no cover - defensive guard
        run = self._process_runs.get(run_id)
        if run is not None:
            self._fail_process_run_launch(run, str(e))
        return False


async def prepare_and_execute_process_run(
    self,
    run_id: str,
    launch_request: ProcessRunLaunchRequest,
) -> None:
    """Resolve launch prerequisites, then execute the delegated run."""
    run = self._process_runs.get(run_id)
    if run is None or run.closed:
        return
    needs_cleanup = True
    prepared = False
    try:
        prepared = await self._prepare_process_run_with_timeout(run_id, launch_request)
        if not prepared:
            return
        await self._execute_process_run(run_id)
        needs_cleanup = False
    finally:
        if needs_cleanup:
            run = self._process_runs.get(run_id)
            if run is not None:
                run.worker = None
                await self._persist_process_run_ui_state()


async def prepare_process_run_launch(
    self,
    run_id: str,
    launch_request: ProcessRunLaunchRequest,
) -> bool:
    """Resolve process/workspace/auth preflight before delegate execution."""
    run = self._process_runs.get(run_id)
    if run is None or run.closed:
        return False

    selected_process = launch_request.process_defn
    process_name = str(launch_request.process_name_override or "").strip()
    synthesized_now = False

    self._set_process_run_launch_stage(
        run,
        "resolving_process",
        note="Resolving process configuration...",
    )
    if selected_process is None and process_name:
        loader = self._create_process_loader()
        try:
            selected_process = loader.load(process_name)
        except Exception as e:
            self._fail_process_run_launch(
                run,
                f"Failed to load process '{process_name}': {e}",
            )
            return False
    if selected_process is None and launch_request.is_adhoc:
        synthesis_goal = str(launch_request.synthesis_goal or "").strip() or launch_request.goal
        self._set_process_run_launch_stage(
            run,
            "resolving_process",
            note="Synthesizing ad hoc process for /run goal...",
        )
        try:
            entry, from_cache = await self._get_or_create_adhoc_process(
                synthesis_goal,
                fresh=bool(launch_request.force_fresh),
            )
        except Exception as e:
            self._fail_process_run_launch(run, f"Ad hoc synthesis failed: {e}")
            return False
        if run.closed:
            return False
        synthesized_now = True
        selected_process = entry.process_defn
        process_name = str(getattr(selected_process, "name", "") or process_name).strip()
        run.is_adhoc = True
        run.recommended_tools = list(getattr(entry, "recommended_tools", []) or [])
        if from_cache:
            self._append_process_run_activity(
                run,
                f"Using cached ad hoc process {process_name}.",
            )
        else:
            self._append_process_run_activity(
                run,
                (
                    f"Synthesized ad hoc process {process_name} "
                    f"with {len(getattr(selected_process, 'phases', []) or [])} phases."
                ),
            )
        if run.recommended_tools:
            self._append_process_run_activity(
                run,
                "Recommended extra tools: " + ", ".join(sorted(run.recommended_tools)),
            )
        cache_key = str(
            getattr(entry, "key", "") or self._adhoc_cache_key(synthesis_goal),
        )
        self._append_process_run_activity(
            run,
            f"Ad hoc process cache: {self._adhoc_cache_path(cache_key)}",
        )
        adhoc_notes = self._adhoc_synthesis_activity_lines(
            entry,
            from_cache=from_cache,
            fresh=bool(launch_request.force_fresh),
        )
        for note in adhoc_notes:
            self._append_process_run_activity(run, note)
    elif selected_process is None and not run.task_id:
        self._fail_process_run_launch(
            run,
            "Unable to resolve process for /run.",
        )
        return False

    if selected_process is not None:
        process_name = str(getattr(selected_process, "name", "") or process_name).strip()
        if process_name and self._is_reserved_process_name(process_name):
            self._fail_process_run_launch(
                run,
                (
                    f"Process '{process_name}' conflicts with a built-in slash command "
                    "and cannot be loaded in TUI."
                ),
            )
            return False
    if not process_name:
        process_name = "process-run"
    run.process_name = process_name
    run.process_defn = selected_process
    run.goal_context_overrides = dict(launch_request.goal_context_overrides)
    if run.is_adhoc and not synthesized_now:
        self._append_process_run_activity(run, "Run mode: synthesized ad hoc process.")
        for note in list(launch_request.adhoc_synthesis_notes):
            self._append_process_run_activity(run, note)
        if run.recommended_tools:
            self._append_process_run_activity(
                run,
                "Recommended extra tools: " + ", ".join(sorted(run.recommended_tools)),
            )
    self._update_process_run_visuals(run)

    self._set_process_run_launch_stage(
        run,
        "provisioning_workspace",
        note="Provisioning run workspace...",
    )
    if launch_request.run_workspace_override is not None:
        run_workspace = Path(launch_request.run_workspace_override).expanduser()
    else:
        run_workspace = await self._choose_process_run_workspace(
            run.run_id,
            process_name,
            run.goal,
        )
        if run_workspace is None:
            self._fail_process_run_launch(
                run,
                "Run cancelled: working folder selection cancelled.",
            )
            return False
    if run.closed:
        return False
    run.run_workspace = run_workspace
    self._append_process_run_activity(run, f"Run workspace: {run_workspace}")
    self._request_workspace_refresh("run-workspace-created", immediate=True)
    try:
        rel_workspace = str(run_workspace.resolve().relative_to(self._workspace.resolve()))
    except Exception:
        rel_workspace = ""
    if rel_workspace and rel_workspace not in {".", "./"}:
        self._ingest_files_panel_from_paths([rel_workspace], operation_hint="create")

    self._set_process_run_launch_stage(
        run,
        "auth_preflight",
        note="Resolving auth preflight...",
    )
    run_auth_overrides = dict(self._run_auth_profile_overrides)
    resolved_auth_overrides, required_auth_resources = (
        await self._resolve_auth_overrides_for_run_start(
            process_defn=selected_process,
            base_overrides=run_auth_overrides,
            run_id=run.run_id,
        )
    )
    if run.closed:
        return False
    if resolved_auth_overrides is None:
        self._fail_process_run_launch(
            run,
            "Run cancelled: unresolved auth requirements.",
        )
        return False
    run.auth_profile_overrides = dict(resolved_auth_overrides)
    run.auth_required_resources = list(required_auth_resources)
    if run.auth_profile_overrides:
        rendered = ", ".join(
            f"{selector}={profile_id}"
            for selector, profile_id in sorted(run.auth_profile_overrides.items())
        )
        self._append_process_run_activity(run, f"Run auth overrides: {rendered}")
    if run.task_id:
        self._append_process_run_activity(
            run,
            f"Resuming task state: {run.task_id}",
        )

    self._set_process_run_launch_stage(
        run,
        "queueing_delegate",
        note="Queueing delegate task...",
    )
    self._refresh_process_run_progress(run)
    return True


async def execute_process_run(self, run_id: str) -> None:
    """Execute one process run and stream updates into its dedicated tab."""
    run = self._process_runs.get(run_id)
    if run is None:
        return
    if run.closed:
        return
    chat = self.query_one("#chat-log", ChatLog)
    events_panel = self.query_one("#events-panel", EventPanel)

    self._set_process_run_status(run, "running")
    run.started_at = time.monotonic()
    run.ended_at = None
    run.launch_error = ""
    run.close_after_cancel = False
    run.cancel_requested_at = 0.0
    run.progress_ui_last_refresh_at = 0.0
    run.paused_started_at = 0.0
    run.paused_accumulated_seconds = 0.0
    self._clear_process_run_user_input_pause(run_id)
    self._clear_process_run_cancel_handler(run_id)
    tab_created_at = float(getattr(run, "launch_tab_created_at", 0.0) or 0.0)
    if tab_created_at > 0:
        tab_to_delegate = max(0.0, run.started_at - tab_created_at)
        log_latency_event(
            logger,
            event="run_tab_to_delegate_start",
            duration_seconds=tab_to_delegate,
            fields={
                "run_id": str(getattr(run, "run_id", "")).strip(),
                "process": str(getattr(run, "process_name", "")).strip(),
                "run_tab_to_delegate_start_ms": int(tab_to_delegate * 1000),
            },
        )
    self._set_process_run_launch_stage(
        run,
        "running",
        note="Run started.",
    )

    try:
        run_context = self._build_process_run_context(
            run.goal,
            workspace=run.run_workspace,
        )
        extra_context = getattr(run, "goal_context_overrides", {})
        if isinstance(extra_context, dict) and extra_context:
            run_context.update(extra_context)
        result = await self._tools.execute(
            "delegate_task",
            {
                "goal": run.goal,
                "context": run_context,
                "wait": True,
                "_approval_mode": "disabled",
                "_process_override": run.process_defn,
                "_read_roots": [str(self._workspace.resolve())],
                "_auth_profile_overrides": dict(
                    getattr(run, "auth_profile_overrides", self._run_auth_profile_overrides),
                ),
                "_auth_required_resources": list(
                    getattr(run, "auth_required_resources", []),
                ),
                "_auth_workspace": str(self._workspace.resolve()),
                "_execution_surface": "tui",
                "_resume_task_id": str(run.task_id or "").strip(),
                "_progress_callback": (
                    lambda data, rid=run_id: self._on_process_progress_event(
                        data, run_id=rid,
                    )
                ),
                "_register_cancel_handler": (
                    lambda payload, rid=run_id: self._register_process_run_cancel_handler(
                        rid,
                        payload,
                    )
                ),
                "_clear_cancel_handler": (
                    lambda rid=run_id: self._clear_process_run_cancel_handler(rid)
                ),
            },
            workspace=run.run_workspace,
        )
        data = getattr(result, "data", None)
        if run.closed:
            return
        if isinstance(data, dict):
            run.task_id = str(data.get("task_id", "") or run.task_id)
            if run.task_id:
                self._persist_process_run_conversation_link(run)
            event_log_path = str(data.get("event_log_path", "")).strip()
            if event_log_path:
                self._append_process_run_activity(
                    run,
                    f"Detailed log: {event_log_path}",
                )
            tasks = data.get("tasks", [])
            if isinstance(tasks, list):
                normalized = self._normalize_process_run_tasks(run, tasks)
                run.tasks = normalized
                self._refresh_process_run_progress(run)
                self._refresh_process_run_outputs(run)
        delegated_status = ""
        if isinstance(data, dict):
            delegated_status = str(data.get("status", "")).strip().lower()
        delegated_failed = delegated_status == "failed"
        delegated_cancelled = delegated_status == "cancelled"
        run_succeeded = (
            bool(result.success)
            and not delegated_failed
            and not delegated_cancelled
        )

        if run_succeeded:
            output = result.output or "Process run completed."
            self._append_process_run_result(run, output, success=True)
            self._set_process_run_status(run, "completed")
            run.ended_at = time.monotonic()
            run.launch_error = ""
            self._log_terminal_stage_duration(run, terminal_state="completed")
            run.launch_last_progress_at = time.monotonic()
            self._refresh_process_run_progress(run)
            self._update_process_run_visuals(run)
            self._refresh_sidebar_progress_summary()
            events_panel.add_event(
                _now_str(), "process_ok", f"{run.process_name} #{run.run_id}",
            )
            chat.add_info(f"Process run [dim]{run.run_id}[/dim] completed.")
        elif delegated_cancelled:
            detail = result.output or "Process run cancelled."
            self._set_process_run_status(run, "cancelled")
            run.ended_at = time.monotonic()
            run.launch_error = "Process run cancelled."
            self._log_terminal_stage_duration(run, terminal_state="cancelled")
            run.launch_last_progress_at = time.monotonic()
            self._append_process_run_result(run, detail, success=False)
            self._refresh_process_run_progress(run)
            self._update_process_run_visuals(run)
            self._refresh_sidebar_progress_summary()
            events_panel.add_event(
                _now_str(), "process_err", f"{run.process_name} #{run.run_id}",
            )
            chat.add_info(f"[bold #f7768e]Process run {run.run_id} cancelled.[/]")
        else:
            if result.success and delegated_failed:
                detail = result.output or "Process run failed."
                error = "Process run failed."
                self._append_process_run_result(run, detail, success=False)
            else:
                error = result.error or result.output or "Process run failed."
                self._append_process_run_result(run, error, success=False)
            self._set_process_run_status(run, "failed")
            run.ended_at = time.monotonic()
            run.launch_error = str(error)
            self._log_terminal_stage_duration(run, terminal_state="failed")
            run.launch_last_progress_at = time.monotonic()
            self._refresh_process_run_progress(run)
            self._update_process_run_visuals(run)
            self._refresh_sidebar_progress_summary()
            events_panel.add_event(
                _now_str(), "process_err", f"{run.process_name} #{run.run_id}",
            )
            chat.add_info(
                f"[bold #f7768e]Process run {run.run_id} failed:[/] {error}"
            )
            self.notify(error, severity="error", timeout=5)
        self._request_workspace_refresh("process-run-finished")
    except asyncio.CancelledError:
        if run.closed:
            return
        if run.status == "paused":
            # Preserve paused state on worker cancellation (for example app
            # shutdown) so the run can be resumed after restart.
            run.launch_error = ""
            self._refresh_process_run_progress(run)
            self._update_process_run_visuals(run)
            self._refresh_sidebar_progress_summary()
            raise
        self._set_process_run_status(run, "cancelled")
        run.ended_at = time.monotonic()
        run.launch_error = "Process run cancelled."
        self._log_terminal_stage_duration(run, terminal_state="cancelled")
        self._append_process_run_result(run, "Process run cancelled.", success=False)
        self._refresh_process_run_progress(run)
        self._update_process_run_visuals(run)
        self._refresh_sidebar_progress_summary()
        events_panel.add_event(
            _now_str(), "process_err", f"{run.process_name} #{run.run_id}",
        )
        chat.add_info(f"[bold #f7768e]Process run {run.run_id} cancelled.[/]")
        raise
    except Exception as e:  # pragma: no cover - defensive guard
        self._append_process_run_result(run, str(e), success=False)
        self._set_process_run_status(run, "failed")
        run.ended_at = time.monotonic()
        run.launch_error = str(e)
        self._log_terminal_stage_duration(run, terminal_state="failed")
        self._refresh_process_run_progress(run)
        self._update_process_run_visuals(run)
        self._refresh_sidebar_progress_summary()
        events_panel.add_event(
            _now_str(), "process_err", f"{run.process_name} #{run.run_id}",
        )
        chat.add_info(f"[bold #f7768e]Process run {run.run_id} failed:[/] {e}")
        self.notify(str(e), severity="error", timeout=5)
    finally:
        if run.status in {"completed", "failed", "cancelled", "force_closed", "cancel_failed"}:
            self._process_run_pending_inject.pop(run_id, None)
        run.worker = None
        self._clear_process_run_cancel_handler(run_id)
        await self._persist_process_run_ui_state()
