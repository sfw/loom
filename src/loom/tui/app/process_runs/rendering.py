"""Process-run task/output rendering and sidebar summary helpers."""

from __future__ import annotations

from pathlib import Path

from loom.processes.phase_alignment import infer_phase_id_for_subtask
from loom.tui.widgets import Sidebar

from .. import rendering as app_rendering
from ..constants import _PROCESS_STATUS_LABEL, _plain_text
from ..models import ProcessRunState
from . import events as process_run_events


def _one_line(text: object | None, max_len: int | None = 180) -> str:
    """Normalize whitespace and cap a string for concise progress rows."""
    return app_rendering.one_line(text, max_len=max_len, plain_text=_plain_text)

def _normalize_process_run_tasks(
    self, run: ProcessRunState, tasks: list[dict]
) -> list[dict]:
    """Keep process task rows stable and focused on the original plan labels."""
    phase_labels: dict[str, str] = {}
    process = getattr(run, "process_defn", None)
    if process is not None:
        for phase in getattr(process, "phases", []):
            phase_id = str(getattr(phase, "id", "")).strip()
            if not phase_id:
                continue
            phase_desc = self._one_line(
                getattr(phase, "description", ""),
                max_len=None,
            )
            phase_labels[phase_id] = phase_desc or phase_id

    task_labels = getattr(run, "task_labels", None)
    if not isinstance(task_labels, dict):
        task_labels = {}
        try:
            run.task_labels = task_labels
        except Exception:
            pass
    phase_map = getattr(run, "subtask_phase_ids", None)
    if not isinstance(phase_map, dict):
        phase_map = {}
        try:
            run.subtask_phase_ids = phase_map
        except Exception:
            pass

    normalized: list[dict] = []
    for row in tasks:
        if not isinstance(row, dict):
            continue
        subtask_id = str(row.get("id", "")).strip()
        raw_status = str(row.get("status", "pending")).strip()
        status = raw_status if raw_status in {
            "pending", "in_progress", "completed", "failed", "skipped",
        } else "pending"
        candidate = self._one_line(row.get("content", ""), max_len=None)
        if not candidate:
            candidate = subtask_id or "subtask"

        if subtask_id in phase_labels:
            label = phase_labels[subtask_id]
            task_labels[subtask_id] = label
            phase_map[subtask_id] = subtask_id
        else:
            if subtask_id:
                existing = str(task_labels.get(subtask_id, "")).strip()
                # Only improve labels while task is active; don't let
                # completion summaries replace the original checklist label.
                if (
                    status in {"pending", "in_progress"}
                    and candidate
                    and (not existing or existing == subtask_id)
                ):
                    task_labels[subtask_id] = candidate
                label = str(task_labels.get(subtask_id, "")).strip() or candidate
            else:
                label = candidate

        row_payload = {
            "id": subtask_id or candidate,
            "status": status,
            "content": label,
        }
        row_phase_id = str(row.get("phase_id", "")).strip()
        if not row_phase_id and subtask_id:
            row_phase_id = str(phase_map.get(subtask_id, "")).strip()
        if row_phase_id:
            row_payload["phase_id"] = row_phase_id
            if subtask_id:
                phase_map[subtask_id] = row_phase_id

        normalized.append(row_payload)
    return normalized

def _process_run_output_rows(self, run: ProcessRunState) -> list[dict]:
    """Build per-deliverable output status rows for the process run pane."""
    process = getattr(run, "process_defn", None)
    if process is None or not hasattr(process, "get_deliverables"):
        return []

    try:
        deliverables_by_phase = process.get_deliverables()
    except Exception:
        deliverables_by_phase = {}

    has_deliverables = (
        isinstance(deliverables_by_phase, dict)
        and bool(deliverables_by_phase)
    )

    if bool(getattr(run, "is_adhoc", False)) and not has_deliverables:
        rows: list[dict] = []
        for idx, task in enumerate(run.tasks, start=1):
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "pending")).strip()
            if status not in {
                "pending", "in_progress", "completed", "failed", "skipped",
            }:
                status = "pending"
            label = self._one_line(task.get("content", ""), max_len=None)
            if not label:
                label = str(task.get("id", "")).strip() or f"step-{idx}"
            rows.append({
                "id": f"adhoc-output-{idx}",
                "status": status,
                "content": f"{label} (expected output)",
            })
        return rows

    if not has_deliverables:
        return []

    ordered_phase_ids: list[str] = []
    phase_labels: dict[str, str] = {}
    for phase in getattr(process, "phases", []):
        phase_id = str(getattr(phase, "id", "")).strip()
        if phase_id:
            ordered_phase_ids.append(phase_id)
            phase_label = self._one_line(
                getattr(phase, "description", ""),
                max_len=None,
            )
            if phase_label:
                phase_labels[phase_id] = phase_label
    for phase_id in deliverables_by_phase.keys():
        if phase_id not in ordered_phase_ids:
            ordered_phase_ids.append(phase_id)

    run_workspace = getattr(run, "run_workspace", None)
    workspace_root = Path(run_workspace) if run_workspace else self._workspace

    phase_statuses: dict[str, list[str]] = {phase_id: [] for phase_id in ordered_phase_ids}
    phase_map = self._process_run_phase_map(run)
    for row in getattr(run, "tasks", []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "pending")).strip()
        if status not in {"pending", "in_progress", "completed", "failed", "skipped"}:
            status = "pending"
        phase_id = self._infer_process_run_task_phase_id(
            run,
            row=row,
            phase_ids=ordered_phase_ids,
            phase_labels=phase_labels,
            deliverables_by_phase=deliverables_by_phase,
            phase_map=phase_map,
        )
        if phase_id and phase_id in phase_statuses:
            phase_statuses[phase_id].append(status)

    rows: list[dict] = []
    for phase_id in ordered_phase_ids:
        phase_deliverables = deliverables_by_phase.get(phase_id) or []
        if not isinstance(phase_deliverables, list):
            continue
        phase_state = self._aggregate_phase_state(phase_statuses.get(phase_id, []))
        for path in phase_deliverables:
            rel_path = str(path).strip()
            if not rel_path:
                continue
            exists = (workspace_root / rel_path).exists()
            if exists:
                status = "completed"
                suffix = ""
            elif phase_state == "in_progress":
                status = "in_progress"
                suffix = " (pending)"
            elif phase_state == "completed":
                status = "failed"
                suffix = " (missing)"
            elif phase_state == "failed":
                status = "failed"
                suffix = " (not produced)"
            elif phase_state == "skipped":
                status = "skipped"
                suffix = " (skipped)"
            else:
                status = "pending"
                suffix = " (planned)"
            rows.append({
                "id": f"{phase_id}:{rel_path}",
                "status": status,
                "content": f"{rel_path} ({phase_id}){suffix}",
            })
    return rows

def _process_run_phase_map(self, run: ProcessRunState) -> dict[str, str]:
    """Return mutable subtask->phase map for the run, creating if missing."""
    phase_map = getattr(run, "subtask_phase_ids", None)
    if isinstance(phase_map, dict):
        return phase_map
    phase_map = {}
    try:
        run.subtask_phase_ids = phase_map
    except Exception:
        pass
    return phase_map

def _aggregate_phase_state(statuses: list[str]) -> str:
    """Aggregate multiple subtask states into one phase-level state."""
    return app_rendering.aggregate_phase_state(statuses)

def _infer_process_run_task_phase_id(
    self,
    run: ProcessRunState,
    *,
    row: dict,
    phase_ids: list[str],
    phase_labels: dict[str, str],
    deliverables_by_phase: dict[str, list[str]],
    phase_map: dict[str, str],
) -> str:
    """Infer phase ID for one task row, preserving stable mappings."""
    phase_set = set(phase_ids)
    subtask_id = str(row.get("id", "")).strip()
    explicit_phase_id = str(row.get("phase_id", "")).strip()
    if explicit_phase_id in phase_set:
        if subtask_id:
            phase_map[subtask_id] = explicit_phase_id
        return explicit_phase_id

    if subtask_id in phase_set:
        phase_map[subtask_id] = subtask_id
        return subtask_id

    if subtask_id:
        existing = str(phase_map.get(subtask_id, "")).strip()
        if existing in phase_set:
            return existing

    content = self._one_line(row.get("content", ""), max_len=None)
    label = ""
    task_labels = getattr(run, "task_labels", {})
    if subtask_id and isinstance(task_labels, dict):
        label = self._one_line(task_labels.get(subtask_id, ""), max_len=None)
    text = " ".join(part for part in [label, content] if part).strip()
    inferred = infer_phase_id_for_subtask(
        subtask_id=subtask_id,
        text=text,
        phase_ids=phase_ids,
        phase_descriptions=phase_labels,
        phase_deliverables=deliverables_by_phase,
    )
    if inferred in phase_set and subtask_id:
        phase_map[subtask_id] = inferred
    return inferred if inferred in phase_set else ""

def _refresh_process_run_outputs(self, run: ProcessRunState) -> None:
    """Refresh per-run output rows in the process pane."""
    if not hasattr(run, "pane") or run.pane is None:
        return
    try:
        rows = self._process_run_output_rows(run)
        run.pane.set_outputs(rows)
    except Exception:
        return

def _subtask_content(
    data: dict,
    subtask_id: str,
    run: ProcessRunState | None = None,
) -> str:
    """Lookup subtask label, preferring stable run-normalized labels."""
    return process_run_events.subtask_content(
        data,
        subtask_id,
        run=run,
        plain_text=_plain_text,
    )

def _format_process_progress_event(
    self,
    data: dict,
    *,
    run: ProcessRunState | None = None,
    context: str = "process_run",
) -> str | None:
    """Format orchestrator progress events into concise chat messages."""
    return process_run_events.format_process_progress_event(
        self,
        data,
        run=run,
        context=context,
    )

def _mark_process_run_failed(self, error: str) -> None:
    """Reflect a failed /run execution in the progress panel."""
    message = error.strip() or "Process run failed."
    if "timed out" in message.lower():
        message = (
            f"{message} Increase \\[execution].delegate_task_timeout_seconds "
            "(or LOOM_DELEGATE_TIMEOUT_SECONDS) for longer runs."
        )
    self._sidebar_cowork_tasks = [
        {
            "id": "process-run",
            "status": "failed",
            "content": f"/run failed: {message}",
        },
    ]
    self._refresh_sidebar_progress_summary()


def _update_sidebar_tasks(self, data: dict) -> None:
    """Update sidebar task progress from a tool result payload."""
    if not data:
        return
    if not isinstance(data, dict):
        return
    tasks = data.get("tasks", [])
    if not tasks and "id" in data:
        tasks = [data]
    normalized: list[dict] = []
    if isinstance(tasks, list):
        for row in tasks:
            if isinstance(row, dict):
                normalized.append(row)
    self._sidebar_cowork_tasks = normalized
    self._refresh_sidebar_progress_summary()

def _summarize_cowork_tasks(self) -> list[dict]:
    """Return a compact one-row summary for cowork delegated tasks."""
    if not self._sidebar_cowork_tasks:
        return []
    total = len(self._sidebar_cowork_tasks)
    in_progress = sum(
        1
        for t in self._sidebar_cowork_tasks
        if str(t.get("status", "")) == "in_progress"
    )
    failed = sum(
        1
        for t in self._sidebar_cowork_tasks
        if str(t.get("status", "")) == "failed"
    )
    completed = sum(
        1
        for t in self._sidebar_cowork_tasks
        if str(t.get("status", "")) == "completed"
    )
    primary = next(
        (
            t
            for t in self._sidebar_cowork_tasks
            if str(t.get("status", "")) == "in_progress"
        ),
        self._sidebar_cowork_tasks[0],
    )
    focus = self._one_line(primary.get("content", ""), 180)
    status = "pending"
    if in_progress:
        status = "in_progress"
    elif failed:
        status = "failed"
    elif completed and completed == total:
        status = "completed"
    content = (
        f"Cowork delegated: {total} task(s) "
        f"({in_progress} active, {failed} failed)"
    )
    if focus:
        content += f" - {focus}"
    return [{
        "id": "cowork-delegated",
        "status": status,
        "content": content,
    }]


def _refresh_sidebar_progress_summary(self) -> None:
    """Render concise sidebar progress: one row per run + cowork summary."""
    try:
        sidebar = self.query_one("#sidebar", Sidebar)
    except Exception:
        return
    rows: list[dict] = []
    for run in sorted(self._process_runs.values(), key=lambda r: r.started_at):
        state = run.status
        row_status = {
            "queued": "pending",
            "running": "in_progress",
            "paused": "in_progress",
            "cancel_requested": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "cancel_failed": "failed",
            "force_closed": "skipped",
            "cancelled": "skipped",
        }.get(state, "pending")
        elapsed = self._format_elapsed(self._elapsed_seconds_for_run(run))
        label = _PROCESS_STATUS_LABEL.get(state, state.title())
        rows.append({
            "id": f"process-run-{run.run_id}",
            "status": row_status,
            "content": f"{run.process_name} #{run.run_id} {label} {elapsed}",
        })
    rows.extend(self._summarize_cowork_tasks())
    sidebar.update_tasks(rows)
    self._sync_process_runs_into_session_state()
