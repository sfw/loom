"""Pure process-run state helpers extracted from the TUI core."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as MM:SS or H:MM:SS."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def elapsed_seconds_for_run(run: Any, *, now: float | None = None) -> float:
    """Return elapsed seconds for a run (live or finalized)."""
    end_now = float(now) if now is not None else time.monotonic()
    end = getattr(run, "ended_at", None)
    end_value = end_now if end is None else float(end)
    elapsed = max(0.0, end_value - float(getattr(run, "started_at", 0.0) or 0.0))
    paused_total = max(
        0.0,
        float(getattr(run, "paused_accumulated_seconds", 0.0) or 0.0),
    )
    paused_started = float(getattr(run, "paused_started_at", 0.0) or 0.0)
    if paused_started > 0.0:
        paused_total += max(0.0, end_value - paused_started)
    return max(0.0, elapsed - paused_total)


def is_process_run_busy_status(status: str) -> bool:
    """Return True while a run is actively consuming execution resources."""
    state = str(status or "").strip().lower()
    return state in {"queued", "running", "cancel_requested"}


def set_process_run_status(run: Any, status: str, *, now: float | None = None) -> None:
    """Apply run status transition and keep paused-time bookkeeping consistent."""
    old_status = str(getattr(run, "status", "") or "").strip().lower()
    new_status = str(status or "").strip().lower()
    if not new_status:
        return
    now_value = float(now) if now is not None else time.monotonic()
    paused_started = float(getattr(run, "paused_started_at", 0.0) or 0.0)
    paused_accum = float(getattr(run, "paused_accumulated_seconds", 0.0) or 0.0)

    if old_status == "paused" and new_status != "paused" and paused_started > 0.0:
        paused_accum += max(0.0, now_value - paused_started)
        paused_started = 0.0

    if new_status == "paused" and old_status != "paused" and paused_started <= 0.0:
        paused_started = now_value

    run.status = new_status
    run.paused_started_at = paused_started
    run.paused_accumulated_seconds = max(0.0, paused_accum)


def normalize_process_run_status(raw_status: object | None) -> str:
    """Map task-engine status strings into process-run pane status values."""
    status = str(raw_status or "").strip().lower()
    if status in {"executing", "planning"}:
        return "running"
    return status


def process_run_launch_stage_label(stage: str, *, stage_labels: dict[str, str]) -> str:
    """Return display label for a launch-stage identifier."""
    return stage_labels.get(stage, "Initializing")


def process_run_stage_rows(
    run: Any,
    *,
    stages: tuple[tuple[str, str], ...],
    stage_index: dict[str, int],
    one_line: Callable[[object | None, int | None], str],
) -> list[dict]:
    """Render launch/provisioning checklist rows while plan tasks are not ready."""
    current = str(getattr(run, "launch_stage", "accepted") or "accepted").strip()
    current_idx = stage_index.get(current, 0)
    status = str(getattr(run, "status", "queued") or "queued").strip()
    rows: list[dict] = []
    for idx, (stage_id, label) in enumerate(stages):
        row_status = "pending"
        if idx < current_idx:
            row_status = "completed"
        elif idx == current_idx:
            if status in {"failed", "cancel_failed"}:
                row_status = "failed"
            elif status in {"cancelled", "force_closed"}:
                row_status = "skipped"
            elif status == "completed":
                row_status = "completed"
            else:
                row_status = "in_progress"
        rows.append({
            "id": f"stage:{stage_id}",
            "status": row_status,
            "content": label,
        })

    if status == "failed":
        detail = one_line(getattr(run, "launch_error", ""), 180) or "Run failed."
        rows.append({
            "id": "stage:failed",
            "status": "failed",
            "content": f"Failed: {detail}",
        })
    elif status == "cancel_failed":
        rows.append({
            "id": "stage:cancel-failed",
            "status": "failed",
            "content": "Cancellation timed out",
        })
    elif status == "cancelled":
        rows.append({
            "id": "stage:cancelled",
            "status": "skipped",
            "content": "Cancelled",
        })
    elif status == "force_closed":
        rows.append({
            "id": "stage:force-closed",
            "status": "skipped",
            "content": "Force-closed",
        })
    elif status == "cancel_requested":
        rows.append({
            "id": "stage:cancel-requested",
            "status": "in_progress",
            "content": "Cancellation requested",
        })
    elif status == "paused":
        rows.append({
            "id": "stage:paused",
            "status": "in_progress",
            "content": "Paused",
        })
    return rows


def process_run_stage_summary_row(
    run: Any,
    *,
    stage_labels: dict[str, str],
    launch_stage_label: Callable[[str], str],
) -> dict | None:
    """Return a compact launch-stage summary row to prepend above task rows."""
    if not hasattr(run, "launch_stage"):
        return None
    stage = str(getattr(run, "launch_stage", "accepted") or "accepted").strip()
    if stage not in stage_labels:
        stage = "accepted"
    status = str(getattr(run, "status", "queued") or "queued").strip()
    if status in {"failed", "cancel_failed"}:
        row_status = "failed"
    elif status in {"cancelled", "force_closed"}:
        row_status = "skipped"
    elif status == "completed":
        row_status = "completed"
    else:
        row_status = "in_progress"
    return {
        "id": "stage:summary",
        "status": row_status,
        "content": f"Launch stage: {launch_stage_label(stage)}",
    }
