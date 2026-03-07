"""Runtime infrastructure helpers for orchestrator.

Owns run-id lifecycle and event emission plumbing that are shared across
subsystems and are not telemetry-specific.
"""

from __future__ import annotations

import uuid

from loom.events.bus import Event
from loom.state.task_state import Task


def task_run_id(task: Task) -> str:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    return str(metadata.get("run_id", "") or "").strip()


def initialize_task_run_id(orchestrator, task: Task) -> str:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    run_id = str(metadata.get("run_id", "") or "").strip()
    if not run_id:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        metadata["run_id"] = run_id
        task.metadata = metadata
        orchestrator._state.save(task)
    orchestrator._active_run_id = run_id
    return run_id


def emit_event(orchestrator, event_type: str, task_id: str, data: dict) -> None:
    payload = dict(data or {})
    payload.setdefault("source_component", "orchestrator")
    run_id = str(payload.get("run_id", "") or "").strip()
    if not run_id:
        run_id = str(getattr(orchestrator, "_active_run_id", "") or "").strip()
    if run_id and not str(payload.get("run_id", "")).strip():
        payload["run_id"] = run_id
    orchestrator._events.emit(Event(
        event_type=event_type,
        task_id=task_id,
        data=payload,
    ))
