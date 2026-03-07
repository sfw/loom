"""Task factory helpers for orchestrator public API."""

from __future__ import annotations

import uuid
from datetime import datetime

from loom.state.task_state import Task


def create_task(
    goal: str,
    workspace: str = "",
    approval_mode: str = "auto",
    callback_url: str = "",
    context: dict | None = None,
    metadata: dict | None = None,
) -> Task:
    """Factory for creating new tasks with a generated ID."""
    return Task(
        id=uuid.uuid4().hex[:8],
        goal=goal,
        workspace=workspace,
        approval_mode=approval_mode,
        callback_url=callback_url,
        context=context or {},
        metadata=metadata or {},
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
