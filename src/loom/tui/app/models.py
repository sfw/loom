"""Data models for the Loom TUI app package."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

    from .widgets import ProcessRunPane


@dataclass(frozen=True)
class SlashCommandSpec:
    """Definition for a slash command shown in help and autocomplete."""

    canonical: str
    description: str
    aliases: tuple[str, ...] = ()
    usage: str = ""


@dataclass
class ProcessRunState:
    """In-memory state for a single process run tab."""

    run_id: str
    process_name: str
    goal: str
    run_workspace: Path
    process_defn: ProcessDefinition | None
    pane_id: str
    pane: ProcessRunPane
    status: str = "queued"
    task_id: str = ""
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    tasks: list[dict] = field(default_factory=list)
    task_labels: dict[str, str] = field(default_factory=dict)
    subtask_phase_ids: dict[str, str] = field(default_factory=dict)
    last_progress_message: str = ""
    last_progress_at: float = 0.0
    activity_log: list[str] = field(default_factory=list)
    result_log: list[dict] = field(default_factory=list)
    worker: object | None = None
    closed: bool = False
    is_adhoc: bool = False
    recommended_tools: list[str] = field(default_factory=list)
    goal_context_overrides: dict[str, Any] = field(default_factory=dict)
    auth_profile_overrides: dict[str, str] = field(default_factory=dict)
    auth_required_resources: list[dict[str, Any]] = field(default_factory=list)
    launch_stage: str = "accepted"
    launch_stage_started_at: float = field(default_factory=time.monotonic)
    launch_last_progress_at: float = field(default_factory=time.monotonic)
    launch_last_heartbeat_at: float = 0.0
    launch_error: str = ""
    launch_tab_created_at: float = 0.0
    launch_silent_warning_emitted: bool = False
    launch_stage_heartbeat_dots: int = 0
    launch_stage_heartbeat_stage: str = ""
    launch_stage_activity_indices: dict[str, int] = field(default_factory=dict)
    close_after_cancel: bool = False
    cancel_requested_at: float = 0.0
    progress_ui_last_refresh_at: float = 0.0
    paused_started_at: float = 0.0
    paused_accumulated_seconds: float = 0.0


@dataclass
class ProcessRunLaunchRequest:
    """Runtime launch request used to preflight and execute one process run."""

    goal: str
    command_prefix: str = "/run"
    process_defn: ProcessDefinition | None = None
    process_name_override: str = ""
    is_adhoc: bool = False
    recommended_tools: list[str] = field(default_factory=list)
    adhoc_synthesis_notes: list[str] = field(default_factory=list)
    goal_context_overrides: dict[str, Any] = field(default_factory=dict)
    resume_task_id: str = ""
    run_workspace_override: Path | None = None
    synthesis_goal: str = ""
    force_fresh: bool = False


@dataclass
class AdhocProcessCacheEntry:
    """Cached ad hoc process synthesized for `/run` goals."""

    key: str
    goal: str
    process_defn: ProcessDefinition
    recommended_tools: list[str] = field(default_factory=list)
    spec: dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.monotonic)


@dataclass
class SteeringDirective:
    """Ephemeral cowork steering directive."""

    id: str
    kind: str
    text: str
    source: str = "slash"
    created_at: float = field(default_factory=time.monotonic)
    status: str = "queued"
