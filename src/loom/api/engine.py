"""Engine lifecycle: wires up all Loom components for the API server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Coroutine
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loom import __version__
from loom.config import Config, load_config
from loom.config_runtime.store import ConfigRuntimeStore
from loom.cowork.approval import ApprovalDecision
from loom.engine.orchestrator import Orchestrator
from loom.events.bus import Event, EventBus, EventPersister
from loom.events.types import (
    TASK_CANCELLED,
    TASK_FAILED,
    TASK_RUN_ACQUIRED,
    TASK_RUN_HEARTBEAT,
    TASK_RUN_RECOVERED,
    TELEMETRY_MODE_CHANGED,
    TELEMETRY_SETTINGS_WARNING,
    TelemetryMode,
)
from loom.events.verbosity import DEFAULT_TELEMETRY_MODE, normalize_telemetry_mode
from loom.events.webhook import WebhookDelivery
from loom.learning.manager import LearningManager
from loom.models.router import ModelRouter
from loom.prompts.assembler import PromptAssembler
from loom.recovery.approval import ApprovalManager
from loom.recovery.questions import QuestionManager
from loom.state.conversation_store import ConversationStore
from loom.state.memory import Database, MemoryManager
from loom.state.task_state import Task, TaskStateManager, TaskStatus
from loom.state.workspaces import WorkspaceRegistry
from loom.tools import create_default_registry
from loom.tools.registry import ToolRegistry, normalize_tool_execution_surface
from loom.utils.concurrency import run_blocking_io
from loom.utils.latency import diagnostics_enabled, log_latency_event

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

logger = logging.getLogger(__name__)
_API_EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS = 0.5
_API_EVENT_LOOP_LAG_WARN_THRESHOLD_SECONDS = 0.05
_TASK_TERMINAL_STATUSES = {
    TaskStatus.COMPLETED.value,
    TaskStatus.FAILED.value,
    TaskStatus.CANCELLED.value,
}
_TASK_INTERRUPTED_STATUSES = {
    TaskStatus.PENDING.value,
    TaskStatus.PLANNING.value,
    TaskStatus.EXECUTING.value,
    TaskStatus.PAUSED.value,
    TaskStatus.WAITING_APPROVAL.value,
}


@dataclass
class ConversationWorker:
    """Tracks one background cowork turn for a persisted conversation."""

    session: object
    task: asyncio.Task


@dataclass
class TaskWorker:
    """Tracks one background task execution worker for a persisted run."""

    run_id: str
    task: asyncio.Task


@dataclass
class ConversationApprovalRequest:
    """Tracks one pending cowork tool approval for a conversation."""

    approval_id: str
    session_id: str
    tool_name: str
    args: dict
    created_at: str
    event: asyncio.Event
    decision: ApprovalDecision | None = None

    def to_dict(self) -> dict[str, object]:
        risk_info = self.args.get("_loom_risk_info")
        if not isinstance(risk_info, dict):
            risk_info = None
        return {
            "approval_id": self.approval_id,
            "conversation_id": self.session_id,
            "tool_name": self.tool_name,
            "args": dict(self.args),
            "risk_info": risk_info,
            "created_at": self.created_at,
        }


class TelemetryPersistConflictError(RuntimeError):
    """Raised when persisted telemetry override conflicts with external edits."""


class TelemetryPersistDisabledError(RuntimeError):
    """Raised when callers request persisted override while feature is disabled."""


class Engine:
    """Holds all Loom components. Created during server lifespan."""

    def __init__(
        self,
        config: Config,
        orchestrator: Orchestrator,
        event_bus: EventBus,
        model_router: ModelRouter,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        state_manager: TaskStateManager,
        prompt_assembler: PromptAssembler,
        database: Database,
        conversation_store: ConversationStore,
        workspace_registry: WorkspaceRegistry,
        config_runtime_store: ConfigRuntimeStore,
        approval_manager: ApprovalManager,
        question_manager: QuestionManager | None,
        webhook_delivery: WebhookDelivery,
        learning_manager: LearningManager,
        event_persister: EventPersister | None = None,
        runtime_role: str = "api",
    ):
        self.config = config
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.model_router = model_router
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        self.state_manager = state_manager
        self.prompt_assembler = prompt_assembler
        self.database = database
        self.conversation_store = conversation_store
        self.workspace_registry = workspace_registry
        self.config_runtime_store = config_runtime_store
        self.approval_manager = approval_manager
        self.question_manager = question_manager
        self.event_persister = event_persister
        self.webhook_delivery = webhook_delivery
        self.learning_manager = learning_manager
        self.runtime_role = str(runtime_role or "api").strip() or "api"
        set_snapshot_mirror_writer = getattr(
            self.orchestrator,
            "set_snapshot_mirror_writer",
            None,
        )
        if callable(set_snapshot_mirror_writer):
            set_snapshot_mirror_writer(self._sync_task_row_snapshot)
        self.started_at = datetime.now(UTC).isoformat()
        self._background_tasks: set[asyncio.Task] = set()
        self._conversation_workers: dict[str, ConversationWorker] = {}
        self._task_workers: dict[str, TaskWorker] = {}
        self._conversation_approvals: dict[str, ConversationApprovalRequest] = {}
        self._runner_id = f"api-{uuid.uuid4().hex[:8]}"
        self._run_lease_seconds = 30
        self._run_heartbeat_interval_seconds = 10
        self._bind_api_tool_registry()
        telemetry_mode_input = str(
            getattr(self.config.telemetry, "configured_mode_input", ""),
        ).strip()
        configured_resolution = normalize_telemetry_mode(
            getattr(self.config.telemetry, "mode", DEFAULT_TELEMETRY_MODE),
            default=DEFAULT_TELEMETRY_MODE,
        )
        input_resolution = normalize_telemetry_mode(
            telemetry_mode_input,
            default=configured_resolution.mode,
        )
        self._telemetry_lock = threading.Lock()
        self._telemetry_configured_mode: TelemetryMode = configured_resolution.mode
        self._telemetry_configured_input = telemetry_mode_input or configured_resolution.mode
        self._telemetry_runtime_override: TelemetryMode | None = None
        self._telemetry_updated_at = datetime.now(UTC).isoformat()
        source_path_text = str(getattr(config, "source_path", "") or "").strip()
        self._telemetry_config_path: Path | None = (
            Path(source_path_text).expanduser() if source_path_text else None
        )
        self._telemetry_config_mtime_ns = self._read_config_mtime_ns(self._telemetry_config_path)
        self.event_bus.set_operator_mode_provider(self.effective_telemetry_mode)
        if input_resolution.warning_code:
            self._emit_telemetry_settings_warning(
                warning_code=input_resolution.warning_code,
                input_value=input_resolution.input_value,
                normalized_mode=configured_resolution.mode,
                source="config.telemetry.mode",
            )
        if diagnostics_enabled():
            probe_task = asyncio.create_task(self._monitor_event_loop_lag())
            self._background_tasks.add(probe_task)
            probe_task.add_done_callback(self._background_tasks.discard)

    def _bind_api_tool_registry(self) -> None:
        """Bind runtime-backed tools needed by API cowork sessions."""
        delegate = self.tool_registry.get("delegate_task")
        if delegate is None:
            return
        try:
            from loom.tools.delegate_task import DelegateTaskTool
        except Exception:
            logger.debug(
                "Unable to import DelegateTaskTool for API session binding",
                exc_info=True,
            )
            return
        if not isinstance(delegate, DelegateTaskTool):
            return

        async def _orchestrator_factory(
            process_override: ProcessDefinition | None = None,
        ) -> Orchestrator:
            return self.create_task_orchestrator(process=process_override)

        delegate.bind(_orchestrator_factory)
        delegate.set_timeout_resolver(
            lambda: int(
                self.config_runtime_store.effective_value(
                    "execution.delegate_task_timeout_seconds",
                ) or 3600,
            ),
        )

    async def shutdown(self) -> None:
        """Graceful cleanup."""
        try:
            await self.pause_active_task_runs_for_shutdown()
        except Exception:
            logger.warning("Failed pausing active task runs during shutdown", exc_info=True)
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
        await self.event_bus.drain(timeout=5.0)
        if self.event_persister is not None:
            await self.event_persister.drain(timeout=5.0)
        await self.model_router.close()
        await self.database.close()

    def conversation_turn_inflight(self, session_id: str) -> bool:
        """Return True when a cowork turn is still running for a session."""
        clean_id = str(session_id or "").strip()
        if not clean_id:
            return False
        worker = self._conversation_workers.get(clean_id)
        return worker is not None and not worker.task.done()

    def task_run_inflight(self, task_id: str) -> bool:
        """Return True when a task execution worker is still running."""
        clean_id = str(task_id or "").strip()
        if not clean_id:
            return False
        worker = self._task_workers.get(clean_id)
        return worker is not None and not worker.task.done()

    def stop_task_worker(self, task_id: str) -> bool:
        """Cancel an active task execution worker, if one exists."""
        clean_id = str(task_id or "").strip()
        if not clean_id:
            return False
        worker = self._task_workers.pop(clean_id, None)
        if worker is None or worker.task.done():
            return False
        worker.task.cancel()
        return True

    def conversation_stop_requested(self, session_id: str) -> bool:
        """Return True when the active cowork turn has a stop flag set."""
        clean_id = str(session_id or "").strip()
        if not clean_id:
            return False
        worker = self._conversation_workers.get(clean_id)
        if worker is None:
            return False
        return bool(getattr(worker.session, "stop_requested", False))

    def conversation_pending_inject_count(self, session_id: str) -> int:
        """Return queued inject instructions for the active cowork turn."""
        clean_id = str(session_id or "").strip()
        if not clean_id:
            return 0
        worker = self._conversation_workers.get(clean_id)
        if worker is None:
            return 0
        count = getattr(worker.session, "pending_inject_instruction_count", 0)
        try:
            return max(0, int(count))
        except Exception:
            return 0

    def conversation_context_status(self, session_id: str) -> dict[str, object] | None:
        """Return live context-pressure status for an active cowork session."""
        clean_id = str(session_id or "").strip()
        if not clean_id:
            return None
        worker = self._conversation_workers.get(clean_id)
        if worker is None:
            return None
        getter = getattr(worker.session, "context_status_snapshot", None)
        if not callable(getter):
            return None
        try:
            snapshot = getter()
        except Exception:
            logger.debug(
                "Failed reading live conversation context status for %s",
                clean_id,
                exc_info=True,
            )
            return None
        return dict(snapshot) if isinstance(snapshot, dict) else None

    def queue_conversation_inject_instruction(self, session_id: str, text: str) -> int:
        """Queue a steering instruction for the active cowork turn."""
        clean_id = str(session_id or "").strip()
        if not self.conversation_turn_inflight(clean_id):
            return -1
        worker = self._conversation_workers.get(clean_id)
        if worker is None:
            return -1
        queue_instruction = getattr(worker.session, "queue_inject_instruction", None)
        if not callable(queue_instruction):
            return -1
        queue_instruction(text)
        return self.conversation_pending_inject_count(clean_id)

    def get_pending_conversation_approval(self, session_id: str) -> dict[str, object] | None:
        """Return one pending cowork approval request for a conversation."""
        clean_id = str(session_id or "").strip()
        if not clean_id:
            return None
        request = self._conversation_approvals.get(clean_id)
        if request is None or request.event.is_set():
            return None
        if not self.conversation_turn_inflight(clean_id):
            self._conversation_approvals.pop(clean_id, None)
            return None
        return request.to_dict()

    def list_pending_conversation_approvals(self) -> list[dict[str, object]]:
        """Return pending cowork approvals for conversations that are still inflight."""
        pending: list[dict[str, object]] = []
        stale_session_ids: list[str] = []
        for session_id, request in self._conversation_approvals.items():
            if request.event.is_set():
                continue
            if not self.conversation_turn_inflight(session_id):
                stale_session_ids.append(session_id)
                continue
            pending.append(request.to_dict())
        for session_id in stale_session_ids:
            self._conversation_approvals.pop(session_id, None)
        pending.sort(key=lambda item: str(item.get("created_at", "") or ""), reverse=True)
        return pending

    def begin_conversation_approval(
        self,
        session_id: str,
        *,
        tool_name: str,
        args: dict,
    ) -> ConversationApprovalRequest:
        """Register a pending cowork approval request."""
        clean_id = str(session_id or "").strip()
        if not clean_id:
            raise ValueError("session_id is required")
        existing = self._conversation_approvals.get(clean_id)
        if existing is not None and not existing.event.is_set():
            raise RuntimeError(f"Conversation approval already pending: {clean_id}")
        request = ConversationApprovalRequest(
            approval_id=uuid.uuid4().hex,
            session_id=clean_id,
            tool_name=str(tool_name or "").strip() or "tool",
            args=dict(args or {}),
            created_at=datetime.now(UTC).isoformat(),
            event=asyncio.Event(),
        )
        self._conversation_approvals[clean_id] = request
        return request

    async def wait_for_conversation_approval(
        self,
        session_id: str,
        approval_id: str,
    ) -> ApprovalDecision:
        """Wait for the pending cowork approval to be resolved."""
        clean_id = str(session_id or "").strip()
        request = self._conversation_approvals.get(clean_id)
        if request is None or request.approval_id != str(approval_id or "").strip():
            raise KeyError(f"Conversation approval not found: {clean_id}/{approval_id}")
        try:
            await request.event.wait()
            return request.decision or ApprovalDecision.DENY
        finally:
            current = self._conversation_approvals.get(clean_id)
            if current is request:
                self._conversation_approvals.pop(clean_id, None)

    def resolve_conversation_approval(
        self,
        session_id: str,
        approval_id: str,
        decision: ApprovalDecision,
    ) -> bool:
        """Resolve one pending cowork approval request."""
        clean_id = str(session_id or "").strip()
        request = self._conversation_approvals.get(clean_id)
        if request is None:
            return False
        if request.approval_id != str(approval_id or "").strip():
            return False
        request.decision = decision
        request.event.set()
        return True

    def request_conversation_stop(self, session_id: str, *, reason: str = "user_requested") -> bool:
        """Cooperatively stop the active cowork turn for a session."""
        clean_id = str(session_id or "").strip()
        if not self.conversation_turn_inflight(clean_id):
            return False
        worker = self._conversation_workers.get(clean_id)
        if worker is None:
            return False
        request_stop = getattr(worker.session, "request_stop", None)
        if not callable(request_stop):
            return False
        request_stop(reason)
        pending = self._conversation_approvals.get(clean_id)
        if pending is not None and not pending.event.is_set():
            pending.decision = ApprovalDecision.DENY
            pending.event.set()
        return True

    def start_conversation_worker(
        self,
        session_id: str,
        session: object,
        worker: Coroutine[object, object, object],
    ) -> asyncio.Task:
        """Start one background cowork turn per session."""
        clean_id = str(session_id or "").strip()
        if not clean_id:
            raise ValueError("session_id is required")
        if self.conversation_turn_inflight(clean_id):
            raise RuntimeError(f"Conversation already processing: {clean_id}")

        task = asyncio.create_task(worker)
        self._conversation_workers[clean_id] = ConversationWorker(
            session=session,
            task=task,
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        def _clear_worker(done_task: asyncio.Task) -> None:
            current = self._conversation_workers.get(clean_id)
            if current is not None and current.task is done_task:
                self._conversation_workers.pop(clean_id, None)
            pending = self._conversation_approvals.get(clean_id)
            if pending is not None and not pending.event.is_set():
                pending.decision = ApprovalDecision.DENY
                pending.event.set()

        task.add_done_callback(_clear_worker)
        return task

    def start_task_worker(
        self,
        *,
        task_id: str,
        run_id: str,
        worker: Coroutine[object, object, object],
    ) -> asyncio.Task:
        """Start one background execution worker per task."""
        clean_id = str(task_id or "").strip()
        if not clean_id:
            raise ValueError("task_id is required")
        if self.task_run_inflight(clean_id):
            raise RuntimeError(f"Task already processing: {clean_id}")

        task = asyncio.create_task(worker)
        self._task_workers[clean_id] = TaskWorker(
            run_id=str(run_id or "").strip(),
            task=task,
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        def _clear_worker(done_task: asyncio.Task) -> None:
            current = self._task_workers.get(clean_id)
            if current is not None and current.task is done_task:
                self._task_workers.pop(clean_id, None)

        task.add_done_callback(_clear_worker)
        return task

    async def _monitor_event_loop_lag(self) -> None:
        """Emit periodic API event-loop lag diagnostics when enabled."""
        expected = time.monotonic() + _API_EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS
        while True:
            await asyncio.sleep(_API_EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS)
            now = time.monotonic()
            lag = max(0.0, now - expected)
            if lag >= _API_EVENT_LOOP_LAG_WARN_THRESHOLD_SECONDS:
                log_latency_event(
                    logger,
                    event="api_event_loop_lag",
                    duration_seconds=lag,
                    fields={"threshold_ms": int(_API_EVENT_LOOP_LAG_WARN_THRESHOLD_SECONDS * 1000)},
                )
            expected = now + _API_EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS

    @staticmethod
    def _read_config_mtime_ns(path: Path | None) -> int | None:
        if path is None or not path.exists():
            return None
        try:
            return int(path.stat().st_mtime_ns)
        except OSError:
            return None

    @staticmethod
    def _upsert_telemetry_mode_toml(text: str, mode: TelemetryMode) -> str:
        lines = text.splitlines()
        section_start: int | None = None
        section_end = len(lines)

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not (stripped.startswith("[") and stripped.endswith("]")):
                continue
            section_name = stripped[1:-1].strip().lower()
            if section_name == "telemetry":
                section_start = idx
                continue
            if section_start is not None and idx > section_start:
                section_end = idx
                break

        mode_line = f'mode = "{mode}"'
        if section_start is None:
            if lines and lines[-1].strip():
                lines.append("")
            lines.extend(["[telemetry]", mode_line])
        else:
            replaced = False
            for idx in range(section_start + 1, section_end):
                stripped = lines[idx].strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key = stripped.split("=", 1)[0].strip().lower()
                if key != "mode":
                    continue
                lines[idx] = mode_line
                replaced = True
                break
            if not replaced:
                lines.insert(section_start + 1, mode_line)

        rendered = "\n".join(lines)
        if text.endswith("\n"):
            rendered += "\n"
        return rendered

    def configured_telemetry_mode(self) -> TelemetryMode:
        with self._telemetry_lock:
            return self._telemetry_configured_mode

    def runtime_telemetry_mode(self) -> TelemetryMode | None:
        with self._telemetry_lock:
            return self._telemetry_runtime_override

    def effective_telemetry_mode(self) -> TelemetryMode:
        with self._telemetry_lock:
            return self._telemetry_runtime_override or self._telemetry_configured_mode

    @staticmethod
    def telemetry_mode_scope() -> str:
        return "process_local"

    def telemetry_settings_snapshot(self) -> dict[str, str]:
        with self._telemetry_lock:
            runtime_override_mode = self._telemetry_runtime_override or ""
            effective_mode = runtime_override_mode or self._telemetry_configured_mode
            return {
                "configured_mode": self._telemetry_configured_mode,
                "runtime_override_mode": runtime_override_mode,
                "effective_mode": effective_mode,
                "scope": self.telemetry_mode_scope(),
                "updated_at": self._telemetry_updated_at,
            }

    def refresh_config_from_runtime_store(self) -> Config:
        """Recompute effective config after runtime config mutations."""
        effective = self.config_runtime_store.effective_config()
        self.config = effective
        self.model_router = ModelRouter.from_config(effective)
        self.tool_registry = create_default_registry(
            effective,
            mcp_startup_mode="background",
        )
        self._bind_api_tool_registry()
        process = None
        if isinstance(getattr(self.orchestrator, "__dict__", None), dict):
            process = self.orchestrator.__dict__.get("_process")
        self.orchestrator = Orchestrator(
            model_router=self.model_router,
            tool_registry=self.tool_registry,
            memory_manager=self.memory_manager,
            prompt_assembler=self.prompt_assembler,
            state_manager=self.state_manager,
            event_bus=self.event_bus,
            config=self.config,
            approval_manager=self.approval_manager,
            question_manager=self.question_manager,
            learning_manager=self.learning_manager,
            process=process,
            snapshot_mirror_writer=self._sync_task_row_snapshot,
            execution_surface=normalize_tool_execution_surface(
                self.runtime_role,
                default="api",
            ),
        )
        return effective

    def reload_config_from_source(self, path: Path | None = None) -> Config:
        """Reload file-backed config and rebuild runtime state."""
        target = path or self.config_runtime_store.source_path()
        loaded = load_config(target)
        self.config_runtime_store.set_config(
            loaded,
            source_path=target if target is not None and target.exists() else None,
        )
        return self.refresh_config_from_runtime_store()

    def runtime_status_snapshot(self) -> dict[str, object]:
        """Return desktop/client-friendly runtime contract information."""
        config_source = self.config_runtime_store.source_path()
        execution_surface = normalize_tool_execution_surface(
            self.runtime_role,
            default="api",
        )
        return {
            "status": "ok",
            "ready": True,
            "runtime_role": self.runtime_role,
            "started_at": self.started_at,
            "version": __version__,
            "config_path": str(config_source) if config_source is not None else "",
            "database_path": str(self.config.database_path),
            "scratch_dir": str(self.config.scratch_path),
            "host": str(self.config.server.host),
            "port": int(self.config.server.port),
            "workspace_default_path": str(self.config.workspace.default_path),
            "tool_availability": self.tool_registry.availability_rows(
                execution_surface=execution_surface,
            ),
        }

    async def activity_summary_snapshot(self) -> dict[str, object]:
        """Return a global snapshot of active desktop-visible backend work."""
        active_conversation_count = sum(
            1
            for session_id, worker in self._conversation_workers.items()
            if not worker.task.done()
            and self.get_pending_conversation_approval(session_id) is None
        )
        active_run_count = 0
        for task_id, worker in self._task_workers.items():
            if worker.task.done():
                continue
            task = await self._load_task_state(task_id)
            if task is not None and task.status not in {
                TaskStatus.PENDING,
                TaskStatus.PLANNING,
                TaskStatus.EXECUTING,
            }:
                continue
            active_run_count += 1
        return {
            "status": "ok",
            "active": bool(active_conversation_count or active_run_count),
            "active_conversation_count": active_conversation_count,
            "active_run_count": active_run_count,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    def _emit_telemetry_settings_warning(
        self,
        *,
        warning_code: str,
        input_value: str,
        normalized_mode: TelemetryMode,
        source: str,
    ) -> None:
        self.event_bus.emit(
            Event(
                event_type=TELEMETRY_SETTINGS_WARNING,
                task_id="system",
                data={
                    "warning_code": str(warning_code or "").strip(),
                    "input_value": str(input_value or "").strip() or "<empty>",
                    "normalized_mode": normalized_mode,
                    "configured_mode": self.configured_telemetry_mode(),
                    "runtime_override_mode": self.runtime_telemetry_mode() or "",
                    "effective_mode": self.effective_telemetry_mode(),
                    "scope": self.telemetry_mode_scope(),
                    "source": str(source or "").strip() or "unknown",
                    "source_component": "api_engine",
                },
            ),
        )

    def _persist_runtime_telemetry_mode(self, mode: TelemetryMode) -> None:
        if not bool(getattr(self.config.telemetry, "persist_runtime_override", False)):
            raise TelemetryPersistDisabledError(
                "Persisted telemetry override is disabled. "
                "Set [telemetry].persist_runtime_override=true to enable persisted writes.",
            )

        path = self._telemetry_config_path
        if path is None or not path.exists():
            raise TelemetryPersistDisabledError(
                "Persisted telemetry override requires a writable loom.toml source path.",
            )

        lock_path = path.with_suffix(f"{path.suffix}.lock")
        temp_path: Path | None = None
        try:
            import fcntl  # POSIX advisory locks
        except ImportError as e:  # pragma: no cover - non-POSIX fallback
            raise TelemetryPersistDisabledError(
                "Persisted telemetry override is unavailable on this platform.",
            ) from e

        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            current_mtime = self._read_config_mtime_ns(path)
            expected_mtime = self._telemetry_config_mtime_ns
            if (
                expected_mtime is not None
                and current_mtime is not None
                and current_mtime != expected_mtime
            ):
                raise TelemetryPersistConflictError(
                    "loom.toml changed since startup; reload config and retry telemetry persist.",
                )

            original_text = path.read_text(encoding="utf-8")
            updated_text = self._upsert_telemetry_mode_toml(original_text, mode)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            try:
                temp_path = Path(tmp_name)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(updated_text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

            self._telemetry_config_mtime_ns = self._read_config_mtime_ns(path)

    def set_runtime_telemetry_mode(
        self,
        *,
        mode_input: object,
        actor: str,
        source: str,
        persist: bool = False,
    ) -> dict[str, str]:
        if not bool(getattr(self.config.telemetry, "runtime_override_enabled", True)):
            raise TelemetryPersistDisabledError(
                "Runtime telemetry override is disabled by config.",
            )

        resolution = normalize_telemetry_mode(
            mode_input,
            default=DEFAULT_TELEMETRY_MODE,
        )
        actor_text = str(actor or "").strip() or "unknown"
        source_text = str(source or "").strip() or "unknown"
        updated_at = datetime.now(UTC).isoformat()

        with self._telemetry_lock:
            previous_runtime = self._telemetry_runtime_override
            previous_configured = self._telemetry_configured_mode
            previous_effective = previous_runtime or self._telemetry_configured_mode
            if persist:
                self._persist_runtime_telemetry_mode(resolution.mode)
                self._telemetry_configured_mode = resolution.mode
                self._telemetry_configured_input = resolution.mode
            self._telemetry_runtime_override = resolution.mode
            self._telemetry_updated_at = updated_at
            runtime_override_mode = self._telemetry_runtime_override or ""
            effective_mode = runtime_override_mode or self._telemetry_configured_mode
            changed = (
                previous_runtime != self._telemetry_runtime_override
                or previous_effective != effective_mode
                or previous_configured != self._telemetry_configured_mode
            )

        if resolution.warning_code:
            self._emit_telemetry_settings_warning(
                warning_code=resolution.warning_code,
                input_value=resolution.input_value,
                normalized_mode=resolution.mode,
                source=source_text,
            )

        if changed:
            self.event_bus.emit(
                Event(
                    event_type=TELEMETRY_MODE_CHANGED,
                    task_id="system",
                    data={
                        "configured_mode": self.configured_telemetry_mode(),
                        "runtime_override_mode": self.runtime_telemetry_mode() or "",
                        "effective_mode": self.effective_telemetry_mode(),
                        "scope": self.telemetry_mode_scope(),
                        "actor": actor_text,
                        "source": source_text,
                        "persisted": bool(persist),
                        "updated_at": updated_at,
                        "source_component": "api_engine",
                    },
                ),
            )
        return self.telemetry_settings_snapshot()

    def create_task_orchestrator(
        self, process: ProcessDefinition | None = None,
    ) -> Orchestrator:
        """Create an isolated orchestrator for a single task execution.

        A fresh prompt assembler and tool registry avoid cross-task mutation
        when a process definition applies tool exclusions or prompt overrides.
        """
        return Orchestrator(
            model_router=self.model_router,
            tool_registry=create_default_registry(
                self.config,
                mcp_startup_mode="background",
            ),
            memory_manager=self.memory_manager,
            prompt_assembler=PromptAssembler(),
            state_manager=self.state_manager,
            event_bus=self.event_bus,
            config=self.config,
            approval_manager=self.approval_manager,
            question_manager=self.question_manager,
            learning_manager=self.learning_manager,
            process=process,
            snapshot_mirror_writer=self._sync_task_row_snapshot,
            execution_surface=normalize_tool_execution_surface(
                self.runtime_role,
                default="api",
            ),
        )

    async def _task_state_exists(self, task_id: str) -> bool:
        return bool(await run_blocking_io(self.state_manager.exists, task_id))

    async def _load_task_state(self, task_id: str) -> Task | None:
        try:
            task = await run_blocking_io(self.state_manager.load, task_id)
        except Exception:
            return None
        return task if isinstance(task, Task) else None

    async def _save_task_state(self, task: Task) -> None:
        await run_blocking_io(self.state_manager.save, task)
        await self._sync_task_row_snapshot(task)

    async def _sync_task_row_snapshot(self, task: Task) -> None:
        raw_status = getattr(task, "status", TaskStatus.FAILED)
        status_value = (
            raw_status.value
            if hasattr(raw_status, "value")
            else str(raw_status or TaskStatus.FAILED.value)
        )
        created_at = str(task.created_at or datetime.now(UTC).isoformat()).strip()
        updated_at = str(task.updated_at or created_at).strip() or created_at
        completed_at = str(task.completed_at or "").strip()
        if not completed_at and status_value in _TASK_TERMINAL_STATUSES:
            completed_at = updated_at
        context_json = json.dumps(task.context, ensure_ascii=False) if task.context else None
        plan_json = (
            json.dumps(asdict(task.plan), ensure_ascii=False)
            if task.plan and (
                task.plan.subtasks
                or int(getattr(task.plan, "version", 1) or 1) != 1
                or str(getattr(task.plan, "last_replanned", "") or "").strip()
            )
            else None
        )
        metadata = task.metadata if isinstance(task.metadata, dict) else None
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        try:
            await self.database.execute(
                """INSERT INTO tasks (
                       id, goal, context, workspace_path, status, plan,
                       created_at, updated_at, state_snapshot_updated_at, completed_at,
                       approval_mode, callback_url, metadata
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       goal=excluded.goal,
                       context=excluded.context,
                       workspace_path=excluded.workspace_path,
                       status=excluded.status,
                       plan=excluded.plan,
                       created_at=excluded.created_at,
                       updated_at=excluded.updated_at,
                       state_snapshot_updated_at=excluded.state_snapshot_updated_at,
                       completed_at=excluded.completed_at,
                       approval_mode=excluded.approval_mode,
                       callback_url=excluded.callback_url,
                       metadata=excluded.metadata""",
                (
                    task.id,
                    task.goal,
                    context_json,
                    task.workspace,
                    status_value,
                    plan_json,
                    created_at,
                    updated_at,
                    updated_at,
                    completed_at or None,
                    task.approval_mode,
                    task.callback_url or None,
                    metadata_json,
                ),
            )
        except Exception:
            logger.debug("Failed syncing task row snapshot for %s", task.id, exc_info=True)

    async def submit_task(
        self,
        *,
        task: Task,
        process: ProcessDefinition | None = None,
        process_name: str = "",
        run_id: str | None = None,
        recovered: bool = False,
    ) -> str:
        """Queue a task run in the background, optionally with durable run metadata."""
        assigned_run_id = str(run_id or f"run-{uuid.uuid4().hex[:12]}").strip()
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["run_id"] = assigned_run_id
        task.metadata = metadata
        await self._save_task_state(task)

        if bool(getattr(self.config.execution, "enable_durable_task_runner", False)):
            existing = await self.database.get_task_run(assigned_run_id)
            if existing is None:
                await self.database.insert_task_run(
                    run_id=assigned_run_id,
                    task_id=task.id,
                    status="queued",
                    process_name=process_name,
                    metadata={"recovered": recovered},
                )

        self.start_task_worker(
            task_id=task.id,
            run_id=assigned_run_id,
            worker=self._execute_task_run(
                task=task,
                run_id=assigned_run_id,
                process=process,
                process_name=process_name,
                recovered=recovered,
            ),
        )
        return assigned_run_id

    async def recover_pending_task_runs(self) -> int:
        """Recover queued/running task runs from durable state."""
        if not bool(getattr(self.config.execution, "enable_durable_task_runner", False)):
            return 0
        recovered_count = 0
        try:
            rows = await self.database.list_recoverable_task_runs(limit=100)
        except Exception:
            logger.warning("Failed loading recoverable task runs", exc_info=True)
            return 0
        for row in rows:
            task_id = str(row.get("task_id", "")).strip()
            run_id = str(row.get("run_id", "")).strip()
            if not task_id or not run_id:
                continue
            task = await self._load_task_state(task_id)
            if task is None:
                logger.warning(
                    "Failed loading state for recoverable task %s",
                    task_id,
                )
                continue
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                continue
            if task.status == TaskStatus.PAUSED:
                continue
            process_name = str(row.get("process_name", "") or "").strip()
            process = await self._resolve_process_definition(
                process_name=process_name,
                workspace=Path(task.workspace) if task.workspace else None,
            )
            await self.submit_task(
                task=task,
                process=process,
                process_name=process_name,
                run_id=run_id,
                recovered=True,
            )
            self.event_bus.emit(
                Event(
                    event_type=TASK_RUN_RECOVERED,
                    task_id=task_id,
                    data={
                        "run_id": run_id,
                        "recovered": True,
                    },
                ),
            )
            recovered_count += 1
        return recovered_count

    async def pause_active_task_runs_for_shutdown(self) -> int:
        """Persist active desktop-owned runs into a resumable paused state."""
        if not bool(getattr(self.config.execution, "enable_durable_task_runner", False)):
            return 0
        try:
            rows = await self.database.list_tasks()
        except Exception:
            logger.warning("Failed loading task rows for shutdown pause", exc_info=True)
            return 0

        paused_count = 0
        for row in rows:
            task_id = str(row.get("id", "") or "").strip()
            if not task_id:
                continue
            task = await self._load_task_state(task_id)
            if task is None:
                logger.warning(
                    "Failed loading task state during shutdown pause: %s",
                    task_id,
                )
                continue

            if task.status not in {
                TaskStatus.EXECUTING,
                TaskStatus.PLANNING,
                TaskStatus.PAUSED,
            }:
                continue

            if task.status in {TaskStatus.EXECUTING, TaskStatus.PLANNING}:
                self.orchestrator.pause_task(task)

            if not isinstance(task.metadata, dict):
                task.metadata = {}
            task.metadata["shutdown_paused"] = True
            task.metadata["shutdown_pause_reason"] = "desktop_shutdown"
            try:
                await self._save_task_state(task)
            except Exception:
                logger.warning(
                    "Failed saving paused task state during shutdown: %s",
                    task_id,
                    exc_info=True,
                )
                continue

            run_id = ""
            if isinstance(task.metadata, dict):
                run_id = str(task.metadata.get("run_id", "") or "").strip()
            if not run_id:
                latest_run = await self.database.get_latest_task_run_for_task(task_id)
                run_id = str((latest_run or {}).get("run_id", "") or "").strip()
            if run_id:
                try:
                    await self.database.requeue_task_run(run_id=run_id)
                except Exception:
                    logger.debug(
                        "Failed requeueing task_run %s during shutdown pause",
                        run_id,
                        exc_info=True,
                    )
            paused_count += 1

        return paused_count

    async def reconcile_interrupted_task_runs(self) -> int:
        """Mark interrupted non-durable runs failed on startup.

        The TUI already treats busy runs restored after a client restart as failed
        unless there is an explicit resume flow. Desktop/loomd needs the same
        behavior when the durable runner is disabled, otherwise stale runs can
        appear resumable even though there is no worker left to continue them.
        """
        if bool(getattr(self.config.execution, "enable_durable_task_runner", False)):
            return 0
        try:
            rows = await self.database.list_tasks()
        except Exception:
            logger.warning(
                "Failed loading task rows for interrupted-run reconciliation",
                exc_info=True,
            )
            return 0

        interrupted_note = (
            "Run interrupted when Loom Desktop closed before recovery was available; "
            "marked failed. Restart the run to continue."
        )
        reconciled_count = 0

        for row in rows:
            task_id = str(row.get("id", "") or "").strip()
            if not task_id:
                continue

            row_status = str(row.get("status", "") or "").strip().lower()
            if row_status in _TASK_TERMINAL_STATUSES:
                continue

            task = await self._load_task_state(task_id)
            if task is None:
                logger.warning(
                    "Failed loading task state for interrupted-run reconciliation: %s",
                    task_id,
                )
                continue

            task_status = str(getattr(task.status, "value", task.status) or "").strip().lower()
            effective_status = task_status or row_status
            if effective_status in _TASK_TERMINAL_STATUSES:
                continue
            if effective_status not in _TASK_INTERRUPTED_STATUSES:
                continue

            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            if not metadata:
                raw_metadata = row.get("metadata")
                if isinstance(raw_metadata, str) and raw_metadata.strip():
                    try:
                        parsed = json.loads(raw_metadata)
                    except Exception:
                        parsed = {}
                    if isinstance(parsed, dict):
                        metadata = parsed

            run_id = str(metadata.get("run_id", "") or "").strip()
            task_run = None
            if run_id:
                try:
                    task_run = await self.database.get_task_run(run_id)
                except Exception:
                    logger.debug(
                        "Failed loading task_run %s during reconciliation",
                        run_id,
                        exc_info=True,
                    )

            if task_run is None:
                try:
                    task_run = await self.database.get_latest_task_run_for_task(task_id)
                except Exception:
                    logger.debug(
                        "Failed loading latest task_run for %s during reconciliation",
                        task_id,
                        exc_info=True,
                    )

            task.status = TaskStatus.FAILED
            task.add_error("system", interrupted_note, resolution="Restart the run to continue.")
            try:
                await self._save_task_state(task)
            except Exception:
                logger.warning(
                    "Failed saving interrupted task state during reconciliation: %s",
                    task_id,
                    exc_info=True,
                )
                continue

            run_id_for_event = run_id
            if task_run is not None:
                task_run_id = str(task_run.get("run_id", "") or "").strip()
                run_status = str(task_run.get("status", "") or "").strip().lower()
                if task_run_id:
                    run_id_for_event = run_id_for_event or task_run_id
                    if run_status not in {"completed", "failed", "cancelled"}:
                        try:
                            await self.database.complete_task_run(
                                run_id=task_run_id,
                                status="failed",
                                last_error=interrupted_note,
                            )
                        except Exception:
                            logger.debug(
                                "Failed completing interrupted task_run %s during reconciliation",
                                task_run_id,
                                exc_info=True,
                            )

            self.event_bus.emit(
                Event(
                    event_type=TASK_FAILED,
                    task_id=task_id,
                    data={
                        "run_id": run_id_for_event,
                        "interrupted": True,
                        "recovered": False,
                        "message": interrupted_note,
                    },
                ),
            )
            reconciled_count += 1

        return reconciled_count

    async def _resolve_process_definition(
        self,
        *,
        process_name: str,
        workspace: Path | None,
    ) -> ProcessDefinition | None:
        if not process_name:
            return None
        try:
            from loom.processes.schema import ProcessLoader

            extra = [Path(p) for p in self.config.process.search_paths]
            loader = ProcessLoader(
                workspace=workspace,
                extra_search_paths=extra,
                require_rule_scope_metadata=bool(
                    getattr(self.config.process, "require_rule_scope_metadata", False),
                ),
                require_v2_contract=bool(
                    getattr(self.config.process, "require_v2_contract", False),
                ),
            )
            return loader.load(process_name)
        except Exception:
            logger.warning(
                "Failed resolving process '%s' during run recovery",
                process_name,
                exc_info=True,
            )
            return None

    async def _execute_task_run(
        self,
        *,
        task: Task,
        run_id: str,
        process: ProcessDefinition | None,
        process_name: str,
        recovered: bool,
    ) -> None:
        lease_enabled = bool(getattr(self.config.execution, "enable_durable_task_runner", False))
        lease_owner = self._runner_id
        heartbeat_task: asyncio.Task | None = None
        try:
            if lease_enabled:
                acquired = await self.database.acquire_task_run_lease(
                    run_id=run_id,
                    lease_owner=lease_owner,
                    lease_seconds=self._run_lease_seconds,
                )
                if not acquired:
                    return
                self.event_bus.emit(
                    Event(
                        event_type=TASK_RUN_ACQUIRED,
                        task_id=task.id,
                        data={"run_id": run_id, "lease_owner": lease_owner},
                    ),
                )
                heartbeat_task = asyncio.create_task(
                    self._run_heartbeat_loop(run_id=run_id, task_id=task.id),
                )

            orchestrator = self.orchestrator
            if process is not None:
                orchestrator = self.create_task_orchestrator(process)

            if recovered and task.plan and task.plan.subtasks:
                for subtask in task.plan.subtasks:
                    if subtask.status.value == "running":
                        subtask.status = type(subtask.status).PENDING
                        subtask.summary = "Recovered after process restart; re-queued."
                await self._save_task_state(task)

            result = await orchestrator.execute_task(
                task,
                reuse_existing_plan=bool(recovered and task.plan and task.plan.subtasks),
            )
            raw_status = getattr(result, "status", TaskStatus.FAILED)
            await self._sync_task_row_snapshot(result)
            if lease_enabled:
                status_map = {
                    TaskStatus.COMPLETED: "completed",
                    TaskStatus.CANCELLED: "cancelled",
                }
                run_status = status_map.get(raw_status, "failed")
                await self.database.complete_task_run(
                    run_id=run_id,
                    status=run_status,
                    last_error="",
                )
        except asyncio.CancelledError:
            cancelled_task = task
            if cancelled_task.status != TaskStatus.CANCELLED:
                persisted_task = await self._load_task_state(task.id)
                if persisted_task is not None and persisted_task.status == TaskStatus.CANCELLED:
                    cancelled_task = persisted_task
            if cancelled_task.status == TaskStatus.CANCELLED:
                cancel_reason = ""
                if isinstance(cancelled_task.metadata, dict):
                    cancel_reason = str(
                        cancelled_task.metadata.get("cancel_reason", "") or "",
                    ).strip()
                try:
                    await self._save_task_state(cancelled_task)
                except Exception:
                    logger.debug(
                        "Failed persisting cancelled task state for %s",
                        cancelled_task.id,
                        exc_info=True,
                    )
                if lease_enabled:
                    try:
                        await self.database.complete_task_run(
                            run_id=run_id,
                            status="cancelled",
                            last_error=cancel_reason,
                        )
                    except Exception:
                        logger.debug(
                            "Failed completing cancelled task run %s",
                            run_id,
                            exc_info=True,
                        )
                self.event_bus.emit(
                    Event(
                        event_type=TASK_CANCELLED,
                        task_id=cancelled_task.id,
                        data={
                            "run_id": run_id,
                            "reason": cancel_reason or "cancel_requested",
                            "outcome": "cancelled",
                        },
                    ),
                )
            elif lease_enabled:
                try:
                    await self.database.requeue_task_run(run_id=run_id)
                except Exception:
                    logger.debug(
                        "Failed to requeue task run %s after cancellation",
                        run_id,
                        exc_info=True,
                    )
            raise
        except Exception as e:
            logger.exception("Task run %s failed: %s", run_id, e)
            try:
                task.status = TaskStatus.FAILED
                await self._save_task_state(task)
            except Exception:
                logger.exception("Failed persisting failure state for task %s", task.id)
            if lease_enabled:
                try:
                    await self.database.complete_task_run(
                        run_id=run_id,
                        status="failed",
                        last_error=f"{type(e).__name__}: {e}",
                    )
                except Exception:
                    logger.exception("Failed completing task run %s", run_id)
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)

    async def _run_heartbeat_loop(self, *, run_id: str, task_id: str) -> None:
        while True:
            await asyncio.sleep(self._run_heartbeat_interval_seconds)
            try:
                ok = await self.database.heartbeat_task_run(
                    run_id=run_id,
                    lease_owner=self._runner_id,
                    lease_seconds=self._run_lease_seconds,
                )
                if not ok:
                    return
                self.event_bus.emit(
                    Event(
                        event_type=TASK_RUN_HEARTBEAT,
                        task_id=task_id,
                        data={"run_id": run_id},
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Heartbeat failed for run %s", run_id, exc_info=True)


async def create_engine(config: Config, *, runtime_role: str = "api") -> Engine:
    """Create and initialize all Loom components."""
    import logging

    logger = logging.getLogger("loom.engine")

    # Database
    db_path = config.database_path
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database = Database(str(db_path))
        await database.initialize()
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        raise RuntimeError(
            f"Cannot initialize database at {db_path}: {e}"
        ) from e

    # State manager
    data_dir = Path(config.workspace.scratch_dir).expanduser()
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("Cannot create data directory: %s", e)
        raise RuntimeError(
            f"Cannot create data directory {data_dir}: {e}"
        ) from e
    state_manager = TaskStateManager(data_dir=data_dir)

    # Memory
    memory_manager = MemoryManager(database)
    conversation_store = ConversationStore(database)
    workspace_registry = WorkspaceRegistry(database)
    config_runtime_store = ConfigRuntimeStore(
        config,
        source_path=(
            Path(config.source_path).expanduser()
            if str(config.source_path or "").strip()
            else None
        ),
    )

    # Models
    model_router = ModelRouter.from_config(config)

    # Tools
    tool_registry = create_default_registry(
        config,
        mcp_startup_mode="background",
    )

    # Prompts
    prompt_assembler = PromptAssembler()

    # Events
    event_bus = EventBus(
        debug_diagnostics_rate_per_minute=int(
            getattr(
                config.telemetry,
                "debug_diagnostics_rate_per_minute",
                120,
            ),
        ),
        debug_diagnostics_burst=int(
            getattr(config.telemetry, "debug_diagnostics_burst", 30),
        ),
    )

    # Event persistence — subscribe to all events and write to SQLite
    event_persister = EventPersister(database)
    event_persister.attach(event_bus)

    # Webhook delivery — subscribe to terminal events
    webhook_delivery = WebhookDelivery()
    webhook_delivery.attach(event_bus)

    # Approval manager
    approval_manager = ApprovalManager(event_bus)
    question_manager: QuestionManager | None = None
    if bool(getattr(config.execution, "ask_user_durable_state_enabled", False)) or bool(
        getattr(config.execution, "ask_user_runtime_blocking_enabled", False),
    ) or bool(getattr(config.execution, "ask_user_api_enabled", False)):
        question_manager = QuestionManager(
            event_bus=event_bus,
            memory_manager=memory_manager,
        )

    # Learning manager
    learning_manager = LearningManager(database)

    # Orchestrator
    orchestrator = Orchestrator(
        model_router=model_router,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        prompt_assembler=prompt_assembler,
        state_manager=state_manager,
        event_bus=event_bus,
        config=config,
        approval_manager=approval_manager,
        question_manager=question_manager,
        learning_manager=learning_manager,
        execution_surface=normalize_tool_execution_surface(
            runtime_role,
            default="api",
        ),
    )

    engine = Engine(
        config=config,
        orchestrator=orchestrator,
        event_bus=event_bus,
        model_router=model_router,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        state_manager=state_manager,
        prompt_assembler=prompt_assembler,
        database=database,
        conversation_store=conversation_store,
        workspace_registry=workspace_registry,
        config_runtime_store=config_runtime_store,
        approval_manager=approval_manager,
        question_manager=question_manager,
        event_persister=event_persister,
        webhook_delivery=webhook_delivery,
        learning_manager=learning_manager,
        runtime_role=runtime_role,
    )
    try:
        await workspace_registry.sync_known_workspaces_from_sources()
    except Exception:
        logger.warning("Workspace registry bootstrap failed during startup", exc_info=True)
    if bool(getattr(config.execution, "enable_durable_task_runner", False)):
        try:
            await engine.recover_pending_task_runs()
        except Exception:
            logger.warning("Task run recovery failed during startup", exc_info=True)
    try:
        await engine.reconcile_interrupted_task_runs()
    except Exception:
        logger.warning("Interrupted task reconciliation failed during startup", exc_info=True)
    return engine
