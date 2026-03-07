"""Engine lifecycle: wires up all Loom components for the API server."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loom.config import Config
from loom.engine.orchestrator import Orchestrator
from loom.events.bus import Event, EventBus, EventPersister
from loom.events.types import (
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
from loom.state.memory import Database, MemoryManager
from loom.state.task_state import Task, TaskStateManager, TaskStatus
from loom.tools import create_default_registry
from loom.tools.registry import ToolRegistry
from loom.utils.latency import diagnostics_enabled, log_latency_event

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition

logger = logging.getLogger(__name__)
_API_EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS = 0.5
_API_EVENT_LOOP_LAG_WARN_THRESHOLD_SECONDS = 0.05


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
        approval_manager: ApprovalManager,
        question_manager: QuestionManager | None,
        webhook_delivery: WebhookDelivery,
        learning_manager: LearningManager,
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
        self.approval_manager = approval_manager
        self.question_manager = question_manager
        self.webhook_delivery = webhook_delivery
        self.learning_manager = learning_manager
        self._background_tasks: set[asyncio.Task] = set()
        self._runner_id = f"api-{uuid.uuid4().hex[:8]}"
        self._run_lease_seconds = 30
        self._run_heartbeat_interval_seconds = 10
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

    async def shutdown(self) -> None:
        """Graceful cleanup."""
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*list(self._background_tasks), return_exceptions=True)
        await self.model_router.close()
        await self.database.close()

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
        )

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
        self.state_manager.save(task)
        try:
            await self.database.update_task_metadata(task.id, metadata)
        except Exception:
            logger.debug("Failed updating task metadata for %s", task.id, exc_info=True)

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

        bg_task = asyncio.create_task(
            self._execute_task_run(
                task=task,
                run_id=assigned_run_id,
                process=process,
                process_name=process_name,
                recovered=recovered,
            ),
        )
        self._background_tasks.add(bg_task)
        bg_task.add_done_callback(self._background_tasks.discard)
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
            if not self.state_manager.exists(task_id):
                continue
            try:
                task = self.state_manager.load(task_id)
            except Exception:
                logger.warning(
                    "Failed loading state for recoverable task %s",
                    task_id,
                    exc_info=True,
                )
                continue
            if task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
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
                self.state_manager.save(task)

            result = await orchestrator.execute_task(
                task,
                reuse_existing_plan=bool(recovered and task.plan and task.plan.subtasks),
            )
            result_id = str(getattr(result, "id", "") or task.id)
            raw_status = getattr(result, "status", TaskStatus.FAILED)
            status_value = (
                raw_status.value
                if hasattr(raw_status, "value")
                else str(raw_status or TaskStatus.FAILED.value)
            )
            await self.database.update_task_status(result_id, status_value)
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
            if lease_enabled:
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
                self.state_manager.save(task)
                await self.database.update_task_status(task.id, TaskStatus.FAILED.value)
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


async def create_engine(config: Config) -> Engine:
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
        approval_manager=approval_manager,
        question_manager=question_manager,
        webhook_delivery=webhook_delivery,
        learning_manager=learning_manager,
    )
    if bool(getattr(config.execution, "enable_durable_task_runner", False)):
        try:
            await engine.recover_pending_task_runs()
        except Exception:
            logger.warning("Task run recovery failed during startup", exc_info=True)
    return engine
