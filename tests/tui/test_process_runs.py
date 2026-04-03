"""TUI process-run lifecycle tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestInitPersistence:
    """Test graceful degradation when database init fails."""

    def test_new_db_error_raises_without_ephemeral_opt_in(self, tmp_path):
        """New DB init failure should be blocking unless --ephemeral is enabled."""
        from loom.__main__ import PersistenceInitError, _init_persistence
        from loom.config import Config, MemoryConfig

        # Point database_path at a path with a null byte (invalid)
        cfg = Config(
            memory=MemoryConfig(
                database_path=str(tmp_path / "no" / "such" / "parent" / "\0invalid" / "loom.db"),
            ),
        )
        with pytest.raises(PersistenceInitError):
            _init_persistence(cfg)

    def test_returns_none_on_db_error_with_ephemeral_opt_in(self, tmp_path):
        """With explicit --ephemeral semantics, fallback returns (None, None)."""
        from loom.__main__ import _init_persistence
        from loom.config import Config, MemoryConfig

        cfg = Config(
            memory=MemoryConfig(
                database_path=str(tmp_path / "no" / "such" / "parent" / "\0invalid" / "loom.db"),
            ),
        )
        db, store = _init_persistence(cfg, allow_ephemeral=True)
        assert db is None
        assert store is None

    def test_returns_valid_on_success(self, tmp_path):
        """Sanity: happy path still works."""
        from loom.__main__ import _init_persistence
        from loom.config import Config, MemoryConfig

        cfg = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "loom.db")),
        )
        db, store = _init_persistence(cfg)
        assert db is not None
        assert store is not None

    def test_existing_db_failure_raises_persistence_error(self, tmp_path, monkeypatch):
        """Existing DB init failures should block startup (no silent ephemeral fallback)."""
        from loom.__main__ import PersistenceInitError, _init_persistence
        from loom.config import Config, MemoryConfig

        db_path = tmp_path / "loom.db"
        db_path.write_text("corrupt", encoding="utf-8")

        async def _boom(_self):
            raise RuntimeError("migration blew up")

        monkeypatch.setattr("loom.state.memory.Database.initialize", _boom)

        cfg = Config(
            memory=MemoryConfig(database_path=str(db_path)),
        )
        with pytest.raises(PersistenceInitError):
            _init_persistence(cfg)

    def test_existing_db_locked_raises_persistence_error(self, tmp_path, monkeypatch):
        """Locked existing DB should be surfaced as blocking startup failure."""
        import sqlite3

        from loom.__main__ import PersistenceInitError, _init_persistence
        from loom.config import Config, MemoryConfig

        db_path = tmp_path / "loom.db"
        db_path.write_text("locked", encoding="utf-8")

        async def _locked(_self):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr("loom.state.memory.Database.initialize", _locked)

        cfg = Config(
            memory=MemoryConfig(database_path=str(db_path)),
        )
        with pytest.raises(PersistenceInitError):
            _init_persistence(cfg)

class TestDelegateTaskUnbound:
    """delegate_task returns a clean error when not bound."""

    @pytest.mark.asyncio
    async def test_unbound_returns_not_available(self):
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        tool = DelegateTaskTool()
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({"goal": "do something"}, ctx)
        assert not result.success
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_missing_goal(self):
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        async def _factory():
            return MagicMock()

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({}, ctx)
        assert not result.success
        assert "goal" in result.error

    @pytest.mark.asyncio
    async def test_success_includes_subtask_progress_payload(self):
        from loom.state.task_state import (
            Plan,
            Subtask,
            SubtaskStatus,
            Task,
            TaskStatus,
        )
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        task = Task(
            id="task-123",
            goal="Analyze Tesla",
            workspace="/tmp",
            status=TaskStatus.COMPLETED,
        )
        task.plan = Plan(subtasks=[
            Subtask(
                id="company-screening",
                description="Company screening",
                phase_id="company-screening",
                status=SubtaskStatus.COMPLETED,
            ),
            Subtask(
                id="financial-analysis",
                description="Financial analysis",
                status=SubtaskStatus.RUNNING,
            ),
            Subtask(
                id="memo",
                description="Memo",
                status=SubtaskStatus.SKIPPED,
            ),
        ])

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(return_value=task)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["task_id"] == "task-123"
        assert result.data["tasks"][0]["status"] == "completed"
        assert result.data["tasks"][0]["phase_id"] == "company-screening"
        assert result.data["tasks"][1]["status"] == "in_progress"
        assert result.data["tasks"][2]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_failed_task_returns_failure_and_task_failed_progress(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        async def _execute(task):
            task.status = TaskStatus.FAILED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        progress: list[dict] = []
        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute(
            {"goal": "Analyze Tesla", "_progress_callback": progress.append},
            ctx,
        )

        assert result.success is False
        assert result.error == "Task execution failed."
        assert isinstance(result.data, dict)
        assert result.data.get("status") == "failed"
        assert any(p.get("event_type") == "task_failed" for p in progress)

    @pytest.mark.asyncio
    async def test_delegate_logs_detailed_event_stream_when_configured(self, tmp_path):
        from loom.config import Config, LoggingConfig
        from loom.events.bus import Event, EventBus
        from loom.events.types import TASK_EXECUTING
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        bus = EventBus()

        async def _execute(task):
            task.status = TaskStatus.EXECUTING
            bus.emit(Event(
                event_type=TASK_EXECUTING,
                task_id=task.id,
                data={"note": "running"},
            ))
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._events = bus
            orchestrator._config = Config(
                logging=LoggingConfig(
                    event_log_path=str(tmp_path / "loom-logs"),
                )
            )
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert isinstance(result.data, dict)
        event_log_path = str(result.data.get("event_log_path", "")).strip()
        assert event_log_path
        log_path = Path(event_log_path)
        assert log_path.exists()

        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(record.get("kind") == "meta" for record in records)
        assert any(record.get("kind") == "event_bus" for record in records)
        assert any(record.get("kind") == "progress_snapshot" for record in records)

    @pytest.mark.asyncio
    async def test_delegate_ignores_non_pathlike_log_config(self, tmp_path, monkeypatch):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        monkeypatch.chdir(tmp_path)

        async def _execute(task):
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._config = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert isinstance(result.data, dict)
        assert "event_log_path" not in result.data
        assert not list(tmp_path.glob("<MagicMock*"))

    @pytest.mark.asyncio
    async def test_progress_callback_includes_event_metadata(self):
        from loom.events.bus import Event, EventBus
        from loom.events.types import SUBTASK_STARTED
        from loom.state.task_state import Plan, Subtask, SubtaskStatus, TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        bus = EventBus()

        async def _execute(_task):
            _task.status = TaskStatus.EXECUTING
            _task.plan = Plan(subtasks=[
                Subtask(
                    id="company-screening",
                    description="Company screening",
                    status=SubtaskStatus.RUNNING,
                ),
            ])
            bus.emit(Event(
                event_type=SUBTASK_STARTED,
                task_id=_task.id,
                data={"subtask_id": "company-screening"},
            ))
            _task.plan.subtasks[0].status = SubtaskStatus.COMPLETED
            _task.status = TaskStatus.COMPLETED
            return _task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._events = bus
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        progress: list[dict] = []
        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))

        result = await tool.execute(
            {"goal": "Analyze Tesla", "_progress_callback": progress.append},
            ctx,
        )

        assert result.success is True
        events = [p for p in progress if p.get("event_type") == SUBTASK_STARTED]
        assert events, "Expected at least one streamed progress event"
        payload = events[0]
        assert payload["event_data"]["subtask_id"] == "company-screening"

    @pytest.mark.asyncio
    async def test_progress_callback_forwards_ask_user_events(self):
        from loom.events.bus import Event, EventBus
        from loom.events.types import ASK_USER_REQUESTED
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        bus = EventBus()

        async def _execute(task):
            task.status = TaskStatus.EXECUTING
            bus.emit(Event(
                event_type=ASK_USER_REQUESTED,
                task_id=task.id,
                data={
                    "subtask_id": "s1",
                    "question_id": "q-1",
                    "question": "Pick runtime",
                    "question_type": "single_choice",
                    "options": [
                        {"id": "python", "label": "Python", "description": ""},
                        {"id": "rust", "label": "Rust", "description": ""},
                    ],
                },
            ))
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._events = bus
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        progress: list[dict] = []
        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute(
            {"goal": "Analyze Tesla", "_progress_callback": progress.append},
            ctx,
        )

        assert result.success is True
        events = [p for p in progress if p.get("event_type") == ASK_USER_REQUESTED]
        assert events
        assert events[0]["event_data"]["question_id"] == "q-1"

    @pytest.mark.asyncio
    async def test_delegate_registers_and_clears_cancel_handler(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        captured: dict[str, object] = {}
        cleared: list[bool] = []

        async def _execute(task):
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.cancel_task = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        def _register(payload):
            captured.update(payload)

        def _clear():
            cleared.append(True)

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute(
            {
                "goal": "Analyze Tesla",
                "_register_cancel_handler": _register,
                "_clear_cancel_handler": _clear,
            },
            ctx,
        )

        assert result.success is True
        assert isinstance(result.data, dict)
        assert captured.get("task_id") == result.data.get("task_id")
        assert callable(captured.get("cancel"))
        assert callable(captured.get("pause"))
        assert callable(captured.get("resume"))
        assert callable(captured.get("inject"))
        assert callable(captured.get("answer_question"))
        assert cleared == [True]

    @pytest.mark.asyncio
    async def test_delegate_cancel_handler_requests_orchestrator_cancel(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        gate = asyncio.Event()
        cancel_holder: dict[str, object] = {}
        registered = asyncio.Event()

        async def _execute(task):
            await gate.wait()
            return task

        def _cancel_task(task):
            task.status = TaskStatus.CANCELLED
            gate.set()

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.cancel_task = MagicMock(side_effect=_cancel_task)
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        def _register(payload):
            cancel_holder.update(payload)
            registered.set()

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        pending = asyncio.create_task(tool.execute(
            {
                "goal": "Analyze Tesla",
                "_register_cancel_handler": _register,
            },
            ctx,
        ))
        await asyncio.wait_for(registered.wait(), timeout=1.0)

        cancel_fn = cancel_holder.get("cancel")
        assert callable(cancel_fn)
        cancel_response = await cancel_fn(wait_timeout_seconds=0.2)
        assert cancel_response["requested"] is True
        assert cancel_response["path"] == "orchestrator"
        assert cancel_response["timeout"] is False

        result = await pending
        assert result.success is False
        assert result.error == "Task execution cancelled."
        assert isinstance(result.data, dict)
        assert result.data.get("status") == "cancelled"

    @pytest.mark.asyncio
    async def test_delegate_cancellation_does_not_cancel_paused_task(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        orchestrator_holder: dict[str, object] = {}

        async def _execute(task):
            task.status = TaskStatus.PAUSED
            await asyncio.Future()
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.cancel_task = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            orchestrator_holder["value"] = orchestrator
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        pending = asyncio.create_task(tool.execute({"goal": "Analyze Tesla"}, ctx))
        await asyncio.sleep(0.05)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        orchestrator = orchestrator_holder["value"]
        orchestrator.cancel_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_delegate_creates_fresh_orchestrator_per_execute(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        created: list[MagicMock] = []

        async def _execute(task):
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            created.append(orchestrator)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))

        first = await tool.execute({"goal": "first"}, ctx)
        second = await tool.execute({"goal": "second"}, ctx)

        assert first.success is True
        assert second.success is True
        assert len(created) == 2
        assert created[0] is not created[1]
        created[0].execute_task.assert_awaited_once()
        created[1].execute_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delegate_applies_internal_approval_mode_override(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        observed_modes: list[str] = []

        async def _execute(task):
            observed_modes.append(task.approval_mode)
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))

        result = await tool.execute(
            {"goal": "Analyze Tesla", "_approval_mode": "disabled"},
            ctx,
        )

        assert result.success is True
        assert observed_modes == ["disabled"]

    @pytest.mark.asyncio
    async def test_delegate_defaults_to_disabled_approval_in_tui_surface(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        observed_modes: list[str] = []

        async def _execute(task):
            observed_modes.append(task.approval_mode)
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))

        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert observed_modes == ["disabled"]

    @pytest.mark.asyncio
    async def test_delegate_keeps_confidence_threshold_default_for_api_surface(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        observed_modes: list[str] = []

        async def _execute(task):
            observed_modes.append(task.approval_mode)
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"), execution_surface="api")

        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert observed_modes == ["confidence_threshold"]

    @pytest.mark.asyncio
    async def test_delegate_resume_reuses_plan_and_resets_incomplete_subtasks(self):
        from loom.state.task_state import Plan, Subtask, SubtaskStatus, Task, TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        task = Task(
            id="cowork-1234",
            goal="Original goal",
            workspace="/tmp",
            status=TaskStatus.FAILED,
            context={"existing": "ctx"},
            plan=Plan(subtasks=[
                Subtask(
                    id="done",
                    description="already done",
                    status=SubtaskStatus.COMPLETED,
                ),
                Subtask(
                    id="retry-me",
                    description="retry this",
                    status=SubtaskStatus.FAILED,
                    summary="failed before",
                    active_issue="timeout",
                    retry_count=2,
                ),
                Subtask(
                    id="blocked",
                    description="blocked by failure",
                    status=SubtaskStatus.SKIPPED,
                    summary="Skipped: blocked",
                ),
            ]),
        )

        observed: dict = {}

        async def _execute(resume_task, **kwargs):
            observed["reuse_existing_plan"] = kwargs.get("reuse_existing_plan")
            observed["statuses"] = [subtask.status for subtask in resume_task.plan.subtasks]
            resume_task.status = TaskStatus.COMPLETED
            return resume_task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._state = MagicMock()
            orchestrator._state.load = MagicMock(return_value=task)
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute(
            {
                "goal": "Updated goal",
                "context": {"new": "value"},
                "_resume_task_id": "cowork-1234",
            },
            ctx,
        )

        assert result.success is True
        assert observed["reuse_existing_plan"] is True
        assert observed["statuses"] == [
            SubtaskStatus.COMPLETED,
            SubtaskStatus.PENDING,
            SubtaskStatus.PENDING,
        ]
        assert task.goal == "Updated goal"
        assert task.context["existing"] == "ctx"
        assert task.context["new"] == "value"

    @pytest.mark.asyncio
    async def test_delegate_resume_rejects_completed_task(self):
        from loom.state.task_state import Task, TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        task = Task(
            id="cowork-done",
            goal="Completed",
            workspace="/tmp",
            status=TaskStatus.COMPLETED,
        )

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._state = MagicMock()
            orchestrator._state.load = MagicMock(return_value=task)
            orchestrator.execute_task = AsyncMock()
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute(
            {"goal": "Completed", "_resume_task_id": "cowork-done"},
            ctx,
        )

        assert result.success is False
        assert "already completed" in (result.error or "").lower()

    def test_delegate_timeout_default(self, monkeypatch):
        from loom.tools.delegate_task import DelegateTaskTool

        monkeypatch.delenv("LOOM_DELEGATE_TIMEOUT_SECONDS", raising=False)
        tool = DelegateTaskTool()
        assert tool.timeout_seconds == 3600

    def test_delegate_timeout_env_override(self, monkeypatch):
        from loom.tools.delegate_task import DelegateTaskTool

        monkeypatch.setenv("LOOM_DELEGATE_TIMEOUT_SECONDS", "7200")
        tool = DelegateTaskTool()
        assert tool.timeout_seconds == 7200

    def test_delegate_timeout_constructor_override(self, monkeypatch):
        from loom.tools.delegate_task import DelegateTaskTool

        monkeypatch.delenv("LOOM_DELEGATE_TIMEOUT_SECONDS", raising=False)
        tool = DelegateTaskTool(timeout_seconds=5400)
        assert tool.timeout_seconds == 5400

class TestSessionResumeFallback:
    """Verify CoworkSession.resume raises on bad IDs."""

    @pytest.mark.asyncio
    async def test_resume_nonexistent_session(self, tmp_path):
        """Resuming a session that doesn't exist should raise."""
        from loom.cowork.session import CoworkSession
        from loom.state.conversation_store import ConversationStore
        from loom.state.memory import Database

        db = Database(tmp_path / "test.db")
        await db.initialize()
        store = ConversationStore(db)

        model = MagicMock()
        model.name = "test"
        tools = MagicMock()
        tools.all_schemas.return_value = []

        session = CoworkSession(
            model=model, tools=tools, store=store,
        )
        with pytest.raises(Exception):
            await session.resume("nonexistent-session-id")

    @pytest.mark.asyncio
    async def test_resume_without_store_raises(self):
        """Resuming with no store should raise."""
        from loom.cowork.session import CoworkSession

        model = MagicMock()
        model.name = "test"
        tools = MagicMock()
        tools.all_schemas.return_value = []

        session = CoworkSession(model=model, tools=tools)
        with pytest.raises(Exception):
            await session.resume("any-id")

class TestCLIResumeNoDB:
    """--resume should fail cleanly when persistence is unavailable."""

    def test_resume_flag_without_db(self, monkeypatch):
        """If _init_persistence returns None, --resume should exit 1."""
        from click.testing import CliRunner

        import loom.cli.commands.root as root_mod
        from loom.__main__ import cli

        monkeypatch.setattr(
            root_mod,
            "_init_persistence",
            lambda cfg, **_kwargs: (None, None),
        )
        # Mock model resolution so we don't need a real config
        monkeypatch.setattr(
            root_mod, "_resolve_model", lambda cfg, name: MagicMock(name="mock"),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--resume", "abc123"],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert "requires database" in result.output

class TestInputSubmitGuard:
    """P0-2: App should only handle Input.Submitted from #user-input."""

    def test_on_user_submit_uses_selector(self):
        """Verify the handler is decorated with the '#user-input' selector."""
        from loom.tui.app import LoomApp

        bindings = getattr(LoomApp.on_user_submit, "_textual_on", [])
        assert bindings, "Expected textual on-handler metadata"
        _event_cls, selector_spec = bindings[0]
        control = selector_spec.get("control", ())
        assert control
        assert "#user-input" in str(control[0])

    def test_setup_screen_stops_input_submitted(self):
        """SetupScreen.on_input_submitted should call event.stop()."""
        from unittest.mock import MagicMock

        from loom.tui.screens.setup import _STEP_DETAILS, SetupScreen

        screen = SetupScreen()
        # Mock _show_step to prevent reactive watcher from doing DOM ops
        screen._show_step = MagicMock()
        screen._step = _STEP_DETAILS
        screen._provider_key = "ollama"

        event = MagicMock()
        event.input.id = "input-model"

        # Mock query_one for the model input focus
        screen.query_one = MagicMock()

        screen.on_input_submitted(event)
        event.stop.assert_called_once()

class TestFinalizeSetupResetsSession:
    """P1-5: Re-running /setup should invalidate the old session."""

    @pytest.mark.asyncio
    async def test_finalize_setup_clears_session(self, monkeypatch):
        """_finalize_setup should invalidate old session and reinitialize."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=None,
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._initialize_session = AsyncMock()

        fake_config = MagicMock()
        fake_model = MagicMock()
        fake_router = MagicMock()
        fake_router.select.return_value = fake_model

        monkeypatch.setattr("loom.config.load_config", lambda: fake_config)
        monkeypatch.setattr(
            "loom.models.router.ModelRouter.from_config",
            lambda cfg: fake_router,
        )

        # Call the undecorated coroutine behind @work.
        await LoomApp._finalize_setup.__wrapped__(app)

        assert app._config is fake_config
        assert app._model is fake_model
        app._store.update_session.assert_awaited_once_with(
            "old-session", is_active=False,
        )
        app._initialize_session.assert_awaited_once()

class TestFilesPanelReset:
    def test_clear_files_panel(self):
        """Session changes should reset the files panel and clear the diff."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=None,
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        panel = MagicMock()
        app.query_one = MagicMock(return_value=panel)

        app._clear_files_panel()

        panel.clear_files.assert_called_once()
        panel.show_diff.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_new_session_clears_files_panel(self):
        """Creating a new session should clear stale file history."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._model = MagicMock()
        app._model.name = "test-model"
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._store.create_session = AsyncMock(return_value="new-session")
        app._bind_session_tools = MagicMock()
        panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return panel
            return chat

        app.query_one = MagicMock(side_effect=_query_one)

        await app._new_session()

        panel.clear_files.assert_called_once()
        panel.show_diff.assert_called_once_with("")

class TestWorkspaceRefresh:
    def test_ingest_files_panel_falls_back_to_summary_markers(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        panel = MagicMock()
        panel._all_entries = []
        app.query_one = MagicMock(return_value=panel)

        count = app._ingest_files_panel_from_paths(
            ["(2 files created)", "(1 files modified)"],
            operation_hint="modify",
        )

        assert count == 2
        entries = panel.update_files.call_args.args[0]
        assert entries[0]["operation"] == "create"
        assert entries[0]["path"] == "(2 files created)"
        assert entries[1]["operation"] == "modify"
        assert entries[1]["path"] == "(1 files modified)"

    def test_update_files_panel_refreshes_sidebar_tree(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        files_panel = MagicMock()
        sidebar = MagicMock()
        app.notify = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return files_panel
            if selector == "#sidebar":
                return sidebar
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        turn = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    name="write_file",
                    args={"path": "new-file.md"},
                    result=SimpleNamespace(success=True, output="ok"),
                ),
            ],
        )

        app._update_files_panel(turn)

        files_panel.update_files.assert_called_once()
        sidebar.refresh_workspace_tree.assert_called_once()

    def test_update_files_panel_refreshes_tree_for_document_write(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        files_panel = MagicMock()
        sidebar = MagicMock()
        app.notify = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return files_panel
            if selector == "#sidebar":
                return sidebar
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        turn = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    name="document_write",
                    args={"path": "memo.md"},
                    result=SimpleNamespace(success=True, output="ok"),
                ),
            ],
        )

        app._update_files_panel(turn)

        files_panel.update_files.assert_not_called()
        sidebar.refresh_workspace_tree.assert_called_once()

    @pytest.mark.asyncio
    async def test_cowork_delegate_progress_updates_chat_section(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        chat.append_delegate_progress_line.return_value = True
        sidebar = MagicMock()
        events_panel = MagicMock()
        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: {
            "#chat-log": chat,
            "#sidebar": sidebar,
            "#events-panel": events_panel,
        }[selector])
        app._append_chat_replay_event = AsyncMock()
        app._request_workspace_refresh = MagicMock()
        app._ingest_files_panel_from_paths = MagicMock(return_value=1)

        await app._on_cowork_delegate_progress_event({
            "tool_call_id": "call_1",
            "caller_tool_name": "delegate_task",
            "event_type": "tool_call_completed",
            "event_data": {
                "tool": "write_file",
                "subtask_id": "scope",
                "files_changed": ["notes.md"],
            },
            "tasks": [
                {"id": "scope", "status": "completed", "content": "Scope work"},
            ],
        })

        chat.add_delegate_progress_section.assert_called()
        chat.append_delegate_progress_line.assert_called_once()
        replay_events = [call.args[0] for call in app._append_chat_replay_event.await_args_list]
        assert replay_events == ["delegate_progress_started"]
        app._request_workspace_refresh.assert_called_once()
        app._ingest_files_panel_from_paths.assert_called_once()

    @pytest.mark.asyncio
    async def test_cowork_delegate_progress_ignores_late_events_after_finalize(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        sidebar = MagicMock()
        events_panel = MagicMock()
        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: {
            "#chat-log": chat,
            "#sidebar": sidebar,
            "#events-panel": events_panel,
        }[selector])
        app._append_chat_replay_event = AsyncMock()
        app._request_workspace_refresh = MagicMock()
        app._ingest_files_panel_from_paths = MagicMock(return_value=0)
        app._active_delegate_streams["call_done"] = {
            "tool_call_id": "call_done",
            "title": "Delegated progress",
            "status": "completed",
            "elapsed_ms": 1200,
            "lines": ["Delegated task completed."],
            "started_at": 1.0,
            "finalized": True,
        }

        await app._on_cowork_delegate_progress_event({
            "tool_call_id": "call_done",
            "caller_tool_name": "delegate_task",
            "event_type": "subtask_started",
            "event_data": {
                "subtask_id": "late",
            },
        })

        chat.add_delegate_progress_section.assert_not_called()
        chat.append_delegate_progress_line.assert_not_called()
        app._append_chat_replay_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cowork_delegate_progress_skips_token_streamed_lines(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        chat.append_delegate_progress_line.return_value = True
        sidebar = MagicMock()
        events_panel = MagicMock()
        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: {
            "#chat-log": chat,
            "#sidebar": sidebar,
            "#events-panel": events_panel,
        }[selector])
        app._append_chat_replay_event = AsyncMock()
        app._request_workspace_refresh = MagicMock()
        app._ingest_files_panel_from_paths = MagicMock(return_value=0)

        await app._on_cowork_delegate_progress_event({
            "tool_call_id": "call_tokens",
            "caller_tool_name": "delegate_task",
            "event_type": "token_streamed",
            "event_data": {
                "subtask_id": "scope",
                "token_count": 12,
            },
        })

        chat.add_delegate_progress_section.assert_called_once()
        chat.append_delegate_progress_line.assert_not_called()
        replay_events = [call.args[0] for call in app._append_chat_replay_event.await_args_list]
        assert replay_events == ["delegate_progress_started"]

    def test_format_process_progress_event_uses_cowork_delegate_terminal_phrasing(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        assert (
            app._format_process_progress_event(
                {"event_type": "task_completed", "event_data": {}},
                context="cowork_delegate",
            )
            == "Delegated task completed."
        )
        assert (
            app._format_process_progress_event(
                {
                    "event_type": "task_failed",
                    "event_data": {"reason": "rate limited"},
                },
                context="cowork_delegate",
            )
            == "Delegated task failed: rate limited"
        )

    def test_render_chat_event_delegate_progress_rehydrates(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        chat.append_delegate_progress_line.return_value = True
        app.query_one = MagicMock(return_value=chat)

        assert app._render_chat_event({
            "event_type": "delegate_progress_started",
            "payload": {
                "tool_call_id": "call_2",
                "title": "Delegated progress",
            },
        })
        assert app._render_chat_event({
            "event_type": "delegate_progress_line",
            "payload": {
                "tool_call_id": "call_2",
                "line": "Started scope.",
            },
        })
        assert app._render_chat_event({
            "event_type": "delegate_progress_finalized",
            "payload": {
                "tool_call_id": "call_2",
                "title": "Delegated progress",
                "status": "completed",
                "elapsed_ms": 320,
                "lines": ["Started scope.", "Completed scope."],
            },
        })

        assert chat.add_delegate_progress_section.call_count >= 2
        chat.append_delegate_progress_line.assert_called_once()
        chat.finalize_delegate_progress_section.assert_called_once()
        stream = app._active_delegate_streams.get("call_2")
        assert isinstance(stream, dict)
        assert stream.get("status") == "completed"

    def test_render_chat_event_tool_completion_updates_existing_widget(self):
        from loom.tui.app import LoomApp
        from loom.tui.widgets.chat_log import ChatLog

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = ChatLog()
        mounted: list = []
        chat.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        chat._scroll_to_end = lambda: None
        app.query_one = MagicMock(return_value=chat)

        assert app._render_chat_event({
            "event_type": "tool_call_started",
            "payload": {
                "tool_name": "web_search",
                "tool_call_id": "call_3",
                "args": {"query": "blink49 canada"},
            },
        })
        assert app._render_chat_event({
            "event_type": "tool_call_completed",
            "payload": {
                "tool_name": "web_search",
                "tool_call_id": "call_3",
                "args": {"query": "blink49 canada"},
                "success": True,
                "elapsed_ms": 280,
                "output": "1. Blink49\n   https://blink49.com",
            },
        })

        assert len(mounted) == 1
        assert getattr(mounted[0], "_success", None) is True

    def test_render_chat_event_turn_interrupted_rehydrates(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        assert app._render_chat_event({
            "event_type": "turn_interrupted",
            "payload": {
                "message": "Stopped current chat execution.",
                "markup": True,
            },
        })
        chat.add_info.assert_called_once_with("Stopped current chat execution.", markup=True)

    def test_render_chat_event_assistant_thinking_rehydrates(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        assert app._render_chat_event({
            "event_type": "assistant_thinking",
            "payload": {
                "text": "Looking up more context.",
                "streaming": True,
            },
        })
        chat.add_live_feedback.assert_called_once_with("Looking up more context.")

    @pytest.mark.asyncio
    async def test_switch_session_clears_files_panel(self, monkeypatch):
        """Switching sessions should clear stale file history."""
        from loom.tui.app import LoomApp

        class FakeSession:
            def __init__(self, *args, **kwargs):
                self.session_id = ""
                self.session_state = SimpleNamespace(turn_count=3)
                self.total_tokens = 99
                self.workspace = Path("/tmp")

            async def resume(self, session_id: str):
                self.session_id = session_id

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._model = MagicMock()
        app._model.name = "test-model"
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._bind_session_tools = MagicMock()
        panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return panel
            return chat

        app.query_one = MagicMock(side_effect=_query_one)

        await app._switch_to_session("new-session")

        panel.clear_files.assert_called_once()
        panel.show_diff.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_new_session_uses_process_system_prompt(self):
        """Creating a new session should preserve active process prompt text."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._model = MagicMock()
        app._model.name = "test-model"
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._store.create_session = AsyncMock(return_value="new-session")
        app._bind_session_tools = MagicMock()
        app._process_defn = SimpleNamespace(
            persona="You are a process specialist.",
            tool_guidance="Use process tool guidance.",
        )
        panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return panel
            return chat

        app.query_one = MagicMock(side_effect=_query_one)

        await app._new_session()

        create_kwargs = app._store.create_session.await_args.kwargs
        prompt = create_kwargs["system_prompt"]
        assert "DOMAIN ROLE" in prompt
        assert "process specialist" in prompt
        assert "DOMAIN TOOL GUIDANCE" in prompt

    @pytest.mark.asyncio
    async def test_switch_session_uses_process_system_prompt(self, monkeypatch):
        """Switching sessions should keep process prompt extensions."""
        from loom.tui.app import LoomApp

        captured_prompt: dict[str, str] = {}

        class FakeSession:
            def __init__(self, *args, **kwargs):
                captured_prompt["value"] = kwargs["system_prompt"]
                self.session_id = ""
                self.session_state = SimpleNamespace(turn_count=3)
                self.total_tokens = 99
                self.workspace = Path("/tmp")

            async def resume(self, session_id: str):
                self.session_id = session_id

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._model = MagicMock()
        app._model.name = "test-model"
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._bind_session_tools = MagicMock()
        app._process_defn = SimpleNamespace(
            persona="You are a process specialist.",
            tool_guidance="Use process tool guidance.",
        )
        panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return panel
            return chat

        app.query_one = MagicMock(side_effect=_query_one)

        await app._switch_to_session("new-session")

        prompt = captured_prompt["value"]
        assert "DOMAIN ROLE" in prompt
        assert "process specialist" in prompt
        assert "DOMAIN TOOL GUIDANCE" in prompt

class TestDelegateBindingProcess:
    @pytest.mark.asyncio
    async def test_delegate_orchestrator_factory_includes_process(self, monkeypatch):
        """delegate_task should spawn process-aware orchestrators when active."""
        from loom.tools.conversation_recall import ConversationRecallTool
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolRegistry
        from loom.tui.app import LoomApp

        captured: dict[str, object] = {}
        created_registry_args: dict[str, object] = {}

        class FakeOrchestrator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("loom.engine.orchestrator.Orchestrator", FakeOrchestrator)
        monkeypatch.setattr("loom.events.bus.EventBus", lambda: MagicMock())
        monkeypatch.setattr(
            "loom.models.router.ModelRouter.from_config",
            lambda _cfg: MagicMock(),
        )
        monkeypatch.setattr("loom.prompts.assembler.PromptAssembler", lambda: MagicMock())
        monkeypatch.setattr("loom.state.memory.MemoryManager", lambda _db: MagicMock())
        monkeypatch.setattr("loom.state.task_state.TaskStateManager", lambda _dir: MagicMock())
        monkeypatch.setattr(
            "loom.tools.create_default_registry",
            lambda _config=None: (
                created_registry_args.update({"config": _config}) or MagicMock()
            ),
        )

        registry = ToolRegistry()
        registered_delegate = DelegateTaskTool()
        registry.register(ConversationRecallTool())
        registry.register(registered_delegate)

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=registry,
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._ensure_persistence_tools()
        app._session = SimpleNamespace(
            session_id="session-id",
            session_state=SimpleNamespace(),
        )
        app._config = SimpleNamespace(
            workspace=SimpleNamespace(scratch_dir="/tmp"),
        )
        app._db = MagicMock()
        app._process_defn = SimpleNamespace(
            name="marketing-strategy",
            tools=SimpleNamespace(excluded=[]),
        )

        assert app._delegate_tool is registered_delegate
        app._bind_session_tools()

        assert app._delegate_tool._factory is not None
        await app._delegate_tool._factory()
        assert captured["process"] is app._process_defn
        assert created_registry_args.get("config") is app._config

    def test_apply_process_tool_policy_reports_missing_required_tools(self):
        """Missing required process tools should produce a visible warning."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        registry = MagicMock()
        registry.has.return_value = False
        app._tools = registry
        app._process_defn = SimpleNamespace(
            tools=SimpleNamespace(excluded=[], required=["missing-tool"]),
        )
        chat = MagicMock()

        app._apply_process_tool_policy(chat)

        chat.add_info.assert_called_once()
        assert "missing-tool" in chat.add_info.call_args.args[0]

class TestStartupSessionResume:
    @pytest.mark.asyncio
    async def test_resolve_startup_resume_target_prefers_explicit(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            resume_session="explicit-session",
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock()

        session_id, auto = await app._resolve_startup_resume_target()

        assert session_id == "explicit-session"
        assert auto is False
        app._store.list_sessions.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_startup_resume_target_uses_latest_workspace(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[
            {"id": "latest-session"},
            {"id": "older-session"},
        ])

        session_id, auto = await app._resolve_startup_resume_target()

        assert session_id == "latest-session"
        assert auto is True
        app._store.list_sessions.assert_awaited_once_with(workspace="/tmp")

    @pytest.mark.asyncio
    async def test_resolve_startup_resume_target_disabled_after_initial_start(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[
            {"id": "latest-session"},
        ])
        app._auto_resume_workspace_on_init = False

        session_id, auto = await app._resolve_startup_resume_target()

        assert session_id is None
        assert auto is False
        app._store.list_sessions.assert_not_called()

    def test_should_show_startup_landing_respects_config_and_resume(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(
                tui=SimpleNamespace(
                    startup_landing_enabled=True,
                    always_open_chat_directly=False,
                ),
            ),
        )
        app._store = MagicMock()

        assert app._should_show_startup_landing(resume_target=None) is True
        assert app._should_show_startup_landing(resume_target="existing-session") is False

        app._config.tui.always_open_chat_directly = True
        assert app._should_show_startup_landing(resume_target=None) is False

        app._config.tui.always_open_chat_directly = False
        app._config.tui.startup_landing_enabled = False
        assert app._should_show_startup_landing(resume_target=None) is False

    @pytest.mark.asyncio
    async def test_on_mount_shows_landing_when_no_resume(self):
        from textual.widgets import Input, Static

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(
                models={"primary": SimpleNamespace(model="kimi-k2.5")},
                tui=SimpleNamespace(
                    startup_landing_enabled=True,
                    always_open_chat_directly=False,
                ),
            ),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[])
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            assert app._startup_landing_active is True
            assert app.query_one("#bottom-stack").has_class("landing")
            assert app.query_one("#bottom-stack").display is False
            workspace_line = app.query_one("#landing-workspace-path", Static)
            assert "workspace:" in str(workspace_line.render()).lower()
            assert "/tmp" in str(workspace_line.render())
            model_line = app.query_one("#landing-model-name", Static)
            assert "model:" in str(model_line.render()).lower()
            assert "kimi-k2.5" in str(model_line.render())
            logo = app.query_one("#landing-logo", Static)
            assert "___" in str(logo.render())
            close_button = app.query_one("#landing-close-btn", Static)
            assert "x" in str(close_button.render()).lower()
            landing_input = app.query_one("#landing-input", Input)
            assert landing_input.placeholder == "Give me a challenge"
            landing_shortcuts = " ".join(
                str(widget.render())
                for widget in app.query("#landing-shortcuts .landing-shortcut")
            ).lower()
            assert "ctrl + p" in landing_shortcuts
            assert "ctrl + c" in landing_shortcuts
            assert "esc" in landing_shortcuts
            assert "goto main window" in landing_shortcuts
            assert "ctrl + a" not in landing_shortcuts
            assert "ctrl + m" not in landing_shortcuts

        app._initialize_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_landing_close_button_enters_workspace_surface(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(
                models={"primary": SimpleNamespace(model="kimi-k2.5")},
                tui=SimpleNamespace(
                    startup_landing_enabled=True,
                    always_open_chat_directly=False,
                ),
            ),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[])
        app._initialize_session = AsyncMock(
            side_effect=lambda **_kwargs: setattr(
                app,
                "_session",
                SimpleNamespace(session_id="landing-session"),
            ),
        )

        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            assert app._startup_landing_active is True
            await pilot.click("#landing-close-btn")
            await pilot.pause()
            assert app._startup_landing_active is False
            assert app.query_one("#bottom-stack").display is True

        app._initialize_session.assert_awaited_once_with(
            allow_auto_resume=False,
            emit_info_messages=False,
        )

    @pytest.mark.asyncio
    async def test_landing_escape_enters_workspace_surface(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(
                models={"primary": SimpleNamespace(model="kimi-k2.5")},
                tui=SimpleNamespace(
                    startup_landing_enabled=True,
                    always_open_chat_directly=False,
                ),
            ),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[])
        app._initialize_session = AsyncMock(
            side_effect=lambda **_kwargs: setattr(
                app,
                "_session",
                SimpleNamespace(session_id="landing-session"),
            ),
        )

        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            assert app._startup_landing_active is True
            await pilot.press("escape")
            await pilot.pause()
            assert app._startup_landing_active is False
            assert app.query_one("#bottom-stack").display is True

        app._initialize_session.assert_awaited_once_with(
            allow_auto_resume=False,
            emit_info_messages=False,
        )

    @pytest.mark.asyncio
    async def test_landing_input_accepts_slash_commands(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        landing_input = SimpleNamespace(value="seeded")
        app.query_one = MagicMock(return_value=landing_input)
        app._enter_workspace_surface = AsyncMock()
        app._reset_slash_tab_cycle = MagicMock()
        app._reset_input_history_navigation = MagicMock()
        app._set_slash_hint = MagicMock()
        app._handle_slash_command = AsyncMock(return_value=True)
        app._append_input_history = MagicMock()
        app._persist_process_run_ui_state = AsyncMock()

        await app._submit_user_text("/help", source="landing")

        app._enter_workspace_surface.assert_awaited_once_with(ensure_session=True)
        app._handle_slash_command.assert_awaited_once_with("/help")
        app._append_input_history.assert_called_once_with("/help")
        app._persist_process_run_ui_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_landing_input_plain_text_dispatches_run_goal(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        landing_input = SimpleNamespace(value="seeded")
        app.query_one = MagicMock(return_value=landing_input)
        app._enter_workspace_surface = AsyncMock()
        app._reset_slash_tab_cycle = MagicMock()
        app._reset_input_history_navigation = MagicMock()
        app._set_slash_hint = MagicMock()
        app._handle_slash_command = AsyncMock(return_value=True)
        app._append_input_history = MagicMock()
        app._persist_process_run_ui_state = AsyncMock()
        app._chat_turn_worker = None

        await app._submit_user_text("analyze cloud infra spend", source="landing")

        app._enter_workspace_surface.assert_awaited_once_with(ensure_session=True)
        app._handle_slash_command.assert_awaited_once_with(
            "/run analyze cloud infra spend"
        )
        app._append_input_history.assert_called_once_with(
            "/run analyze cloud infra spend"
        )
        app._persist_process_run_ui_state.assert_awaited_once()
        assert app._chat_turn_worker is None

    @pytest.mark.asyncio
    async def test_landing_slash_hint_is_anchored_and_persists_refresh(self):
        from textual.containers import VerticalScroll
        from textual.widgets import Static

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(
                models={"primary": SimpleNamespace(model="kimi-k2.5")},
                tui=SimpleNamespace(
                    startup_landing_enabled=True,
                    always_open_chat_directly=False,
                ),
            ),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[])
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            assert app._startup_landing_active is True

            await pilot.press("/")
            await pilot.pause()

            landing_hint = app.query_one("#landing-slash-hint", VerticalScroll)
            landing_hint_body = app.query_one("#landing-slash-hint-body", Static)
            chat_hint = app.query_one("#slash-hint", VerticalScroll)
            assert landing_hint.display is True
            assert landing_hint_body.display is True
            assert "slash commands" in str(landing_hint_body.render()).lower()
            assert chat_hint.display is False

            app._refresh_hint_panel()
            await pilot.pause()

            assert landing_hint.display is True
            assert "slash commands" in str(landing_hint_body.render()).lower()

    @pytest.mark.asyncio
    async def test_on_mount_bypass_opens_chat_directly(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(
                tui=SimpleNamespace(
                    startup_landing_enabled=True,
                    always_open_chat_directly=True,
                ),
            ),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[])
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            assert app._startup_landing_active is False

        app._initialize_session.assert_awaited_once_with(
            startup_resume=(None, False),
            allow_auto_resume=False,
            emit_info_messages=True,
        )

    @pytest.mark.asyncio
    async def test_initialize_session_auto_resume_restores_input_history(self, monkeypatch):
        from loom.tui.app import LoomApp

        class FakeSession:
            def __init__(self, **_kwargs):
                self.total_tokens = 12
                self.session_id = "latest-session"
                self.session_state = SimpleNamespace(
                    turn_count=4,
                    ui_state={
                        "input_history": {
                            "version": 1,
                            "items": ["/help", "summarize the doc"],
                        },
                    },
                )
                self.messages = [{"role": "system", "content": "system"}]

            async def resume(self, _session_id: str) -> None:
                self.messages = [
                    {"role": "system", "content": "system"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "summarize the doc"},
                ]

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._resolve_startup_resume_target = AsyncMock(return_value=("latest-session", True))
        app._refresh_tool_registry = MagicMock()
        app._refresh_process_command_index = MagicMock()
        app._ensure_persistence_tools = MagicMock()
        app._apply_process_tool_policy = MagicMock()
        app._build_system_prompt = MagicMock(return_value="system")
        app._bind_session_tools = MagicMock()
        app._restore_process_run_tabs = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._tools.list_tools = MagicMock(return_value=[])

        chat = MagicMock()
        status = SimpleNamespace(workspace_name="", model_name="", process_name="")
        input_widget = SimpleNamespace(focus=MagicMock())

        def _query_one(selector: str, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            if selector == "#user-input":
                return input_widget
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        await app._initialize_session()

        assert app._input_history == ["/help", "summarize the doc"]

    @pytest.mark.asyncio
    async def test_initialize_session_auto_resume_replays_chat_history(self, monkeypatch):
        from loom.tui.app import LoomApp

        class FakeSession:
            def __init__(self, **_kwargs):
                self.total_tokens = 12
                self.session_id = "latest-session"
                self.session_state = SimpleNamespace(turn_count=4, ui_state={})
                self.messages = [{"role": "system", "content": "system"}]

            async def resume(self, _session_id: str) -> None:
                self.messages = [{"role": "system", "content": "system"}]

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._store.get_chat_events = AsyncMock(return_value=[
            {
                "seq": 1,
                "event_type": "user_message",
                "payload": {"text": "hello"},
            },
            {
                "seq": 2,
                "event_type": "assistant_text",
                "payload": {"text": "world", "markup": False},
            },
        ])
        app._resolve_startup_resume_target = AsyncMock(return_value=("latest-session", True))
        app._refresh_tool_registry = MagicMock()
        app._refresh_process_command_index = MagicMock()
        app._ensure_persistence_tools = MagicMock()
        app._apply_process_tool_policy = MagicMock()
        app._build_system_prompt = MagicMock(return_value="system")
        app._bind_session_tools = MagicMock()
        app._restore_process_run_tabs = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._tools.list_tools = MagicMock(return_value=[])

        chat = MagicMock()
        chat.children = []
        status = SimpleNamespace(workspace_name="", model_name="", process_name="")
        input_widget = SimpleNamespace(focus=MagicMock())

        def _query_one(selector: str, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            if selector == "#user-input":
                return input_widget
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        await app._initialize_session()

        chat.add_user_message.assert_any_call("hello")
        chat.add_model_text.assert_any_call("world", markup=False)

    @pytest.mark.asyncio
    async def test_initialize_session_refreshes_process_index_in_background(self, monkeypatch):
        from loom.tui.app import LoomApp

        class FakeSession:
            def __init__(self, **_kwargs):
                self.total_tokens = 0
                self.session_id = "s1"
                self.session_state = SimpleNamespace(turn_count=0, ui_state={})
                self.messages = []

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = None
        app._resolve_startup_resume_target = AsyncMock(return_value=(None, False))
        app._refresh_tool_registry = MagicMock()
        app._refresh_process_command_index = MagicMock()
        app._ensure_persistence_tools = MagicMock()
        app._apply_process_tool_policy = MagicMock()
        app._build_system_prompt = MagicMock(return_value="system")
        app._bind_session_tools = MagicMock()
        app._hydrate_chat_history_for_active_session = AsyncMock()
        app._restore_process_run_tabs = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._tools.list_tools = MagicMock(return_value=[])
        app._reset_cowork_steering_state = MagicMock()
        app._hydrate_input_history_from_session = MagicMock()

        chat = MagicMock()
        status = SimpleNamespace(workspace_name="", model_name="", process_name="")
        input_widget = SimpleNamespace(focus=MagicMock())

        def _query_one(selector: str, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            if selector == "#user-input":
                return input_widget
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        to_thread_calls: list[tuple] = []

        async def _fake_to_thread(func, *args, **kwargs):
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr("loom.tui.app.asyncio.to_thread", _fake_to_thread)

        await app._initialize_session()

        app._refresh_tool_registry.assert_called_once()
        app._refresh_process_command_index.assert_called_once_with(
            chat=chat,
            notify_conflicts=True,
            background=True,
            force=True,
        )
        assert to_thread_calls

    @pytest.mark.asyncio
    async def test_new_session_without_existing_session_initializes_once(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._session = None
        app._initialize_session = AsyncMock(
            side_effect=lambda **_kwargs: setattr(
                app,
                "_session",
                SimpleNamespace(session_id="abc123"),
            ),
        )
        app._enter_workspace_surface = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        await app._new_session()

        app._initialize_session.assert_awaited_once_with(
            allow_auto_resume=False,
            emit_info_messages=False,
        )
        app._enter_workspace_surface.assert_awaited_once_with(ensure_session=False)
        chat.add_info.assert_called_once()

    def test_start_workspace_watch_does_not_scan_synchronously(self):
        from loom.tui.app import LoomApp

        class RunningHarness(LoomApp):
            @property
            def is_running(self) -> bool:  # pragma: no cover - property shim
                return True

        app = RunningHarness(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._stop_workspace_watch = MagicMock()
        app._tui_realtime_refresh_enabled = MagicMock(return_value=True)
        app._tui_workspace_watch_backend = MagicMock(return_value="poll")
        app._tui_workspace_poll_interval_seconds = MagicMock(return_value=2.0)
        app._compute_workspace_signature = MagicMock(
            side_effect=AssertionError("workspace signature scan should not run inline"),
        )
        timer = MagicMock()
        app.set_interval = MagicMock(return_value=timer)

        app._start_workspace_watch()

        app._compute_workspace_signature.assert_not_called()
        app.set_interval.assert_called_once_with(
            2.0,
            app._on_workspace_poll_tick,
        )
        assert app._workspace_poll_timer is timer
        assert app._workspace_signature is None

class TestChatReplayHydration:
    class _PerfChat:
        def __init__(self):
            self.children = []

        def add_user_message(self, _text):
            return None

        def add_model_text(self, _text, *, markup=False):
            _ = markup
            return None

        def add_tool_call(
            self,
            _tool_name,
            _args,
            *,
            success=None,
            elapsed_ms=0,
            output="",
            error="",
        ):
            _ = (success, elapsed_ms, output, error)
            return None

        def add_content_indicator(self, _blocks):
            return None

        def add_turn_separator(
            self,
            _tool_count,
            _tokens,
            _model,
            *,
            tokens_per_second=0.0,
            latency_ms=0,
            total_time_ms=0,
            context_tokens=0,
            context_messages=0,
            omitted_messages=0,
            recall_index_used=False,
        ):
            _ = (
                tokens_per_second,
                latency_ms,
                total_time_ms,
                context_tokens,
                context_messages,
                omitted_messages,
                recall_index_used,
            )
            return None

        def add_info(self, _text, *, markup=True):
            _ = markup
            return None

    @pytest.mark.asyncio
    async def test_hydrate_chat_history_skips_bad_row_and_keeps_rendering(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(session_id="session-1")
        app._store = MagicMock()
        app._store.get_chat_events = AsyncMock(return_value=[
            {"seq": 1, "event_type": "user_message", "payload": {"text": "boom"}},
            {
                "seq": 2,
                "event_type": "assistant_text",
                "payload": {"text": "still renders", "markup": False},
            },
        ])
        app._store.synthesize_chat_events_from_turns = AsyncMock(return_value=[])

        chat = MagicMock()
        chat.children = []
        chat.add_user_message.side_effect = RuntimeError("bad row")
        app.query_one = MagicMock(return_value=chat)

        await app._hydrate_chat_history_for_active_session()

        chat.add_model_text.assert_called_once_with("still renders", markup=False)

    @pytest.mark.asyncio
    async def test_append_chat_replay_event_trims_with_continuity_sentinel(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._chat_resume_max_rendered_rows = MagicMock(return_value=2)

        chat = MagicMock()
        chat.children = []
        app.query_one = MagicMock(return_value=chat)

        await app._append_chat_replay_event(
            "user_message",
            {"text": "one"},
            persist=False,
        )
        await app._append_chat_replay_event(
            "user_message",
            {"text": "two"},
            persist=False,
        )
        await app._append_chat_replay_event(
            "user_message",
            {"text": "three"},
            persist=False,
        )

        assert [row["payload"]["text"] for row in app._chat_replay_events] == ["two", "three"]
        sentinel = chat.add_info.call_args_list[-1].args[0]
        assert "Transcript window truncated" in sentinel

    @pytest.mark.asyncio
    async def test_append_chat_replay_event_does_not_render_by_default(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        chat.children = []
        app.query_one = MagicMock(return_value=chat)

        await app._append_chat_replay_event(
            "user_message",
            {"text": "hello"},
            persist=False,
        )

        assert app._chat_replay_events[-1]["event_type"] == "user_message"
        chat.add_user_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_hydrate_perf_latest_300_rows_under_target(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(session_id="session-1")
        app._store = MagicMock()
        app._store.get_chat_events = AsyncMock(return_value=[
            {
                "seq": idx + 1,
                "event_type": "user_message",
                "payload": {"text": f"row {idx + 1}"},
            }
            for idx in range(300)
        ])
        app._store.synthesize_chat_events_from_turns = AsyncMock(return_value=[])

        chat = self._PerfChat()
        app.query_one = MagicMock(return_value=chat)

        started = asyncio.get_running_loop().time()
        await app._hydrate_chat_history_for_active_session()
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < 0.8

    @pytest.mark.asyncio
    async def test_chat_hydrate_perf_older_200_rows_under_target(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(session_id="session-1")
        app._store = MagicMock()

        latest = [
            {
                "seq": seq,
                "event_type": "user_message",
                "payload": {"text": f"latest {seq}"},
            }
            for seq in range(201, 401)
        ]
        older = [
            {
                "seq": seq,
                "event_type": "user_message",
                "payload": {"text": f"older {seq}"},
            }
            for seq in range(1, 201)
        ]

        async def _get_chat_events(
            _session_id: str,
            *,
            before_seq: int | None = None,
            after_seq: int | None = None,
            limit: int = 200,
        ):
            _ = (after_seq, limit)
            if before_seq is None:
                return latest
            return older if before_seq == 201 else []

        app._store.get_chat_events = AsyncMock(side_effect=_get_chat_events)
        app._store.synthesize_chat_events_from_turns = AsyncMock(return_value=[])

        chat = self._PerfChat()
        app.query_one = MagicMock(return_value=chat)
        await app._hydrate_chat_history_for_active_session()

        started = asyncio.get_running_loop().time()
        loaded = await app._load_older_chat_history()
        elapsed = asyncio.get_running_loop().time() - started

        assert loaded is True
        assert elapsed < 0.35

class TestRunStartAuthResolution:
    def test_collect_required_auth_resources_uses_required_tool_set_when_declared(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools = SimpleNamespace(
            list_tools=lambda: ["tool_required", "tool_non_required", "tool_excluded"],
            get=lambda name: {
                "tool_required": SimpleNamespace(
                    auth_requirements=[
                        {"provider": "ga_provider", "source": "api"},
                    ]
                ),
                "tool_non_required": SimpleNamespace(
                    auth_requirements=[
                        {"provider": "non_required_provider", "source": "api"},
                    ]
                ),
                "tool_excluded": SimpleNamespace(
                    auth_requirements=[
                        {"provider": "excluded_provider", "source": "api"},
                    ]
                ),
            }.get(name),
        )
        process_defn = SimpleNamespace(
            auth=SimpleNamespace(required=[{"provider": "notion", "source": "mcp"}]),
            tools=SimpleNamespace(
                required=["tool_required"],
                excluded=["tool_excluded"],
            ),
        )

        required = app._collect_required_auth_resources_for_process(process_defn)
        selectors = {
            (item["provider"], item.get("source", "api"))
            for item in required
        }
        assert ("notion", "mcp") in selectors
        assert ("ga_provider", "api") in selectors
        assert ("non_required_provider", "api") not in selectors
        assert ("excluded_provider", "api") not in selectors

    @pytest.mark.asyncio
    async def test_blocking_unresolved_auth_opens_manager_and_retries(self, monkeypatch):
        from loom.auth.runtime import (
            AuthResourceRequirement,
            UnresolvedAuthResource,
            UnresolvedAuthResourcesError,
        )
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._collect_required_auth_resources_for_process = MagicMock(return_value=[
            {"provider": "notion", "source": "api"},
        ])
        app._prompt_auth_choice = AsyncMock(return_value="Open Auth Manager")
        app._open_auth_manager_for_run_start = AsyncMock(return_value=True)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._config = SimpleNamespace(mcp=SimpleNamespace(servers={}))

        unresolved_error = UnresolvedAuthResourcesError(
            "blocked",
            unresolved=[
                UnresolvedAuthResource(
                    provider="notion",
                    source="api",
                    reason="auth_missing",
                    message="token missing",
                )
            ],
            defaults_user={},
            defaults_workspace={},
            explicit_overrides={},
            required_resources=[AuthResourceRequirement(provider="notion", source="api")],
        )
        fake_context = SimpleNamespace(
            profile_for_provider=lambda provider: (
                SimpleNamespace(profile_id="notion_dev")
                if provider == "notion" else None
            )
        )
        call_count = {"value": 0}

        def _fake_build_run_auth_context(**_kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise unresolved_error
            return fake_context

        monkeypatch.setattr(
            "loom.auth.runtime.build_run_auth_context",
            _fake_build_run_auth_context,
        )

        overrides, required_resources = await app._resolve_auth_overrides_for_run_start(
            process_defn=SimpleNamespace(),
            base_overrides={},
        )

        assert required_resources == [{"provider": "notion", "source": "api"}]
        assert overrides == {"notion": "notion_dev"}
        app._open_auth_manager_for_run_start.assert_awaited_once()
        assert call_count["value"] == 2
