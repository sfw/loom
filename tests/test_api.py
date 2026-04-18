"""Tests for the Loom API server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

import loom.api.routes as routes_mod
import loom.auth.oauth_profiles as oauth_profiles_mod
from loom.api.engine import Engine
from loom.api.server import LOCAL_CORS_ORIGIN_REGEX, create_app
from loom.auth.config import (
    AuthConfig,
    AuthProfile,
    load_merged_auth_config,
    write_auth_file,
)
from loom.auth.resources import (
    AuthBinding,
    AuthResource,
    AuthResourcesStore,
    load_workspace_auth_resources,
    write_workspace_auth_resources,
)
from loom.config import (
    Config,
    ExecutionConfig,
    MCPOAuthConfig,
    MCPServerConfig,
    ProcessConfig,
    TelemetryConfig,
)
from loom.config_runtime import ConfigRuntimeStore
from loom.cowork.session import CoworkTurn, ToolCallEvent
from loom.engine.orchestrator import Orchestrator
from loom.events.bus import Event, EventBus
from loom.events.types import (
    APPROVAL_RECEIVED,
    APPROVAL_REQUESTED,
    MODEL_INVOCATION,
    STEER_INSTRUCTION,
    TASK_COMPLETED,
    TASK_CREATED,
    TASK_EXECUTING,
    TASK_RUN_HEARTBEAT,
    TELEMETRY_MODE_CHANGED,
    TELEMETRY_SETTINGS_WARNING,
    TOKEN_STREAMED,
)
from loom.events.webhook import WebhookDelivery
from loom.integrations.mcp.oauth import MCPOAuthRefreshResult
from loom.learning.manager import LearningManager
from loom.mcp.config import load_mcp_file, write_mcp_file
from loom.models.router import ModelRouter
from loom.prompts.assembler import PromptAssembler
from loom.recovery.approval import ApprovalManager, ApprovalRequest
from loom.recovery.questions import QuestionManager
from loom.state.conversation_store import ConversationStore
from loom.state.memory import Database, MemoryEntry, MemoryManager
from loom.state.task_state import (
    Plan,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStateManager,
    TaskStatus,
)
from loom.state.workspaces import WorkspaceRegistry
from loom.tools import create_default_registry
from loom.tools.registry import ToolAvailabilityReason, ToolAvailabilityStatus, ToolResult

# --- Test Fixtures ---


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
async def database(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
async def memory_manager(database):
    return MemoryManager(database)


@pytest.fixture
async def conversation_store(database):
    return ConversationStore(database)


@pytest.fixture
async def workspace_registry(database):
    return WorkspaceRegistry(database)


@pytest.fixture
def state_manager(tmp_path):
    return TaskStateManager(data_dir=tmp_path / "state")


@pytest.fixture
def tool_registry():
    return create_default_registry()


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock(spec=Orchestrator)
    async def _default_execute(task, reuse_existing_plan=False):
        return task

    def _default_cancel(task):
        task.status = TaskStatus.CANCELLED
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["cancel_reason"] = "cancel_requested"

    def _default_pause(task):
        if task.status in (TaskStatus.EXECUTING, TaskStatus.PLANNING):
            task.status = TaskStatus.PAUSED

    def _default_resume(task):
        if task.status == TaskStatus.PAUSED:
            task.status = TaskStatus.EXECUTING

    orch.execute_task = AsyncMock(side_effect=_default_execute)
    orch.cancel_task = MagicMock(side_effect=_default_cancel)
    orch.pause_task = MagicMock(side_effect=_default_pause)
    orch.resume_task = MagicMock(side_effect=_default_resume)
    return orch


@pytest.fixture
def engine(
    event_bus,
    database,
    memory_manager,
    conversation_store,
    state_manager,
    workspace_registry,
    tool_registry,
    mock_orchestrator,
    tmp_path,
):
    config = Config(
        execution=ExecutionConfig(
            ask_user_v2_enabled=True,
            ask_user_runtime_blocking_enabled=True,
            ask_user_durable_state_enabled=True,
            ask_user_api_enabled=True,
        ),
        source_path=str(tmp_path / "loom.toml"),
    )
    return Engine(
        config=config,
        orchestrator=mock_orchestrator,
        event_bus=event_bus,
        model_router=ModelRouter(),
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        state_manager=state_manager,
        prompt_assembler=PromptAssembler(),
        database=database,
        conversation_store=conversation_store,
        workspace_registry=workspace_registry,
        config_runtime_store=ConfigRuntimeStore(config),
        approval_manager=ApprovalManager(event_bus),
        question_manager=QuestionManager(event_bus, memory_manager),
        webhook_delivery=WebhookDelivery(),
        learning_manager=LearningManager(database),
    )


@pytest.fixture
def app(engine):
    """Create a test app with the engine pre-injected (bypasses lifespan)."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from loom.api.routes import router

    app = FastAPI()
    app.state.engine = engine
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await app.state.engine.shutdown()


def test_create_app_allows_packaged_desktop_origins() -> None:
    app = create_app()
    middleware = next(
        layer
        for layer in app.user_middleware
        if layer.cls.__name__ == "CORSMiddleware"
    )
    assert middleware.kwargs["allow_origin_regex"] == LOCAL_CORS_ORIGIN_REGEX

    allowed = re.compile(LOCAL_CORS_ORIGIN_REGEX)
    assert allowed.match("http://127.0.0.1:1420")
    assert allowed.match("http://localhost:1420")
    assert allowed.match("https://tauri.localhost")
    assert allowed.match("tauri://localhost")
    assert not allowed.match("https://example.com")


def _make_task(
    state_manager,
    task_id="test-1",
    goal="Test goal",
    status=TaskStatus.PENDING,
    metadata: dict | None = None,
):
    """Create and persist a task for testing."""
    task = Task(
        id=task_id,
        goal=goal,
        status=status,
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
        metadata=metadata or {},
    )
    state_manager.save(task)
    return task


def _make_task_with_plan(state_manager, task_id="test-1"):
    """Create and persist a task with a plan."""
    task = Task(
        id=task_id,
        goal="Test goal",
        status=TaskStatus.EXECUTING,
        plan=Plan(
            subtasks=[
                Subtask(id="s1", description="Step 1", status=SubtaskStatus.COMPLETED),
                Subtask(id="s2", description="Step 2", depends_on=["s1"]),
            ],
            version=1,
        ),
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
    )
    state_manager.save(task)
    return task


def test_engine_binds_delegate_task_for_api_cowork(engine):
    from loom.tools.delegate_task import DelegateTaskTool

    delegate = engine.tool_registry.get("delegate_task")

    assert isinstance(delegate, DelegateTaskTool)
    assert callable(getattr(delegate, "_factory", None))


class TestInterruptedRunReconciliation:
    @pytest.mark.asyncio
    async def test_marks_non_durable_interrupted_runs_failed(
        self,
        event_bus,
        database,
        memory_manager,
        conversation_store,
        state_manager,
        workspace_registry,
        tool_registry,
        mock_orchestrator,
        tmp_path,
    ):
        config = Config(
            execution=ExecutionConfig(
                enable_durable_task_runner=False,
                ask_user_v2_enabled=True,
                ask_user_runtime_blocking_enabled=True,
                ask_user_durable_state_enabled=True,
                ask_user_api_enabled=True,
            ),
            source_path=str(tmp_path / "loom.toml"),
        )
        engine = Engine(
            config=config,
            orchestrator=mock_orchestrator,
            event_bus=event_bus,
            model_router=ModelRouter(),
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            state_manager=state_manager,
            prompt_assembler=PromptAssembler(),
            database=database,
            conversation_store=conversation_store,
            workspace_registry=workspace_registry,
            config_runtime_store=ConfigRuntimeStore(config),
            approval_manager=ApprovalManager(event_bus),
            question_manager=QuestionManager(event_bus, memory_manager),
            webhook_delivery=WebhookDelivery(),
            learning_manager=LearningManager(database),
        )
        task = _make_task(
            state_manager,
            task_id="run-1",
            goal="Interrupted run",
            status=TaskStatus.EXECUTING,
            metadata={"run_id": "exec-run-1"},
        )
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path="/tmp/workspace",
            status=TaskStatus.EXECUTING.value,
            metadata=task.metadata,
        )

        reconciled = await engine.reconcile_interrupted_task_runs()

        assert reconciled == 1
        updated_row = await database.get_task(task.id)
        assert updated_row is not None
        assert updated_row["status"] == TaskStatus.FAILED.value
        updated_task = state_manager.load(task.id)
        assert updated_task.status == TaskStatus.FAILED
        assert updated_task.errors_encountered
        assert "marked failed" in updated_task.errors_encountered[-1].error

    @pytest.mark.asyncio
    async def test_marks_existing_task_run_failed_when_reconciling(
        self,
        event_bus,
        database,
        memory_manager,
        conversation_store,
        state_manager,
        workspace_registry,
        tool_registry,
        mock_orchestrator,
        tmp_path,
    ):
        config = Config(
            execution=ExecutionConfig(
                enable_durable_task_runner=False,
                ask_user_v2_enabled=True,
                ask_user_runtime_blocking_enabled=True,
                ask_user_durable_state_enabled=True,
                ask_user_api_enabled=True,
            ),
            source_path=str(tmp_path / "loom.toml"),
        )
        engine = Engine(
            config=config,
            orchestrator=mock_orchestrator,
            event_bus=event_bus,
            model_router=ModelRouter(),
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            state_manager=state_manager,
            prompt_assembler=PromptAssembler(),
            database=database,
            conversation_store=conversation_store,
            workspace_registry=workspace_registry,
            config_runtime_store=ConfigRuntimeStore(config),
            approval_manager=ApprovalManager(event_bus),
            question_manager=QuestionManager(event_bus, memory_manager),
            webhook_delivery=WebhookDelivery(),
            learning_manager=LearningManager(database),
        )
        task = _make_task(
            state_manager,
            task_id="run-2",
            goal="Interrupted durable-looking run",
            status=TaskStatus.PLANNING,
            metadata={"run_id": "exec-run-2"},
        )
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path="/tmp/workspace",
            status=TaskStatus.PLANNING.value,
            metadata=task.metadata,
        )
        await database.insert_task_run(
            run_id="exec-run-2",
            task_id=task.id,
            status="running",
            process_name="seo-geo-review",
        )

        reconciled = await engine.reconcile_interrupted_task_runs()

        assert reconciled == 1
        task_run = await database.get_task_run("exec-run-2")
        assert task_run is not None
        assert task_run["status"] == "failed"

    @pytest.mark.asyncio
    async def test_skips_reconciliation_when_durable_runner_enabled(
        self,
        event_bus,
        database,
        memory_manager,
        conversation_store,
        state_manager,
        workspace_registry,
        tool_registry,
        mock_orchestrator,
        tmp_path,
    ):
        config = Config(
            execution=ExecutionConfig(
                enable_durable_task_runner=True,
                ask_user_v2_enabled=True,
                ask_user_runtime_blocking_enabled=True,
                ask_user_durable_state_enabled=True,
                ask_user_api_enabled=True,
            ),
            source_path=str(tmp_path / "loom.toml"),
        )
        durable_engine = Engine(
            config=config,
            orchestrator=mock_orchestrator,
            event_bus=event_bus,
            model_router=ModelRouter(),
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            state_manager=state_manager,
            prompt_assembler=PromptAssembler(),
            database=database,
            conversation_store=conversation_store,
            workspace_registry=workspace_registry,
            config_runtime_store=ConfigRuntimeStore(config),
            approval_manager=ApprovalManager(event_bus),
            question_manager=QuestionManager(event_bus, memory_manager),
            webhook_delivery=WebhookDelivery(),
            learning_manager=LearningManager(database),
        )
        task = _make_task(
            state_manager,
            task_id="run-3",
            goal="Recoverable run",
            status=TaskStatus.EXECUTING,
            metadata={"run_id": "exec-run-3"},
        )
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path="/tmp/workspace",
            status=TaskStatus.EXECUTING.value,
            metadata=task.metadata,
        )

        reconciled = await durable_engine.reconcile_interrupted_task_runs()

        assert reconciled == 0
        row = await database.get_task(task.id)
        assert row is not None
        assert row["status"] == TaskStatus.EXECUTING.value

    @pytest.mark.asyncio
    async def test_shutdown_pause_marks_active_runs_paused_and_requeues(
        self,
        engine,
        database,
        state_manager,
    ):
        def _pause_task_side_effect(task):
            if not isinstance(task.metadata, dict):
                task.metadata = {}
            task.metadata["paused_from_status"] = task.status.value
            task.status = TaskStatus.PAUSED
            state_manager.save(task)

        engine.orchestrator.pause_task.side_effect = _pause_task_side_effect
        task = _make_task(
            state_manager,
            task_id="run-shutdown-1",
            goal="Shutdown pause run",
            status=TaskStatus.EXECUTING,
            metadata={"run_id": "exec-run-shutdown-1", "process": "seo-geo-review"},
        )
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path="/tmp/workspace",
            status=TaskStatus.EXECUTING.value,
            metadata=task.metadata,
        )
        await database.insert_task_run(
            run_id="exec-run-shutdown-1",
            task_id=task.id,
            status="running",
            process_name="seo-geo-review",
        )

        paused = await engine.pause_active_task_runs_for_shutdown()

        assert paused == 1
        updated_task = state_manager.load(task.id)
        assert updated_task.status == TaskStatus.PAUSED
        assert updated_task.metadata["shutdown_paused"] is True
        task_run = await database.get_task_run("exec-run-shutdown-1")
        assert task_run is not None
        assert task_run["status"] == "queued"

    @pytest.mark.asyncio
    async def test_durable_recovery_skips_paused_shutdown_runs(
        self,
        event_bus,
        database,
        memory_manager,
        conversation_store,
        state_manager,
        workspace_registry,
        tool_registry,
        mock_orchestrator,
        tmp_path,
    ):
        config = Config(
            execution=ExecutionConfig(
                enable_durable_task_runner=True,
                ask_user_v2_enabled=True,
                ask_user_runtime_blocking_enabled=True,
                ask_user_durable_state_enabled=True,
                ask_user_api_enabled=True,
            ),
            source_path=str(tmp_path / "loom.toml"),
        )
        durable_engine = Engine(
            config=config,
            orchestrator=mock_orchestrator,
            event_bus=event_bus,
            model_router=ModelRouter(),
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            state_manager=state_manager,
            prompt_assembler=PromptAssembler(),
            database=database,
            conversation_store=conversation_store,
            workspace_registry=workspace_registry,
            config_runtime_store=ConfigRuntimeStore(config),
            approval_manager=ApprovalManager(event_bus),
            question_manager=QuestionManager(event_bus, memory_manager),
            webhook_delivery=WebhookDelivery(),
            learning_manager=LearningManager(database),
        )
        task = _make_task(
            state_manager,
            task_id="run-paused-1",
            goal="Paused durable run",
            status=TaskStatus.PAUSED,
            metadata={"run_id": "exec-run-paused-1", "process": "seo-geo-review"},
        )
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path="/tmp/workspace",
            status=TaskStatus.PAUSED.value,
            metadata=task.metadata,
        )
        await database.insert_task_run(
            run_id="exec-run-paused-1",
            task_id=task.id,
            status="queued",
            process_name="seo-geo-review",
        )

        recovered = await durable_engine.recover_pending_task_runs()

        assert recovered == 0
        mock_orchestrator.execute_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_resume_run_spawns_paused_durable_worker(
        self,
        client,
        engine,
        database,
        state_manager,
    ):
        def _resume_task_side_effect(task):
            if task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.EXECUTING
                state_manager.save(task)

        engine.orchestrator.resume_task.side_effect = _resume_task_side_effect
        task = _make_task(
            state_manager,
            task_id="run-resume-1",
            goal="Resume durable run",
            status=TaskStatus.PAUSED,
            metadata={"run_id": "exec-run-resume-1", "process": "seo-geo-review"},
        )
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path="/tmp/workspace",
            status=TaskStatus.PAUSED.value,
            metadata=task.metadata,
        )
        await database.insert_task_run(
            run_id="exec-run-resume-1",
            task_id=task.id,
            status="queued",
            process_name="seo-geo-review",
        )
        engine.submit_task = AsyncMock(return_value="exec-run-resume-1")  # type: ignore[method-assign]
        engine._resolve_process_definition = AsyncMock(return_value=None)  # type: ignore[method-assign]

        response = await client.post("/runs/run-resume-1/resume")

        assert response.status_code == 200
        engine.submit_task.assert_awaited_once()
        await_args = engine.submit_task.await_args.kwargs
        assert await_args["run_id"] == "exec-run-resume-1"
        assert await_args["process_name"] == "seo-geo-review"
        assert await_args["recovered"] is False

    @pytest.mark.asyncio
    async def test_resume_run_spawns_worker_when_task_run_is_running_but_worker_missing(
        self,
        client,
        engine,
        database,
        state_manager,
    ):
        def _resume_task_side_effect(task):
            if task.status == TaskStatus.PAUSED:
                task.status = TaskStatus.EXECUTING
                state_manager.save(task)

        engine.orchestrator.resume_task.side_effect = _resume_task_side_effect
        task = _make_task(
            state_manager,
            task_id="run-resume-stale-1",
            goal="Resume stale paused run",
            status=TaskStatus.PAUSED,
            metadata={"run_id": "exec-run-resume-stale-1", "process": "seo-geo-review"},
        )
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path="/tmp/workspace",
            status=TaskStatus.PAUSED.value,
            metadata=task.metadata,
        )
        await database.insert_task_run(
            run_id="exec-run-resume-stale-1",
            task_id=task.id,
            status="running",
            process_name="seo-geo-review",
        )
        engine.submit_task = AsyncMock(return_value="exec-run-resume-stale-1")  # type: ignore[method-assign]
        engine._resolve_process_definition = AsyncMock(return_value=None)  # type: ignore[method-assign]

        response = await client.post("/runs/run-resume-stale-1/resume")

        assert response.status_code == 200
        engine.submit_task.assert_awaited_once()
        await_args = engine.submit_task.await_args.kwargs
        assert await_args["run_id"] == "exec-run-resume-stale-1"
        assert await_args["process_name"] == "seo-geo-review"
        assert await_args["recovered"] is False


# --- Health & System ---


class TestSystemEndpoints:
    @pytest.mark.asyncio
    async def test_health(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["runtime_role"] == "api"

    @pytest.mark.asyncio
    async def test_runtime(self, client):
        response = await client.get("/runtime")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        assert data["runtime_role"] == "api"
        assert "database_path" in data
        assert "tool_availability" in data
        assert isinstance(data["tool_availability"], list)

    @pytest.mark.asyncio
    async def test_setup_status_reports_first_run_guidance(
        self,
        client,
        monkeypatch,
        tmp_path,
    ):
        config_dir = tmp_path / ".loom"
        config_path = config_dir / "loom.toml"
        monkeypatch.setattr(routes_mod.setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(routes_mod.setup_mod, "CONFIG_PATH", config_path)

        response = await client.get("/setup/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["needs_setup"] is True
        assert payload["config_path"] == str(config_path)
        assert payload["providers"]
        assert payload["role_presets"]["all"]

    @pytest.mark.asyncio
    async def test_setup_discover_models_returns_provider_results(
        self,
        client,
        monkeypatch,
    ):
        monkeypatch.setattr(
            routes_mod.setup_mod,
            "discover_models",
            lambda provider, base_url, api_key="": ["model-a", "model-b"],
        )

        response = await client.post(
            "/setup/discover-models",
            json={
                "provider": "openai_compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key": "",
            },
        )

        assert response.status_code == 200
        assert response.json()["models"] == ["model-a", "model-b"]

    @pytest.mark.asyncio
    async def test_setup_complete_writes_config_and_reloads_runtime_models(
        self,
        client,
        monkeypatch,
        tmp_path,
    ):
        config_dir = tmp_path / ".loom"
        config_path = config_dir / "loom.toml"
        monkeypatch.setattr(routes_mod.setup_mod, "CONFIG_DIR", config_dir)
        monkeypatch.setattr(routes_mod.setup_mod, "CONFIG_PATH", config_path)

        response = await client.post(
            "/setup/complete",
            json={
                "models": [
                    {
                        "name": "primary",
                        "provider": "ollama",
                        "base_url": "http://localhost:11434",
                        "model": "qwen3:14b",
                        "api_key": "",
                        "roles": [
                            "planner",
                            "executor",
                            "extractor",
                            "verifier",
                            "compactor",
                        ],
                        "max_tokens": 8192,
                        "temperature": 0.1,
                    },
                ],
            },
        )

        assert response.status_code == 200
        assert config_path.exists()
        written = config_path.read_text(encoding="utf-8")
        assert '[models.primary]' in written
        assert 'model = "qwen3:14b"' in written

        status_response = await client.get("/setup/status")
        assert status_response.status_code == 200
        assert status_response.json()["needs_setup"] is False

        models_response = await client.get("/models")
        assert models_response.status_code == 200
        assert any(row["name"] == "primary" for row in models_response.json())

    @pytest.mark.asyncio
    async def test_activity_summary_reports_global_backend_work(
        self,
        client,
        engine,
        conversation_store,
        database,
        tmp_path,
    ):
        workspace_path = tmp_path / "activity-ws"
        workspace_path.mkdir()
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="activity-model",
        )
        await database.insert_task(
            task_id="task-activity-1",
            goal="Track global activity",
            workspace_path=str(workspace_path),
            status="executing",
        )

        conversation_gate = asyncio.Event()
        run_gate = asyncio.Event()

        class FakeConversationSession:
            stop_requested = False

        conversation_task = engine.start_conversation_worker(
            session_id,
            FakeConversationSession(),
            conversation_gate.wait(),
        )
        run_task = engine.start_task_worker(
            task_id="task-activity-1",
            run_id="run-activity-1",
            worker=run_gate.wait(),
        )

        try:
            response = await client.get("/activity")
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "ok"
            assert payload["active"] is True
            assert payload["active_conversation_count"] == 1
            assert payload["active_run_count"] == 1
            assert payload["updated_at"]
        finally:
            conversation_gate.set()
            run_gate.set()
            await asyncio.gather(conversation_task, run_task, return_exceptions=True)

        response = await client.get("/activity")
        assert response.status_code == 200
        payload = response.json()
        assert payload["active"] is False
        assert payload["active_conversation_count"] == 0
        assert payload["active_run_count"] == 0

    @pytest.mark.asyncio
    async def test_activity_summary_ignores_paused_run_workers(
        self,
        client,
        engine,
        state_manager,
    ):
        paused_task = _make_task(
            state_manager,
            task_id="task-activity-paused",
            goal="Paused task",
            status=TaskStatus.PAUSED,
        )
        pause_gate = asyncio.Event()
        run_task = engine.start_task_worker(
            task_id=paused_task.id,
            run_id="run-activity-paused",
            worker=pause_gate.wait(),
        )

        try:
            response = await client.get("/activity")
            assert response.status_code == 200
            payload = response.json()
            assert payload["active"] is False
            assert payload["active_run_count"] == 0
        finally:
            pause_gate.set()
            await asyncio.gather(run_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_models(self, client):
        response = await client.get("/models")
        assert response.status_code == 200
        rows = response.json()
        assert isinstance(rows, list)
        if rows:
            assert "temperature" in rows[0]
            assert "max_tokens" in rows[0]

    @pytest.mark.asyncio
    async def test_patch_model_persists_temperature_and_max_tokens(
        self,
        client,
        engine,
        tmp_path,
    ):
        config_path = tmp_path / "loom.toml"
        config_path.write_text(
            """
[server]
host = "127.0.0.1"
port = 9000

[models.primary]
provider = "openai_compatible"
base_url = "http://localhost:1234/v1"
model = "gpt-4.1"
max_tokens = 8192
temperature = 0.1
roles = ["planner", "executor"]
""".strip()
            + "\n",
            encoding="utf-8",
        )
        engine.reload_config_from_source(config_path)

        response = await client.patch(
            "/models/primary",
            json={
                "temperature": 1.0,
                "max_tokens": 4096,
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["name"] == "primary"
        assert payload["temperature"] == 1.0
        assert payload["max_tokens"] == 4096

        updated = config_path.read_text(encoding="utf-8")
        assert "temperature = 1.0" in updated
        assert "max_tokens = 4096" in updated

    @pytest.mark.asyncio
    async def test_tools(self, client):
        response = await client.get("/tools")
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "name" in tools[0]
        assert "auth_mode" in tools[0]
        assert "auth_required" in tools[0]
        assert "auth_requirements" in tools[0]
        assert "execution_surfaces" in tools[0]
        assert "availability_state" in tools[0]
        assert "runnable" in tools[0]
        assert "availability_reasons" in tools[0]
        ask_user = next((row for row in tools if row.get("name") == "ask_user"), None)
        assert isinstance(ask_user, dict)
        assert ask_user.get("execution_surfaces") == ["tui"]

    @pytest.mark.asyncio
    async def test_config(self, client):
        response = await client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "server" in data
        assert "execution" in data

    @pytest.mark.asyncio
    async def test_slo_disabled_by_default(self, client):
        response = await client.get("/slo")
        assert response.status_code == 200
        payload = response.json()
        assert payload["enabled"] is False


class TestWorkspaceFirstEndpoints:
    @pytest.mark.asyncio
    async def test_list_workspaces_discovers_existing_tasks_and_sessions(
        self,
        client,
        database,
        conversation_store,
    ):
        await database.insert_task(
            task_id="task-ws-1",
            goal="Explore workspace",
            workspace_path="/tmp/work-a",
            status="executing",
        )
        await conversation_store.create_session(
            workspace="/tmp/work-a",
            model_name="test-model",
        )

        response = await client.get("/workspaces")
        assert response.status_code == 200
        rows = response.json()
        assert len(rows) == 1
        assert rows[0]["canonical_path"].endswith("/tmp/work-a")
        assert rows[0]["run_count"] == 1
        assert rows[0]["conversation_count"] == 1

    @pytest.mark.asyncio
    async def test_create_workspace_and_overview(
        self,
        client,
        tmp_path,
        database,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "desktop-ws"
        workspace_path.mkdir()
        create_response = await client.post(
            "/workspaces",
            json={"path": str(workspace_path), "display_name": "Desktop WS"},
        )
        assert create_response.status_code == 201
        workspace = create_response.json()
        assert workspace["display_name"] == "Desktop WS"
        workspace_id = workspace["id"]

        await database.insert_task(
            task_id="task-ws-2",
            goal="Ship feature",
            workspace_path=str(workspace_path),
            status="pending",
        )
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="test-model",
        )
        await conversation_store.link_run(session_id, "task-ws-2")
        synced = await workspace_registry.get(workspace_id)
        assert synced is not None

        overview_response = await client.get(f"/workspaces/{workspace_id}/overview")
        assert overview_response.status_code == 200
        payload = overview_response.json()
        assert payload["workspace"]["id"] == workspace_id
        assert payload["counts"]["runs"] == 1
        assert payload["counts"]["conversations"] == 1
        assert payload["recent_conversations"][0]["linked_run_ids"] == ["task-ws-2"]

    @pytest.mark.asyncio
    async def test_workspace_overview_uses_batched_relationship_queries(
        self,
        client,
        tmp_path,
        database,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "desktop-ws-batched"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None

        await database.insert_task(
            task_id="task-ws-batched-1",
            goal="Ship feature",
            workspace_path=str(workspace_path),
            status="completed",
        )
        await database.insert_task_run(
            run_id="run-ws-batched-1",
            task_id="task-ws-batched-1",
            status="completed",
            process_name="batch-process",
        )
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="test-model",
        )
        await conversation_store.link_run(session_id, "task-ws-batched-1")

        monkeypatch.setattr(
            engine.database,
            "get_latest_task_run_for_task",
            AsyncMock(side_effect=AssertionError("per-run lookup should not be used")),
        )
        monkeypatch.setattr(
            engine.conversation_store,
            "list_linked_conversations",
            AsyncMock(side_effect=AssertionError("per-run link lookup should not be used")),
        )
        monkeypatch.setattr(
            engine.conversation_store,
            "list_linked_runs",
            AsyncMock(side_effect=AssertionError("per-session link lookup should not be used")),
        )

        overview_response = await client.get(f"/workspaces/{workspace['id']}/overview")
        assert overview_response.status_code == 200
        payload = overview_response.json()
        assert payload["recent_runs"][0]["execution_run_id"] == "run-ws-batched-1"
        assert payload["recent_runs"][0]["linked_conversation_ids"] == [session_id]
        assert payload["recent_conversations"][0]["linked_run_ids"] == ["task-ws-batched-1"]

    @pytest.mark.asyncio
    async def test_workspace_summaries_prefer_live_terminal_status_for_active_counts(
        self,
        client,
        tmp_path,
        database,
        state_manager,
        workspace_registry,
    ):
        workspace_path = tmp_path / "live-status-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None

        await database.insert_task(
            task_id="task-live-status-1",
            goal="Finalize the report",
            workspace_path=str(workspace_path),
            status="executing",
        )
        _make_task(
            state_manager,
            task_id="task-live-status-1",
            goal="Finalize the report",
            status=TaskStatus.COMPLETED,
        )

        list_response = await client.get("/workspaces")
        assert list_response.status_code == 200
        list_payload = list_response.json()
        live_row = next(row for row in list_payload if row["id"] == workspace["id"])
        assert live_row["active_run_count"] == 0

        overview_response = await client.get(f"/workspaces/{workspace['id']}/overview")
        assert overview_response.status_code == 200
        overview_payload = overview_response.json()
        assert overview_payload["workspace"]["active_run_count"] == 0
        assert overview_payload["recent_runs"][0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_unified_approvals_list_and_reply(
        self,
        client,
        engine,
        monkeypatch,
        tmp_path,
        database,
        memory_manager,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "approval-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Approval WS",
        )
        assert workspace is not None

        await database.insert_task(
            task_id="task-approval-1",
            goal="Review deployment change",
            workspace_path=str(workspace_path),
            status="executing",
        )
        await memory_manager.upsert_pending_task_question(
            question_id="q_approval_1",
            task_id="task-approval-1",
            subtask_id="subtask-question",
            request_payload={
                "question": "Proceed with production deploy?",
                "question_type": "single_choice",
                "options": [
                    {"id": "yes", "label": "Yes"},
                    {"id": "no", "label": "No"},
                ],
                "context_note": "Waiting on operator approval.",
            },
        )
        approval_waiter = asyncio.create_task(
            engine.approval_manager.request_approval(
                ApprovalRequest(
                    task_id="task-approval-1",
                    subtask_id="subtask-approve",
                    reason="Touches deployment settings",
                    proposed_action="Apply production config update",
                    risk_level="high",
                ),
            ),
        )
        await asyncio.sleep(0.05)

        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="approval-model",
        )
        engine.begin_conversation_approval(
            session_id,
            tool_name="shell_execute",
            args={"command": "rm -rf build"},
        )
        monkeypatch.setattr(
            engine,
            "conversation_turn_inflight",
            lambda conversation_id: conversation_id == session_id,
        )

        list_response = await client.get(f"/approvals?workspace_id={workspace['id']}")
        assert list_response.status_code == 200
        rows = list_response.json()
        kinds = {row["kind"] for row in rows}
        assert {"task_approval", "task_question", "conversation_approval"} <= kinds

        task_item = next(row for row in rows if row["kind"] == "task_approval")
        question_item = next(row for row in rows if row["kind"] == "task_question")
        conversation_item = next(row for row in rows if row["kind"] == "conversation_approval")

        task_reply = await client.post(
            f"/approvals/{task_item['id']}/reply",
            json={"decision": "approve", "reason": "Looks safe."},
        )
        assert task_reply.status_code == 200
        assert await approval_waiter is True

        question_reply = await client.post(
            f"/approvals/{question_item['id']}/reply",
            json={
                "decision": "answer",
                "response_type": "answered",
                "selected_option_ids": ["yes"],
                "selected_labels": ["Yes"],
                "custom_response": "Yes",
            },
        )
        assert question_reply.status_code == 200
        assert question_reply.json()["status"] == "answered"

        conversation_reply = await client.post(
            f"/approvals/{conversation_item['id']}/reply",
            json={"decision": "approve_all"},
        )
        assert conversation_reply.status_code == 200

        refreshed = await client.get(f"/approvals?workspace_id={workspace['id']}")
        assert refreshed.status_code == 200
        assert refreshed.json() == []

    @pytest.mark.asyncio
    async def test_reply_task_approval_without_subtask_id(
        self,
        client,
        engine,
        tmp_path,
        database,
        workspace_registry,
    ):
        workspace_path = tmp_path / "approval-no-subtask-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Approval No Subtask WS",
        )
        assert workspace is not None

        await database.insert_task(
            task_id="task-approval-no-subtask",
            goal="Review top-level action",
            workspace_path=str(workspace_path),
            status="executing",
        )
        approval_waiter = asyncio.create_task(
            engine.approval_manager.request_approval(
                ApprovalRequest(
                    task_id="task-approval-no-subtask",
                    subtask_id="",
                    reason="Review whole-run action",
                    proposed_action="Approve top-level continuation",
                    risk_level="medium",
                ),
            ),
        )
        await asyncio.sleep(0.05)

        list_response = await client.get(f"/approvals?workspace_id={workspace['id']}")
        assert list_response.status_code == 200
        task_item = next(row for row in list_response.json() if row["kind"] == "task_approval")
        assert task_item["id"] == "task:task-approval-no-subtask"

        task_reply = await client.post(
            f"/approvals/{task_item['id']}/reply",
            json={"decision": "approve", "reason": "Proceed."},
        )
        assert task_reply.status_code == 200
        assert task_reply.json()["task_id"] == "task-approval-no-subtask"
        assert task_reply.json()["subtask_id"] == ""
        assert await approval_waiter is True

    @pytest.mark.asyncio
    async def test_workspace_overview_counts_pending_conversation_approvals(
        self,
        client,
        engine,
        monkeypatch,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "pending-overview-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Pending Overview WS",
        )
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="overview-model",
        )
        engine.begin_conversation_approval(
            session_id,
            tool_name="write_file",
            args={"path": "README.md"},
        )
        monkeypatch.setattr(
            engine,
            "conversation_turn_inflight",
            lambda conversation_id: conversation_id == session_id,
        )

        response = await client.get(f"/workspaces/{workspace['id']}/overview")
        assert response.status_code == 200
        assert response.json()["pending_approvals_count"] == 1

    @pytest.mark.asyncio
    async def test_workspace_overview_ignores_stale_pending_conversation_approvals(
        self,
        client,
        engine,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "stale-pending-overview-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Stale Pending Overview WS",
        )
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="overview-model",
        )
        request = engine.begin_conversation_approval(
            session_id,
            tool_name="write_file",
            args={"path": "README.md"},
        )

        overview_response = await client.get(f"/workspaces/{workspace['id']}/overview")
        assert overview_response.status_code == 200
        assert overview_response.json()["pending_approvals_count"] == 0

        approvals_response = await client.get(f"/approvals?workspace_id={workspace['id']}")
        assert approvals_response.status_code == 200
        assert approvals_response.json() == []

        status_response = await client.get(f"/conversations/{session_id}/status")
        assert status_response.status_code == 200
        assert status_response.json()["awaiting_approval"] is False
        assert status_response.json()["pending_approval"] is None

        resolve_response = await client.post(
            f"/conversations/{session_id}/approvals/{request.approval_id}",
            json={"decision": "approve"},
        )
        assert resolve_response.status_code == 404

    @pytest.mark.asyncio
    async def test_workspace_inventory_includes_processes_tools_and_mcp(
        self,
        client,
        tmp_path,
        workspace_registry,
    ):
        workspace_path = tmp_path / "inventory-ws"
        workspace_path.mkdir()
        process_dir = workspace_path / "loom-processes"
        process_dir.mkdir()
        (process_dir / "custom-inventory.yaml").write_text(
            "\n".join([
                "name: custom-inventory",
                "version: 1.0.0",
                "description: Custom workspace process",
                "author: Loom Test",
            ]),
            encoding="utf-8",
        )
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Inventory WS",
        )
        assert workspace is not None

        response = await client.get(f"/workspaces/{workspace['id']}/inventory")
        assert response.status_code == 200
        payload = response.json()
        assert payload["workspace"]["id"] == workspace["id"]
        assert payload["counts"]["tools"] > 0
        assert payload["counts"]["processes"] > 0
        assert payload["mcp_servers"] == []
        assert any(row["name"] == "custom-inventory" for row in payload["processes"])
        assert any(row["name"] == "ask_user" for row in payload["tools"])

    @pytest.mark.asyncio
    async def test_workspace_integrations_reports_management_grade_mcp_and_account_state(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(
            "LOOM_TEST_NOTION_TOKEN",
            json.dumps({"access_token": "token-123", "token_type": "Bearer"}),
        )

        workspace_path = tmp_path / "integrations-ws"
        workspace_path.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()
        write_mcp_file(
            loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.example",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
                "linear": MCPServerConfig(
                    type="remote",
                    url="https://mcp.linear.example",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )

        home_loom_dir = tmp_path / ".loom"
        home_loom_dir.mkdir(exist_ok=True)
        write_auth_file(
            home_loom_dir / "auth.toml",
            AuthConfig(
                profiles={
                    "notion_marketing": AuthProfile(
                        profile_id="notion_marketing",
                        provider="notion",
                        mode="oauth2_pkce",
                        account_label="Notion Marketing",
                        mcp_server="notion",
                        token_ref="${LOOM_TEST_NOTION_TOKEN}",
                    ),
                },
            ),
        )
        write_workspace_auth_resources(
            loom_dir / "auth.resources.toml",
            AuthResourcesStore(
                resources={
                    "resource-mcp-notion": AuthResource(
                        resource_id="resource-mcp-notion",
                        resource_kind="mcp",
                        resource_key="notion",
                        display_name="MCP: notion",
                        provider="notion",
                        source="mcp",
                        status="active",
                    ),
                },
                bindings={
                    "binding-notion": AuthBinding(
                        binding_id="binding-notion",
                        resource_id="resource-mcp-notion",
                        profile_id="notion_marketing",
                        status="active",
                    ),
                },
                workspace_defaults={
                    "resource-mcp-notion": "notion_marketing",
                },
            ),
        )

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Integrations WS",
        )
        assert workspace is not None

        response = await client.get(f"/workspaces/{workspace['id']}/integrations")
        assert response.status_code == 200
        payload = response.json()

        assert payload["workspace"]["id"] == workspace["id"]
        assert payload["counts"]["mcp_servers"] == 2
        assert payload["counts"]["accounts"] == 1

        notion = next(row for row in payload["mcp_servers"] if row["alias"] == "notion")
        assert notion["source"] == "workspace"
        assert notion["trust_state"] == "review_recommended"
        assert notion["approval_required"] is True
        assert notion["approval_state"] == "pending"
        assert notion["runtime_state"] == "pending_approval"
        assert notion["effective_account"]["profile_id"] == "notion_marketing"
        assert notion["auth_state"]["state"] == "ready"

        linear = next(row for row in payload["mcp_servers"] if row["alias"] == "linear")
        assert linear["runtime_state"] == "pending_approval"
        assert (
            "Approve this workspace-defined remote server before connecting "
            "an account."
        ) in linear["remediation"]

        account = payload["accounts"][0]
        assert account["profile_id"] == "notion_marketing"
        assert account["effective_for_mcp_servers"] == ["notion"]
        assert account["used_by_mcp_servers"] == ["notion"]
        assert account["auth_state"]["state"] == "ready"
        assert account["writable_storage_kind"] == "env"

    @pytest.mark.asyncio
    async def test_workspace_mcp_approve_endpoint_updates_management_state(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        workspace_path = tmp_path / "approval-ws"
        workspace_path.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()
        write_mcp_file(
            loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.example",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Approval WS",
        )
        assert workspace is not None

        response = await client.post(f"/workspaces/{workspace['id']}/mcp/notion/approve")
        assert response.status_code == 200
        payload = response.json()
        assert payload["alias"] == "notion"
        assert payload["approval_required"] is True
        assert payload["approval_state"] == "approved"

    @pytest.mark.asyncio
    async def test_workspace_mcp_create_update_disable_and_delete_endpoints(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        workspace_path = tmp_path / "manage-mcp-ws"
        workspace_path.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Manage MCP WS",
        )
        assert workspace is not None

        create = await client.post(
            f"/workspaces/{workspace['id']}/mcp",
            json={
                "alias": "filesystem",
                "type": "local",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "cwd": str(workspace_path),
                "timeout_seconds": 45,
                "enabled": True,
            },
        )
        assert create.status_code == 200
        created = create.json()
        assert created["alias"] == "filesystem"
        assert created["type"] == "local"
        assert created["command"] == "npx"
        assert created["args"] == ["-y", "@modelcontextprotocol/server-filesystem"]
        assert created["timeout_seconds"] == 45

        patch = await client.patch(
            f"/workspaces/{workspace['id']}/mcp/filesystem",
            json={
                "command": "uvx",
                "args": ["mcp-filesystem"],
                "cwd": "",
                "enabled": False,
            },
        )
        assert patch.status_code == 200
        updated = patch.json()
        assert updated["command"] == "uvx"
        assert updated["args"] == ["mcp-filesystem"]
        assert updated["enabled"] is False

        enable = await client.post(
            f"/workspaces/{workspace['id']}/mcp/filesystem/enable"
        )
        assert enable.status_code == 200
        assert enable.json()["enabled"] is True

        disable = await client.post(
            f"/workspaces/{workspace['id']}/mcp/filesystem/disable"
        )
        assert disable.status_code == 200
        assert disable.json()["enabled"] is False

        stored = load_mcp_file(loom_dir / "mcp.toml")
        filesystem = stored.servers["filesystem"]
        assert filesystem.command == "uvx"
        assert filesystem.args == ["mcp-filesystem"]
        assert filesystem.enabled is False

        delete = await client.delete(
            f"/workspaces/{workspace['id']}/mcp/filesystem"
        )
        assert delete.status_code == 200
        assert delete.json()["status"] == "ok"
        assert "filesystem" not in load_mcp_file(loom_dir / "mcp.toml").servers

    @pytest.mark.asyncio
    async def test_workspace_auth_sync_drafts_endpoint_creates_draft_accounts(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        workspace_path = tmp_path / "draft-sync-ws"
        workspace_path.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()
        write_mcp_file(
            loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.example",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Draft Sync WS",
        )
        assert workspace is not None

        response = await client.post(f"/workspaces/{workspace['id']}/auth/accounts/sync-drafts")
        assert response.status_code == 200
        payload = response.json()
        assert payload["created_drafts"] >= 1
        draft = next(
            row for row in payload["integrations"]["accounts"] if row["mcp_server"] == "notion"
        )
        assert draft["status"] == "draft"
        assert draft["mode"] == "oauth2_pkce"

    @pytest.mark.asyncio
    async def test_workspace_mcp_select_account_binds_and_sets_default(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        home_loom_dir = tmp_path / ".loom"
        home_loom_dir.mkdir()
        write_auth_file(
            home_loom_dir / "auth.toml",
            AuthConfig(
                profiles={
                    "notion_personal": AuthProfile(
                        profile_id="notion_personal",
                        provider="notion",
                        mode="oauth2_pkce",
                        account_label="Notion Personal",
                        token_ref="keychain://loom/notion/notion_personal/tokens",
                    ),
                },
            ),
        )

        workspace_path = tmp_path / "select-account-ws"
        workspace_path.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()
        write_mcp_file(
            loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.example",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )
        write_workspace_auth_resources(
            loom_dir / "auth.resources.toml",
            AuthResourcesStore(
                resources={
                    "resource-mcp-notion": AuthResource(
                        resource_id="resource-mcp-notion",
                        resource_kind="mcp",
                        resource_key="notion",
                        display_name="MCP: notion",
                        provider="notion",
                        source="mcp",
                        status="active",
                    ),
                },
            ),
        )

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Select Account WS",
        )
        assert workspace is not None

        approve = await client.post(f"/workspaces/{workspace['id']}/mcp/notion/approve")
        assert approve.status_code == 200

        response = await client.post(
            f"/workspaces/{workspace['id']}/mcp/notion/accounts/notion_personal/select"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["alias"] == "notion"
        assert payload["status"] == "ok"

        store = load_workspace_auth_resources(loom_dir / "auth.resources.toml")
        assert store.workspace_defaults["resource-mcp-notion"] == "notion_personal"
        assert any(
            binding.resource_id == "resource-mcp-notion"
            and binding.profile_id == "notion_personal"
            and binding.status == "active"
            for binding in store.bindings.values()
        )

        integrations = await client.get(f"/workspaces/{workspace['id']}/integrations")
        assert integrations.status_code == 200
        notion = next(
            row
            for row in integrations.json()["mcp_servers"]
            if row["alias"] == "notion"
        )
        assert notion["effective_account"]["profile_id"] == "notion_personal"

    @pytest.mark.asyncio
    async def test_workspace_mcp_reconnect_refreshes_expired_legacy_token_first(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        home_loom_dir = tmp_path / ".loom"
        home_loom_dir.mkdir()
        write_mcp_file(
            home_loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.com/mcp",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )

        workspace_path = tmp_path / "reconnect-legacy-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Reconnect Legacy WS",
        )
        assert workspace is not None

        monkeypatch.setattr(
            routes_mod,
            "oauth_state_for_alias",
            lambda alias, server=None: {
                "state": "expired",
                "has_token": True,
                "expired": True,
            },
        )
        refresh_mock = MagicMock(
            return_value=MCPOAuthRefreshResult(
                status="refreshed",
                reason="",
                refreshed=True,
            ),
        )
        reconnect_mock = MagicMock(
            return_value=SimpleNamespace(
                alias="notion",
                status="healthy",
                last_error="",
            ),
        )
        monkeypatch.setattr(routes_mod, "refresh_mcp_oauth_token", refresh_mock)
        monkeypatch.setattr(routes_mod, "runtime_reconnect_alias", reconnect_mock)

        response = await client.post(
            f"/workspaces/{workspace['id']}/mcp/notion/reconnect"
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["message"] == (
            "Refreshed legacy OAuth token and Reconnected MCP server notion."
        )
        refresh_mock.assert_called_once()
        reconnect_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_mcp_reconnect_surfaces_helpful_legacy_refresh_error(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        home_loom_dir = tmp_path / ".loom"
        home_loom_dir.mkdir()
        write_mcp_file(
            home_loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.com/mcp",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )

        workspace_path = tmp_path / "reconnect-legacy-error-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Reconnect Legacy Error WS",
        )
        assert workspace is not None

        monkeypatch.setattr(
            routes_mod,
            "oauth_state_for_alias",
            lambda alias, server=None: {
                "state": "expired",
                "has_token": True,
                "expired": True,
            },
        )
        monkeypatch.setattr(
            routes_mod,
            "refresh_mcp_oauth_token",
            lambda alias, server=None, force=False: MCPOAuthRefreshResult(
                status="failed",
                reason="Refresh token is missing.",
                refreshed=False,
            ),
        )

        response = await client.post(
            f"/workspaces/{workspace['id']}/mcp/notion/reconnect"
        )

        assert response.status_code == 400
        assert "Create and connect a Loom account instead." in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_workspace_mcp_test_uses_runtime_verification_for_remote_servers(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        home_loom_dir = tmp_path / ".loom"
        home_loom_dir.mkdir()
        write_mcp_file(
            home_loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.com/mcp",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )

        workspace_path = tmp_path / "remote-test-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Remote Test WS",
        )
        assert workspace is not None

        connect_mock = MagicMock(
            return_value=SimpleNamespace(
                alias="notion",
                status="healthy",
                last_error="",
            ),
        )
        list_tools_mock = MagicMock(
            return_value=[
                {"name": "search"},
                {"name": "query_database"},
            ],
        )
        monkeypatch.setattr(routes_mod, "runtime_connect_alias", connect_mock)
        monkeypatch.setattr(routes_mod, "runtime_list_tools_alias", list_tools_mock)

        response = await client.post(
            f"/workspaces/{workspace['id']}/mcp/notion/test"
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["tool_count"] == 2
        assert payload["tool_names"] == ["query_database", "search"]
        assert payload["message"] == (
            "Verified remote MCP server notion and discovered 2 tools."
        )
        connect_mock.assert_called_once()
        list_tools_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_auth_account_create_update_archive_and_restore(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        workspace_path = tmp_path / "account-manage-ws"
        workspace_path.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Account Manage WS",
        )
        assert workspace is not None

        create = await client.post(
            f"/workspaces/{workspace['id']}/auth/accounts",
            json={
                "profile_id": "notion_personal",
                "provider": "notion",
                "mode": "oauth2_pkce",
                "account_label": "Notion Personal",
                "mcp_server": "notion",
                "status": "draft",
            },
        )
        assert create.status_code == 200
        created = create.json()
        assert created["profile_id"] == "notion_personal"
        assert created["status"] == "draft"
        assert created["auth_state"]["state"] == "draft"

        update = await client.patch(
            f"/workspaces/{workspace['id']}/auth/accounts/notion_personal",
            json={
                "account_label": "Notion Personal Updated",
                "scopes": ["read", "search"],
            },
        )
        assert update.status_code == 200
        updated = update.json()
        assert updated["account_label"] == "Notion Personal Updated"

        archive = await client.post(
            f"/workspaces/{workspace['id']}/auth/accounts/notion_personal/archive"
        )
        assert archive.status_code == 200
        assert archive.json()["status"] == "archived"
        assert archive.json()["auth_state"]["state"] == "archived"

        restore = await client.post(
            f"/workspaces/{workspace['id']}/auth/accounts/notion_personal/restore"
        )
        assert restore.status_code == 200
        assert restore.json()["status"] == "ready"

        merged = load_merged_auth_config(workspace=workspace_path)
        stored = merged.config.profiles["notion_personal"]
        assert stored.account_label == "Notion Personal Updated"
        assert stored.scopes == ["read", "search"]
        assert stored.status == "ready"
        assert stored.token_ref == "keychain://loom/notion/notion_personal/tokens"

    @pytest.mark.asyncio
    async def test_workspace_auth_account_login_start_and_complete(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        stored: dict[str, str] = {}
        captured_start: dict[str, object] = {}

        class _FakeSecretResolver:
            def resolve(self, secret_ref: str) -> str:
                return stored.get(secret_ref, "")

            def validate_writable(self, secret_ref: str) -> None:
                stored["validated_ref"] = secret_ref

            def store(self, secret_ref: str, secret_value: str) -> None:
                stored["stored_ref"] = secret_ref
                stored[secret_ref] = secret_value

        monkeypatch.setattr(oauth_profiles_mod, "SecretResolver", _FakeSecretResolver)

        def _fake_start_auth(
            self,
            *,
            provider,
            preferred_port,
            open_browser,
            allow_manual_fallback,
            manual_redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        ):
            captured_start["open_browser"] = open_browser
            return SimpleNamespace(
                state="flow-1",
                authorization_url="https://auth.example/authorize",
                redirect_uri="http://127.0.0.1:8765/oauth/callback",
                expires_at_unix=1_763_000_000,
                callback_mode="loopback",
                browser_error="",
            )

        monkeypatch.setattr("loom.oauth.engine.OAuthEngine.start_auth", _fake_start_auth)
        monkeypatch.setattr(
            "loom.oauth.engine.OAuthEngine.callback_received",
            lambda self, *, state: True,
        )
        monkeypatch.setattr(
            "loom.oauth.engine.OAuthEngine.finish_auth",
            lambda self, *, provider, state, timeout_seconds=1: {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )

        home_auth_dir = tmp_path / ".loom"
        home_auth_dir.mkdir()
        write_auth_file(
            home_auth_dir / "auth.toml",
            AuthConfig(
                profiles={
                    "notion_marketing": AuthProfile(
                        profile_id="notion_marketing",
                        provider="notion",
                        mode="oauth2_pkce",
                        account_label="Notion Marketing",
                        mcp_server="notion",
                        token_ref="keychain://loom/notion/notion_marketing/tokens",
                        metadata={
                            "oauth_authorization_endpoint": "https://auth.example/authorize",
                            "oauth_token_endpoint": "https://auth.example/token",
                            "oauth_client_id": "loom-desktop",
                        },
                        status="draft",
                    ),
                },
            ),
        )

        workspace_path = tmp_path / "oauth-ws"
        workspace_path.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()
        write_mcp_file(
            loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.example",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )
        write_workspace_auth_resources(
            loom_dir / "auth.resources.toml",
            AuthResourcesStore(
                resources={
                    "resource-mcp-notion": AuthResource(
                        resource_id="resource-mcp-notion",
                        resource_kind="mcp",
                        resource_key="notion",
                        display_name="MCP: notion",
                        provider="notion",
                        source="mcp",
                        status="active",
                    ),
                },
                bindings={
                    "binding-notion": AuthBinding(
                        binding_id="binding-notion",
                        resource_id="resource-mcp-notion",
                        profile_id="notion_marketing",
                        status="active",
                    ),
                },
                workspace_defaults={"resource-mcp-notion": "notion_marketing"},
            ),
        )

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="OAuth WS",
        )
        assert workspace is not None

        start = await client.post(
            f"/workspaces/{workspace['id']}/auth/accounts/notion_marketing/login/start"
        )
        assert start.status_code == 200
        assert start.json()["flow_id"] == "flow-1"
        assert captured_start["open_browser"] is True

        complete = await client.post(
            f"/workspaces/{workspace['id']}/auth/accounts/notion_marketing/login/complete",
            json={"flow_id": "flow-1"},
        )
        assert complete.status_code == 200
        payload = complete.json()
        assert payload["status"] == "completed"
        assert payload["account"]["profile_id"] == "notion_marketing"
        assert payload["account"]["auth_state"]["state"] == "ready"
        assert stored["validated_ref"] == "keychain://loom/notion/notion_marketing/tokens"
        assert stored["stored_ref"] == "keychain://loom/notion/notion_marketing/tokens"

    @pytest.mark.asyncio
    async def test_workspace_auth_account_login_start_uses_mcp_oauth_metadata_fallback(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        captured: dict[str, object] = {}

        def _fake_start_auth(
            self,
            *,
            provider,
            preferred_port,
            open_browser,
            allow_manual_fallback,
            manual_redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        ):
            captured["provider"] = provider
            return SimpleNamespace(
                state="flow-mcp-fallback",
                authorization_url="https://auth.example/authorize",
                redirect_uri="http://127.0.0.1:8765/oauth/callback",
                expires_at_unix=1_763_000_000,
                callback_mode="loopback",
                browser_error="",
            )

        monkeypatch.setattr("loom.oauth.engine.OAuthEngine.start_auth", _fake_start_auth)
        monkeypatch.setattr(
            routes_mod,
            "resolve_mcp_oauth_provider",
            lambda **kwargs: SimpleNamespace(
                authorization_endpoint="https://auth.example/authorize",
                token_endpoint="https://auth.example/token",
                client_id="loom-desktop",
                scopes=tuple(kwargs.get("scopes", ()) or ()),
                authorize_params={"resource": "https://mcp.notion.example"},
                token_params={},
            ),
        )

        home_auth_dir = tmp_path / ".loom"
        home_auth_dir.mkdir()
        write_auth_file(
            home_auth_dir / "auth.toml",
            AuthConfig(
                profiles={
                    "notion_personal": AuthProfile(
                        profile_id="notion_personal",
                        provider="notion",
                        mode="oauth2_pkce",
                        account_label="Notion Personal",
                        mcp_server="notion",
                        token_ref="keychain://loom/notion/notion_personal/tokens",
                        status="draft",
                    ),
                },
            ),
        )

        workspace_path = tmp_path / "oauth-fallback-ws"
        workspace_path.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()
        write_mcp_file(
            loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.example",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read", "search"]),
                ),
            },
        )
        write_workspace_auth_resources(
            loom_dir / "auth.resources.toml",
            AuthResourcesStore(
                resources={
                    "resource-mcp-notion": AuthResource(
                        resource_id="resource-mcp-notion",
                        resource_kind="mcp",
                        resource_key="notion",
                        display_name="MCP: notion",
                        provider="notion",
                        source="mcp",
                        status="active",
                    ),
                },
                bindings={
                    "binding-notion": AuthBinding(
                        binding_id="binding-notion",
                        resource_id="resource-mcp-notion",
                        profile_id="notion_personal",
                        status="active",
                    ),
                },
                workspace_defaults={"resource-mcp-notion": "notion_personal"},
            ),
        )

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="OAuth Fallback WS",
        )
        assert workspace is not None

        approve = await client.post(f"/workspaces/{workspace['id']}/mcp/notion/approve")
        assert approve.status_code == 200

        start = await client.post(
            f"/workspaces/{workspace['id']}/auth/accounts/notion_personal/login/start"
        )
        assert start.status_code == 200
        assert start.json()["flow_id"] == "flow-mcp-fallback"

        provider = captured["provider"]
        assert provider.authorization_endpoint == "https://auth.example/authorize"
        assert provider.token_endpoint == "https://auth.example/token"
        assert provider.client_id == "loom-desktop"
        assert provider.scopes == ("read", "search")

    @pytest.mark.asyncio
    async def test_workspace_auth_account_login_complete_keeps_pending_loopback_flow(
        self,
        client,
        tmp_path,
        workspace_registry,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))

        callback_ready = {"value": False}
        stored: dict[str, str] = {}

        class _FakeSecretResolver:
            def resolve(self, secret_ref: str) -> str:
                return stored.get(secret_ref, "")

            def validate_writable(self, secret_ref: str) -> None:
                stored["validated_ref"] = secret_ref

            def store(self, secret_ref: str, secret_value: str) -> None:
                stored["stored_ref"] = secret_ref
                stored[secret_ref] = secret_value

        monkeypatch.setattr(oauth_profiles_mod, "SecretResolver", _FakeSecretResolver)

        monkeypatch.setattr(
            "loom.oauth.engine.OAuthEngine.start_auth",
            lambda self, **kwargs: SimpleNamespace(
                state="flow-pending",
                authorization_url="https://auth.example/authorize",
                redirect_uri="http://127.0.0.1:8765/oauth/callback",
                expires_at_unix=1_763_000_000,
                callback_mode="loopback",
                browser_error="",
            ),
        )
        monkeypatch.setattr(
            "loom.oauth.engine.OAuthEngine.callback_received",
            lambda self, *, state: callback_ready["value"],
        )
        monkeypatch.setattr(
            "loom.oauth.engine.OAuthEngine.finish_auth",
            lambda self, *, provider, state, timeout_seconds=1: {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        )

        home_auth_dir = tmp_path / ".loom"
        home_auth_dir.mkdir()
        write_auth_file(
            home_auth_dir / "auth.toml",
            AuthConfig(
                profiles={
                    "notion_personal": AuthProfile(
                        profile_id="notion_personal",
                        provider="notion",
                        mode="oauth2_pkce",
                        account_label="Notion Personal",
                        mcp_server="notion",
                        token_ref="keychain://loom/notion/notion_personal/tokens",
                        metadata={
                            "oauth_authorization_endpoint": "https://auth.example/authorize",
                            "oauth_token_endpoint": "https://auth.example/token",
                            "oauth_client_id": "loom-desktop",
                        },
                        status="draft",
                    ),
                },
            ),
        )

        workspace_path = tmp_path / "oauth-pending-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="OAuth Pending WS",
        )
        assert workspace is not None

        start = await client.post(
            f"/workspaces/{workspace['id']}/auth/accounts/notion_personal/login/start"
        )
        assert start.status_code == 200
        assert start.json()["flow_id"] == "flow-pending"

        pending = await client.post(
            f"/workspaces/{workspace['id']}/auth/accounts/notion_personal/login/complete",
            json={"flow_id": "flow-pending"},
        )
        assert pending.status_code == 200
        assert pending.json()["status"] == "pending"

        callback_ready["value"] = True
        complete = await client.post(
            f"/workspaces/{workspace['id']}/auth/accounts/notion_personal/login/complete",
            json={"flow_id": "flow-pending"},
        )
        assert complete.status_code == 200
        assert complete.json()["status"] == "completed"

    @pytest.mark.asyncio
    async def test_workspace_search_queries_real_workspace_surfaces(
        self,
        client,
        tmp_path,
        database,
        conversation_store,
        workspace_registry,
        memory_manager,
        state_manager,
        monkeypatch,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        home_loom_dir = tmp_path / ".loom"
        home_loom_dir.mkdir(exist_ok=True)
        write_auth_file(
            home_loom_dir / "auth.toml",
            AuthConfig(
                profiles={
                    "auth_search_profile": AuthProfile(
                        profile_id="auth_search_profile",
                        provider="notion",
                        mode="oauth2_pkce",
                        account_label="Auth Search Account",
                        mcp_server="notion",
                        token_ref="keychain://loom/notion/auth_search_profile/tokens",
                        status="ready",
                    ),
                },
            ),
        )

        workspace_path = tmp_path / "search-ws"
        workspace_path.mkdir()
        process_dir = workspace_path / "loom-processes"
        process_dir.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()
        (process_dir / "auth-audit.yaml").write_text(
            "\n".join([
                "name: auth-audit",
                "version: 1.0.0",
                "description: Authentication auditing process",
            ]),
            encoding="utf-8",
        )
        artifact_path = workspace_path / "auth-report.md"
        artifact_text = "Authentication audit notes"
        artifact_path.write_text(artifact_text, encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()
        write_mcp_file(
            loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.example",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )
        write_workspace_auth_resources(
            loom_dir / "auth.resources.toml",
            AuthResourcesStore(
                resources={
                    "resource-mcp-notion": AuthResource(
                        resource_id="resource-mcp-notion",
                        resource_kind="mcp",
                        resource_key="notion",
                        display_name="MCP: notion",
                        provider="notion",
                        source="mcp",
                        status="active",
                    ),
                },
                bindings={
                    "binding-auth-search": AuthBinding(
                        binding_id="binding-auth-search",
                        resource_id="resource-mcp-notion",
                        profile_id="auth_search_profile",
                        status="active",
                    ),
                },
                workspace_defaults={
                    "resource-mcp-notion": "auth_search_profile",
                },
            ),
        )

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Search WS",
        )
        assert workspace is not None

        await database.insert_task(
            task_id="task-search-1",
            goal="Audit auth boundary",
            workspace_path=str(workspace_path),
            status="executing",
            metadata={
                "artifact_seals": {
                    "auth-report.md": {
                        "path": "auth-report.md",
                        "sha256": artifact_sha,
                        "size_bytes": len(artifact_text.encode("utf-8")),
                        "tool": "document_write",
                        "subtask_id": "audit",
                        "sealed_at": "2026-03-23T10:30:00",
                    },
                },
                "process": "auth-audit",
            },
        )
        await database.insert_event(
            task_id="task-search-1",
            correlation_id="corr-auth",
            run_id="exec-auth-1",
            event_type="task_executing",
            data={"message": "Investigating auth flows"},
            sequence=1,
        )
        state_manager.save_evidence_records(
            "task-search-1",
            [{
                "evidence_id": "EV-AUTH-1",
                "task_id": "task-search-1",
                "subtask_id": "audit",
                "phase_id": "phase-auth",
                "tool": "document_write",
                "evidence_kind": "artifact",
                "artifact_workspace_relpath": "auth-report.md",
                "artifact_sha256": artifact_sha,
                "artifact_size_bytes": len(artifact_text.encode("utf-8")),
                "facets": {"category": "deliverable", "target": "authentication"},
                "created_at": "2026-03-23T10:31:00",
            }],
        )

        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="auth-model",
        )
        await conversation_store.update_session(
            session_id,
            session_state={"title": "Authentication thread"},
        )
        await conversation_store.append_turn(
            session_id,
            1,
            "assistant",
            "We should review the authentication boundary first.",
        )

        await memory_manager.upsert_pending_task_question(
            question_id="q-search-1",
            task_id="task-search-1",
            subtask_id="audit",
            request_payload={
                "question": "Proceed with auth migration?",
                "question_type": "single_choice",
                "options": [{"id": "yes", "label": "Yes"}],
            },
        )

        response = await client.get(
            f"/workspaces/{workspace['id']}/search?q=auth&limit_per_group=3",
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["query"] == "auth"
        assert payload["total_results"] >= 5
        assert payload["conversations"][0]["conversation_id"] == session_id
        assert payload["runs"][0]["run_id"] == "task-search-1"
        assert payload["approvals"][0]["approval_item_id"].startswith("question:")
        assert payload["artifacts"][0]["path"] == "auth-report.md"
        assert payload["files"][0]["path"] == "auth-report.md"
        assert payload["processes"][0]["title"] == "auth-audit"
        assert any(
            item["item_id"] == "auth_search_profile"
            and item["title"] == "Auth Search Account"
            for item in payload["accounts"]
        )
        assert payload["tools"]

    @pytest.mark.asyncio
    async def test_global_search_queries_across_workspaces(
        self,
        client,
        tmp_path,
        database,
        conversation_store,
        workspace_registry,
    ):
        alpha_path = tmp_path / "coach-alpha"
        beta_path = tmp_path / "coach-beta"
        alpha_path.mkdir()
        beta_path.mkdir()
        (beta_path / "coach-notes.md").write_text(
            "Coach outreach notes for Edmonton clubs.",
            encoding="utf-8",
        )

        alpha_workspace = await workspace_registry.ensure_workspace(
            str(alpha_path),
            display_name="Coach Alpha",
        )
        beta_workspace = await workspace_registry.ensure_workspace(
            str(beta_path),
            display_name="Coach Beta",
        )
        assert alpha_workspace is not None
        assert beta_workspace is not None

        alpha_session_id = await conversation_store.create_session(
            workspace=str(alpha_path),
            model_name="kimi-k2.5",
        )
        await conversation_store.update_session(
            alpha_session_id,
            session_state={"title": "Coach scouting thread"},
        )
        await conversation_store.append_turn(
            alpha_session_id,
            1,
            "assistant",
            "Coach shortlist for Alberta and Edmonton.",
        )

        await database.insert_task(
            task_id="task-coach-1",
            goal="Coach outreach plan",
            workspace_path=str(beta_path),
            status="completed",
            metadata={},
        )
        await database.insert_event(
            task_id="task-coach-1",
            correlation_id="corr-coach-1",
            run_id="exec-coach-1",
            event_type="task_completed",
            data={"summary": "Coach outreach report completed"},
            sequence=1,
        )

        response = await client.get("/search?q=coach&limit_per_group=5")
        assert response.status_code == 200
        payload = response.json()
        assert payload["workspace"] is None
        assert payload["query"] == "coach"
        assert {row["workspace_id"] for row in payload["workspaces"]} == {
            alpha_workspace["id"],
            beta_workspace["id"],
        }
        assert any(
            row["conversation_id"] == alpha_session_id
            and row["workspace_display_name"] == "Coach Alpha"
            for row in payload["conversations"]
        )
        assert any(
            row["run_id"] == "task-coach-1"
            and row["workspace_display_name"] == "Coach Beta"
            for row in payload["runs"]
        )
        assert any(
            row["path"] == "coach-notes.md"
            and row["workspace_display_name"] == "Coach Beta"
            for row in payload["files"]
        )

    @pytest.mark.asyncio
    async def test_workspace_search_matches_conversation_older_turn_content(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "conversation-search-ws"
        workspace_path.mkdir()

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Conversation Search WS",
        )
        assert workspace is not None

        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="kimi-k2.5",
        )
        await conversation_store.update_session(
            session_id,
            session_state={"title": "Tennis improvements"},
        )
        await conversation_store.append_turn(
            session_id,
            1,
            "user",
            "I need help fixing my serve toss consistency.",
        )
        await conversation_store.append_turn(
            session_id,
            2,
            "assistant",
            "Let's break the problem into toss drills and timing.",
        )
        await conversation_store.append_turn(
            session_id,
            3,
            "assistant",
            "We can also talk through a weekly practice plan.",
        )

        response = await client.get(
            f"/workspaces/{workspace['id']}/search?q=serve&limit_per_group=5",
        )
        assert response.status_code == 200
        payload = response.json()
        assert any(
            row["conversation_id"] == session_id
            for row in payload["conversations"]
        )

    @pytest.mark.asyncio
    async def test_workspace_search_matches_conversation_content_beyond_last_twelve_turns(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "conversation-search-deep-ws"
        workspace_path.mkdir()

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Conversation Search Deep WS",
        )
        assert workspace is not None

        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="kimi-k2.5",
        )
        await conversation_store.update_session(
            session_id,
            session_state={"title": "SEO follow-up"},
        )

        await conversation_store.append_turn(
            session_id,
            1,
            "user",
            "Please help me diagnose canonicalization drift for this site.",
        )
        for turn_number in range(2, 16):
            await conversation_store.append_turn(
                session_id,
                turn_number,
                "assistant",
                f"Filler turn {turn_number}",
            )

        response = await client.get(
            f"/workspaces/{workspace['id']}/search?q=canonicalization&limit_per_group=5",
        )
        assert response.status_code == 200
        payload = response.json()
        assert any(
            row["conversation_id"] == session_id
            for row in payload["conversations"]
        )

    @pytest.mark.asyncio
    async def test_workspace_files_list_and_preview(
        self,
        client,
        tmp_path,
        workspace_registry,
    ):
        workspace_path = tmp_path / "workspace-files-ws"
        docs_path = workspace_path / "docs"
        docs_path.mkdir(parents=True)
        readme_path = workspace_path / "README.md"
        readme_path.write_text("# Hello\n\nWorkspace preview text.\n", encoding="utf-8")
        table_path = docs_path / "notes.csv"
        table_path.write_text("name,status\nalpha,done\nbeta,pending\n", encoding="utf-8")

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Workspace Files WS",
        )
        assert workspace is not None

        root_response = await client.get(f"/workspaces/{workspace['id']}/files")
        assert root_response.status_code == 200
        root_rows = root_response.json()
        assert root_rows[0]["is_dir"] is True
        assert root_rows[0]["path"] == "docs"
        assert root_rows[1]["path"] == "README.md"

        nested_response = await client.get(
            f"/workspaces/{workspace['id']}/files",
            params={"directory": "docs"},
        )
        assert nested_response.status_code == 200
        nested_rows = nested_response.json()
        assert nested_rows[0]["path"] == "docs/notes.csv"

        text_preview = await client.get(
            f"/workspaces/{workspace['id']}/files/preview",
            params={"path": "README.md"},
        )
        assert text_preview.status_code == 200
        text_payload = text_preview.json()
        assert text_payload["preview_kind"] == "text"
        assert "Workspace preview text." in text_payload["text_content"]

        table_preview = await client.get(
            f"/workspaces/{workspace['id']}/files/preview",
            params={"path": "docs/notes.csv"},
        )
        assert table_preview.status_code == 200
        table_payload = table_preview.json()
        assert table_payload["preview_kind"] == "table"
        assert table_payload["table"]["columns"] == ["name", "status"]
        assert table_payload["table"]["rows"][0] == ["alpha", "done"]

    @pytest.mark.asyncio
    async def test_workspace_path_search_skips_hidden_entries(
        self,
        client,
        tmp_path,
        workspace_registry,
    ):
        workspace_path = tmp_path / "workspace-path-search-ws"
        (workspace_path / "src").mkdir(parents=True)
        (workspace_path / ".git").mkdir()
        (workspace_path / "src" / "app.tsx").write_text("export {};\n", encoding="utf-8")
        (workspace_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (workspace_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Workspace Path Search WS",
        )
        assert workspace is not None

        response = await client.get(
            f"/workspaces/{workspace['id']}/paths/search",
            params={"q": "", "limit": 20},
        )
        assert response.status_code == 200
        paths = [row["path"] for row in response.json()]
        assert "src" in paths
        assert "src/app.tsx" in paths
        assert ".env" not in paths
        assert ".git" not in paths
        assert ".git/config" not in paths

    @pytest.mark.asyncio
    async def test_workspace_file_preview_rejects_escape_path(
        self,
        client,
        tmp_path,
        workspace_registry,
    ):
        workspace_path = tmp_path / "workspace-files-escape-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None

        response = await client.get(
            f"/workspaces/{workspace['id']}/files/preview",
            params={"path": "../secret.txt"},
        )
        assert response.status_code == 400


    @pytest.mark.asyncio
    async def test_conversation_endpoints(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Chat WS",
        )
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        await conversation_store.append_turn(session_id, 1, "user", "hello")
        await conversation_store.append_turn(session_id, 2, "assistant", "hi")
        await conversation_store.append_chat_event(session_id, "message", {"text": "hello"})

        list_response = await client.get(f"/workspaces/{workspace['id']}/conversations")
        assert list_response.status_code == 200
        rows = list_response.json()
        assert len(rows) == 1
        assert rows[0]["id"] == session_id

        detail_response = await client.get(f"/conversations/{session_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["workspace"]["id"] == workspace["id"]

        messages_response = await client.get(f"/conversations/{session_id}/messages")
        assert messages_response.status_code == 200
        messages = messages_response.json()
        assert len(messages) == 2
        assert messages[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_post_conversation_message_starts_background_turn(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "chat-send-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Chat Send WS",
        )
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        engine.model_router.add_provider(
            "chat-model",
            SimpleNamespace(name="chat-model", roles=["executor"], tier=1),
        )

        class FakeCoworkSession:
            def __init__(self, *args, store=None, session_id="", **kwargs):
                self._store = store
                self._session_id = session_id

            async def resume(self, session_id: str) -> None:
                self._session_id = session_id

            async def send_streaming(
                self,
                user_message: str,
                *,
                message_metadata: dict[str, object] | None = None,
            ):
                turn_count = await self._store.get_turn_count(self._session_id)
                await self._store.append_turn(
                    self._session_id,
                    turn_count + 1,
                    "user",
                    user_message,
                )
                await self._store.append_turn(
                    self._session_id,
                    turn_count + 2,
                    "assistant",
                    "Background reply",
                )
                yield CoworkTurn(
                    text="Background reply",
                    tool_calls=[],
                    tokens_used=12,
                    model="chat-model",
                    latency_ms=10,
                    total_time_ms=20,
                    tokens_per_second=1.2,
                    context_tokens=0,
                    context_messages=0,
                    omitted_messages=0,
                    recall_index_used=False,
                )

        monkeypatch.setattr("loom.api.routes._cowork_session_cls", lambda: FakeCoworkSession)

        response = await client.post(
            f"/conversations/{session_id}/messages",
            json={"message": "Hello from desktop", "role": "user"},
        )
        assert response.status_code == 202

        for _ in range(20):
            turns = await conversation_store.get_turns(session_id, limit=10)
            if len(turns) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("background conversation turn did not persist expected turns")

        assert turns[-2]["content"] == "Hello from desktop"
        assert turns[-1]["content"] == "Background reply"
        events = await conversation_store.get_chat_events(session_id)
        event_types = [row["event_type"] for row in events]
        assert "user_message" in event_types
        assert "assistant_text" in event_types

    @pytest.mark.asyncio
    async def test_post_conversation_message_persists_attachment_metadata_and_indicator(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "chat-attachments-ws"
        workspace_path.mkdir()
        (workspace_path / "src").mkdir()
        (workspace_path / "src" / "app.tsx").write_text(
            "export const app = true;\n",
            encoding="utf-8",
        )
        (workspace_path / "docs").mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        scratch_image = engine.config.scratch_path / "pasted-image.png"
        scratch_image.parent.mkdir(parents=True, exist_ok=True)
        scratch_image.write_bytes(b"\x89PNG\r\n\x1a\n")

        engine.model_router.add_provider(
            "chat-model",
            SimpleNamespace(name="chat-model", roles=["executor"], tier=1),
        )
        captured: dict[str, object] = {}

        class FakeCoworkSession:
            def __init__(self, *args, store=None, session_id="", **kwargs):
                self._store = store
                self._session_id = session_id

            async def resume(self, session_id: str) -> None:
                self._session_id = session_id

            async def send_streaming(
                self,
                user_message: str,
                *,
                message_metadata: dict[str, object] | None = None,
            ):
                captured["message_metadata"] = dict(message_metadata or {})
                turn_count = await self._store.get_turn_count(self._session_id)
                await self._store.append_turn(
                    self._session_id,
                    turn_count + 1,
                    "user",
                    user_message,
                    metadata=dict(message_metadata or {}),
                )
                await self._store.append_turn(
                    self._session_id,
                    turn_count + 2,
                    "assistant",
                    "Attachment-aware reply",
                )
                yield CoworkTurn(
                    text="Attachment-aware reply",
                    tool_calls=[],
                    tokens_used=12,
                    model="chat-model",
                    latency_ms=10,
                    total_time_ms=20,
                    tokens_per_second=1.2,
                    context_tokens=0,
                    context_messages=0,
                    omitted_messages=0,
                    recall_index_used=False,
                )

        monkeypatch.setattr("loom.api.routes._cowork_session_cls", lambda: FakeCoworkSession)

        response = await client.post(
            f"/conversations/{session_id}/messages",
            json={
                "message": "Please review this screenshot",
                "role": "user",
                "workspace_paths": ["src/app.tsx", "docs"],
                "workspace_files": ["src/app.tsx"],
                "workspace_directories": ["docs"],
                "content_blocks": [{
                    "type": "image",
                    "source_path": str(scratch_image),
                    "media_type": "image/png",
                    "text_fallback": "Screenshot",
                }],
            },
        )
        assert response.status_code == 202

        for _ in range(20):
            turns = await conversation_store.get_turns(session_id, limit=10)
            if len(turns) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("background conversation attachment turn did not persist expected turns")

        assert captured["message_metadata"] == {
            "workspace_paths": ["src/app.tsx", "docs"],
            "workspace_files": ["src/app.tsx"],
            "workspace_directories": ["docs"],
            "content_blocks": [{
                "type": "image",
                "source_path": str(scratch_image),
                "media_type": "image/png",
                "width": 0,
                "height": 0,
                "size_bytes": 0,
                "text_fallback": "Screenshot",
            }],
        }
        assert json.loads(turns[-2]["metadata"])["workspace_paths"] == ["src/app.tsx", "docs"]
        events = await conversation_store.get_chat_events(session_id)
        assert "content_indicator" in [row["event_type"] for row in events]

    @pytest.mark.asyncio
    async def test_post_conversation_message_rejects_invalid_explicit_context_attachments(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-attachments-invalid-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        outside_image = tmp_path / "outside.png"
        outside_image.write_bytes(b"\x89PNG\r\n\x1a\n")

        response = await client.post(
            f"/conversations/{session_id}/messages",
            json={
                "message": "Please inspect this",
                "role": "user",
                "workspace_paths": ["../secret.txt"],
                "content_blocks": [{
                    "type": "image",
                    "source_path": str(outside_image),
                    "media_type": "image/png",
                    "text_fallback": "Screenshot",
                }],
            },
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "workspace_paths[0]" in detail
        assert "content_blocks[0]" in detail

    @pytest.mark.asyncio
    async def test_post_conversation_message_builds_workspace_auth_context_for_cowork_session(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "chat-mcp-auth-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Chat MCP Auth WS",
        )
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        engine.model_router.add_provider(
            "chat-model",
            SimpleNamespace(name="chat-model", roles=["executor"], tier=1),
        )
        engine.config.mcp.servers["notion"] = MCPServerConfig(
            type="remote",
            url="https://mcp.notion.com/mcp",
        )

        expected_auth_context = SimpleNamespace(selected_by_mcp_alias={"notion": object()})
        captured: dict[str, object] = {}

        def fake_build_run_auth_context(
            *,
            workspace,
            metadata,
            explicit_auth_path=None,
            required_resources=None,
            available_mcp_aliases=None,
        ):
            captured["workspace"] = workspace
            captured["metadata"] = metadata
            captured["explicit_auth_path"] = explicit_auth_path
            captured["required_resources"] = required_resources
            captured["available_mcp_aliases"] = available_mcp_aliases
            return expected_auth_context

        class FakeCoworkSession:
            def __init__(self, *args, auth_context=None, store=None, session_id="", **kwargs):
                captured["session_auth_context"] = auth_context
                self._store = store
                self._session_id = session_id

            async def resume(self, session_id: str) -> None:
                self._session_id = session_id

            async def send_streaming(
                self,
                user_message: str,
                *,
                message_metadata: dict[str, object] | None = None,
            ):
                turn_count = await self._store.get_turn_count(self._session_id)
                await self._store.append_turn(
                    self._session_id,
                    turn_count + 1,
                    "user",
                    user_message,
                )
                await self._store.append_turn(
                    self._session_id,
                    turn_count + 2,
                    "assistant",
                    "MCP-aware reply",
                )
                yield CoworkTurn(
                    text="MCP-aware reply",
                    tool_calls=[],
                    tokens_used=12,
                    model="chat-model",
                    latency_ms=10,
                    total_time_ms=20,
                    tokens_per_second=1.2,
                    context_tokens=0,
                    context_messages=0,
                    omitted_messages=0,
                    recall_index_used=False,
                )

        monkeypatch.setattr(routes_mod, "build_run_auth_context", fake_build_run_auth_context)
        monkeypatch.setattr("loom.api.routes._cowork_session_cls", lambda: FakeCoworkSession)

        response = await client.post(
            f"/conversations/{session_id}/messages",
            json={"message": "Can you use Notion?", "role": "user"},
        )
        assert response.status_code == 202
        assert captured["workspace"] == workspace_path.expanduser()
        assert captured["metadata"] == {}
        assert captured["explicit_auth_path"] is None
        assert captured["required_resources"] is None
        assert captured["available_mcp_aliases"] == {"notion"}
        assert captured["session_auth_context"] is expected_auth_context

    @pytest.mark.asyncio
    async def test_post_conversation_message_uses_workspace_mcp_registry_for_cowork_session(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "chat-workspace-mcp-ws"
        workspace_path.mkdir()
        loom_dir = workspace_path / ".loom"
        loom_dir.mkdir()
        write_mcp_file(
            loom_dir / "mcp.toml",
            {
                "notion": MCPServerConfig(
                    type="remote",
                    url="https://mcp.notion.com/mcp",
                    oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
                ),
            },
        )
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Chat Workspace MCP WS",
        )
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        engine.model_router.add_provider(
            "chat-model",
            SimpleNamespace(name="chat-model", roles=["executor"], tier=1),
        )
        engine.config.mcp.servers.clear()

        expected_auth_context = SimpleNamespace(selected_by_mcp_alias={"notion": object()})
        workspace_registry_tooling = SimpleNamespace(name="workspace-registry")
        captured: dict[str, object] = {}
        closed: list[object] = []

        def fake_build_run_auth_context(
            *,
            workspace,
            metadata,
            explicit_auth_path=None,
            required_resources=None,
            available_mcp_aliases=None,
        ):
            captured["workspace"] = workspace
            captured["available_mcp_aliases"] = available_mcp_aliases
            return expected_auth_context

        def fake_build_workspace_mcp_runtime_registry(*, engine, workspace):
            captured["registry_workspace"] = workspace
            return workspace_registry_tooling

        class FakeCoworkSession:
            def __init__(
                self,
                *args,
                tools=None,
                auth_context=None,
                store=None,
                session_id="",
                **kwargs,
            ):
                captured["session_tools"] = tools
                captured["session_auth_context"] = auth_context
                self._store = store
                self._session_id = session_id

            async def resume(self, session_id: str) -> None:
                self._session_id = session_id

            async def send_streaming(
                self,
                user_message: str,
                *,
                message_metadata: dict[str, object] | None = None,
            ):
                turn_count = await self._store.get_turn_count(self._session_id)
                await self._store.append_turn(
                    self._session_id,
                    turn_count + 1,
                    "user",
                    user_message,
                )
                await self._store.append_turn(
                    self._session_id,
                    turn_count + 2,
                    "assistant",
                    "Workspace MCP reply",
                )
                yield CoworkTurn(
                    text="Workspace MCP reply",
                    tool_calls=[],
                    tokens_used=12,
                    model="chat-model",
                    latency_ms=10,
                    total_time_ms=20,
                    tokens_per_second=1.2,
                    context_tokens=0,
                    context_messages=0,
                    omitted_messages=0,
                    recall_index_used=False,
                )

        monkeypatch.setattr(routes_mod, "build_run_auth_context", fake_build_run_auth_context)
        monkeypatch.setattr(
            routes_mod,
            "_build_workspace_mcp_runtime_registry",
            fake_build_workspace_mcp_runtime_registry,
        )
        monkeypatch.setattr(
            routes_mod,
            "_close_runtime_registry",
            lambda registry: closed.append(registry),
        )
        monkeypatch.setattr("loom.api.routes._cowork_session_cls", lambda: FakeCoworkSession)

        response = await client.post(
            f"/conversations/{session_id}/messages",
            json={"message": "Can you use Notion here?", "role": "user"},
        )
        assert response.status_code == 202

        for _ in range(20):
            turns = await conversation_store.get_turns(session_id, limit=10)
            if len(turns) >= 2 and closed:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("workspace-scoped conversation turn did not finish")

        assert captured["workspace"] == workspace_path.expanduser()
        assert captured["available_mcp_aliases"] == {"notion"}
        assert captured["registry_workspace"] == {
            "canonical_path": str(workspace_path.expanduser()),
        }
        assert captured["session_tools"] is workspace_registry_tooling
        assert captured["session_auth_context"] is expected_auth_context
        assert closed == [workspace_registry_tooling]

    @pytest.mark.asyncio
    async def test_conversation_events_and_stop_status_endpoints(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "chat-stop-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        await conversation_store.append_chat_event(
            session_id,
            "assistant_text",
            {"text": "hello replay"},
        )

        events_response = await client.get(f"/conversations/{session_id}/events")
        assert events_response.status_code == 200
        events_payload = events_response.json()
        assert events_payload[0]["event_type"] == "assistant_text"

        engine.model_router.add_provider(
            "chat-model",
            SimpleNamespace(name="chat-model", roles=["executor"], tier=1),
        )

        stop_flag = asyncio.Event()

        class StoppableCoworkSession:
            def __init__(self, *args, **kwargs):
                self._stop_requested = False

            @property
            def stop_requested(self) -> bool:
                return self._stop_requested

            def request_stop(self, reason: str = "user_requested") -> None:
                self._stop_requested = True
                stop_flag.set()

            async def resume(self, session_id: str) -> None:
                return None

            async def send_streaming(
                self,
                user_message: str,
                *,
                message_metadata: dict[str, object] | None = None,
            ):
                await stop_flag.wait()
                return
                yield  # pragma: no cover

        monkeypatch.setattr("loom.api.routes._cowork_session_cls", lambda: StoppableCoworkSession)

        start_response = await client.post(
            f"/conversations/{session_id}/messages",
            json={"message": "Please start", "role": "user"},
        )
        assert start_response.status_code == 202

        for _ in range(20):
            status_response = await client.get(f"/conversations/{session_id}/status")
            assert status_response.status_code == 200
            if status_response.json()["processing"] is True:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("conversation never reported processing state")

        stop_response = await client.post(f"/conversations/{session_id}/stop")
        assert stop_response.status_code == 200

        for _ in range(20):
            status_response = await client.get(f"/conversations/{session_id}/status")
            payload = status_response.json()
            if payload["processing"] is False:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("conversation stop did not clear processing state")

    @pytest.mark.asyncio
    async def test_conversation_stream_uses_stable_chat_event_name(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-stream-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        await conversation_store.append_chat_event(
            session_id,
            "assistant_text",
            {"text": "hello from stream"},
        )

        async with client.stream(
            "GET",
            f"/conversations/{session_id}/stream?follow=false",
        ) as response:
            current_event = ""
            current_id = ""
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("event: "):
                    current_event = line.removeprefix("event: ").strip()
                    continue
                if line.startswith("id: "):
                    current_id = line.removeprefix("id: ").strip()
                    continue
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line.removeprefix("data: "))
                assert current_event == "chat_event"
                assert current_id == str(payload["seq"])
                assert payload["event_type"] == "assistant_text"
                assert payload["payload"]["text"] == "hello from stream"
                break

    @pytest.mark.asyncio
    async def test_conversation_stream_respects_last_event_id_cursor(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-stream-cursor-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        seq1 = await conversation_store.append_chat_event(
            session_id,
            "assistant_text",
            {"text": "first"},
        )
        seq2 = await conversation_store.append_chat_event(
            session_id,
            "assistant_text",
            {"text": "second"},
        )

        async with client.stream(
            "GET",
            f"/conversations/{session_id}/stream?follow=false",
            headers={"last-event-id": str(seq1)},
        ) as response:
            seen_payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                seen_payloads.append(json.loads(line.removeprefix("data: ")))

        assert [payload["seq"] for payload in seen_payloads] == [seq2]

    @pytest.mark.asyncio
    async def test_conversation_stream_subscribes_before_replaying_events(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "chat-stream-race-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        original_get_transcript_page = engine.conversation_store.get_transcript_page
        original_subscribe_all = engine.event_bus.subscribe_all
        call_order: list[str] = []

        async def tracking_get_transcript_page(
            session_id_arg: str,
            *,
            before_seq: int | None = None,
            before_turn: int | None = None,
            after_seq: int | None = None,
            limit: int = 200,
        ):
            if (
                session_id_arg == session_id
                and before_seq is None
                and before_turn is None
                and (after_seq or 0) == 0
            ):
                call_order.append("replay")
            return await original_get_transcript_page(
                session_id_arg,
                before_seq=before_seq,
                before_turn=before_turn,
                after_seq=after_seq,
                limit=limit,
            )

        def tracking_subscribe_all(handler):
            call_order.append("subscribe")
            return original_subscribe_all(handler)

        monkeypatch.setattr(
            engine.conversation_store,
            "get_transcript_page",
            tracking_get_transcript_page,
        )
        monkeypatch.setattr(engine.event_bus, "subscribe_all", tracking_subscribe_all)

        async with client.stream(
            "GET",
            f"/conversations/{session_id}/stream?follow=false",
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    break

        assert call_order[:2] == ["subscribe", "replay"]

    @pytest.mark.asyncio
    async def test_conversation_messages_support_latest_and_before_turn(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-messages-page-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        for turn_number in range(1, 251):
            role = "assistant" if turn_number % 2 == 0 else "user"
            await conversation_store.append_turn(
                session_id,
                turn_number,
                role,
                f"message {turn_number}",
            )

        latest_response = await client.get(
            f"/conversations/{session_id}/messages?latest=true&limit=100",
        )
        assert latest_response.status_code == 200
        latest_rows = latest_response.json()
        assert latest_rows[0]["turn_number"] == 151
        assert latest_rows[-1]["turn_number"] == 250

        older_response = await client.get(
            f"/conversations/{session_id}/messages?before_turn=151&limit=100",
        )
        assert older_response.status_code == 200
        older_rows = older_response.json()
        assert older_rows[0]["turn_number"] == 51
        assert older_rows[-1]["turn_number"] == 150

    @pytest.mark.asyncio
    async def test_conversation_events_after_seq_without_new_rows_returns_empty_page(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-events-incremental-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        await conversation_store.append_turn(
            session_id,
            1,
            "user",
            "hello",
        )
        await conversation_store.append_turn(
            session_id,
            2,
            "assistant",
            "hi there",
        )
        seq = await conversation_store.append_chat_event(
            session_id,
            "assistant_text",
            {"text": "durable reply"},
        )

        response = await client.get(
            f"/conversations/{session_id}/events?after_seq={seq}",
        )
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_conversation_events_pagination_stitches_with_stream_cursor(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-events-page-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        for event_type, payload in [
            ("user_message", {"text": "hello"}),
            ("assistant_thinking", {"text": "thinking 1"}),
            ("assistant_text", {"text": "reply 1"}),
            ("tool_call_started", {"tool_name": "read_file", "tool_call_id": "call-1"}),
            (
                "tool_call_completed",
                {
                    "tool_name": "read_file",
                    "tool_call_id": "call-1",
                    "success": True,
                },
            ),
            ("turn_separator", {"tokens": 12, "tool_count": 1}),
            ("user_message", {"text": "follow-up"}),
            ("assistant_thinking", {"text": "thinking 2"}),
            ("assistant_text", {"text": "reply 2"}),
        ]:
            await conversation_store.append_chat_event(session_id, event_type, payload)

        latest_response = await client.get(
            f"/conversations/{session_id}/events?limit=4",
        )
        assert latest_response.status_code == 200
        latest_rows = latest_response.json()
        assert [row["seq"] for row in latest_rows] == [6, 7, 8, 9]

        older_response = await client.get(
            f"/conversations/{session_id}/events?before_seq=6&limit=5",
        )
        assert older_response.status_code == 200
        older_rows = older_response.json()
        assert [row["seq"] for row in older_rows] == [1, 2, 3, 4, 5]

        stitched = [*older_rows, *latest_rows]
        assert [row["seq"] for row in stitched] == list(range(1, 10))

        async with client.stream(
            "GET",
            f"/conversations/{session_id}/stream?follow=false&after_seq=5",
        ) as response:
            streamed_payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                streamed_payloads.append(json.loads(line.removeprefix("data: ")))

        assert [payload["seq"] for payload in streamed_payloads] == [6, 7, 8, 9]
        assert [payload["event_type"] for payload in streamed_payloads] == [
            "turn_separator",
            "user_message",
            "assistant_thinking",
            "assistant_text",
        ]

    @pytest.mark.asyncio
    async def test_conversation_events_expand_prefix_for_streamed_assistant_text(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-events-prefix-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        for event_type, payload in [
            ("user_message", {"text": "show me a code block"}),
            ("assistant_text", {"text": "```py\npri", "streaming": True}),
            ("assistant_text", {"text": "nt('hi')\n```", "streaming": True}),
            ("turn_separator", {"tokens": 8, "tool_count": 0}),
        ]:
            await conversation_store.append_chat_event(session_id, event_type, payload)

        response = await client.get(
            f"/conversations/{session_id}/events?limit=2",
        )
        assert response.status_code == 200
        rows = response.json()

        assert [row["seq"] for row in rows] == [1, 2, 3, 4]
        assert rows[0]["event_type"] == "user_message"
        assert rows[1]["payload"]["text"] == "```py\npri"
        assert rows[2]["payload"]["text"] == "nt('hi')\n```"

    @pytest.mark.asyncio
    async def test_conversation_stream_initial_replay_expands_prefix_for_streamed_assistant_text(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-stream-prefix-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        for event_type, payload in [
            ("user_message", {"text": "show me a code block"}),
            ("assistant_text", {"text": "```py\npri", "streaming": True}),
            ("assistant_text", {"text": "nt('hi')\n```", "streaming": True}),
            ("turn_separator", {"tokens": 8, "tool_count": 0}),
        ]:
            await conversation_store.append_chat_event(session_id, event_type, payload)

        async with client.stream(
            "GET",
            f"/conversations/{session_id}/stream?follow=false",
        ) as response:
            seen_payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                seen_payloads.append(json.loads(line.removeprefix("data: ")))

        assert [payload["seq"] for payload in seen_payloads] == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_conversation_events_expand_prefix_across_multiple_pages(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-events-multipage-prefix-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        await conversation_store.append_chat_event(
            session_id,
            "user_message",
            {"text": "show me markdown"},
        )
        for chunk in [
            "```markdown\n",
            "# Heading\n\n",
            "## Section\n\n",
            "- one\n",
            "- two\n",
            "```\n",
            "\nThanks!",
        ]:
            await conversation_store.append_chat_event(
                session_id,
                "assistant_text",
                {"text": chunk, "streaming": True},
            )
        await conversation_store.append_chat_event(
            session_id,
            "turn_separator",
            {"tokens": 12, "tool_count": 0},
        )

        response = await client.get(
            f"/conversations/{session_id}/events?limit=2",
        )
        assert response.status_code == 200
        rows = response.json()

        assert [row["seq"] for row in rows] == list(range(1, 10))
        assert rows[0]["event_type"] == "user_message"
        assert rows[1]["payload"]["text"] == "```markdown\n"
        assert rows[7]["payload"]["text"] == "\nThanks!"
        assert rows[8]["event_type"] == "turn_separator"

    @pytest.mark.asyncio
    async def test_conversation_events_keep_prefix_expansion_bounded_for_long_streams(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-events-bounded-prefix-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        await conversation_store.append_chat_event(
            session_id,
            "user_message",
            {"text": "show me a very long streamed answer"},
        )
        for index in range(1, 401):
            await conversation_store.append_chat_event(
                session_id,
                "assistant_text",
                {"text": f"chunk {index}\n", "streaming": True},
            )
        await conversation_store.append_chat_event(
            session_id,
            "turn_separator",
            {"tokens": 400, "tool_count": 0},
        )

        response = await client.get(
            f"/conversations/{session_id}/events?limit=10",
        )
        assert response.status_code == 200
        rows = response.json()

        assert 10 <= len(rows) <= 74
        assert rows[0]["seq"] > 1
        assert rows[-1]["seq"] == 402

    @pytest.mark.asyncio
    async def test_conversation_stream_initial_replay_keeps_prefix_expansion_bounded(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-stream-bounded-prefix-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        await conversation_store.append_chat_event(
            session_id,
            "user_message",
            {"text": "show me a very long streamed answer"},
        )
        for index in range(1, 701):
            await conversation_store.append_chat_event(
                session_id,
                "assistant_text",
                {"text": f"chunk {index}\n", "streaming": True},
            )
        await conversation_store.append_chat_event(
            session_id,
            "turn_separator",
            {"tokens": 700, "tool_count": 0},
        )

        async with client.stream(
            "GET",
            f"/conversations/{session_id}/stream?follow=false",
        ) as response:
            seen_payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                seen_payloads.append(json.loads(line.removeprefix("data: ")))

        assert 500 <= len(seen_payloads) <= 564
        assert int(seen_payloads[0]["seq"]) > 1
        assert seen_payloads[-1]["seq"] == 702

    @pytest.mark.asyncio
    async def test_conversation_events_before_turn_bridges_hybrid_history(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-events-hybrid-page-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        for turn_number in range(1, 251):
            role = "assistant" if turn_number % 2 == 0 else "user"
            await conversation_store.append_turn(
                session_id,
                turn_number,
                role,
                f"message {turn_number}",
            )

        for event_type, payload in [
            ("user_message", {"text": "message 249"}),
            ("assistant_text", {"text": "message 250"}),
        ]:
            await conversation_store.append_chat_event(session_id, event_type, payload)

        latest_response = await client.get(
            f"/conversations/{session_id}/events?limit=10",
        )
        assert latest_response.status_code == 200
        latest_rows = latest_response.json()
        assert len(latest_rows) == 10
        assert latest_rows[0]["turn_number"] == 241
        assert latest_rows[-1]["turn_number"] == 250
        assert latest_rows[-1]["payload"]["text"] == "message 250"

        older_without_turn_response = await client.get(
            f"/conversations/{session_id}/events?before_seq=1&limit=10",
        )
        assert older_without_turn_response.status_code == 200
        assert older_without_turn_response.json() == []

        older_with_turn_response = await client.get(
            f"/conversations/{session_id}/events?before_seq=1&before_turn=151&limit=100",
        )
        assert older_with_turn_response.status_code == 200
        older_rows = older_with_turn_response.json()
        assert older_rows[0]["turn_number"] == 51
        assert older_rows[-1]["turn_number"] == 150

    @pytest.mark.asyncio
    async def test_conversation_events_before_turn_without_seq_returns_legacy_page(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-events-before-turn-only-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        for turn_number in range(1, 7):
            role = "assistant" if turn_number % 2 == 0 else "user"
            await conversation_store.append_turn(
                session_id,
                turn_number,
                role,
                f"message {turn_number}",
            )

        response = await client.get(
            f"/conversations/{session_id}/events?before_turn=5&limit=10",
        )
        assert response.status_code == 200
        rows = response.json()
        assert [row["turn_number"] for row in rows] == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_conversation_status_ignores_journal_only_pending_prompt(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-question-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        await conversation_store.append_chat_event(
            session_id,
            "tool_call_completed",
            {
                "tool_name": "ask_user",
                "tool_call_id": "ask-1",
                "success": True,
                "data": {
                    "question": "Which language should we use?",
                    "question_type": "single_choice",
                    "options_v2": [
                        {"id": "python", "label": "Python", "description": "Fastest path"},
                        {"id": "rust", "label": "Rust", "description": "Higher control"},
                    ],
                    "allow_custom_response": False,
                    "awaiting_input": True,
                },
            },
        )

        response = await client.get(f"/conversations/{session_id}/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["awaiting_user_input"] is False
        assert payload["pending_prompt"] is None

        await conversation_store.append_chat_event(
            session_id,
            "user_message",
            {"text": "Use Python."},
        )
        response = await client.get(f"/conversations/{session_id}/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["awaiting_user_input"] is False
        assert payload["pending_prompt"] is None

    @pytest.mark.asyncio
    async def test_conversation_events_ignore_uncovered_partial_journal_rows(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-events-coverage-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        await conversation_store.append_turn(session_id, 1, "user", "older covered")
        await conversation_store.append_turn(session_id, 2, "assistant", "assistant newer")
        await conversation_store.append_turn(session_id, 3, "user", "fresh uncovered")
        await conversation_store.append_turn(session_id, 4, "assistant", "fresh answer")

        await conversation_store.append_chat_event(
            session_id,
            "user_message",
            {"text": "older covered"},
            journal_through_turn=1,
        )
        await conversation_store.append_chat_event(
            session_id,
            "assistant_text",
            {"text": "stale partial row"},
        )

        response = await client.get(f"/conversations/{session_id}/events?limit=20")
        assert response.status_code == 200
        rows = response.json()

        assert any(
            row.get("turn_number") == 4
            and row.get("event_type") == "assistant_text"
            and row.get("payload", {}).get("text") == "fresh answer"
            for row in rows
        )
        assert all(
            row.get("payload", {}).get("text") != "stale partial row"
            for row in rows
        )

    @pytest.mark.asyncio
    async def test_conversation_events_do_not_trust_legacy_journal_without_coverage(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-events-legacy-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        await conversation_store.append_turn(session_id, 1, "user", "older covered")
        await conversation_store.append_turn(session_id, 2, "assistant", "assistant newer")
        await conversation_store.append_turn(session_id, 3, "user", "fresh uncovered")
        await conversation_store.append_turn(session_id, 4, "assistant", "fresh answer")

        await conversation_store.append_chat_event(
            session_id,
            "user_message",
            {"text": "older covered"},
        )
        await conversation_store.append_chat_event(
            session_id,
            "assistant_text",
            {"text": "legacy partial row"},
        )

        response = await client.get(f"/conversations/{session_id}/events?limit=20")
        assert response.status_code == 200
        rows = response.json()

        assert any(
            row.get("turn_number") == 4
            and row.get("event_type") == "assistant_text"
            and row.get("payload", {}).get("text") == "fresh answer"
            for row in rows
        )
        assert all(
            row.get("payload", {}).get("text") != "legacy partial row"
            for row in rows
        )

    @pytest.mark.asyncio
    async def test_conversation_status_synthesizes_pending_prompt_from_turns(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-question-turns-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        await conversation_store.append_turn(session_id, 1, "user", "Help me choose a stack.")
        await conversation_store.append_turn(
            session_id,
            2,
            "assistant",
            "",
            tool_calls=[{
                "id": "ask-turn-1",
                "function": {
                    "name": "ask_user",
                    "arguments": json.dumps({
                        "question": "Which stack do you prefer?",
                        "question_type": "single_choice",
                        "options": ["Python", "Rust"],
                        "allow_custom_response": False,
                    }),
                },
            }],
        )
        await conversation_store.append_turn(
            session_id,
            3,
            "tool",
            ToolResult(
                success=True,
                output="QUESTION: Which stack do you prefer?",
                data={
                    "question": "Which stack do you prefer?",
                    "question_type": "single_choice",
                    "options": ["Python", "Rust"],
                    "allow_custom_response": False,
                    "awaiting_input": True,
                },
            ).to_json(),
            tool_call_id="ask-turn-1",
            tool_name="ask_user",
        )

        response = await client.get(f"/conversations/{session_id}/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["awaiting_user_input"] is True
        assert payload["pending_prompt"]["question"] == "Which stack do you prefer?"
        assert [option["label"] for option in payload["pending_prompt"]["options"]] == [
            "Python",
            "Rust",
        ]

    @pytest.mark.asyncio
    async def test_conversation_approval_resolution_endpoints(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "chat-approval-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        engine.model_router.add_provider(
            "chat-model",
            SimpleNamespace(name="chat-model", roles=["executor"], tier=1),
        )
        observed: dict[str, str] = {}

        class ApprovalCoworkSession:
            def __init__(self, *args, approver=None, **kwargs):
                self._approver = approver
                self._stop_requested = False

            @property
            def stop_requested(self) -> bool:
                return self._stop_requested

            def request_stop(self, reason: str = "user_requested") -> None:
                self._stop_requested = True

            async def resume(self, session_id: str) -> None:
                return None

            async def send_streaming(
                self,
                user_message: str,
                *,
                message_metadata: dict[str, object] | None = None,
            ):
                start = ToolCallEvent(
                    name="shell_execute",
                    args={"command": "touch risky.txt"},
                    tool_call_id="tc-approval",
                )
                yield start
                decision = await self._approver.check(
                    "shell_execute",
                    {"command": "touch risky.txt"},
                )
                observed["decision"] = decision.value
                completed = ToolCallEvent(
                    name="shell_execute",
                    args={"command": "touch risky.txt"},
                    tool_call_id="tc-approval",
                )
                if decision.value == "deny":
                    completed.result = ToolResult.fail("Tool call 'shell_execute' denied by user.")
                else:
                    completed.result = ToolResult.ok("Approved.")
                completed.elapsed_ms = 1
                yield completed
                if not self._stop_requested:
                    yield CoworkTurn(
                        text="Approved and continued",
                        tool_calls=[],
                        tokens_used=8,
                        model="chat-model",
                        latency_ms=4,
                        total_time_ms=8,
                        tokens_per_second=1.0,
                        context_tokens=0,
                        context_messages=0,
                        omitted_messages=0,
                        recall_index_used=False,
                    )

        monkeypatch.setattr("loom.api.routes._cowork_session_cls", lambda: ApprovalCoworkSession)

        response = await client.post(
            f"/conversations/{session_id}/messages",
            json={"message": "Run the risky command", "role": "user"},
        )
        assert response.status_code == 202

        approval_id = ""
        for _ in range(30):
            status_response = await client.get(f"/conversations/{session_id}/status")
            assert status_response.status_code == 200
            payload = status_response.json()
            if payload["awaiting_approval"] is True:
                approval_id = payload["pending_approval"]["approval_id"]
                assert payload["pending_approval"]["tool_name"] == "shell_execute"
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("conversation never reported pending approval")

        resolve_response = await client.post(
            f"/conversations/{session_id}/approvals/{approval_id}",
            json={"decision": "approve"},
        )
        assert resolve_response.status_code == 200

        for _ in range(30):
            status_response = await client.get(f"/conversations/{session_id}/status")
            payload = status_response.json()
            if payload["processing"] is False:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("conversation approval resolution did not finish the turn")

        assert observed["decision"] == "approve"
        events = await conversation_store.get_chat_events(session_id)
        event_types = [row["event_type"] for row in events]
        assert "approval_requested" in event_types
        assert "approval_resolved" in event_types

    @pytest.mark.asyncio
    async def test_conversation_stop_unblocks_pending_approval(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "chat-approval-stop-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        engine.model_router.add_provider(
            "chat-model",
            SimpleNamespace(name="chat-model", roles=["executor"], tier=1),
        )
        observed: dict[str, str] = {}

        class ApprovalStopCoworkSession:
            def __init__(self, *args, approver=None, **kwargs):
                self._approver = approver
                self._stop_requested = False

            @property
            def stop_requested(self) -> bool:
                return self._stop_requested

            def request_stop(self, reason: str = "user_requested") -> None:
                self._stop_requested = True

            async def resume(self, session_id: str) -> None:
                return None

            async def send_streaming(
                self,
                user_message: str,
                *,
                message_metadata: dict[str, object] | None = None,
            ):
                start = ToolCallEvent(
                    name="shell_execute",
                    args={"command": "rm -rf not-actually"},
                    tool_call_id="tc-stop",
                )
                yield start
                decision = await self._approver.check(
                    "shell_execute",
                    {"command": "rm -rf not-actually"},
                )
                observed["decision"] = decision.value
                completed = ToolCallEvent(
                    name="shell_execute",
                    args={"command": "rm -rf not-actually"},
                    tool_call_id="tc-stop",
                )
                completed.result = ToolResult.fail("Tool call 'shell_execute' denied by user.")
                completed.elapsed_ms = 1
                yield completed
                return
                yield  # pragma: no cover

        monkeypatch.setattr(
            "loom.api.routes._cowork_session_cls",
            lambda: ApprovalStopCoworkSession,
        )

        response = await client.post(
            f"/conversations/{session_id}/messages",
            json={"message": "Do the dangerous thing", "role": "user"},
        )
        assert response.status_code == 202

        for _ in range(30):
            status_response = await client.get(f"/conversations/{session_id}/status")
            assert status_response.status_code == 200
            payload = status_response.json()
            if payload["awaiting_approval"] is True:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("conversation never reported pending approval for stop path")

        stop_response = await client.post(f"/conversations/{session_id}/stop")
        assert stop_response.status_code == 200

        for _ in range(30):
            status_response = await client.get(f"/conversations/{session_id}/status")
            payload = status_response.json()
            if payload["processing"] is False:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("conversation stop did not unblock pending approval")

        assert observed["decision"] == "deny"

    @pytest.mark.asyncio
    async def test_conversation_inject_instruction_queues_for_active_turn(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "chat-inject-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        engine.model_router.add_provider(
            "chat-model",
            SimpleNamespace(name="chat-model", roles=["executor"], tier=1),
        )
        observed: dict[str, str] = {}
        injected = asyncio.Event()

        class InjectableCoworkSession:
            def __init__(self, *args, **kwargs):
                self._pending: list[str] = []
                self._stop_requested = False

            @property
            def stop_requested(self) -> bool:
                return self._stop_requested

            @property
            def pending_inject_instruction_count(self) -> int:
                return len(self._pending)

            def request_stop(self, reason: str = "user_requested") -> None:
                self._stop_requested = True

            def queue_inject_instruction(self, text: str) -> None:
                self._pending.append(str(text))
                injected.set()

            async def resume(self, session_id: str) -> None:
                return None

            async def send_streaming(
                self,
                user_message: str,
                *,
                message_metadata: dict[str, object] | None = None,
            ):
                await injected.wait()
                observed["instruction"] = self._pending.pop(0)
                yield CoworkTurn(
                    text="Instruction applied",
                    tool_calls=[],
                    tokens_used=6,
                    model="chat-model",
                    latency_ms=4,
                    total_time_ms=8,
                    tokens_per_second=1.0,
                    context_tokens=0,
                    context_messages=0,
                    omitted_messages=0,
                    recall_index_used=False,
                )

        monkeypatch.setattr("loom.api.routes._cowork_session_cls", lambda: InjectableCoworkSession)

        start_response = await client.post(
            f"/conversations/{session_id}/messages",
            json={"message": "Start the turn", "role": "user"},
        )
        assert start_response.status_code == 202

        for _ in range(30):
            status_response = await client.get(f"/conversations/{session_id}/status")
            assert status_response.status_code == 200
            if status_response.json()["processing"] is True:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("conversation never reported processing state for inject path")

        inject_response = await client.post(
            f"/conversations/{session_id}/inject",
            json={"instruction": "Focus on the failing tests first."},
        )
        assert inject_response.status_code == 200
        inject_payload = inject_response.json()
        assert inject_payload["pending_inject_count"] >= 1

        for _ in range(30):
            status_response = await client.get(f"/conversations/{session_id}/status")
            payload = status_response.json()
            if payload["processing"] is False:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("conversation inject did not finish the turn")

        assert observed["instruction"] == "Focus on the failing tests first."
        events = await conversation_store.get_chat_events(session_id)
        event_types = [row["event_type"] for row in events]
        assert "steering_instruction" in event_types

    @pytest.mark.asyncio
    async def test_conversation_inject_requires_active_turn(
        self,
        client,
        tmp_path,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "chat-inject-idle-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )

        response = await client.post(
            f"/conversations/{session_id}/inject",
            json={"instruction": "Please pivot."},
        )
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_run_endpoints_and_workspace_settings(
        self,
        client,
        tmp_path,
        database,
        conversation_store,
        workspace_registry,
    ):
        workspace_path = tmp_path / "run-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Run WS",
        )
        assert workspace is not None
        await database.insert_task(
            task_id="task-run-1",
            goal="Run task",
            workspace_path=str(workspace_path),
            status="executing",
            metadata={"process": "ship-it"},
        )
        await database.insert_task_run(
            run_id="exec-run-1",
            task_id="task-run-1",
            status="running",
            process_name="ship-it",
        )
        await database.insert_event(
            task_id="task-run-1",
            correlation_id="corr-1",
            run_id="exec-run-1",
            event_type="task_executing",
            data={"message": "started"},
            sequence=1,
        )
        session_id = await conversation_store.create_session(
            workspace=str(workspace_path),
            model_name="chat-model",
        )
        await conversation_store.link_run(session_id, "task-run-1")

        list_response = await client.get(f"/workspaces/{workspace['id']}/runs")
        assert list_response.status_code == 200
        runs = list_response.json()
        assert runs[0]["id"] == "task-run-1"
        assert runs[0]["execution_run_id"] == "exec-run-1"
        assert runs[0]["linked_conversation_ids"] == [session_id]

        detail_response = await client.get("/runs/task-run-1")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["task_run"]["run_id"] == "exec-run-1"

        timeline_response = await client.get("/runs/task-run-1/timeline")
        assert timeline_response.status_code == 200
        timeline = timeline_response.json()
        assert timeline[0]["event_type"] == "task_executing"
        assert timeline[0]["data"]["message"] == "started"

        settings_get = await client.get("/settings")
        assert settings_get.status_code == 200
        settings_payload = settings_get.json()
        assert settings_payload["basic"]
        assert settings_payload["advanced"]

        settings_patch = await client.patch(
            "/settings",
            json={"values": {"execution.ask_user_policy": "fail_closed"}},
        )
        assert settings_patch.status_code == 200
        patched = settings_patch.json()
        basic_paths = {row["path"]: row["effective"] for row in patched["basic"]}
        assert basic_paths["execution.ask_user_policy"] == "fail_closed"

    @pytest.mark.asyncio
    async def test_run_endpoints_fall_back_to_task_snapshot_when_task_row_is_missing(
        self,
        client,
        tmp_path,
        database,
        state_manager,
        workspace_registry,
    ):
        workspace_path = tmp_path / "run-state-first-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Run State First WS",
        )
        assert workspace is not None

        task = Task(
            id="task-run-state-only-1",
            goal="State-backed run",
            status=TaskStatus.EXECUTING,
            workspace=str(workspace_path),
            metadata={"process": "state-first", "run_id": "exec-run-state-only-1"},
        )
        state_manager.save(task)
        await database.insert_task_run(
            run_id="exec-run-state-only-1",
            task_id=task.id,
            status="running",
            process_name="state-first",
        )
        await database.insert_event(
            task_id=task.id,
            correlation_id="corr-state-1",
            run_id="exec-run-state-only-1",
            event_type="task_executing",
            data={"message": "still running"},
            sequence=1,
        )

        detail_response = await client.get(f"/runs/{task.id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["task"]["id"] == task.id
        assert detail["task"]["status"] == TaskStatus.EXECUTING.value
        assert detail["task"]["workspace_path"] == str(workspace_path)
        assert detail["task_run"]["run_id"] == "exec-run-state-only-1"

        timeline_response = await client.get(f"/runs/{task.id}/timeline")
        assert timeline_response.status_code == 200
        timeline = timeline_response.json()
        assert timeline[0]["event_type"] == "task_executing"
        assert timeline[0]["data"]["message"] == "still running"

        workspace_settings_patch = await client.patch(
            f"/workspaces/{workspace['id']}/settings",
            json={"overrides": {"layout": {"rail": "expanded"}}},
        )
        assert workspace_settings_patch.status_code == 200
        assert workspace_settings_patch.json()["overrides"]["layout"]["rail"] == "expanded"

    @pytest.mark.asyncio
    async def test_run_detail_includes_failure_analysis_for_verification_driven_failures(
        self,
        client,
        tmp_path,
        database,
        state_manager,
        workspace_registry,
    ):
        workspace_path = tmp_path / "run-failure-analysis-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Run Failure Analysis WS",
        )
        assert workspace is not None

        task = Task(
            id="task-run-failure-analysis-1",
            goal="Collect delegate contact channels",
            status=TaskStatus.FAILED,
            workspace=str(workspace_path),
            metadata={"process": "ad-hoc"},
            plan=Plan(subtasks=[
                Subtask(
                    id="discover-contact-channels",
                    description="Discover contact channels",
                    status=SubtaskStatus.FAILED,
                ),
            ]),
        )
        state_manager.save(task)

        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path=str(workspace_path),
            status="failed",
            metadata={"process": "ad-hoc"},
        )
        await database.insert_task_run(
            run_id="exec-run-failure-analysis-1",
            task_id=task.id,
            status="failed",
            process_name="ad-hoc",
        )
        await database.insert_event(
            task_id=task.id,
            correlation_id="corr-failure-analysis-1",
            run_id="exec-run-failure-analysis-1",
            event_type="verification_failed",
            data={
                "subtask_id": "discover-contact-channels",
                "reason_code": "hard_invariant_failed",
                "outcome": "fail",
            },
            sequence=1,
        )
        await database.insert_event(
            task_id=task.id,
            correlation_id="corr-failure-analysis-1",
            run_id="exec-run-failure-analysis-1",
            event_type="tool_call_completed",
            data={
                "subtask_id": "discover-contact-channels",
                "tool": "web_fetch",
                "success": False,
                "error": "Anti-bot denied (HTTP 999): https://www.linkedin.com/in/example-1/",
            },
            sequence=2,
        )
        await database.insert_event(
            task_id=task.id,
            correlation_id="corr-failure-analysis-1",
            run_id="exec-run-failure-analysis-1",
            event_type="tool_call_completed",
            data={
                "subtask_id": "discover-contact-channels",
                "tool": "web_fetch",
                "success": False,
                "error": "Anti-bot denied (HTTP 999): https://www.linkedin.com/in/example-2/",
            },
            sequence=3,
        )
        await database.insert_event(
            task_id=task.id,
            correlation_id="corr-failure-analysis-1",
            run_id="exec-run-failure-analysis-1",
            event_type="subtask_failed",
            data={
                "subtask_id": "discover-contact-channels",
                "feedback": (
                    "Verification required >75% of profiles to have LinkedIn "
                    "information, but 0% had it."
                ),
                "reason_code": "hard_invariant_failed",
            },
            sequence=4,
        )
        await database.insert_event(
            task_id=task.id,
            correlation_id="corr-failure-analysis-1",
            run_id="exec-run-failure-analysis-1",
            event_type="telemetry_run_summary",
            data={
                "verification_reason_counts": {"hard_invariant_failed": 1},
                "remediation_lifecycle_counts": {
                    "queued": 0,
                    "attempt": 0,
                    "resolved": 0,
                    "failed": 0,
                    "expired": 0,
                },
            },
            sequence=5,
        )
        await database.insert_event(
            task_id=task.id,
            correlation_id="corr-failure-analysis-1",
            run_id="exec-run-failure-analysis-1",
            event_type="task_failed",
            data={
                "failed_subtasks": ["discover-contact-channels"],
                "reason": "subtask_failure",
            },
            sequence=6,
        )

        response = await client.get(f"/runs/{task.id}")
        assert response.status_code == 200
        detail = response.json()
        failure = detail["failure_analysis"]
        assert failure["failing_subtask_id"] == "discover-contact-channels"
        assert failure["failing_subtask_label"] == "Discover contact channels"
        assert failure["primary_reason_code"] == "hard_invariant_failed"
        assert "0% had it" in failure["summary"]
        assert "HTTP 999" in failure["summary"]
        assert "hard invariant verification failure" in failure["remediation"]["why_not_remedied"]

    def test_reason_family_maps_method_failures_to_unconfirmed_data(self):
        from loom.api.routes import _reason_family

        assert _reason_family("tool_upstream_unavailable", "") == "unconfirmed_data"

    @pytest.mark.asyncio
    async def test_auto_subfolder_run_stays_grouped_under_parent_workspace(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
    ):
        workspace_path = tmp_path / "desktop-launch-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Desktop Launch WS",
        )
        assert workspace is not None

        response = await client.post(
            "/tasks",
            json={
                "goal": "Review the site",
                "workspace": str(workspace_path),
                "approval_mode": "auto",
                "auto_subfolder": True,
            },
        )
        assert response.status_code == 201
        payload = response.json()

        task_row = await database.get_task(payload["task_id"])
        assert task_row is not None
        task_workspace = Path(str(task_row["workspace_path"]))
        assert task_workspace.parent == workspace_path.resolve()
        assert task_workspace.is_dir()

        metadata = json.loads(str(task_row.get("metadata") or "{}"))
        assert metadata["source_workspace_root"] == str(workspace_path)
        assert metadata["run_workspace_mode"] == "scoped_subfolder"
        assert metadata["run_workspace_relative"] == task_workspace.name

        overview = await client.get(f"/workspaces/{workspace['id']}/overview")
        assert overview.status_code == 200
        recent_runs = overview.json()["recent_runs"]
        assert any(row["id"] == payload["task_id"] for row in recent_runs)

        detail = await client.get(f"/runs/{payload['task_id']}")
        assert detail.status_code == 200
        detail_payload = detail.json()
        assert detail_payload["workspace_id"] == workspace["id"]
        assert detail_payload["workspace"]["id"] == workspace["id"]
        assert detail_payload["workspace"]["canonical_path"] == str(workspace_path.resolve())
        assert detail_payload["workspace_path"] == str(task_workspace)

        workspace_list = await client.get("/workspaces")
        assert workspace_list.status_code == 200
        canonical_paths = {row["canonical_path"] for row in workspace_list.json()}
        assert str(workspace_path.resolve()) in canonical_paths
        assert str(task_workspace) not in canonical_paths

    @pytest.mark.asyncio
    async def test_auto_subfolder_task_questions_use_parent_workspace_context(
        self,
        client,
        tmp_path,
        database,
        memory_manager,
        workspace_registry,
    ):
        workspace_path = tmp_path / "auto-subfolder-approval-ws"
        workspace_path.mkdir()
        scoped_run_path = workspace_path / "scoped-run"
        scoped_run_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Auto Subfolder Approval WS",
        )
        assert workspace is not None

        await database.insert_task(
            task_id="task-auto-subfolder-question-1",
            goal="Need operator guidance",
            workspace_path=str(scoped_run_path),
            status="executing",
            metadata={
                "source_workspace_root": str(workspace_path.resolve()),
                "run_workspace_relative": "scoped-run",
                "run_workspace_mode": "scoped_subfolder",
            },
        )
        await memory_manager.upsert_pending_task_question(
            question_id="q_auto_subfolder_1",
            task_id="task-auto-subfolder-question-1",
            subtask_id="ask-user",
            request_payload={
                "question": "Keep going?",
                "question_type": "single_choice",
                "options": [
                    {"id": "yes", "label": "Yes"},
                    {"id": "no", "label": "No"},
                ],
                "context_note": "Need a quick confirmation.",
            },
        )

        response = await client.get(f"/approvals?workspace_id={workspace['id']}")

        assert response.status_code == 200
        rows = response.json()
        question_item = next(row for row in rows if row["kind"] == "task_question")
        assert question_item["task_id"] == "task-auto-subfolder-question-1"
        assert question_item["workspace_id"] == workspace["id"]
        assert question_item["workspace_path"] == str(workspace_path.resolve())

    @pytest.mark.asyncio
    async def test_auto_subfolder_process_launch_loads_process_from_parent_workspace(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
    ):
        workspace_path = tmp_path / "process-launch-ws"
        workspace_path.mkdir()
        process_dir = workspace_path / "loom-processes"
        process_dir.mkdir()
        (process_dir / "custom-process.yaml").write_text(
            "\n".join([
                "name: custom-process",
                "version: 1.0.0",
                "description: Custom process from parent workspace",
            ]),
            encoding="utf-8",
        )
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Process Launch WS",
        )
        assert workspace is not None

        response = await client.post(
            "/tasks",
            json={
                "goal": "Audit the workspace",
                "workspace": str(workspace_path),
                "process": "custom-process",
                "approval_mode": "auto",
                "auto_subfolder": True,
            },
        )
        assert response.status_code == 201
        payload = response.json()

        task_row = await database.get_task(payload["task_id"])
        assert task_row is not None
        metadata = json.loads(str(task_row.get("metadata") or "{}"))
        assert metadata["process"] == "custom-process"
        assert metadata["source_workspace_root"] == str(workspace_path)
        assert Path(str(task_row["workspace_path"])).parent == workspace_path.resolve()

    @pytest.mark.asyncio
    async def test_scoped_run_records_attached_read_scope_for_selected_context(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
    ):
        workspace_path = tmp_path / "attached-context-ws"
        workspace_path.mkdir()
        reference_dir = workspace_path / "reference-skill"
        reference_dir.mkdir()
        source_dir = workspace_path / "source-data"
        source_dir.mkdir()
        report_path = source_dir / "report.md"
        report_path.write_text("Attached report", encoding="utf-8")
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Attached Context WS",
        )
        assert workspace is not None

        response = await client.post(
            "/tasks",
            json={
                "goal": "Use the attached source material",
                "workspace": str(workspace_path),
                "approval_mode": "auto",
                "context": {
                    "workspace_paths": ["reference-skill", "source-data/report.md"],
                    "workspace_files": ["source-data/report.md"],
                    "workspace_directories": ["reference-skill"],
                },
                "auto_subfolder": True,
            },
        )
        assert response.status_code == 201
        payload = response.json()

        task_row = await database.get_task(payload["task_id"])
        assert task_row is not None
        metadata = json.loads(str(task_row.get("metadata") or "{}"))
        assert metadata["source_workspace_root"] == str(workspace_path)
        assert metadata["read_roots"] == [str(reference_dir.resolve())]
        assert metadata["attached_read_path_map"] == {
            "reference-skill": str(reference_dir.resolve()),
            "source-data/report.md": str(report_path.resolve()),
        }

    @pytest.mark.asyncio
    async def test_restart_scoped_run_reuses_same_task_and_parent_workspace_grouping(
        self,
        client,
        tmp_path,
        database,
        state_manager,
        workspace_registry,
        engine,
    ):
        workspace_path = tmp_path / "restart-scoped-ws"
        workspace_path.mkdir()
        reference_dir = workspace_path / "reference-skill"
        reference_dir.mkdir()
        source_dir = workspace_path / "source-data"
        source_dir.mkdir()
        report_path = source_dir / "report.md"
        report_path.write_text("Source report", encoding="utf-8")
        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Restart Scoped WS",
        )
        assert workspace is not None
        engine.submit_task = AsyncMock(
            side_effect=lambda **kwargs: kwargs.get("run_id", "run-created"),
        )

        create_response = await client.post(
            "/tasks",
            json={
                "goal": "Can you convert all this data into a microsite?",
                "workspace": str(workspace_path),
                "approval_mode": "auto",
                "context": {
                    "workspace_paths": ["reference-skill", "source-data/report.md"],
                    "workspace_files": ["source-data/report.md"],
                    "workspace_directories": ["reference-skill"],
                },
                "auto_subfolder": True,
            },
        )
        assert create_response.status_code == 201
        create_payload = create_response.json()
        original_task_id = create_payload["task_id"]
        original_run_id = str(create_payload.get("run_id") or "run-created")

        original_row = await database.get_task(original_task_id)
        assert original_row is not None
        original_metadata = json.loads(str(original_row.get("metadata") or "{}"))
        assert original_metadata["source_workspace_root"] == str(workspace_path)
        assert original_metadata["run_workspace_mode"] == "scoped_subfolder"
        assert original_metadata["read_roots"] == [str(reference_dir.resolve())]
        assert original_metadata["attached_read_path_map"] == {
            "reference-skill": str(reference_dir.resolve()),
            "source-data/report.md": str(report_path.resolve()),
        }
        stale_seal = {
            "path": "source-data/report.md",
            "sha256": "stale-seal-sha",
            "size_bytes": len("Source report"),
            "tool": "document_write",
            "subtask_id": "failed-step",
            "sealed_at": "2026-03-27T10:00:00+00:00",
        }
        original_metadata["run_id"] = original_run_id
        original_metadata["artifact_seals"] = {
            "source-data/report.md": stale_seal,
        }
        await database.update_task_metadata(original_task_id, original_metadata)

        saved_task = state_manager.load(original_task_id)
        saved_task.metadata["run_id"] = original_run_id
        fresh_seal = {
            **stale_seal,
            "sha256": "fresh-resealed-sha",
            "sealed_at": "2026-03-27T11:00:00+00:00",
            "tool": "edit_file",
            "tool_call_id": "edit_file:12",
            "resealed_after_mutation": True,
        }
        saved_task.metadata["artifact_seals"] = {
            "source-data/report.md": fresh_seal,
        }
        saved_task.metadata["state_only_marker"] = "preserve-me"
        saved_task.plan = Plan(
            subtasks=[
                Subtask(
                    id="done-step",
                    description="Already done",
                    status=SubtaskStatus.COMPLETED,
                    summary="Preserved",
                ),
                Subtask(
                    id="failed-step",
                    description="Needs retry",
                    status=SubtaskStatus.FAILED,
                    summary="Old failure",
                    retry_count=2,
                    active_issue="approval timeout",
                ),
                Subtask(
                    id="blocked-step",
                    description="Blocked downstream",
                    status=SubtaskStatus.BLOCKED,
                    summary="Blocked",
                    active_issue="dependency failed",
                ),
            ],
        )
        saved_task.status = TaskStatus.FAILED
        saved_task.completed_at = "2026-03-28T00:00:00+00:00"
        state_manager.save(saved_task)
        await database.update_task_plan(
            original_task_id,
            json.dumps(asdict(saved_task.plan)),
        )

        await database.update_task_status(original_task_id, "failed")
        engine.submit_task.reset_mock()

        restart_response = await client.post(f"/runs/{original_task_id}/restart")
        assert restart_response.status_code == 200
        restart_payload = restart_response.json()
        assert restart_payload["task_id"] == original_task_id
        assert restart_payload["run_id"].startswith("run-")
        assert restart_payload["run_id"] != original_run_id

        restarted_row = await database.get_task(original_task_id)
        assert restarted_row is not None
        restarted_workspace = Path(str(restarted_row["workspace_path"]))
        assert restarted_workspace == Path(str(original_row["workspace_path"]))

        restarted_metadata = json.loads(str(restarted_row.get("metadata") or "{}"))
        assert restarted_metadata["source_workspace_root"] == str(workspace_path)
        assert restarted_metadata["run_workspace_mode"] == "scoped_subfolder"
        assert restarted_metadata["run_workspace_relative"] == Path(
            str(original_row["workspace_path"]),
        ).name
        assert restarted_metadata["read_roots"] == [str(reference_dir.resolve())]
        assert restarted_metadata["attached_read_path_map"] == {
            "reference-skill": str(reference_dir.resolve()),
            "source-data/report.md": str(report_path.resolve()),
        }
        assert restarted_metadata["run_id"] == restart_payload["run_id"]
        assert restarted_metadata["restarted_from_run_id"] == original_run_id
        assert restarted_metadata["restart_count"] == 1
        assert restarted_metadata["artifact_seals"]["source-data/report.md"]["sha256"] == (
            "fresh-resealed-sha"
        )
        assert restarted_metadata["artifact_seals"]["source-data/report.md"]["sealed_at"] == (
            "2026-03-27T11:00:00+00:00"
        )
        assert restarted_metadata["state_only_marker"] == "preserve-me"

        restarted_context = json.loads(str(restarted_row.get("context") or "{}"))
        assert restarted_context == {
            "workspace_paths": ["reference-skill", "source-data/report.md"],
            "workspace_files": ["source-data/report.md"],
            "workspace_directories": ["reference-skill"],
        }
        assert restarted_row["status"] == "pending"
        assert restarted_row["completed_at"] in (None, "")
        assert json.loads(str(restarted_row["plan"]))["subtasks"][0]["status"] == "completed"
        assert json.loads(str(restarted_row["plan"]))["subtasks"][1]["status"] == "pending"
        assert json.loads(str(restarted_row["plan"]))["subtasks"][2]["status"] == "pending"

        restarted_task = state_manager.load(original_task_id)
        assert restarted_task.plan.subtasks[0].status == SubtaskStatus.COMPLETED
        assert restarted_task.plan.subtasks[1].status == SubtaskStatus.PENDING
        assert restarted_task.plan.subtasks[1].retry_count == 0
        assert restarted_task.plan.subtasks[1].active_issue == ""
        assert restarted_task.plan.subtasks[2].status == SubtaskStatus.PENDING
        assert restarted_task.decisions_log[-1] == "Resumed execution from prior task state."
        engine.submit_task.assert_awaited_once()
        assert engine.submit_task.await_args.kwargs["task"].id == original_task_id
        assert engine.submit_task.await_args.kwargs["run_id"] == restart_payload["run_id"]
        assert engine.submit_task.await_args.kwargs["recovered"] is True

        overview = await client.get(f"/workspaces/{workspace['id']}/overview")
        assert overview.status_code == 200
        recent_run_ids = {row["id"] for row in overview.json()["recent_runs"]}
        assert original_task_id in recent_run_ids

        workspace_list = await client.get("/workspaces")
        assert workspace_list.status_code == 200
        canonical_paths = {row["canonical_path"] for row in workspace_list.json()}
        assert str(workspace_path.resolve()) in canonical_paths
        assert str(restarted_workspace) not in canonical_paths

    @pytest.mark.asyncio
    async def test_submit_task_syncs_finished_metadata_and_plan_back_to_task_row(
        self,
        engine,
        database,
        state_manager,
        mock_orchestrator,
        tmp_path,
    ):
        workspace_path = tmp_path / "submit-task-sync-ws"
        workspace_path.mkdir()

        async def _complete_with_resealed_output(task, reuse_existing_plan=False):
            task.plan = Plan(
                subtasks=[
                    Subtask(
                        id="write-report",
                        description="Write the report",
                        status=SubtaskStatus.COMPLETED,
                        summary="Created report.md",
                    ),
                ],
            )
            task.status = TaskStatus.COMPLETED
            task.metadata["artifact_seals"] = {
                "report.md": {
                    "path": "report.md",
                    "sha256": "fresh-sync-sha",
                    "size_bytes": 42,
                    "tool": "document_write",
                    "subtask_id": "write-report",
                    "sealed_at": "2026-03-28T12:00:00+00:00",
                    "resealed_after_mutation": True,
                },
            }
            state_manager.save(task)
            return task

        mock_orchestrator.execute_task.side_effect = _complete_with_resealed_output
        task = Task(
            id="task-submit-sync-1",
            goal="Sync final metadata",
            status=TaskStatus.PENDING,
            workspace=str(workspace_path),
            metadata={"run_id": "exec-run-sync-1"},
        )
        state_manager.create(task)
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path=task.workspace,
            status=task.status.value,
            metadata=task.metadata,
        )

        await engine.submit_task(task=task, run_id="exec-run-sync-1")

        for _ in range(40):
            if not engine.task_run_inflight(task.id):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("task worker never finished")

        updated_row = await database.get_task(task.id)
        assert updated_row is not None
        assert updated_row["status"] == TaskStatus.COMPLETED.value
        updated_metadata = json.loads(str(updated_row.get("metadata") or "{}"))
        assert updated_metadata["artifact_seals"]["report.md"]["sha256"] == "fresh-sync-sha"
        assert updated_metadata["artifact_seals"]["report.md"]["resealed_after_mutation"] is True
        updated_plan = json.loads(str(updated_row.get("plan") or "{}"))
        assert updated_plan["subtasks"][0]["status"] == SubtaskStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_cancel_run_stops_active_worker_without_pause(
        self,
        client,
        database,
        state_manager,
        engine,
        mock_orchestrator,
    ):
        async def _hold_execution(task, reuse_existing_plan=False):
            await asyncio.sleep(60)
            return task

        mock_orchestrator.execute_task.side_effect = _hold_execution
        task = Task(
            id="task-run-cancel-live-1",
            goal="Cancel me",
            status=TaskStatus.PENDING,
            workspace="/tmp/run-cancel-live",
        )
        state_manager.create(task)
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path=task.workspace,
            status=task.status.value,
        )
        await engine.submit_task(task=task)

        for _ in range(20):
            if engine.task_run_inflight(task.id):
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("task worker never started")

        cancel_response = await client.post(f"/runs/{task.id}/cancel")
        assert cancel_response.status_code == 200
        assert cancel_response.json()["stop_requested"] is True
        mock_orchestrator.cancel_task.assert_called()

        for _ in range(40):
            row = await database.get_task(task.id)
            if row is not None and str(row.get("status", "") or "").strip().lower() == "cancelled":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("task row never became cancelled")

        assert engine.task_run_inflight(task.id) is False
        latest_task_run = await database.get_latest_task_run_for_task(task.id)
        assert latest_task_run is not None
        assert str(latest_task_run.get("status", "") or "").strip().lower() == "cancelled"

    @pytest.mark.asyncio
    async def test_scoped_run_artifacts_are_rebased_to_parent_workspace_paths(
        self,
        client,
        tmp_path,
        database,
        state_manager,
        workspace_registry,
    ):
        workspace_path = tmp_path / "scoped-artifacts-ws"
        workspace_path.mkdir()
        run_workspace = workspace_path / "review-the-site"
        run_workspace.mkdir()
        artifact_relpath = ".loom_artifacts/fetched/subtask-a/reference.pdf"
        artifact_path = run_workspace / artifact_relpath
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_bytes = b"%PDF-1.4 scoped artifact\n"
        artifact_path.write_bytes(artifact_bytes)
        artifact_sha = hashlib.sha256(artifact_bytes).hexdigest()

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Scoped Artifacts WS",
        )
        assert workspace is not None

        await database.insert_task(
            task_id="task-scoped-artifacts-1",
            goal="Review the site",
            workspace_path=str(run_workspace),
            status="completed",
            metadata={
                "source_workspace_root": str(workspace_path),
                "run_workspace_relative": run_workspace.name,
                "run_workspace_mode": "scoped_subfolder",
                "artifact_seals": {
                    artifact_relpath: {
                        "path": artifact_relpath,
                        "sha256": artifact_sha,
                        "size_bytes": len(artifact_bytes),
                        "tool": "web_fetch",
                        "subtask_id": "fetch-reference",
                        "sealed_at": "2026-03-27T18:00:00",
                    },
                },
            },
        )
        state_manager.save_evidence_records(
            "task-scoped-artifacts-1",
            [
                {
                    "evidence_id": "EV-SCOPED-ARTIFACT-1",
                    "task_id": "task-scoped-artifacts-1",
                    "subtask_id": "fetch-reference",
                    "phase_id": "research",
                    "tool": "web_fetch",
                    "evidence_kind": "artifact",
                    "artifact_workspace_relpath": artifact_relpath,
                    "artifact_sha256": artifact_sha,
                    "artifact_size_bytes": len(artifact_bytes),
                    "facets": {"category": "fetched_artifact"},
                    "created_at": "2026-03-27T18:00:01",
                },
            ],
        )

        run_artifacts = await client.get("/runs/task-scoped-artifacts-1/artifacts")
        assert run_artifacts.status_code == 200
        run_payload = run_artifacts.json()
        assert run_payload == [
            {
                "path": f"{run_workspace.name}/{artifact_relpath}",
                "category": "fetched_artifact",
                "source": "seal+evidence",
                "sha256": artifact_sha,
                "size_bytes": len(artifact_bytes),
                "exists_on_disk": True,
                "is_intermediate": False,
                "created_at": "2026-03-27T18:00:01",
                "tool_name": "web_fetch",
                "subtask_ids": ["fetch-reference"],
                "phase_ids": ["research"],
                "facets": {"category": "fetched_artifact"},
            },
        ]

        workspace_artifacts = await client.get(f"/workspaces/{workspace['id']}/artifacts")
        assert workspace_artifacts.status_code == 200
        workspace_payload = workspace_artifacts.json()
        assert len(workspace_payload) == 1
        assert workspace_payload[0]["path"] == f"{run_workspace.name}/{artifact_relpath}"
        assert workspace_payload[0]["latest_run_id"] == "task-scoped-artifacts-1"
        assert workspace_payload[0]["run_ids"] == ["task-scoped-artifacts-1"]
        assert workspace_payload[0]["run_count"] == 1

    @pytest.mark.asyncio
    async def test_scoped_run_artifacts_keep_attached_source_paths_at_workspace_root(
        self,
        client,
        tmp_path,
        database,
        state_manager,
        workspace_registry,
    ):
        workspace_path = tmp_path / "scoped-source-artifacts-ws"
        workspace_path.mkdir()
        source_dir = workspace_path / "seo-geo-review"
        source_dir.mkdir()
        source_file = source_dir / "audit-scope.md"
        source_text = "# Audit Scope\n\nSource context.\n"
        source_file.write_text(source_text, encoding="utf-8")
        source_bytes = source_text.encode("utf-8")
        source_sha = hashlib.sha256(source_bytes).hexdigest()

        run_workspace = workspace_path / "report-microsite"
        run_workspace.mkdir()

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Scoped Source Artifacts WS",
        )
        assert workspace is not None

        await database.insert_task(
            task_id="task-scoped-source-artifacts-1",
            goal="Make the results into a microsite",
            workspace_path=str(run_workspace),
            status="completed",
            context={
                "workspace_paths": [source_dir.name],
                "workspace_directories": [source_dir.name],
            },
            metadata={
                "source_workspace_root": str(workspace_path),
                "run_workspace_relative": run_workspace.name,
                "run_workspace_mode": "scoped_subfolder",
            },
        )
        state_manager.save_evidence_records(
            "task-scoped-source-artifacts-1",
            [
                {
                    "evidence_id": "EV-SCOPED-SOURCE-REL-1",
                    "task_id": "task-scoped-source-artifacts-1",
                    "subtask_id": "review-source-folder",
                    "phase_id": "research",
                    "tool": "read_file",
                    "evidence_kind": "artifact",
                    "artifact_workspace_relpath": f"{source_dir.name}/{source_file.name}",
                    "artifact_sha256": source_sha,
                    "artifact_size_bytes": len(source_bytes),
                    "created_at": "2026-03-27T18:10:00",
                },
                {
                    "evidence_id": "EV-SCOPED-SOURCE-ABS-1",
                    "task_id": "task-scoped-source-artifacts-1",
                    "subtask_id": "review-source-folder",
                    "phase_id": "research",
                    "tool": "read_file",
                    "evidence_kind": "artifact",
                    "artifact_workspace_relpath": str(source_file.resolve()),
                    "artifact_sha256": source_sha,
                    "artifact_size_bytes": len(source_bytes),
                    "created_at": "2026-03-27T18:10:01",
                },
            ],
        )

        run_artifacts = await client.get("/runs/task-scoped-source-artifacts-1/artifacts")
        assert run_artifacts.status_code == 200
        assert run_artifacts.json() == [
            {
                "path": f"{source_dir.name}/{source_file.name}",
                "category": "document",
                "source": "evidence",
                "sha256": source_sha,
                "size_bytes": len(source_bytes),
                "exists_on_disk": True,
                "is_intermediate": False,
                "created_at": "2026-03-27T18:10:01",
                "tool_name": "read_file",
                "subtask_ids": ["review-source-folder"],
                "phase_ids": ["research"],
                "facets": {},
            },
        ]

        workspace_artifacts = await client.get(f"/workspaces/{workspace['id']}/artifacts")
        assert workspace_artifacts.status_code == 200
        workspace_payload = workspace_artifacts.json()
        assert len(workspace_payload) == 1
        assert workspace_payload[0]["path"] == f"{source_dir.name}/{source_file.name}"
        assert workspace_payload[0]["exists_on_disk"] is True
        assert workspace_payload[0]["latest_run_id"] == "task-scoped-source-artifacts-1"
        assert workspace_payload[0]["run_ids"] == ["task-scoped-source-artifacts-1"]
        assert workspace_payload[0]["run_count"] == 1

    @pytest.mark.asyncio
    async def test_run_artifacts_merge_task_seals_and_evidence(
        self,
        client,
        tmp_path,
        database,
        state_manager,
        workspace_registry,
    ):
        workspace_path = tmp_path / "run-artifacts-ws"
        workspace_path.mkdir()
        report_path = workspace_path / "report.md"
        report_text = "# Ship Report\n\nEverything passed.\n"
        report_path.write_text(report_text, encoding="utf-8")
        report_sha = hashlib.sha256(report_text.encode("utf-8")).hexdigest()

        intermediate_path = (
            workspace_path
            / ".loom"
            / "phase-artifacts"
            / "run-a"
            / "phase-a"
            / "worker-a"
            / "notes.md"
        )
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        intermediate_text = "Worker notes"
        intermediate_path.write_text(intermediate_text, encoding="utf-8")
        intermediate_sha = hashlib.sha256(intermediate_text.encode("utf-8")).hexdigest()

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Artifacts WS",
        )
        assert workspace is not None

        await database.insert_task(
            task_id="task-run-artifacts-1",
            goal="Generate deliverables",
            workspace_path=str(workspace_path),
            status="completed",
            metadata={
                "artifact_seals": {
                    "report.md": {
                        "path": "report.md",
                        "sha256": report_sha,
                        "size_bytes": len(report_text.encode("utf-8")),
                        "tool": "document_write",
                        "subtask_id": "finalize-report",
                        "sealed_at": "2026-03-23T10:00:00",
                    },
                },
            },
        )
        state_manager.save_evidence_records(
            "task-run-artifacts-1",
            [
                {
                    "evidence_id": "EV-REPORT-1",
                    "task_id": "task-run-artifacts-1",
                    "subtask_id": "finalize-report",
                    "phase_id": "final",
                    "tool": "document_write",
                    "evidence_kind": "artifact",
                    "artifact_workspace_relpath": "report.md",
                    "artifact_sha256": report_sha,
                    "artifact_size_bytes": len(report_text.encode("utf-8")),
                    "facets": {"category": "deliverable", "target": "report"},
                    "created_at": "2026-03-23T10:01:00",
                },
                {
                    "evidence_id": "EV-PHASE-1",
                    "task_id": "task-run-artifacts-1",
                    "subtask_id": "phase-a-worker",
                    "phase_id": "phase-a",
                    "tool": "write_file",
                    "evidence_kind": "artifact",
                    "artifact_workspace_relpath": (
                        ".loom/phase-artifacts/run-a/phase-a/worker-a/notes.md"
                    ),
                    "artifact_sha256": intermediate_sha,
                    "artifact_size_bytes": len(intermediate_text.encode("utf-8")),
                    "facets": {"category": "worker_notes"},
                    "created_at": "2026-03-23T09:00:00",
                },
            ],
        )

        response = await client.get("/runs/task-run-artifacts-1/artifacts")
        assert response.status_code == 200
        payload = response.json()
        assert len(payload) == 2

        report = next(item for item in payload if item["path"] == "report.md")
        assert report["source"] == "seal+evidence"
        assert report["category"] == "deliverable"
        assert report["exists_on_disk"] is True
        assert report["is_intermediate"] is False
        assert report["tool_name"] == "document_write"
        assert report["phase_ids"] == ["final"]
        assert report["subtask_ids"] == ["finalize-report"]
        assert report["facets"]["target"] == "report"

        intermediate = next(
            item
            for item in payload
            if item["path"] == ".loom/phase-artifacts/run-a/phase-a/worker-a/notes.md"
        )
        assert intermediate["source"] == "evidence"
        assert intermediate["category"] == "intermediate"
        assert intermediate["exists_on_disk"] is True
        assert intermediate["is_intermediate"] is True
        assert intermediate["phase_ids"] == ["phase-a"]

    @pytest.mark.asyncio
    async def test_run_artifacts_prefer_fresher_state_seals_over_stale_task_row(
        self,
        client,
        tmp_path,
        database,
        state_manager,
    ):
        workspace_path = tmp_path / "run-artifacts-state-precedence-ws"
        workspace_path.mkdir()
        report_path = workspace_path / "report.md"
        report_text = "# Fresh Report\n"
        report_path.write_text(report_text, encoding="utf-8")
        fresh_sha = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
        stale_sha = hashlib.sha256(b"stale").hexdigest()

        await database.insert_task(
            task_id="task-run-artifacts-state-precedence-1",
            goal="Generate deliverables",
            workspace_path=str(workspace_path),
            status="completed",
            metadata={
                "artifact_seals": {
                    "report.md": {
                        "path": "report.md",
                        "sha256": stale_sha,
                        "size_bytes": 5,
                        "tool": "document_write",
                        "subtask_id": "finalize-report",
                        "sealed_at": "2026-03-23T10:00:00+00:00",
                    },
                },
            },
        )
        state_manager.save(
            Task(
                id="task-run-artifacts-state-precedence-1",
                goal="Generate deliverables",
                status=TaskStatus.COMPLETED,
                workspace=str(workspace_path),
                metadata={
                    "artifact_seals": {
                        "report.md": {
                            "path": "report.md",
                            "sha256": fresh_sha,
                            "size_bytes": len(report_text.encode("utf-8")),
                            "tool": "edit_file",
                            "subtask_id": "finalize-report",
                            "sealed_at": "2026-03-23T11:00:00+00:00",
                            "resealed_after_mutation": True,
                        },
                    },
                },
            ),
        )

        response = await client.get("/runs/task-run-artifacts-state-precedence-1/artifacts")
        assert response.status_code == 200
        payload = response.json()
        assert len(payload) == 1
        assert payload[0]["path"] == "report.md"
        assert payload[0]["sha256"] == fresh_sha
        assert payload[0]["source"] == "seal"

    @pytest.mark.asyncio
    async def test_run_artifacts_load_from_task_snapshot_when_task_row_is_missing(
        self,
        client,
        tmp_path,
        state_manager,
    ):
        workspace_path = tmp_path / "run-artifacts-state-only-ws"
        workspace_path.mkdir()
        artifact_path = workspace_path / "report.md"
        artifact_text = "# Snapshot Authority\n"
        artifact_path.write_text(artifact_text, encoding="utf-8")
        artifact_sha = hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()

        state_manager.save(
            Task(
                id="task-run-artifacts-state-only-1",
                goal="Generate deliverables",
                status=TaskStatus.COMPLETED,
                workspace=str(workspace_path),
                metadata={
                    "artifact_seals": {
                        "report.md": {
                            "path": "report.md",
                            "sha256": artifact_sha,
                            "size_bytes": len(artifact_text.encode("utf-8")),
                            "tool": "document_write",
                            "subtask_id": "finalize-report",
                            "sealed_at": "2026-03-23T12:00:00+00:00",
                        },
                    },
                },
            ),
        )

        response = await client.get("/runs/task-run-artifacts-state-only-1/artifacts")
        assert response.status_code == 200
        assert response.json() == [
            {
                "path": "report.md",
                "category": "document",
                "source": "seal",
                "sha256": artifact_sha,
                "size_bytes": len(artifact_text.encode("utf-8")),
                "exists_on_disk": True,
                "is_intermediate": False,
                "created_at": "2026-03-23T12:00:00+00:00",
                "tool_name": "document_write",
                "subtask_ids": ["finalize-report"],
                "phase_ids": [],
                "facets": {},
            },
        ]

    @pytest.mark.asyncio
    async def test_delete_run_removes_task_snapshot_and_evidence(
        self,
        client,
        tmp_path,
        database,
        state_manager,
    ):
        workspace_path = tmp_path / "delete-run-ws"
        workspace_path.mkdir()
        task_id = "task-run-delete-1"
        task = Task(
            id=task_id,
            goal="Delete this run",
            status=TaskStatus.COMPLETED,
            workspace=str(workspace_path),
            metadata={"run_id": "exec-run-delete-1"},
        )
        state_manager.save(task)
        state_manager.save_evidence_records(
            task_id,
            [
                {
                    "evidence_id": "EV-DELETE-1",
                    "task_id": task_id,
                    "subtask_id": "cleanup",
                    "tool": "document_write",
                    "created_at": "2026-03-23T12:30:00+00:00",
                },
            ],
        )
        await database.insert_task(
            task_id=task_id,
            goal=task.goal,
            workspace_path=task.workspace,
            status=task.status.value,
            metadata=task.metadata,
        )
        await database.insert_task_run(
            run_id="exec-run-delete-1",
            task_id=task_id,
            status="completed",
            process_name="cleanup",
        )
        await database.insert_event(
            task_id=task_id,
            correlation_id="corr-delete-1",
            run_id="exec-run-delete-1",
            event_type="task_completed",
            data={"message": "done"},
            sequence=1,
        )

        response = await client.delete(f"/runs/{task_id}")
        assert response.status_code == 200
        assert await database.get_task(task_id) is None
        assert await database.get_task_run("exec-run-delete-1") is None
        assert (state_manager._data_dir / "tasks" / task_id).exists() is False

    @pytest.mark.asyncio
    async def test_workspace_artifacts_aggregate_across_runs(
        self,
        client,
        tmp_path,
        database,
        state_manager,
        workspace_registry,
    ):
        workspace_path = tmp_path / "workspace-artifacts-ws"
        workspace_path.mkdir()
        shared_path = workspace_path / "report.md"
        shared_text = "Latest report"
        shared_path.write_text(shared_text, encoding="utf-8")
        shared_sha = hashlib.sha256(shared_text.encode("utf-8")).hexdigest()

        workspace = await workspace_registry.ensure_workspace(
            str(workspace_path),
            display_name="Workspace Artifacts WS",
        )
        assert workspace is not None

        await database.insert_task(
            task_id="task-run-artifacts-a",
            goal="First artifact run",
            workspace_path=str(workspace_path),
            status="completed",
        )
        await database.insert_task(
            task_id="task-run-artifacts-b",
            goal="Second artifact run",
            workspace_path=str(workspace_path),
            status="completed",
        )
        state_manager.save_evidence_records(
            "task-run-artifacts-a",
            [{
                "evidence_id": "EV-WS-1",
                "task_id": "task-run-artifacts-a",
                "subtask_id": "draft",
                "phase_id": "phase-a",
                "tool": "write_file",
                "evidence_kind": "artifact",
                "artifact_workspace_relpath": "report.md",
                "artifact_sha256": shared_sha,
                "artifact_size_bytes": len(shared_text.encode("utf-8")),
                "created_at": "2026-03-23T08:00:00",
            }],
        )
        state_manager.save_evidence_records(
            "task-run-artifacts-b",
            [{
                "evidence_id": "EV-WS-2",
                "task_id": "task-run-artifacts-b",
                "subtask_id": "finalize",
                "phase_id": "phase-b",
                "tool": "document_write",
                "evidence_kind": "artifact",
                "artifact_workspace_relpath": "report.md",
                "artifact_sha256": shared_sha,
                "artifact_size_bytes": len(shared_text.encode("utf-8")),
                "facets": {"category": "deliverable"},
                "created_at": "2026-03-23T09:30:00",
            }],
        )

        response = await client.get(f"/workspaces/{workspace['id']}/artifacts")
        assert response.status_code == 200
        payload = response.json()
        assert len(payload) == 1
        artifact = payload[0]
        assert artifact["path"] == "report.md"
        assert artifact["run_count"] == 2
        assert artifact["latest_run_id"] == "task-run-artifacts-b"
        assert set(artifact["run_ids"]) == {"task-run-artifacts-a", "task-run-artifacts-b"}
        assert artifact["category"] == "deliverable"
        assert set(artifact["phase_ids"]) == {"phase-a", "phase-b"}
        assert artifact["exists_on_disk"] is True

    @pytest.mark.asyncio
    async def test_run_control_and_message_endpoints(
        self,
        client,
        database,
        state_manager,
        mock_orchestrator,
        memory_manager,
    ):
        await database.insert_task(
            task_id="task-run-control-1",
            goal="Control task",
            workspace_path="/tmp/run-control",
            status="executing",
        )
        _make_task(
            state_manager,
            task_id="task-run-control-1",
            goal="Control task",
            status=TaskStatus.EXECUTING,
        )

        pause_response = await client.post("/runs/task-run-control-1/pause")
        assert pause_response.status_code == 200
        mock_orchestrator.pause_task.assert_called_once()

        resume_response = await client.post("/runs/task-run-control-1/resume")
        assert resume_response.status_code == 200
        mock_orchestrator.resume_task.assert_called_once()

        message_response = await client.post(
            "/runs/task-run-control-1/message",
            json={"message": "Need a quick progress update", "role": "user"},
        )
        assert message_response.status_code == 200
        entries = await memory_manager.query(
            task_id="task-run-control-1",
            entry_type="user_instruction",
        )
        assert any("quick progress update" in str(entry.detail or "") for entry in entries)

        cancel_response = await client.post("/runs/task-run-control-1/cancel")
        assert cancel_response.status_code == 200
        mock_orchestrator.cancel_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_instruction_history_remains_available_after_completion(
        self,
        client,
        database,
        memory_manager,
    ):
        await database.insert_task(
            task_id="task-run-history-1",
            goal="Completed task",
            workspace_path="/tmp/run-history",
            status="completed",
        )
        await memory_manager.store(
            MemoryEntry(
                task_id="task-run-history-1",
                entry_type="user_instruction",
                summary="Narrow scope",
                detail="Only focus on Alberta results.",
                tags="conversation",
            ),
        )

        response = await client.get("/tasks/task-run-history-1/conversation")

        assert response.status_code == 200
        payload = response.json()
        assert len(payload) == 1
        assert payload[0]["message"] == "Only focus on Alberta results."
        assert payload[0]["tags"] == "conversation"

    @pytest.mark.asyncio
    async def test_run_stream_uses_workspace_first_event_shape(self, client, database):
        await database.insert_task(
            task_id="task-run-stream-1",
            goal="Run stream task",
            workspace_path="/tmp/run-stream",
            status="executing",
        )
        await database.insert_event(
            task_id="task-run-stream-1",
            correlation_id="corr-1",
            run_id="exec-run-1",
            event_type=TASK_RUN_HEARTBEAT,
            data={"run_id": "exec-run-1", "status": "executing"},
            sequence=1,
        )
        await database.insert_event(
            task_id="task-run-stream-1",
            correlation_id="corr-1",
            run_id="exec-run-1",
            event_type=TASK_COMPLETED,
            data={"status": "completed"},
            sequence=2,
        )

        seen: list[dict[str, object]] = []
        async with client.stream(
            "GET",
            "/runs/task-run-stream-1/stream?follow=false",
        ) as response:
            current_event = ""
            current_id = ""
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("event: "):
                    current_event = line.removeprefix("event: ").strip()
                    continue
                if line.startswith("id: "):
                    current_id = line.removeprefix("id: ").strip()
                    continue
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line.removeprefix("data: "))
                if current_event == "run_event":
                    seen.append(payload)
                    assert current_id.isdigit()
                if payload.get("terminal") is True:
                    break

        assert seen
        assert seen[0]["event_type"] == TASK_RUN_HEARTBEAT
        assert seen[-1]["event_type"] == TASK_COMPLETED
        assert seen[-1]["terminal"] is True

    @pytest.mark.asyncio
    async def test_run_stream_respects_last_event_id_cursor(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
    ):
        workspace_path = tmp_path / "run-stream-cursor-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-run-stream-cursor-1",
            goal="Run stream cursor task",
            workspace_path=str(workspace_path),
            status="executing",
        )
        first_id = await database.insert_event(
            task_id="task-run-stream-cursor-1",
            correlation_id="corr-1",
            run_id="exec-run-1",
            event_type="task_executing",
            data={"message": "started", "status": "executing"},
            sequence=1,
        )
        await database.insert_event(
            task_id="task-run-stream-cursor-1",
            correlation_id="corr-1",
            run_id="exec-run-1",
            event_type="task_paused",
            data={"message": "paused", "status": "paused"},
            sequence=2,
        )

        async with client.stream(
            "GET",
            "/runs/task-run-stream-cursor-1/stream?follow=false",
            headers={"last-event-id": str(first_id)},
        ) as response:
            seen_ids: list[int] = []
            seen_payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("id: "):
                    seen_ids.append(int(line.removeprefix("id: ").strip()))
                    continue
                if not line.startswith("data: "):
                    continue
                seen_payloads.append(json.loads(line.removeprefix("data: ")))

        assert seen_ids == [2]
        assert [payload["event_type"] for payload in seen_payloads] == ["task_paused"]
        assert seen_payloads[0]["status"] == "paused"
        assert seen_payloads[0]["streaming"] is False

    @pytest.mark.asyncio
    async def test_run_stream_replays_recent_in_memory_events_before_sqlite_flush(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
        engine,
    ):
        workspace_path = tmp_path / "run-stream-history-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-run-stream-history-1",
            goal="Run stream history task",
            workspace_path=str(workspace_path),
            status="executing",
        )

        first = Event(
            event_type="task_executing",
            task_id="task-run-stream-history-1",
            data={"run_id": "exec-run-1", "status": "executing"},
        )
        second = Event(
            event_type="task_paused",
            task_id="task-run-stream-history-1",
            data={"run_id": "exec-run-1", "status": "paused"},
        )
        engine.event_bus.emit(first)
        engine.event_bus.emit(second)

        async with client.stream(
            "GET",
            "/runs/task-run-stream-history-1/stream?follow=false",
            headers={"last-event-id": str(first.data.get("sequence", 0))},
        ) as response:
            seen_ids: list[int] = []
            seen_payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("id: "):
                    seen_ids.append(int(line.removeprefix("id: ").strip()))
                    continue
                if not line.startswith("data: "):
                    continue
                seen_payloads.append(json.loads(line.removeprefix("data: ")))

        assert seen_ids == [2]
        assert [payload["event_type"] for payload in seen_payloads] == ["task_paused"]
        assert seen_payloads[0]["streaming"] is False

    @pytest.mark.asyncio
    async def test_run_stream_delivers_live_events_without_requerying_sqlite(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "run-stream-live-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-run-stream-live-1",
            goal="Run stream live task",
            workspace_path=str(workspace_path),
            status="executing",
        )

        original_query_events = engine.database.query_events
        query_call_count = 0

        async def tracking_query_events(*args, **kwargs):
            nonlocal query_call_count
            query_call_count += 1
            return await original_query_events(*args, **kwargs)

        monkeypatch.setattr(engine.database, "query_events", tracking_query_events)

        async def emit_later():
            await asyncio.sleep(0.02)
            engine.event_bus.emit(
                Event(
                    event_type=TASK_COMPLETED,
                    task_id="task-run-stream-live-1",
                    data={"run_id": "exec-run-1", "status": "completed"},
                ),
            )

        emit_task = asyncio.create_task(emit_later())
        payload = None
        async with client.stream(
            "GET",
            "/runs/task-run-stream-live-1/stream",
        ) as response:
            current_event = ""
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("event: "):
                    current_event = line.removeprefix("event: ").strip()
                    continue
                if current_event == "run_event" and line.startswith("data: "):
                    payload = json.loads(line.removeprefix("data: "))
                    break
        await emit_task

        assert payload is not None
        assert payload["event_type"] == TASK_COMPLETED
        assert query_call_count == 1

    @pytest.mark.asyncio
    async def test_run_timeline_include_noise_false_filters_noise_events(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
    ):
        workspace_path = tmp_path / "run-timeline-noise-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-run-timeline-noise-1",
            goal="Run timeline noise task",
            workspace_path=str(workspace_path),
            status="executing",
        )
        await database.insert_event(
            task_id="task-run-timeline-noise-1",
            correlation_id="corr-1",
            run_id="exec-run-noise-1",
            event_type=TASK_EXECUTING,
            data={"status": "executing"},
            sequence=1,
        )
        await database.insert_event(
            task_id="task-run-timeline-noise-1",
            correlation_id="corr-1",
            run_id="exec-run-noise-1",
            event_type=TASK_RUN_HEARTBEAT,
            data={"status": "executing"},
            sequence=2,
        )
        await database.insert_event(
            task_id="task-run-timeline-noise-1",
            correlation_id="corr-1",
            run_id="exec-run-noise-1",
            event_type="token_streamed",
            data={"token": "hello"},
            sequence=3,
        )

        response = await client.get(
            "/runs/task-run-timeline-noise-1/timeline?include_noise=false&limit=10",
        )

        assert response.status_code == 200
        payload = response.json()
        assert [row["event_type"] for row in payload] == [TASK_EXECUTING]

    @pytest.mark.asyncio
    async def test_run_timeline_include_noise_false_keeps_retry_scheduled_model_events(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
    ):
        workspace_path = tmp_path / "run-timeline-model-retry-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-run-model-retry-1",
            goal="Run model retry task",
            workspace_path=str(workspace_path),
            status="executing",
        )
        await database.insert_event(
            task_id="task-run-model-retry-1",
            correlation_id="corr-1",
            run_id="exec-run-model-retry-1",
            event_type=TASK_EXECUTING,
            data={"status": "executing"},
            sequence=1,
        )
        await database.insert_event(
            task_id="task-run-model-retry-1",
            correlation_id="corr-1",
            run_id="exec-run-model-retry-1",
            event_type=MODEL_INVOCATION,
            data={
                "phase": "done",
                "retry_scheduled": True,
                "retry_delay_seconds": 8.0,
                "http_status": 429,
                "model_error_code": "engine_overloaded_error",
            },
            sequence=2,
        )
        await database.insert_event(
            task_id="task-run-model-retry-1",
            correlation_id="corr-1",
            run_id="exec-run-model-retry-1",
            event_type=TOKEN_STREAMED,
            data={"token": "hello"},
            sequence=3,
        )

        response = await client.get(
            "/runs/task-run-model-retry-1/timeline?include_noise=false&limit=10",
        )

        assert response.status_code == 200
        payload = response.json()
        assert [row["event_type"] for row in payload] == [
            TASK_EXECUTING,
            MODEL_INVOCATION,
        ]

    @pytest.mark.asyncio
    async def test_run_stream_include_noise_false_filters_noise_events(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
    ):
        workspace_path = tmp_path / "run-stream-noise-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-run-stream-noise-1",
            goal="Run stream noise task",
            workspace_path=str(workspace_path),
            status="executing",
        )
        await database.insert_event(
            task_id="task-run-stream-noise-1",
            correlation_id="corr-1",
            run_id="exec-run-noise-1",
            event_type=TASK_EXECUTING,
            data={"status": "executing"},
            sequence=1,
        )
        await database.insert_event(
            task_id="task-run-stream-noise-1",
            correlation_id="corr-1",
            run_id="exec-run-noise-1",
            event_type=TASK_RUN_HEARTBEAT,
            data={"status": "executing"},
            sequence=2,
        )
        await database.insert_event(
            task_id="task-run-stream-noise-1",
            correlation_id="corr-1",
            run_id="exec-run-noise-1",
            event_type="token_streamed",
            data={"token": "hello"},
            sequence=3,
        )
        await database.insert_event(
            task_id="task-run-stream-noise-1",
            correlation_id="corr-1",
            run_id="exec-run-noise-1",
            event_type=TASK_COMPLETED,
            data={"status": "completed"},
            sequence=4,
        )

        async with client.stream(
            "GET",
            "/runs/task-run-stream-noise-1/stream?include_noise=false&follow=false",
        ) as response:
            payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue
                payloads.append(json.loads(line.removeprefix("data: ")))

        assert [payload["event_type"] for payload in payloads] == [
            TASK_EXECUTING,
            TASK_COMPLETED,
        ]

    @pytest.mark.asyncio
    async def test_run_stream_overflow_recovers_from_recent_history_without_dropping_events(
        self,
        tmp_path,
        database,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "run-stream-overflow-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-run-stream-overflow-1",
            goal="Run stream overflow task",
            workspace_path=str(workspace_path),
            status="executing",
        )

        from loom.api import routes as api_routes

        monkeypatch.setattr(api_routes, "_STREAM_QUEUE_MAXSIZE", 1)
        monkeypatch.setattr(api_routes, "EventSourceResponse", lambda generator: generator)
        request = SimpleNamespace(
            headers={},
            app=SimpleNamespace(state=SimpleNamespace(engine=engine)),
            is_disconnected=AsyncMock(return_value=False),
        )

        stream = await api_routes.stream_run_events(
            request,
            "task-run-stream-overflow-1",
            follow=True,
        )
        assert await anext(stream) == {"comment": "open"}

        engine.event_bus.emit(
            Event(
                event_type=TASK_EXECUTING,
                task_id="task-run-stream-overflow-1",
                data={"run_id": "exec-run-overflow-1", "status": "executing"},
            ),
        )
        first_item = await anext(stream)
        first_payload = json.loads(first_item["data"])
        assert first_payload["sequence"] == 1

        engine.event_bus.emit(
            Event(
                event_type=TASK_RUN_HEARTBEAT,
                task_id="task-run-stream-overflow-1",
                data={"run_id": "exec-run-overflow-1", "status": "executing"},
            ),
        )
        engine.event_bus.emit(
            Event(
                event_type=TASK_COMPLETED,
                task_id="task-run-stream-overflow-1",
                data={"run_id": "exec-run-overflow-1", "status": "completed"},
            ),
        )

        second_item = await anext(stream)
        third_item = await anext(stream)
        second_payload = json.loads(second_item["data"])
        third_payload = json.loads(third_item["data"])

        assert [second_payload["sequence"], third_payload["sequence"]] == [2, 3]
        assert [second_payload["event_type"], third_payload["event_type"]] == [
            TASK_RUN_HEARTBEAT,
            TASK_COMPLETED,
        ]
        assert third_payload["terminal"] is True

    @pytest.mark.asyncio
    async def test_notification_stream_delivers_live_events_without_requerying_sqlite(
        self,
        tmp_path,
        database,
        workspace_registry,
        engine,
        monkeypatch,
    ):
        workspace_path = tmp_path / "notification-stream-live-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-notification-stream-live-1",
            goal="Notification stream live task",
            workspace_path=str(workspace_path),
            status="executing",
        )

        original_query = engine.database.query
        query_call_count = 0

        async def tracking_query(*args, **kwargs):
            nonlocal query_call_count
            query_call_count += 1
            return await original_query(*args, **kwargs)

        monkeypatch.setattr(engine.database, "query", tracking_query)
        from loom.api import routes as api_routes

        monkeypatch.setattr(api_routes, "EventSourceResponse", lambda generator: generator)
        request = SimpleNamespace(
            headers={},
            app=SimpleNamespace(state=SimpleNamespace(engine=engine)),
            is_disconnected=AsyncMock(return_value=False),
        )

        async def emit_later():
            await asyncio.sleep(0.02)
            engine.event_bus.emit(
                Event(
                    event_type=APPROVAL_REQUESTED,
                    task_id="task-notification-stream-live-1",
                    data={"summary": "Need approval"},
                ),
            )

        emit_task = asyncio.create_task(emit_later())
        payload = None
        stream = await api_routes.stream_notifications(
            request,
            workspace_id=workspace["id"],
            follow=True,
        )
        async for item in stream:
            if item.get("event") == "notification":
                payload = json.loads(item["data"])
                break
        await emit_task

        assert payload is not None
        assert payload["event_type"] == APPROVAL_REQUESTED
        assert query_call_count == 1

    @pytest.mark.asyncio
    async def test_notification_stream_respects_last_event_id_cursor(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
    ):
        workspace_path = tmp_path / "notification-stream-cursor-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-notification-stream-1",
            goal="Notification stream task",
            workspace_path=str(workspace_path),
            status="executing",
        )
        first_id = await database.insert_event(
            task_id="task-notification-stream-1",
            correlation_id="corr-1",
            run_id="exec-run-1",
            event_type="approval_requested",
            data={"summary": "Need approval"},
            sequence=1,
        )
        second_id = await database.insert_event(
            task_id="task-notification-stream-1",
            correlation_id="corr-1",
            run_id="exec-run-1",
            event_type="approval_received",
            data={"summary": "Approved"},
            sequence=2,
        )

        async with client.stream(
            "GET",
            f"/notifications/stream?workspace_id={workspace['id']}&follow=false",
            headers={"last-event-id": str(first_id)},
        ) as response:
            seen_ids: list[int] = []
            seen_payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("id: "):
                    seen_ids.append(int(line.removeprefix("id: ").strip()))
                    continue
                if not line.startswith("data: "):
                    continue
                seen_payloads.append(json.loads(line.removeprefix("data: ")))

        assert seen_ids == [second_id]
        assert [payload["event_type"] for payload in seen_payloads] == ["approval_received"]
        assert seen_payloads[0]["workspace_id"] == workspace["id"]
        assert seen_payloads[0]["stream_id"] == second_id

    @pytest.mark.asyncio
    async def test_notification_stream_replays_recent_in_memory_events_before_sqlite_flush(
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
        engine,
    ):
        workspace_path = tmp_path / "notification-stream-history-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-notification-stream-history-1",
            goal="Notification stream history task",
            workspace_path=str(workspace_path),
            status="executing",
        )

        first = Event(
            event_type=APPROVAL_REQUESTED,
            task_id="task-notification-stream-history-1",
            data={"summary": "Need approval"},
        )
        second = Event(
            event_type=APPROVAL_RECEIVED,
            task_id="task-notification-stream-history-1",
            data={"summary": "Approved"},
        )
        engine.event_bus.emit(first)
        engine.event_bus.emit(second)

        async with client.stream(
            "GET",
            f"/notifications/stream?workspace_id={workspace['id']}&follow=false",
            headers={"last-event-id": f"event:{first.data.get('event_id', '')}"},
        ) as response:
            seen_ids: list[str] = []
            seen_payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("id: "):
                    seen_ids.append(line.removeprefix("id: ").strip())
                    continue
                if not line.startswith("data: "):
                    continue
                seen_payloads.append(json.loads(line.removeprefix("data: ")))

        assert seen_ids == [f"event:{second.data.get('event_id', '')}"]
        assert [payload["event_type"] for payload in seen_payloads] == [APPROVAL_RECEIVED]
        assert seen_payloads[0]["workspace_id"] == workspace["id"]
        assert seen_payloads[0]["stream_id"] is None

    @pytest.mark.asyncio
    async def test_notification_stream_initial_connect_replays_recent_in_memory_events_before_sqlite_flush(  # noqa: E501
        self,
        client,
        tmp_path,
        database,
        workspace_registry,
        engine,
    ):
        workspace_path = tmp_path / "notification-stream-initial-history-ws"
        workspace_path.mkdir()
        workspace = await workspace_registry.ensure_workspace(str(workspace_path))
        assert workspace is not None
        await database.insert_task(
            task_id="task-notification-stream-initial-history-1",
            goal="Notification initial history task",
            workspace_path=str(workspace_path),
            status="executing",
        )

        first = Event(
            event_type=APPROVAL_REQUESTED,
            task_id="task-notification-stream-initial-history-1",
            data={"summary": "Need approval"},
        )
        second = Event(
            event_type=APPROVAL_RECEIVED,
            task_id="task-notification-stream-initial-history-1",
            data={"summary": "Approved"},
        )
        engine.event_bus.emit(first)
        engine.event_bus.emit(second)

        async with client.stream(
            "GET",
            f"/notifications/stream?workspace_id={workspace['id']}&follow=false",
        ) as response:
            seen_ids: list[str] = []
            seen_payloads: list[dict[str, object]] = []
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("id: "):
                    seen_ids.append(line.removeprefix("id: ").strip())
                    continue
                if not line.startswith("data: "):
                    continue
                seen_payloads.append(json.loads(line.removeprefix("data: ")))

        assert seen_ids == [
            f"event:{first.data.get('event_id', '')}",
            f"event:{second.data.get('event_id', '')}",
        ]
        assert [payload["event_type"] for payload in seen_payloads] == [
            APPROVAL_REQUESTED,
            APPROVAL_RECEIVED,
        ]
        assert all(payload["workspace_id"] == workspace["id"] for payload in seen_payloads)
        assert all(payload["stream_id"] is None for payload in seen_payloads)


class TestTelemetrySettings:
    @pytest.mark.asyncio
    async def test_get_telemetry_settings_defaults(self, client):
        response = await client.get("/settings/telemetry")
        assert response.status_code == 200
        payload = response.json()
        assert payload["configured_mode"] == "active"
        assert payload["effective_mode"] == "active"
        assert payload["scope"] == "process_local"

    @pytest.mark.asyncio
    async def test_patch_telemetry_settings_disabled_by_default(self, client):
        response = await client.patch(
            "/settings/telemetry",
            json={"mode": "debug", "persist": False},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_patch_telemetry_settings_requires_admin_token(self, client, engine):
        engine.config = Config(
            telemetry=TelemetryConfig(
                mode="active",
                configured_mode_input="active",
                runtime_override_enabled=True,
                runtime_override_api_enabled=True,
                runtime_override_api_token="local-secret",
            ),
        )
        response = await client.patch(
            "/settings/telemetry",
            json={"mode": "debug", "persist": False},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_patch_telemetry_settings_accepts_bearer_admin_token(self, client, engine):
        engine.config = Config(
            telemetry=TelemetryConfig(
                mode="active",
                configured_mode_input="active",
                runtime_override_enabled=True,
                runtime_override_api_enabled=True,
                runtime_override_api_token="local-secret",
            ),
        )
        response = await client.patch(
            "/settings/telemetry",
            json={"mode": "debug", "persist": False},
            headers={"authorization": "Bearer local-secret"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["runtime_override_mode"] == "debug"

    @pytest.mark.asyncio
    async def test_patch_telemetry_settings_requires_loopback_origin(self, client, engine):
        engine.config = Config(
            telemetry=TelemetryConfig(
                mode="active",
                configured_mode_input="active",
                runtime_override_enabled=True,
                runtime_override_api_enabled=True,
                runtime_override_api_token="local-secret",
            ),
        )
        response = await client.patch(
            "/settings/telemetry",
            json={"mode": "debug", "persist": False},
            headers={
                "x-loom-admin-token": "local-secret",
                "x-forwarded-for": "203.0.113.20",
            },
        )
        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_patch_telemetry_settings_updates_mode_and_emits_audit_event(
        self,
        client,
        engine,
        event_bus,
    ):
        engine.config = Config(
            telemetry=TelemetryConfig(
                mode="active",
                configured_mode_input="active",
                runtime_override_enabled=True,
                runtime_override_api_enabled=True,
                runtime_override_api_token="local-secret",
            ),
        )
        observed = []
        event_bus.subscribe_all(lambda event: observed.append(event))
        response = await client.patch(
            "/settings/telemetry",
            json={"mode": "debug", "persist": False},
            headers={"x-loom-admin-token": "local-secret"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["configured_mode"] == "active"
        assert payload["runtime_override_mode"] == "debug"
        assert payload["effective_mode"] == "debug"
        mode_events = [
            event
            for event in observed
            if event.event_type == TELEMETRY_MODE_CHANGED
        ]
        assert mode_events
        assert mode_events[-1].data.get("effective_mode") == "debug"

    @pytest.mark.asyncio
    async def test_patch_telemetry_settings_normalizes_alias_and_emits_warning(
        self,
        client,
        engine,
        event_bus,
    ):
        engine.config = Config(
            telemetry=TelemetryConfig(
                mode="active",
                configured_mode_input="active",
                runtime_override_enabled=True,
                runtime_override_api_enabled=True,
                runtime_override_api_token="local-secret",
            ),
        )
        observed = []
        event_bus.subscribe_all(lambda event: observed.append(event))
        response = await client.patch(
            "/settings/telemetry",
            json={"mode": "internal_only", "persist": False},
            headers={"x-loom-admin-token": "local-secret"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["runtime_override_mode"] == "all_typed"
        warnings = [
            event
            for event in observed
            if event.event_type == TELEMETRY_SETTINGS_WARNING
        ]
        assert warnings
        assert warnings[-1].data.get("warning_code") == "telemetry_mode_alias_normalized"

    @pytest.mark.asyncio
    async def test_patch_telemetry_settings_persist_disabled_returns_400(self, client, engine):
        engine.config = Config(
            telemetry=TelemetryConfig(
                mode="active",
                configured_mode_input="active",
                runtime_override_enabled=True,
                runtime_override_api_enabled=True,
                runtime_override_api_token="local-secret",
                persist_runtime_override=False,
            ),
        )
        assert engine.runtime_telemetry_mode() is None
        assert engine.effective_telemetry_mode() == "active"
        response = await client.patch(
            "/settings/telemetry",
            json={"mode": "debug", "persist": True},
            headers={"x-loom-admin-token": "local-secret"},
        )
        assert response.status_code == 400
        assert engine.runtime_telemetry_mode() is None
        assert engine.effective_telemetry_mode() == "active"


class TestTelemetrySinkFiltering:
    @pytest.mark.asyncio
    async def test_stream_off_mode_keeps_passthrough_and_hides_tokens(
        self,
        client,
        engine,
        event_bus,
        state_manager,
    ):
        _make_task(
            state_manager,
            task_id="stream-1",
            status=TaskStatus.EXECUTING,
        )
        engine.set_runtime_telemetry_mode(
            mode_input="off",
            actor="test",
            source="test-suite",
            persist=False,
        )

        async def emit_events():
            await asyncio.sleep(0.02)
            event_bus.emit(
                Event(
                    event_type=TOKEN_STREAMED,
                    task_id="stream-1",
                    data={"token": "x", "subtask_id": "s1", "model": "m"},
                ),
            )
            event_bus.emit(
                Event(
                    event_type=TASK_RUN_HEARTBEAT,
                    task_id="stream-1",
                    data={"run_id": "run-1"},
                ),
            )
            event_bus.emit(
                Event(
                    event_type=TASK_COMPLETED,
                    task_id="stream-1",
                    data={},
                ),
            )

        seen_event_types: list[str] = []
        emitter = asyncio.create_task(emit_events())
        async with client.stream("GET", "/tasks/stream-1/stream") as response:
            current_event = ""
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("event: "):
                    current_event = line.removeprefix("event: ").strip()
                    continue
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line.removeprefix("data: "))
                seen_event_types.append(current_event or str(payload.get("event_type", "")))
                if current_event == TASK_COMPLETED:
                    break
        await emitter

        assert TASK_RUN_HEARTBEAT in seen_event_types
        assert TOKEN_STREAMED not in seen_event_types

    @pytest.mark.asyncio
    async def test_off_mode_does_not_suppress_compliance_persistence(
        self,
        engine,
        event_bus,
        state_manager,
    ):
        from loom.events.bus import EventPersister

        _make_task(
            state_manager,
            task_id="persist-1",
            status=TaskStatus.EXECUTING,
        )
        EventPersister(engine.database).attach(event_bus)
        engine.set_runtime_telemetry_mode(
            mode_input="off",
            actor="test",
            source="test-suite",
            persist=False,
        )
        event_bus.emit(
            Event(
                event_type=TASK_EXECUTING,
                task_id="persist-1",
                data={},
            ),
        )
        await event_bus.drain(timeout=1.0)
        rows = await engine.database.query_events("persist-1", event_type=TASK_EXECUTING)
        assert rows


# --- Task CRUD ---


class TestTaskCreate:
    @pytest.mark.asyncio
    async def test_create_task(self, client):
        response = await client.post("/tasks", json={
            "goal": "Write hello world",
        })
        assert response.status_code == 201
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_task_with_workspace(self, client, tmp_path):
        workspace = str(tmp_path)
        response = await client.post("/tasks", json={
            "goal": "Build a feature",
            "workspace": workspace,
        })
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_create_task_invalid_workspace(self, client):
        response = await client.post("/tasks", json={
            "goal": "Build",
            "workspace": "/nonexistent/path/xyz",
        })
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_create_task_missing_goal(self, client):
        response = await client.post("/tasks", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_task_emits_task_created_event(self, client, event_bus):
        events = []
        event_bus.subscribe_all(lambda event: events.append(event))
        response = await client.post("/tasks", json={"goal": "Emit telemetry"})
        assert response.status_code == 201
        task_id = response.json()["task_id"]
        created_events = [
            event for event in events
            if event.event_type == TASK_CREATED and event.task_id == task_id
        ]
        assert created_events
        payload = created_events[-1].data
        assert payload.get("goal") == "Emit telemetry"
        assert str(payload.get("run_id", "")).strip()

    @pytest.mark.asyncio
    async def test_create_task_forces_api_execution_surface(
        self,
        client,
        state_manager,
    ):
        response = await client.post(
            "/tasks",
            json={
                "goal": "Write hello world",
                "metadata": {"execution_surface": "tui"},
            },
        )
        assert response.status_code == 201
        task_id = str(response.json()["task_id"])
        task = state_manager.load(task_id)
        assert task.metadata["execution_surface"] == "api"

    @pytest.mark.asyncio
    async def test_create_task_returns_structured_unresolved_auth_payload(
        self,
        client,
        tmp_path,
    ):
        process_path = tmp_path / "auth-required.process.yaml"
        process_path.write_text(
            """
name: auth-required-process
version: "1.0"
auth:
  required:
    - provider: notion
      source: api
""",
            encoding="utf-8",
        )
        auth_path = tmp_path / "auth.toml"
        auth_path.write_text(
            """
[auth.profiles.notion_dev]
provider = "notion"
mode = "api_key"
secret_ref = "env://NOTION_TOKEN_DEV"

[auth.profiles.notion_prod]
provider = "notion"
mode = "api_key"
secret_ref = "env://NOTION_TOKEN_PROD"
""",
            encoding="utf-8",
        )

        response = await client.post(
            "/tasks",
            json={
                "goal": "Do auth-sensitive work",
                "workspace": str(tmp_path),
                "process": str(process_path),
                "metadata": {"auth_config_path": str(auth_path)},
            },
        )
        assert response.status_code == 400
        payload = response.json()
        assert isinstance(payload.get("detail"), dict)
        detail = payload["detail"]
        assert detail["code"] == "auth_unresolved"
        unresolved = detail.get("unresolved", [])
        assert isinstance(unresolved, list)
        assert unresolved
        assert unresolved[0]["provider"] == "notion"
        assert unresolved[0]["reason"] == "ambiguous"


class TestTaskGet:
    @pytest.mark.asyncio
    async def test_get_task(self, client, state_manager):
        _make_task(state_manager)
        response = await client.get("/tasks/test-1")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-1"
        assert data["goal"] == "Test goal"

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, client):
        response = await client.get("/tasks/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_task_with_plan(self, client, state_manager):
        _make_task_with_plan(state_manager)
        response = await client.get("/tasks/test-1")
        assert response.status_code == 200
        data = response.json()
        assert data["plan"] is not None
        assert len(data["plan"]["subtasks"]) == 2
        assert data["progress"]["total_subtasks"] == 2
        assert data["progress"]["completed"] == 1


class TestTaskCancel:
    @pytest.mark.asyncio
    async def test_cancel_task(self, client, state_manager, mock_orchestrator):
        _make_task(state_manager, status=TaskStatus.EXECUTING)
        response = await client.delete("/tasks/test-1")
        assert response.status_code == 200
        mock_orchestrator.cancel_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, client):
        response = await client.delete("/tasks/nonexistent")
        assert response.status_code == 404


class TestTaskSteer:
    @pytest.mark.asyncio
    async def test_steer_running_task(self, client, state_manager):
        _make_task(state_manager, status=TaskStatus.EXECUTING)
        response = await client.patch("/tasks/test-1", json={
            "instruction": "Use PostgreSQL instead of MySQL",
        })
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_steer_emits_steer_instruction_event(self, client, state_manager, event_bus):
        _make_task(state_manager, status=TaskStatus.EXECUTING)
        events = []
        event_bus.subscribe_all(lambda event: events.append(event))
        instruction = "Use PostgreSQL instead of MySQL"
        response = await client.patch("/tasks/test-1", json={"instruction": instruction})
        assert response.status_code == 200
        steer_events = [
            event for event in events
            if event.event_type == STEER_INSTRUCTION and event.task_id == "test-1"
        ]
        assert steer_events
        assert steer_events[-1].data.get("instruction_chars") == len(instruction)

    @pytest.mark.asyncio
    async def test_steer_completed_task(self, client, state_manager):
        _make_task(state_manager, status=TaskStatus.COMPLETED)
        response = await client.patch("/tasks/test-1", json={
            "instruction": "Change approach",
        })
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_steer_paused_task(self, client, state_manager):
        _make_task(state_manager, status=TaskStatus.PAUSED)
        response = await client.patch("/tasks/test-1", json={
            "instruction": "Apply this once resumed",
        })
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_steer_not_found(self, client):
        response = await client.patch("/tasks/nonexistent", json={
            "instruction": "test",
        })
        assert response.status_code == 404


class TestTaskApprove:
    @pytest.mark.asyncio
    async def test_approve(self, client, state_manager):
        _make_task_with_plan(state_manager)
        response = await client.post("/tasks/test-1/approve", json={
            "subtask_id": "s2",
            "approved": True,
            "reason": "Looks good",
        })
        assert response.status_code == 200
        assert response.json()["approved"] is True

    @pytest.mark.asyncio
    async def test_approve_not_found(self, client):
        response = await client.post("/tasks/nonexistent/approve", json={
            "subtask_id": "s1",
            "approved": True,
        })
        assert response.status_code == 404


class TestTaskQuestions:
    @pytest.mark.asyncio
    async def test_list_pending_questions(self, client, state_manager, memory_manager):
        _make_task(
            state_manager,
            status=TaskStatus.EXECUTING,
            metadata={"execution_surface": "tui"},
        )
        await memory_manager.upsert_pending_task_question(
            question_id="q-1",
            task_id="test-1",
            subtask_id="s1",
            request_payload={
                "question_id": "q-1",
                "question": "Choose stack",
                "question_type": "single_choice",
                "options": [
                    {"id": "py", "label": "Python", "description": ""},
                    {"id": "rs", "label": "Rust", "description": ""},
                ],
            },
        )

        response = await client.get("/tasks/test-1/questions")
        assert response.status_code == 200
        rows = response.json()
        assert len(rows) == 1
        assert rows[0]["question_id"] == "q-1"
        assert rows[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_answer_question(self, client, state_manager, memory_manager):
        _make_task(
            state_manager,
            status=TaskStatus.EXECUTING,
            metadata={"execution_surface": "tui"},
        )
        await memory_manager.upsert_pending_task_question(
            question_id="q-2",
            task_id="test-1",
            subtask_id="s1",
            request_payload={
                "question_id": "q-2",
                "question": "Choose stack",
                "question_type": "single_choice",
                "allow_custom_response": False,
                "options": [
                    {"id": "py", "label": "Python", "description": ""},
                    {"id": "rs", "label": "Rust", "description": ""},
                ],
            },
        )

        response = await client.post(
            "/tasks/test-1/questions/q-2/answer",
            json={
                "selected_option_ids": ["py"],
                "source": "api",
                "answered_by": "qa",
                "client_id": "test-client",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "answered"
        assert payload["answer_payload"]["selected_option_ids"] == ["py"]
        assert payload["answer_payload"]["selected_labels"] == ["Python"]
        assert payload["answer_payload"]["answered_by"] == "qa"
        assert payload["answer_payload"]["client_id"] == "test-client"

    @pytest.mark.asyncio
    async def test_answer_question_unknown_returns_404(self, client, state_manager):
        _make_task(
            state_manager,
            status=TaskStatus.EXECUTING,
            metadata={"execution_surface": "tui"},
        )
        response = await client.post(
            "/tasks/test-1/questions/missing/answer",
            json={"custom_response": "hello"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_answer_non_pending_timeout_returns_409(
        self,
        client,
        state_manager,
        memory_manager,
    ):
        _make_task(
            state_manager,
            status=TaskStatus.EXECUTING,
            metadata={"execution_surface": "tui"},
        )
        await memory_manager.upsert_pending_task_question(
            question_id="q-3",
            task_id="test-1",
            subtask_id="s1",
            request_payload={
                "question_id": "q-3",
                "question": "Provide details",
                "question_type": "free_text",
                "allow_custom_response": True,
            },
        )
        await memory_manager.resolve_task_question(
            task_id="test-1",
            question_id="q-3",
            status="timeout",
            answer_payload={
                "question_id": "q-3",
                "response_type": "timeout",
                "selected_option_ids": [],
                "selected_labels": [],
                "custom_response": "",
                "answered_at": "2026-03-01T00:00:00+00:00",
                "source": "policy_default",
            },
        )

        response = await client.post(
            "/tasks/test-1/questions/q-3/answer",
            json={"custom_response": "late answer"},
        )
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_duplicate_answer_submission_is_idempotent(
        self,
        client,
        state_manager,
        memory_manager,
    ):
        _make_task(
            state_manager,
            status=TaskStatus.EXECUTING,
            metadata={"execution_surface": "tui"},
        )
        await memory_manager.upsert_pending_task_question(
            question_id="q-4",
            task_id="test-1",
            subtask_id="s1",
            request_payload={
                "question_id": "q-4",
                "question": "Choose stack",
                "question_type": "single_choice",
                "allow_custom_response": False,
                "options": [
                    {"id": "py", "label": "Python", "description": ""},
                    {"id": "rs", "label": "Rust", "description": ""},
                ],
            },
        )
        await memory_manager.resolve_task_question(
            task_id="test-1",
            question_id="q-4",
            status="answered",
            answer_payload={
                "question_id": "q-4",
                "response_type": "single_choice",
                "selected_option_ids": ["py"],
                "selected_labels": ["Python"],
                "custom_response": "",
                "answered_at": "2026-03-01T00:00:00+00:00",
                "source": "api",
            },
        )

        response = await client.post(
            "/tasks/test-1/questions/q-4/answer",
            json={
                "selected_option_ids": ["rs"],
                "source": "api",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "answered"
        assert payload["answer_payload"]["selected_option_ids"] == ["py"]

    @pytest.mark.asyncio
    async def test_question_endpoints_blocked_for_non_tui_surface(
        self,
        client,
        state_manager,
        memory_manager,
    ):
        _make_task(
            state_manager,
            status=TaskStatus.EXECUTING,
            metadata={"execution_surface": "cli"},
        )
        await memory_manager.upsert_pending_task_question(
            question_id="q-cli",
            task_id="test-1",
            subtask_id="s1",
            request_payload={
                "question_id": "q-cli",
                "question": "Should not be exposed",
                "question_type": "free_text",
            },
        )
        list_response = await client.get("/tasks/test-1/questions")
        assert list_response.status_code == 409
        answer_response = await client.post(
            "/tasks/test-1/questions/q-cli/answer",
            json={"custom_response": "nope"},
        )
        assert answer_response.status_code == 409


class TestTaskFeedback:
    @pytest.mark.asyncio
    async def test_feedback(self, client, state_manager):
        _make_task(state_manager, status=TaskStatus.EXECUTING)
        response = await client.post("/tasks/test-1/feedback", json={
            "feedback": "Use type hints everywhere",
        })
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_feedback_not_found(self, client):
        response = await client.post("/tasks/nonexistent/feedback", json={
            "feedback": "test",
        })
        assert response.status_code == 404


# --- Subtask Endpoints ---


class TestSubtasks:
    @pytest.mark.asyncio
    async def test_list_subtasks(self, client, state_manager):
        _make_task_with_plan(state_manager)
        response = await client.get("/tasks/test-1/subtasks")
        assert response.status_code == 200
        subtasks = response.json()
        assert len(subtasks) == 2
        assert subtasks[0]["id"] == "s1"

    @pytest.mark.asyncio
    async def test_list_subtasks_no_plan(self, client, state_manager):
        _make_task(state_manager)
        response = await client.get("/tasks/test-1/subtasks")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_get_subtask(self, client, state_manager):
        _make_task_with_plan(state_manager)
        response = await client.get("/tasks/test-1/subtasks/s1")
        assert response.status_code == 200
        assert response.json()["id"] == "s1"
        assert response.json()["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_subtask_not_found(self, client, state_manager):
        _make_task_with_plan(state_manager)
        response = await client.get("/tasks/test-1/subtasks/nonexistent")
        assert response.status_code == 404


# --- Memory ---


class TestMemoryEndpoints:
    @pytest.mark.asyncio
    async def test_query_empty_memory(self, client, state_manager):
        _make_task(state_manager)
        response = await client.get("/tasks/test-1/memory")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_search_memory(self, client):
        response = await client.get("/memory/search?q=test&task_id=t1")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_search_no_task_id(self, client):
        response = await client.get("/memory/search?q=test")
        assert response.status_code == 200
        assert response.json() == []


# --- Split-brain fix: create -> list consistency ---


class TestTaskListAfterCreate:
    @pytest.mark.asyncio
    async def test_create_then_list_shows_task(self, client, database):
        """Verify that after POST /tasks, GET /tasks returns the new task."""
        response = await client.post("/tasks", json={
            "goal": "Integration test goal",
        })
        assert response.status_code == 201
        task_id = response.json()["task_id"]

        list_response = await client.get("/tasks")
        assert list_response.status_code == 200
        tasks = list_response.json()
        task_ids = [t["task_id"] for t in tasks]
        assert task_id in task_ids

    @pytest.mark.asyncio
    async def test_create_then_list_status_transitions(self, client, database):
        """Verify task appears with correct initial status in /tasks list."""
        response = await client.post("/tasks", json={"goal": "Status test"})
        assert response.status_code == 201
        task_id = response.json()["task_id"]

        list_response = await client.get("/tasks?status=pending")
        assert list_response.status_code == 200
        tasks = list_response.json()
        task_ids = [t["task_id"] for t in tasks]
        assert task_id in task_ids


# --- CLI cancel uses DELETE ---


class TestCancelUsesDelete:
    @pytest.mark.asyncio
    async def test_cancel_via_delete(self, client, state_manager, mock_orchestrator):
        """Verify cancel endpoint is DELETE /tasks/{id}."""
        _make_task(state_manager, status=TaskStatus.EXECUTING)
        response = await client.delete("/tasks/test-1")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        mock_orchestrator.cancel_task.assert_called_once()


class TestTaskPauseResume:
    @pytest.mark.asyncio
    async def test_pause_task(self, client, state_manager, mock_orchestrator):
        _make_task(state_manager, status=TaskStatus.EXECUTING)
        response = await client.post("/tasks/test-1/pause")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        mock_orchestrator.pause_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_task(self, client, state_manager, mock_orchestrator):
        _make_task(state_manager, status=TaskStatus.PAUSED)
        response = await client.post("/tasks/test-1/resume")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        mock_orchestrator.resume_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_run_uses_canonical_task_state_when_task_row_missing(
        self,
        client,
        state_manager,
        mock_orchestrator,
    ):
        _make_task(state_manager, status=TaskStatus.EXECUTING)
        response = await client.post("/runs/test-1/pause")
        assert response.status_code == 200
        assert response.json()["task_status"] == TaskStatus.PAUSED.value
        mock_orchestrator.pause_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_run_uses_canonical_task_state_when_task_row_missing(
        self,
        client,
        state_manager,
        mock_orchestrator,
    ):
        _make_task(state_manager, status=TaskStatus.PAUSED)
        response = await client.post("/runs/test-1/resume")
        assert response.status_code == 200
        assert response.json()["task_status"] == TaskStatus.EXECUTING.value
        mock_orchestrator.resume_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_message_uses_canonical_task_state_when_task_row_missing(
        self,
        client,
        state_manager,
    ):
        _make_task(state_manager, status=TaskStatus.EXECUTING)
        response = await client.post(
            "/runs/test-1/message",
            json={"role": "user", "message": "Please continue."},
        )
        assert response.status_code == 200
        assert response.json()["task_id"] == "test-1"

    @pytest.mark.asyncio
    async def test_pause_invalid_status(self, client, state_manager):
        _make_task(state_manager, status=TaskStatus.COMPLETED)
        response = await client.post("/tasks/test-1/pause")
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_resume_invalid_status(self, client, state_manager):
        _make_task(state_manager, status=TaskStatus.FAILED)
        response = await client.post("/tasks/test-1/resume")
        assert response.status_code == 409


# --- Process field in task creation ---


class TestProcessField:
    @pytest.mark.asyncio
    async def test_create_task_with_invalid_process(self, client, tmp_path):
        """Submitting an unknown process name returns 400."""
        response = await client.post("/tasks", json={
            "goal": "Test",
            "workspace": str(tmp_path),
            "process": "nonexistent-process-xyz",
        })
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_create_task_without_process(self, client):
        """Creating a task without process still works."""
        response = await client.post("/tasks", json={
            "goal": "No process",
        })
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_create_task_uses_default_process_when_omitted(
        self,
        client,
        engine,
        monkeypatch,
    ):
        """When request omits process, config.process.default should apply."""
        engine.config = Config(
            process=ProcessConfig(default="default-process"),
        )
        captured: dict[str, object] = {}

        class DummyLoader:
            def __init__(self, workspace=None, extra_search_paths=None, **kwargs):
                captured["workspace"] = workspace
                captured["extra_search_paths"] = extra_search_paths
                captured["kwargs"] = kwargs

            def load(self, name):
                captured["name"] = name
                return SimpleNamespace(
                    name=name,
                    tools=SimpleNamespace(required=[], excluded=[]),
                )

        monkeypatch.setattr("loom.processes.schema.ProcessLoader", DummyLoader)

        response = await client.post("/tasks", json={"goal": "Use default process"})

        assert response.status_code == 201
        assert captured["name"] == "default-process"

    @pytest.mark.asyncio
    async def test_create_task_explicit_process_overrides_default(
        self,
        client,
        engine,
        monkeypatch,
    ):
        """Explicit request process should override config.process.default."""
        engine.config = Config(
            process=ProcessConfig(default="default-process"),
        )
        captured: dict[str, object] = {}

        class DummyLoader:
            def __init__(self, workspace=None, extra_search_paths=None, **kwargs):
                captured["workspace"] = workspace
                captured["extra_search_paths"] = extra_search_paths
                captured["kwargs"] = kwargs

            def load(self, name):
                captured["name"] = name
                return SimpleNamespace(
                    name=name,
                    tools=SimpleNamespace(required=[], excluded=[]),
                )

        monkeypatch.setattr("loom.processes.schema.ProcessLoader", DummyLoader)

        response = await client.post("/tasks", json={
            "goal": "Use explicit process",
            "process": "explicit-process",
        })

        assert response.status_code == 201
        assert captured["name"] == "explicit-process"

    @pytest.mark.asyncio
    async def test_create_task_uses_configured_process_search_paths(
        self,
        client,
        engine,
        tmp_path,
        monkeypatch,
    ):
        """API process loader should honor config.process.search_paths."""
        extra_dir = tmp_path / "custom-processes"
        extra_dir.mkdir()
        engine.config = Config(
            process=ProcessConfig(search_paths=[str(extra_dir)]),
        )

        captured: dict[str, object] = {}

        class DummyLoader:
            def __init__(self, workspace=None, extra_search_paths=None, **kwargs):
                captured["workspace"] = workspace
                captured["extra_search_paths"] = extra_search_paths
                captured["kwargs"] = kwargs

            def load(self, name):
                # Any non-None object is fine for route wiring.
                return {"name": name}

        monkeypatch.setattr("loom.processes.schema.ProcessLoader", DummyLoader)

        response = await client.post("/tasks", json={
            "goal": "Test process search paths",
            "workspace": str(tmp_path),
            "process": "demo-process",
        })

        assert response.status_code == 201
        assert captured["workspace"] == tmp_path
        assert captured["extra_search_paths"] == [extra_dir]

    @pytest.mark.asyncio
    async def test_create_task_rejects_missing_required_tools(
        self,
        client,
        monkeypatch,
    ):
        """Process-required missing tools should return HTTP 400 immediately."""
        captured: dict[str, object] = {}

        class DummyLoader:
            def __init__(self, workspace=None, extra_search_paths=None, **kwargs):
                captured["workspace"] = workspace
                captured["extra_search_paths"] = extra_search_paths
                captured["kwargs"] = kwargs

            def load(self, name):
                return SimpleNamespace(
                    name=name,
                    tools=SimpleNamespace(
                        required=["definitely-missing-tool"],
                        excluded=[],
                    ),
                )

        monkeypatch.setattr("loom.processes.schema.ProcessLoader", DummyLoader)

        response = await client.post("/tasks", json={
            "goal": "Reject missing required tools",
            "process": "demo-process",
        })

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "requires missing tool" in detail
        assert "definitely-missing-tool" in detail

    def test_required_auth_resources_for_process_uses_required_tool_set_when_declared(self):
        from loom.api.routes import _required_auth_resources_for_process

        included_tool = SimpleNamespace(
            auth_requirements=[
                {
                    "provider": "ga_provider",
                    "source": "api",
                    "modes": ["api_key"],
                }
            ]
        )
        non_required_tool = SimpleNamespace(
            auth_requirements=[
                {
                    "provider": "non_required_provider",
                    "source": "api",
                }
            ]
        )
        excluded_tool = SimpleNamespace(
            auth_requirements=[
                {
                    "provider": "should_not_appear",
                    "source": "api",
                }
            ]
        )

        process_def = SimpleNamespace(
            auth=SimpleNamespace(required=[{"provider": "notion", "source": "mcp"}]),
            tools=SimpleNamespace(required=["tool_included"], excluded=["tool_excluded"]),
        )
        registry = SimpleNamespace(
            list_tools=lambda: ["tool_included", "tool_non_required", "tool_excluded"],
            get=lambda name: {
                "tool_included": included_tool,
                "tool_non_required": non_required_tool,
                "tool_excluded": excluded_tool,
            }.get(name),
        )

        required = _required_auth_resources_for_process(
            process_def,
            tool_registry=registry,
        )
        selectors = {
            (item["provider"], item.get("source", "api"))
            for item in required
        }
        assert ("notion", "mcp") in selectors
        assert ("ga_provider", "api") in selectors
        assert ("non_required_provider", "api") not in selectors
        assert ("should_not_appear", "api") not in selectors


class TestBackgroundExecution:
    @pytest.mark.asyncio
    async def test_process_task_uses_isolated_orchestrator(self):
        """Process-backed task execution should not mutate shared orchestrator."""
        from loom.api.routes import _execute_in_background

        shared_orch = MagicMock()
        shared_orch._process = "BASE"
        shared_orch._prompts = SimpleNamespace(process="BASE")
        shared_orch.execute_task = AsyncMock()

        isolated_orch = MagicMock()
        result = SimpleNamespace(id="task-1", status=TaskStatus.COMPLETED)
        isolated_orch.execute_task = AsyncMock(return_value=result)

        engine = MagicMock()
        engine.orchestrator = shared_orch
        engine.create_task_orchestrator = MagicMock(return_value=isolated_orch)
        engine._sync_task_row_snapshot = AsyncMock()
        engine.state_manager.save = MagicMock()

        task = SimpleNamespace(id="task-1", status=TaskStatus.PENDING)

        await _execute_in_background(engine, task, process_def={"name": "proc"})

        engine.create_task_orchestrator.assert_called_once_with({"name": "proc"})
        isolated_orch.execute_task.assert_awaited_once_with(task)
        shared_orch.execute_task.assert_not_awaited()
        assert shared_orch._process == "BASE"
        assert shared_orch._prompts.process == "BASE"

    @pytest.mark.asyncio
    async def test_non_process_task_uses_shared_orchestrator(self):
        """Non-process task execution should keep using the shared orchestrator."""
        from loom.api.routes import _execute_in_background

        shared_orch = MagicMock()
        result = SimpleNamespace(id="task-2", status=TaskStatus.COMPLETED)
        shared_orch.execute_task = AsyncMock(return_value=result)

        engine = MagicMock()
        engine.orchestrator = shared_orch
        engine.create_task_orchestrator = MagicMock()
        engine._sync_task_row_snapshot = AsyncMock()
        engine.state_manager.save = MagicMock()

        task = SimpleNamespace(id="task-2", status=TaskStatus.PENDING)
        await _execute_in_background(engine, task, process_def=None)

        engine.create_task_orchestrator.assert_not_called()
        shared_orch.execute_task.assert_awaited_once_with(task)
        engine._sync_task_row_snapshot.assert_awaited_once_with(result)


class TestEngineOrchestratorFactory:
    def test_create_task_orchestrator_isolated_components(self, engine):
        """Per-task factory should create fresh prompt/tools and pass process."""
        process_def = SimpleNamespace(
            name="process",
            tools=SimpleNamespace(excluded=[]),
        )
        task_orch = engine.create_task_orchestrator(process=process_def)

        assert task_orch is not engine.orchestrator
        assert task_orch._tools is not engine.tool_registry
        assert task_orch._prompts is not engine.prompt_assembler
        assert task_orch._process == process_def

    def test_create_task_orchestrator_rejects_missing_required_tools(self, engine):
        """Process-required tools must exist at orchestrator creation time."""
        process_def = SimpleNamespace(
            name="process",
            tools=SimpleNamespace(
                excluded=[],
                required=["definitely-missing-tool"],
            ),
        )

        with pytest.raises(ValueError, match="requires missing tool"):
            engine.create_task_orchestrator(process=process_def)

    def test_create_task_orchestrator_reports_required_tool_unavailability_reason(
        self,
        engine,
        monkeypatch,
    ):
        process_def = SimpleNamespace(
            name="process",
            tools=SimpleNamespace(
                excluded=[],
                required=["openai_codex"],
            ),
        )
        registry = MagicMock()
        registry.list_tools.return_value = []
        registry.availability.return_value = ToolAvailabilityStatus(
            state="unavailable",
            reasons=(
                ToolAvailabilityReason(
                    code="binary_not_found",
                    message="Binary not found: codex",
                ),
            ),
        )
        monkeypatch.setattr(
            "loom.api.engine.create_default_registry",
            lambda *_args, **_kwargs: registry,
        )

        with pytest.raises(ValueError, match="Binary not found: codex"):
            engine.create_task_orchestrator(process=process_def)

        registry.list_tools.assert_called_once_with(
            runnable_only=True,
            execution_surface="api",
        )
        registry.availability.assert_called_once_with(
            "openai_codex",
            execution_surface="api",
        )

    def test_refresh_config_from_runtime_store_rebuilds_runtime_tooling(
        self,
        engine,
        monkeypatch,
        tmp_path,
    ):
        refreshed_config = Config(
            execution=ExecutionConfig(
                ask_user_v2_enabled=True,
                ask_user_runtime_blocking_enabled=True,
                ask_user_durable_state_enabled=True,
                ask_user_api_enabled=True,
                tool_binary_overrides={"codex": "/custom/codex"},
            ),
            source_path=str(tmp_path / "updated-loom.toml"),
        )
        monkeypatch.setattr(
            engine.config_runtime_store,
            "effective_config",
            lambda: refreshed_config,
        )

        refreshed_registry = MagicMock()
        refreshed_registry.get.return_value = None
        monkeypatch.setattr(
            "loom.api.engine.create_default_registry",
            lambda *_args, **_kwargs: refreshed_registry,
        )

        created: dict[str, object] = {}

        def _fake_orchestrator(**kwargs):
            created.update(kwargs)
            return SimpleNamespace(
                _process=kwargs.get("process"),
                _tools=kwargs.get("tool_registry"),
                _config=kwargs.get("config"),
                _prompts=kwargs.get("prompt_assembler"),
            )

        monkeypatch.setattr("loom.api.engine.Orchestrator", _fake_orchestrator)
        engine.orchestrator = SimpleNamespace(_process="BASE_PROCESS")

        result = engine.refresh_config_from_runtime_store()

        assert result is refreshed_config
        assert engine.config is refreshed_config
        assert engine.tool_registry is refreshed_registry
        assert engine.orchestrator._tools is refreshed_registry
        assert engine.orchestrator._config is refreshed_config
        assert created["execution_surface"] == "api"
        assert created["process"] == "BASE_PROCESS"

    @pytest.mark.asyncio
    async def test_create_task_orchestrator_mirrors_saved_snapshots_to_task_rows(
        self,
        engine,
        database,
        tmp_path,
    ):
        workspace_path = tmp_path / "orchestrator-mirror-ws"
        workspace_path.mkdir()
        task_orch = engine.create_task_orchestrator()
        task = Task(
            id="task-orchestrator-mirror-1",
            goal="Mirror my state",
            status=TaskStatus.EXECUTING,
            workspace=str(workspace_path),
            metadata={"run_id": "exec-run-orchestrator-1"},
            plan=Plan(
                subtasks=[
                    Subtask(
                        id="mirror-step",
                        description="Sync the mirror",
                        status=SubtaskStatus.RUNNING,
                    ),
                ],
            ),
        )

        await task_orch._save_task_state(task)

        row = await database.get_task(task.id)
        assert row is not None
        assert row["status"] == TaskStatus.EXECUTING.value
        assert row["workspace_path"] == str(workspace_path)
        assert json.loads(str(row.get("metadata") or "{}"))["run_id"] == "exec-run-orchestrator-1"
        assert json.loads(str(row.get("plan") or "{}"))["subtasks"][0]["status"] == (
            SubtaskStatus.RUNNING.value
        )
