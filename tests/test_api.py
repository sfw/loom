"""Tests for the Loom API server."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from loom.api.engine import Engine
from loom.config import Config, ProcessConfig
from loom.engine.orchestrator import Orchestrator
from loom.events.bus import EventBus
from loom.events.webhook import WebhookDelivery
from loom.learning.manager import LearningManager
from loom.models.router import ModelRouter
from loom.prompts.assembler import PromptAssembler
from loom.recovery.approval import ApprovalManager
from loom.state.memory import Database, MemoryManager
from loom.state.task_state import (
    Plan,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStateManager,
    TaskStatus,
)
from loom.tools import create_default_registry

# --- Test Fixtures ---


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
async def database(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    await db.initialize()
    return db


@pytest.fixture
async def memory_manager(database):
    return MemoryManager(database)


@pytest.fixture
def state_manager(tmp_path):
    return TaskStateManager(data_dir=tmp_path / "state")


@pytest.fixture
def tool_registry():
    return create_default_registry()


@pytest.fixture
def mock_orchestrator():
    orch = MagicMock(spec=Orchestrator)
    orch.execute_task = AsyncMock()
    orch.cancel_task = MagicMock()
    return orch


@pytest.fixture
def engine(
    event_bus,
    database,
    memory_manager,
    state_manager,
    tool_registry,
    mock_orchestrator,
):
    return Engine(
        config=Config(),
        orchestrator=mock_orchestrator,
        event_bus=event_bus,
        model_router=ModelRouter(),
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        state_manager=state_manager,
        prompt_assembler=PromptAssembler(),
        database=database,
        approval_manager=ApprovalManager(event_bus),
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


def _make_task(state_manager, task_id="test-1", goal="Test goal", status=TaskStatus.PENDING):
    """Create and persist a task for testing."""
    task = Task(
        id=task_id,
        goal=goal,
        status=status,
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
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


# --- Health & System ---


class TestSystemEndpoints:
    @pytest.mark.asyncio
    async def test_health(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_models(self, client):
        response = await client.get("/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_tools(self, client):
        response = await client.get("/tools")
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "name" in tools[0]

    @pytest.mark.asyncio
    async def test_config(self, client):
        response = await client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "server" in data
        assert "execution" in data


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
    async def test_steer_completed_task(self, client, state_manager):
        _make_task(state_manager, status=TaskStatus.COMPLETED)
        response = await client.patch("/tasks/test-1", json={
            "instruction": "Change approach",
        })
        assert response.status_code == 409

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
            def __init__(self, workspace=None, extra_search_paths=None):
                captured["workspace"] = workspace
                captured["extra_search_paths"] = extra_search_paths

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
        engine.database.update_task_status = AsyncMock()
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
        engine.database.update_task_status = AsyncMock()
        engine.state_manager.save = MagicMock()

        task = SimpleNamespace(id="task-2", status=TaskStatus.PENDING)
        await _execute_in_background(engine, task, process_def=None)

        engine.create_task_orchestrator.assert_not_called()
        shared_orch.execute_task.assert_awaited_once_with(task)
        engine.database.update_task_status.assert_awaited_once_with(
            "task-2", TaskStatus.COMPLETED.value,
        )


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
