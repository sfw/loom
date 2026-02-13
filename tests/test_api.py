"""Tests for the Loom API server."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from loom.api.engine import Engine
from loom.config import Config
from loom.engine.orchestrator import Orchestrator
from loom.events.bus import EventBus
from loom.models.router import ModelRouter
from loom.prompts.assembler import PromptAssembler
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
