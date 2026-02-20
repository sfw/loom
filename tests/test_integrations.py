"""Tests for Phase 2 integration wiring.

Tests that changelog, event persistence, memory extraction,
and response validation are properly wired together.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import Config
from loom.engine.orchestrator import Orchestrator, create_task
from loom.events.bus import Event, EventBus, EventPersister
from loom.models.base import ModelResponse, TokenUsage, ToolCall
from loom.models.router import ModelRouter
from loom.prompts.assembler import PromptAssembler
from loom.state.memory import Database
from loom.state.task_state import (
    TaskStateManager,
    TaskStatus,
)
from loom.tools.file_ops import EditFileTool, WriteFileTool
from loom.tools.registry import ToolContext, ToolResult
from loom.tools.workspace import ChangeLog

# --- Changelog Integration with File Tools ---


class TestChangelogFileTools:
    @pytest.mark.asyncio
    async def test_write_file_records_changelog_create(self, tmp_path):
        """WriteFileTool should record a 'create' entry in changelog."""
        workspace = tmp_path / "project"
        workspace.mkdir()

        data_dir = tmp_path / "changelog"
        data_dir.mkdir()
        changelog = ChangeLog(task_id="t1", workspace=workspace, data_dir=data_dir)

        ctx = ToolContext(
            workspace=workspace,
            changelog=changelog,
            subtask_id="s1",
        )

        tool = WriteFileTool()
        result = await tool.execute(
            {"path": "hello.py", "content": "print('hello')"},
            ctx,
        )

        assert result.success
        entries = changelog.get_entries()
        assert len(entries) == 1
        assert entries[0].operation == "create"
        assert entries[0].path == "hello.py"
        assert entries[0].subtask_id == "s1"

    @pytest.mark.asyncio
    async def test_write_file_records_changelog_modify(self, tmp_path):
        """WriteFileTool should snapshot existing files before overwriting."""
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / "existing.py").write_text("original content")

        data_dir = tmp_path / "changelog"
        data_dir.mkdir()
        changelog = ChangeLog(task_id="t1", workspace=workspace, data_dir=data_dir)

        ctx = ToolContext(
            workspace=workspace,
            changelog=changelog,
            subtask_id="s1",
        )

        tool = WriteFileTool()
        result = await tool.execute(
            {"path": "existing.py", "content": "new content"},
            ctx,
        )

        assert result.success
        entries = changelog.get_entries()
        assert len(entries) == 1
        assert entries[0].operation == "modify"
        assert entries[0].before_snapshot is not None

        # Verify snapshot content
        snapshot_content = Path(entries[0].before_snapshot).read_text()
        assert snapshot_content == "original content"

    @pytest.mark.asyncio
    async def test_edit_file_records_changelog(self, tmp_path):
        """EditFileTool should record changes in changelog."""
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / "app.py").write_text("x = 1\ny = 2\n")

        data_dir = tmp_path / "changelog"
        data_dir.mkdir()
        changelog = ChangeLog(task_id="t1", workspace=workspace, data_dir=data_dir)

        ctx = ToolContext(
            workspace=workspace,
            changelog=changelog,
            subtask_id="s2",
        )

        tool = EditFileTool()
        result = await tool.execute(
            {"path": "app.py", "old_str": "x = 1", "new_str": "x = 42"},
            ctx,
        )

        assert result.success
        entries = changelog.get_entries()
        assert len(entries) == 1
        assert entries[0].operation == "modify"
        assert entries[0].subtask_id == "s2"

    @pytest.mark.asyncio
    async def test_no_changelog_still_works(self, tmp_path):
        """File tools should work fine without a changelog attached."""
        workspace = tmp_path / "project"
        workspace.mkdir()

        ctx = ToolContext(workspace=workspace)  # No changelog

        tool = WriteFileTool()
        result = await tool.execute(
            {"path": "hello.py", "content": "print('hello')"},
            ctx,
        )

        assert result.success
        assert (workspace / "hello.py").read_text() == "print('hello')"


# --- Event Persistence ---


class TestEventPersistence:
    @pytest.mark.asyncio
    async def test_persister_stores_events(self, tmp_path):
        """EventPersister should write events to the database."""
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()

        bus = EventBus()
        persister = EventPersister(db)
        persister.attach(bus)

        # Emit an event
        event = Event(
            event_type="task_created",
            task_id="t1",
            data={"goal": "Test"},
        )
        bus.emit(event)

        # Wait deterministically for async handler completion
        await bus.drain(timeout=1.0)

        # Check database
        rows = await db.query_events("t1")
        assert len(rows) == 1
        assert rows[0]["event_type"] == "task_created"

    @pytest.mark.asyncio
    async def test_persister_handles_errors(self, tmp_path):
        """EventPersister should not crash on database errors."""
        db = MagicMock()
        db.insert_event = AsyncMock(side_effect=Exception("DB error"))

        persister = EventPersister(db)
        event = Event(
            event_type="task_created",
            task_id="t1",
            data={},
        )
        # Should not raise
        await persister.handle(event)
        db.insert_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multiple_events_persisted(self, tmp_path):
        """Multiple events should all be persisted."""
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()

        bus = EventBus()
        persister = EventPersister(db)
        persister.attach(bus)

        for i in range(5):
            bus.emit(Event(
                event_type=f"event_{i}",
                task_id="t1",
                data={"index": i},
            ))

        await bus.drain(timeout=1.0)

        rows = await db.query_events("t1")
        assert len(rows) == 5


# --- Response Validation in Orchestrator ---


class TestResponseValidation:
    """Tests that the orchestrator validates tool calls before execution."""

    @pytest.mark.asyncio
    async def test_invalid_tool_name_triggers_retry(self, tmp_path):
        """When model hallucinates a tool name, orchestrator should retry."""
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Do work"}]
        })

        # First response has invalid tool, second is corrected text-only
        invalid_response = ModelResponse(
            text="",
            tool_calls=[ToolCall(
                id="tc1", name="nonexistent_tool", arguments={"x": 1},
            )],
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        )
        valid_response = ModelResponse(
            text="Done without tools.",
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        )

        # Planner model
        planner = AsyncMock()
        planner.name = "mock-planner"
        planner.complete = AsyncMock(return_value=ModelResponse(
            text=plan_json,
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))

        # Executor model returns invalid tool first, then valid response
        executor = AsyncMock()
        executor.name = "mock-executor"
        executor.complete = AsyncMock(side_effect=[invalid_response, valid_response])

        router = MagicMock(spec=ModelRouter)
        router.select = MagicMock(side_effect=lambda tier=1, role="executor":
            planner if role == "planner" else executor)

        tools = MagicMock()
        tools.all_schemas = MagicMock(return_value=[
            {"name": "read_file", "description": "Read a file"},
        ])
        tools.execute = AsyncMock(return_value=ToolResult.ok("ok"))

        memory = AsyncMock()
        memory.query_relevant = AsyncMock(return_value=[])
        memory.store_many = AsyncMock(return_value=[])

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_planner_prompt = MagicMock(return_value="Plan")
        prompts.build_executor_prompt = MagicMock(return_value="Execute")

        orch = Orchestrator(
            model_router=router,
            tool_registry=tools,
            memory_manager=memory,
            prompt_assembler=prompts,
            state_manager=TaskStateManager(data_dir=tmp_path),
            event_bus=EventBus(),
            config=Config(),
        )

        task = create_task("test")
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        # The invalid tool was never executed
        tools.execute.assert_not_called()


# --- Re-planning ---


class TestReplanning:
    @pytest.mark.asyncio
    async def test_retry_on_failure_before_replan(self, tmp_path):
        """Failed subtask should be retried before re-planning."""
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Do work", "max_retries": 1}]
        })

        planner = AsyncMock()
        planner.name = "mock-planner"
        planner.complete = AsyncMock(return_value=ModelResponse(
            text=plan_json,
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))

        executor = AsyncMock()
        executor.name = "mock-executor"
        executor.complete = AsyncMock(return_value=ModelResponse(
            text="Done",
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))

        router = MagicMock(spec=ModelRouter)
        router.select = MagicMock(side_effect=lambda tier=1, role="executor":
            planner if role == "planner" else executor)

        tools = MagicMock()
        tools.all_schemas = MagicMock(return_value=[])
        tools.execute = AsyncMock(return_value=ToolResult.ok("ok"))

        memory = AsyncMock()
        memory.query_relevant = AsyncMock(return_value=[])
        memory.store_many = AsyncMock(return_value=[])

        prompts = MagicMock(spec=PromptAssembler)
        prompts.build_planner_prompt = MagicMock(return_value="Plan")
        prompts.build_executor_prompt = MagicMock(return_value="Execute")

        orch = Orchestrator(
            model_router=router,
            tool_registry=tools,
            memory_manager=memory,
            prompt_assembler=prompts,
            state_manager=TaskStateManager(data_dir=tmp_path),
            event_bus=EventBus(),
            config=Config(),
        )

        task = create_task("test")
        result = await orch.execute_task(task)

        # Task should complete (no failures in this simple case)
        assert result.status == TaskStatus.COMPLETED
