"""Tests for the task tracker tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.registry import ToolContext
from loom.tools.task_tracker import TaskTrackerTool


@pytest.fixture
def tool():
    return TaskTrackerTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


class TestTaskTrackerAdd:
    async def test_add_task(self, tool, ctx):
        result = await tool.execute({"action": "add", "content": "Fix bug #42"}, ctx)
        assert result.success
        assert "#1" in result.output
        assert "Fix bug #42" in result.output

    async def test_add_increments_id(self, tool, ctx):
        await tool.execute({"action": "add", "content": "Task 1"}, ctx)
        result = await tool.execute({"action": "add", "content": "Task 2"}, ctx)
        assert "#2" in result.output

    async def test_add_empty_fails(self, tool, ctx):
        result = await tool.execute({"action": "add", "content": ""}, ctx)
        assert not result.success


class TestTaskTrackerUpdate:
    async def test_update_status(self, tool, ctx):
        await tool.execute({"action": "add", "content": "Task 1"}, ctx)
        result = await tool.execute(
            {"action": "update", "task_id": 1, "status": "completed"}, ctx
        )
        assert result.success
        assert "completed" in result.output

    async def test_update_nonexistent(self, tool, ctx):
        result = await tool.execute(
            {"action": "update", "task_id": 99, "status": "completed"}, ctx
        )
        assert not result.success

    async def test_update_invalid_status(self, tool, ctx):
        await tool.execute({"action": "add", "content": "Task 1"}, ctx)
        result = await tool.execute(
            {"action": "update", "task_id": 1, "status": "invalid"}, ctx
        )
        assert not result.success

    async def test_update_no_task_id(self, tool, ctx):
        result = await tool.execute({"action": "update", "status": "completed"}, ctx)
        assert not result.success


class TestTaskTrackerList:
    async def test_list_empty(self, tool, ctx):
        result = await tool.execute({"action": "list"}, ctx)
        assert result.success
        assert "No tasks" in result.output

    async def test_list_with_tasks(self, tool, ctx):
        await tool.execute({"action": "add", "content": "First"}, ctx)
        await tool.execute({"action": "add", "content": "Second"}, ctx)
        await tool.execute(
            {"action": "update", "task_id": 1, "status": "completed"}, ctx
        )

        result = await tool.execute({"action": "list"}, ctx)
        assert result.success
        assert "1/2 done" in result.output
        assert "[x]" in result.output  # completed
        assert "[ ]" in result.output  # pending

    async def test_list_data_includes_counts(self, tool, ctx):
        await tool.execute({"action": "add", "content": "A"}, ctx)
        await tool.execute({"action": "add", "content": "B"}, ctx)
        await tool.execute(
            {"action": "update", "task_id": 1, "status": "in_progress"}, ctx
        )

        result = await tool.execute({"action": "list"}, ctx)
        assert result.data["counts"]["in_progress"] == 1
        assert result.data["counts"]["pending"] == 1


class TestTaskTrackerClear:
    async def test_clear(self, tool, ctx):
        await tool.execute({"action": "add", "content": "A"}, ctx)
        await tool.execute({"action": "add", "content": "B"}, ctx)
        result = await tool.execute({"action": "clear"}, ctx)
        assert result.success
        assert "2" in result.output

        # List should be empty after clear
        list_result = await tool.execute({"action": "list"}, ctx)
        assert "No tasks" in list_result.output


class TestTaskTrackerSchema:
    def test_schema(self, tool):
        schema = tool.schema()
        assert schema["name"] == "task_tracker"
        assert "action" in schema["parameters"]["properties"]
        assert "action" in schema["parameters"]["required"]


class TestUnknownAction:
    async def test_unknown(self, tool, ctx):
        result = await tool.execute({"action": "explode"}, ctx)
        assert not result.success
