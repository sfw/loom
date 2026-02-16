"""Tests for the MCP server integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from loom.integrations.mcp_server import (
    EXECUTE_TASK_SCHEMA,
    LIST_TASKS_SCHEMA,
    TASK_STATUS_SCHEMA,
    LoomMCPServer,
)


class TestMCPToolSchemas:
    def test_execute_task_schema_has_required_fields(self):
        assert "goal" in EXECUTE_TASK_SCHEMA["properties"]
        assert "required" in EXECUTE_TASK_SCHEMA
        assert "goal" in EXECUTE_TASK_SCHEMA["required"]

    def test_task_status_schema_requires_task_id(self):
        assert "task_id" in TASK_STATUS_SCHEMA["properties"]
        assert "task_id" in TASK_STATUS_SCHEMA["required"]

    def test_list_tasks_schema(self):
        assert "status_filter" in LIST_TASKS_SCHEMA["properties"]


class TestMCPServer:
    def test_list_tools(self):
        server = LoomMCPServer()
        tools = server.list_tools()

        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert "loom_execute_task" in names
        assert "loom_task_status" in names
        assert "loom_list_tasks" in names

    def test_list_tools_have_descriptions(self):
        server = LoomMCPServer()
        for tool in server.list_tools():
            assert "description" in tool
            assert len(tool["description"]) > 10

    def test_list_tools_have_input_schemas(self):
        server = LoomMCPServer()
        for tool in server.list_tools():
            assert "inputSchema" in tool
            assert "properties" in tool["inputSchema"]

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        server = LoomMCPServer()
        result = await server.call_tool("nonexistent", {})

        assert len(result) == 1
        data = json.loads(result[0]["text"])
        assert "error" in data

    @pytest.mark.asyncio
    async def test_execute_task_no_wait(self):
        server = LoomMCPServer()

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"task_id": "t1", "status": "pending"}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await server.call_tool("loom_execute_task", {
                "goal": "Test task",
                "wait": False,
            })

        data = json.loads(result[0]["text"])
        assert data["task_id"] == "t1"

    @pytest.mark.asyncio
    async def test_task_status(self):
        server = LoomMCPServer()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "t1",
            "status": "completed",
            "goal": "Test",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await server.call_tool("loom_task_status", {"task_id": "t1"})

        data = json.loads(result[0]["text"])
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_task_status_not_found(self):
        server = LoomMCPServer()

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await server.call_tool("loom_task_status", {"task_id": "missing"})

        data = json.loads(result[0]["text"])
        assert "error" in data

    @pytest.mark.asyncio
    async def test_list_tasks(self):
        server = LoomMCPServer()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"task_id": "t1", "status": "completed"},
            {"task_id": "t2", "status": "running"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await server.call_tool("loom_list_tasks", {})

        data = json.loads(result[0]["text"])
        assert data["count"] == 2

    @pytest.mark.asyncio
    async def test_list_tasks_with_filter(self):
        server = LoomMCPServer()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"task_id": "t1", "status": "completed"},
            {"task_id": "t2", "status": "running"},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await server.call_tool("loom_list_tasks", {
                "status_filter": "completed",
            })

        data = json.loads(result[0]["text"])
        assert data["count"] == 1
        assert data["tasks"][0]["status"] == "completed"
