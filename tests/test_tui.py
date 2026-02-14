"""Tests for the TUI API client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.tui.api_client import LoomAPIClient


class TestLoomAPIClient:
    def test_init_default_url(self):
        client = LoomAPIClient()
        assert client._base_url == "http://localhost:9000"

    def test_init_custom_url(self):
        client = LoomAPIClient("http://myhost:8080/")
        assert client._base_url == "http://myhost:8080"

    @pytest.mark.asyncio
    async def test_close_when_no_client(self):
        api = LoomAPIClient()
        await api.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_list_tasks(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [{"task_id": "t1", "status": "running"}]
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        api._client = mock_client

        tasks = await api.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t1"
        mock_client.get.assert_called_once_with("/tasks")

    @pytest.mark.asyncio
    async def test_get_task(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"task_id": "t1", "status": "completed"}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        api._client = mock_client

        task = await api.get_task("t1")
        assert task["task_id"] == "t1"

    @pytest.mark.asyncio
    async def test_create_task(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"task_id": "t1", "status": "pending"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        api._client = mock_client

        result = await api.create_task("Build a CLI", workspace="/tmp/proj")
        assert result["task_id"] == "t1"
        mock_client.post.assert_called_once_with("/tasks", json={
            "goal": "Build a CLI",
            "approval_mode": "auto",
            "workspace": "/tmp/proj",
        })

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_client.delete = AsyncMock(return_value=mock_response)
        api._client = mock_client

        result = await api.cancel_task("t1")
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_steer(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_client.patch = AsyncMock(return_value=mock_response)
        api._client = mock_client

        result = await api.steer("t1", "Use TypeScript")
        assert result["status"] == "ok"
        mock_client.patch.assert_called_once_with(
            "/tasks/t1", json={"instruction": "Use TypeScript"}
        )

    @pytest.mark.asyncio
    async def test_approve(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "approved": True}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        api._client = mock_client

        result = await api.approve("t1", "s1", approved=True)
        assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_health(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        api._client = mock_client

        result = await api.health()
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_get_subtasks(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "s1", "status": "completed"},
            {"id": "s2", "status": "pending"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        api._client = mock_client

        result = await api.get_subtasks("t1")
        assert len(result) == 2
