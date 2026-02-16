"""Async HTTP client for the Loom REST API.

Used by the TUI and CLI to interact with the running Loom server.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


class LoomAPIClient:
    """Thin async client wrapping the Loom REST API."""

    def __init__(self, base_url: str = "http://localhost:9000"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # --- Task CRUD ---

    async def list_tasks(self) -> list[dict]:
        client = await self._get_client()
        r = await client.get("/tasks")
        r.raise_for_status()
        return r.json()

    async def get_task(self, task_id: str) -> dict:
        client = await self._get_client()
        r = await client.get(f"/tasks/{task_id}")
        r.raise_for_status()
        return r.json()

    async def create_task(
        self,
        goal: str,
        workspace: str | None = None,
        approval_mode: str = "auto",
        callback_url: str | None = None,
    ) -> dict:
        client = await self._get_client()
        payload: dict = {"goal": goal, "approval_mode": approval_mode}
        if workspace:
            payload["workspace"] = workspace
        if callback_url:
            payload["callback_url"] = callback_url
        r = await client.post("/tasks", json=payload)
        r.raise_for_status()
        return r.json()

    async def cancel_task(self, task_id: str) -> dict:
        client = await self._get_client()
        r = await client.delete(f"/tasks/{task_id}")
        r.raise_for_status()
        return r.json()

    async def steer(self, task_id: str, instruction: str) -> dict:
        client = await self._get_client()
        r = await client.patch(
            f"/tasks/{task_id}", json={"instruction": instruction}
        )
        r.raise_for_status()
        return r.json()

    async def approve(
        self, task_id: str, subtask_id: str, approved: bool = True
    ) -> dict:
        client = await self._get_client()
        r = await client.post(
            f"/tasks/{task_id}/approve",
            json={"subtask_id": subtask_id, "approved": approved},
        )
        r.raise_for_status()
        return r.json()

    async def get_subtasks(self, task_id: str) -> list[dict]:
        client = await self._get_client()
        r = await client.get(f"/tasks/{task_id}/subtasks")
        r.raise_for_status()
        return r.json()

    async def health(self) -> dict:
        client = await self._get_client()
        r = await client.get("/health")
        r.raise_for_status()
        return r.json()

    # --- SSE Streaming ---

    async def stream_task_events(self, task_id: str) -> AsyncIterator[dict]:
        """Subscribe to task-specific SSE stream."""
        client = await self._get_client()
        async with client.stream("GET", f"/tasks/{task_id}/stream") as response:
            async for line in response.aiter_lines():
                if not line.strip() or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    try:
                        yield json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

    # --- Memory & Feedback ---

    async def get_memory(self, task_id: str) -> list[dict]:
        """Fetch memory entries for a task."""
        client = await self._get_client()
        r = await client.get(f"/tasks/{task_id}/memory")
        r.raise_for_status()
        return r.json()

    async def submit_feedback(self, task_id: str, feedback: str) -> dict:
        """Submit feedback for a task."""
        client = await self._get_client()
        r = await client.post(
            f"/tasks/{task_id}/feedback",
            json={"feedback": feedback},
        )
        r.raise_for_status()
        return r.json()

    # --- Conversation ---

    async def get_conversation_history(self, task_id: str) -> list[dict]:
        """Fetch conversation history for a task."""
        client = await self._get_client()
        r = await client.get(f"/tasks/{task_id}/conversation")
        r.raise_for_status()
        return r.json()

    async def send_message(self, task_id: str, message: str) -> dict:
        """Send a conversation message to a running task."""
        client = await self._get_client()
        r = await client.post(
            f"/tasks/{task_id}/message",
            json={"message": message},
        )
        r.raise_for_status()
        return r.json()

