"""Webhook delivery for task events.

Delivers terminal events (completed, failed, cancelled) to registered
callback URLs with retry and exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from loom.events.bus import Event, EventBus

logger = logging.getLogger(__name__)

# Events that trigger webhook delivery
TERMINAL_EVENTS = {"task_completed", "task_failed", "task_cancelled"}


class WebhookDelivery:
    """Delivers events to callback URLs via HTTP POST.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s).
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._callback_urls: dict[str, str] = {}  # task_id -> url

    def register(self, task_id: str, callback_url: str) -> None:
        """Register a callback URL for a task."""
        self._callback_urls[task_id] = callback_url

    def unregister(self, task_id: str) -> None:
        """Remove callback URL for a task."""
        self._callback_urls.pop(task_id, None)

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to terminal events on the event bus."""
        event_bus.subscribe_all(self._handle_event)

    async def _handle_event(self, event: Event) -> None:
        """Check if event should trigger webhook delivery."""
        if event.event_type not in TERMINAL_EVENTS:
            return

        url = self._callback_urls.get(event.task_id)
        if not url:
            return

        await self.deliver(url, event)
        self.unregister(event.task_id)

    async def deliver(self, callback_url: str, event: Event) -> bool:
        """POST event to callback URL with retry.

        Returns True if delivery succeeded, False otherwise.
        """
        payload = {
            "task_id": event.task_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "data": event.data,
        }

        for attempt in range(self._max_retries):
            try:
                success = await self._post(callback_url, payload)
                if success:
                    return True
            except Exception as e:
                logger.warning(
                    "Webhook delivery attempt %d failed for %s: %s",
                    attempt + 1, callback_url, e,
                )

            if attempt < self._max_retries - 1:
                delay = self._base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

        logger.error(
            "Webhook delivery failed after %d attempts for %s",
            self._max_retries, callback_url,
        )
        return False

    @staticmethod
    async def _post(url: str, payload: dict[str, Any]) -> bool:
        """HTTP POST to webhook URL.

        Uses httpx if available, falls back to aiohttp, or urllib.
        """
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=10.0)
                response.raise_for_status()
                return True
        except ImportError:
            pass

        # Fallback: use urllib in a thread (no external dependency required)
        import json
        import urllib.request

        def _sync_post():
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return 200 <= resp.status < 300

        return await asyncio.to_thread(_sync_post)
