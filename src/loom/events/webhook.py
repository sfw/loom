"""Webhook delivery for task events.

Delivers terminal events (completed, failed, cancelled) to registered
callback URLs with retry and exponential backoff.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any
from urllib.parse import urlsplit

from loom.events.bus import Event, EventBus
from loom.events.types import (
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_FAILED,
    WEBHOOK_DELIVERY_ATTEMPTED,
    WEBHOOK_DELIVERY_DROPPED,
    WEBHOOK_DELIVERY_FAILED,
    WEBHOOK_DELIVERY_SUCCEEDED,
)

logger = logging.getLogger(__name__)

# Events that trigger webhook delivery
TERMINAL_EVENTS = {TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED}


class WebhookDelivery:
    """Delivers events to callback URLs via HTTP POST.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s).
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._callback_urls: dict[str, str] = {}  # task_id -> url
        self._event_bus: EventBus | None = None

    def register(self, task_id: str, callback_url: str) -> None:
        """Register a callback URL for a task.

        Validates the URL to prevent SSRF attacks against internal networks.
        """
        from loom.tools.web import is_safe_url
        safe, reason = is_safe_url(callback_url)
        if not safe:
            logger.warning("Rejected callback URL for task %s: %s", task_id, reason)
            self._emit_delivery_event(
                task_id,
                WEBHOOK_DELIVERY_DROPPED,
                {
                    **self._safe_target_metadata(callback_url),
                    "reason": f"unsafe_url:{reason}",
                    "source_component": "webhook",
                },
            )
            return
        self._callback_urls[task_id] = callback_url

    def unregister(self, task_id: str) -> None:
        """Remove callback URL for a task."""
        self._callback_urls.pop(task_id, None)

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to terminal events on the event bus."""
        self._event_bus = event_bus
        event_bus.subscribe_all(self._handle_event)

    async def _handle_event(self, event: Event) -> None:
        """Check if event should trigger webhook delivery."""
        if event.event_type not in TERMINAL_EVENTS:
            return

        url = self._callback_urls.get(event.task_id)
        if not url:
            self._emit_delivery_event(
                event.task_id,
                WEBHOOK_DELIVERY_DROPPED,
                {
                    "delivery_target_host": "",
                    "delivery_target_hash": "",
                    "reason": "unregistered",
                    "terminal_event_type": event.event_type,
                    "source_component": "webhook",
                },
            )
            return

        await self.deliver(url, event)
        self.unregister(event.task_id)

    async def deliver(self, callback_url: str, event: Event) -> bool:
        """POST event to callback URL with retry.

        Returns True if delivery succeeded, False otherwise.
        """
        target_meta = self._safe_target_metadata(callback_url)
        payload = {
            "task_id": event.task_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "data": event.data,
        }

        last_status_code = 0
        last_error_class = ""
        for attempt in range(self._max_retries):
            attempt_num = attempt + 1
            self._emit_delivery_event(
                event.task_id,
                WEBHOOK_DELIVERY_ATTEMPTED,
                {
                    **target_meta,
                    "attempt": attempt_num,
                    "max_retries": self._max_retries,
                    "terminal_event_type": event.event_type,
                    "source_component": "webhook",
                },
            )
            try:
                success = await self._post(callback_url, payload)
                if success:
                    self._emit_delivery_event(
                        event.task_id,
                        WEBHOOK_DELIVERY_SUCCEEDED,
                        {
                            **target_meta,
                            "attempt": attempt_num,
                            "terminal_event_type": event.event_type,
                            "source_component": "webhook",
                        },
                    )
                    return True
            except Exception as e:
                last_error_class = type(e).__name__
                last_status_code = self._status_code_from_exception(e)
                logger.warning(
                    "Webhook delivery attempt %d failed for %s: %s",
                    attempt_num,
                    target_meta.get("delivery_target_host", "<unknown>"),
                    e,
                )

            if attempt < self._max_retries - 1:
                delay = self._base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

        logger.error(
            "Webhook delivery failed after %d attempts for %s",
            self._max_retries,
            target_meta.get("delivery_target_host", "<unknown>"),
        )
        self._emit_delivery_event(
            event.task_id,
            WEBHOOK_DELIVERY_FAILED,
            {
                **target_meta,
                "attempts": self._max_retries,
                "status_code": last_status_code,
                "error_class": last_error_class,
                "terminal_event_type": event.event_type,
                "source_component": "webhook",
            },
        )
        return False

    @staticmethod
    def _safe_target_metadata(raw_url: str) -> dict[str, str]:
        text = str(raw_url or "").strip()
        if not text:
            return {"delivery_target_host": "", "delivery_target_hash": ""}
        try:
            parsed = urlsplit(text)
        except ValueError:
            return {"delivery_target_host": "", "delivery_target_hash": ""}
        host = str(parsed.hostname or "").strip().lower()
        canonical = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        hashed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return {
            "delivery_target_host": host,
            "delivery_target_hash": hashed,
        }

    @staticmethod
    def _status_code_from_exception(error: Exception) -> int:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        try:
            parsed = int(status)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    def _emit_delivery_event(
        self,
        task_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        if self._event_bus is None:
            return
        self._event_bus.emit(Event(
            event_type=event_type,
            task_id=task_id,
            data=dict(payload or {}),
        ))

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
