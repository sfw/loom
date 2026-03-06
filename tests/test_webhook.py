"""Tests for webhook delivery."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from loom.events.bus import Event, EventBus
from loom.events.types import (
    WEBHOOK_DELIVERY_ATTEMPTED,
    WEBHOOK_DELIVERY_DROPPED,
    WEBHOOK_DELIVERY_FAILED,
    WEBHOOK_DELIVERY_SUCCEEDED,
)
from loom.events.webhook import TERMINAL_EVENTS, WebhookDelivery


class TestWebhookDelivery:
    def test_register_and_unregister(self):
        wh = WebhookDelivery()
        wh.register("t1", "http://example.com/webhook")
        assert "t1" in wh._callback_urls
        wh.unregister("t1")
        assert "t1" not in wh._callback_urls

    def test_terminal_events_defined(self):
        assert "task_completed" in TERMINAL_EVENTS
        assert "task_failed" in TERMINAL_EVENTS
        assert "task_cancelled" in TERMINAL_EVENTS

    @pytest.mark.asyncio
    async def test_handle_event_ignores_non_terminal(self):
        wh = WebhookDelivery()
        wh.register("t1", "http://example.com/webhook")
        wh.deliver = AsyncMock()

        event = Event(
            event_type="subtask_started",
            task_id="t1",
            data={},
        )
        await wh._handle_event(event)
        wh.deliver.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_event_ignores_unregistered_task(self):
        wh = WebhookDelivery()
        wh.deliver = AsyncMock()

        event = Event(
            event_type="task_completed",
            task_id="t1",
            data={},
        )
        await wh._handle_event(event)
        wh.deliver.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_event_delivers_for_registered_task(self):
        wh = WebhookDelivery()
        wh.register("t1", "http://example.com/webhook")
        wh.deliver = AsyncMock(return_value=True)

        event = Event(
            event_type="task_completed",
            task_id="t1",
            data={"completed": 3, "total": 3},
        )
        await wh._handle_event(event)
        wh.deliver.assert_called_once_with("http://example.com/webhook", event)

    @pytest.mark.asyncio
    async def test_handle_event_unregisters_after_delivery(self):
        wh = WebhookDelivery()
        wh.register("t1", "http://example.com/webhook")
        wh.deliver = AsyncMock(return_value=True)

        event = Event(
            event_type="task_completed",
            task_id="t1",
            data={},
        )
        await wh._handle_event(event)
        assert "t1" not in wh._callback_urls

    @pytest.mark.asyncio
    async def test_deliver_success(self):
        wh = WebhookDelivery()
        wh._post = AsyncMock(return_value=True)

        event = Event(event_type="task_completed", task_id="t1", data={})
        result = await wh.deliver("http://example.com/webhook", event)

        assert result is True
        wh._post.assert_called_once()

    @pytest.mark.asyncio
    async def test_deliver_retries_on_failure(self):
        wh = WebhookDelivery(max_retries=3, base_delay=0.01)
        wh._post = AsyncMock(side_effect=[False, False, True])

        event = Event(event_type="task_completed", task_id="t1", data={})
        result = await wh.deliver("http://example.com/webhook", event)

        assert result is True
        assert wh._post.call_count == 3

    @pytest.mark.asyncio
    async def test_deliver_fails_after_max_retries(self):
        wh = WebhookDelivery(max_retries=2, base_delay=0.01)
        wh._post = AsyncMock(return_value=False)

        event = Event(event_type="task_failed", task_id="t1", data={})
        result = await wh.deliver("http://example.com/webhook", event)

        assert result is False
        assert wh._post.call_count == 2

    @pytest.mark.asyncio
    async def test_deliver_handles_exceptions(self):
        wh = WebhookDelivery(max_retries=2, base_delay=0.01)
        wh._post = AsyncMock(side_effect=Exception("Connection refused"))

        event = Event(event_type="task_failed", task_id="t1", data={})
        result = await wh.deliver("http://example.com/webhook", event)

        assert result is False

    def test_attach_subscribes_to_event_bus(self):
        bus = EventBus()
        wh = WebhookDelivery()
        wh.attach(bus)

        assert len(bus._global_handlers) == 1

    @pytest.mark.asyncio
    async def test_emits_delivery_lifecycle_events(self):
        bus = EventBus()
        wh = WebhookDelivery(max_retries=2, base_delay=0.0)
        wh.attach(bus)
        wh.register("t1", "https://example.com/webhook?token=secret")
        wh._post = AsyncMock(return_value=True)

        events = []
        bus.subscribe_all(lambda event: events.append(event))
        await wh._handle_event(Event(event_type="task_completed", task_id="t1", data={}))

        event_types = [event.event_type for event in events]
        assert WEBHOOK_DELIVERY_ATTEMPTED in event_types
        assert WEBHOOK_DELIVERY_SUCCEEDED in event_types
        attempt = next(event for event in events if event.event_type == WEBHOOK_DELIVERY_ATTEMPTED)
        assert attempt.data.get("delivery_target_host") == "example.com"
        assert str(attempt.data.get("delivery_target_hash", "")).strip()

    @pytest.mark.asyncio
    async def test_emits_failed_delivery_event_after_retries(self):
        bus = EventBus()
        wh = WebhookDelivery(max_retries=2, base_delay=0.0)
        wh.attach(bus)
        wh.register("t2", "https://example.com/callback")
        wh._post = AsyncMock(return_value=False)
        events = []
        bus.subscribe_all(lambda event: events.append(event))

        await wh._handle_event(Event(event_type="task_failed", task_id="t2", data={}))

        event_types = [event.event_type for event in events]
        assert WEBHOOK_DELIVERY_FAILED in event_types
        failed = next(event for event in events if event.event_type == WEBHOOK_DELIVERY_FAILED)
        assert failed.data.get("attempts") == 2
        assert failed.data.get("delivery_target_host") == "example.com"

    def test_register_emits_dropped_event_for_unsafe_url(self):
        bus = EventBus()
        wh = WebhookDelivery()
        wh.attach(bus)
        events = []
        bus.subscribe_all(lambda event: events.append(event))

        wh.register("t3", "http://localhost:8000/callback?secret=1")

        event_types = [event.event_type for event in events]
        assert WEBHOOK_DELIVERY_DROPPED in event_types
