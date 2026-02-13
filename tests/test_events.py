"""Tests for the event bus and event types."""

from __future__ import annotations

import asyncio

from loom.events.bus import Event, EventBus
from loom.events.types import (
    SUBTASK_COMPLETED,
    SUBTASK_STARTED,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLANNING,
)


class TestEvent:
    def test_auto_timestamp(self):
        event = Event(event_type="test", task_id="t1")
        assert event.timestamp != ""

    def test_explicit_timestamp(self):
        event = Event(event_type="test", task_id="t1", timestamp="2025-01-01T00:00:00")
        assert event.timestamp == "2025-01-01T00:00:00"

    def test_default_data(self):
        event = Event(event_type="test", task_id="t1")
        assert event.data == {}

    def test_with_data(self):
        event = Event(event_type="test", task_id="t1", data={"key": "value"})
        assert event.data["key"] == "value"


class TestEventBus:
    def test_subscribe_and_emit_sync(self):
        bus = EventBus()
        received = []

        def handler(event: Event):
            received.append(event)

        bus.subscribe("task_completed", handler)
        bus.emit(Event(event_type="task_completed", task_id="t1"))

        assert len(received) == 1
        assert received[0].task_id == "t1"

    def test_subscribe_does_not_receive_other_types(self):
        bus = EventBus()
        received = []

        def handler(event: Event):
            received.append(event)

        bus.subscribe("task_completed", handler)
        bus.emit(Event(event_type="task_failed", task_id="t1"))

        assert len(received) == 0

    def test_subscribe_all_receives_everything(self):
        bus = EventBus()
        received = []

        def handler(event: Event):
            received.append(event)

        bus.subscribe_all(handler)
        bus.emit(Event(event_type="task_completed", task_id="t1"))
        bus.emit(Event(event_type="task_failed", task_id="t2"))

        assert len(received) == 2
        assert received[0].event_type == "task_completed"
        assert received[1].event_type == "task_failed"

    def test_multiple_handlers(self):
        bus = EventBus()
        r1 = []
        r2 = []

        bus.subscribe("test", lambda e: r1.append(e))
        bus.subscribe("test", lambda e: r2.append(e))

        bus.emit(Event(event_type="test", task_id="t1"))

        assert len(r1) == 1
        assert len(r2) == 1

    def test_recent_events(self):
        bus = EventBus()
        for i in range(5):
            bus.emit(Event(event_type="test", task_id=f"t{i}"))

        recent = bus.recent_events(limit=3)
        assert len(recent) == 3
        assert recent[0].task_id == "t2"
        assert recent[2].task_id == "t4"

    def test_recent_events_default_limit(self):
        bus = EventBus()
        for i in range(10):
            bus.emit(Event(event_type="test", task_id=f"t{i}"))

        recent = bus.recent_events()
        assert len(recent) == 10

    def test_history_cap(self):
        bus = EventBus()
        bus._max_history = 5
        for i in range(10):
            bus.emit(Event(event_type="test", task_id=f"t{i}"))

        assert len(bus._history) == 5
        assert bus._history[0].task_id == "t5"

    def test_clear(self):
        bus = EventBus()
        bus.subscribe("test", lambda e: None)
        bus.subscribe_all(lambda e: None)
        bus.emit(Event(event_type="test", task_id="t1"))

        bus.clear()
        assert len(bus._handlers) == 0
        assert len(bus._global_handlers) == 0
        assert len(bus._history) == 0

    def test_handler_error_does_not_break_emit(self):
        bus = EventBus()
        received = []

        def bad_handler(event: Event):
            raise RuntimeError("boom")

        def good_handler(event: Event):
            received.append(event)

        bus.subscribe("test", bad_handler)
        bus.subscribe("test", good_handler)

        bus.emit(Event(event_type="test", task_id="t1"))
        assert len(received) == 1

    def test_event_type_constants(self):
        """Verify event type string values match expected names."""
        assert TASK_PLANNING == "task_planning"
        assert TASK_EXECUTING == "task_executing"
        assert TASK_COMPLETED == "task_completed"
        assert TASK_FAILED == "task_failed"
        assert SUBTASK_STARTED == "subtask_started"
        assert SUBTASK_COMPLETED == "subtask_completed"

    def test_async_handler_called(self):
        """Test that async handlers are invoked when an event loop is running."""
        bus = EventBus()
        received = []

        async def async_handler(event: Event):
            received.append(event)

        bus.subscribe("test", async_handler)

        async def run():
            bus.emit(Event(event_type="test", task_id="t1"))
            # Give the fire-and-forget task a chance to complete
            await asyncio.sleep(0.05)

        asyncio.run(run())
        assert len(received) == 1

    def test_mixed_sync_and_async_handlers(self):
        bus = EventBus()
        sync_received = []
        async_received = []

        def sync_handler(event: Event):
            sync_received.append(event)

        async def async_handler(event: Event):
            async_received.append(event)

        bus.subscribe("test", sync_handler)
        bus.subscribe("test", async_handler)

        async def run():
            bus.emit(Event(event_type="test", task_id="t1"))
            await asyncio.sleep(0.05)

        asyncio.run(run())
        assert len(sync_received) == 1
        assert len(async_received) == 1
