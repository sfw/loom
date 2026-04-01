"""Tests for the event bus and event types."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from loom.events.bus import Event, EventBus, EventPersister
from loom.events.contracts import validate_payload_shape
from loom.events.types import (
    ACTIVE_EVENT_TYPES,
    ASK_USER_ANSWERED,
    ASK_USER_REQUESTED,
    EVENT_LIFECYCLE,
    EVENT_NAME_TO_TYPE,
    SUBTASK_BLOCKED,
    SUBTASK_COMPLETED,
    SUBTASK_POLICY_RECONCILED,
    SUBTASK_STARTED,
    TASK_COMPLETED,
    TASK_CREATED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLANNING,
    TASK_RUN_HEARTBEAT,
    TELEMETRY_DIAGNOSTIC,
    TOKEN_STREAMED,
    VERIFICATION_PASSED,
)
from loom.events.verbosity import (
    normalize_telemetry_mode,
    should_deliver_operator,
    should_persist_compliance,
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
            bus.emit(Event(event_type=TASK_COMPLETED, task_id=f"t{i}"))

        recent = bus.recent_events(limit=3)
        assert len(recent) == 3
        assert recent[0].task_id == "t2"
        assert recent[2].task_id == "t4"

    def test_recent_events_default_limit(self):
        bus = EventBus()
        for i in range(10):
            bus.emit(Event(event_type=TASK_COMPLETED, task_id=f"t{i}"))

        recent = bus.recent_events()
        assert len(recent) == 10

    def test_history_cap(self):
        bus = EventBus()
        bus._max_history = 5
        for i in range(10):
            bus.emit(Event(event_type=TASK_COMPLETED, task_id=f"t{i}"))

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
        assert ASK_USER_REQUESTED == "ask_user_requested"
        assert ASK_USER_ANSWERED == "ask_user_answered"

    def test_catalog_lifecycle_covers_all_declared_events(self):
        assert set(EVENT_NAME_TO_TYPE.values()) == set(EVENT_LIFECYCLE.keys())
        assert TASK_CREATED in ACTIVE_EVENT_TYPES

    def test_normalize_telemetry_mode_alias(self):
        resolved = normalize_telemetry_mode("internal_only")
        assert resolved.mode == "all_typed"
        assert resolved.warning_code == "telemetry_mode_alias_normalized"

    def test_should_deliver_operator_preserves_passthrough_in_off_mode(self):
        assert should_deliver_operator(TASK_RUN_HEARTBEAT, "off") is True
        assert should_deliver_operator(TOKEN_STREAMED, "off") is False
        assert should_deliver_operator(TOKEN_STREAMED, "all_typed") is True
        assert should_deliver_operator(TELEMETRY_DIAGNOSTIC, "all_typed") is False
        assert should_deliver_operator(TELEMETRY_DIAGNOSTIC, "debug") is True

    def test_should_persist_compliance_keeps_unknown_events(self):
        assert should_persist_compliance("mystery_event") is True

    def test_emit_enriches_payload_with_audit_fields(self):
        bus = EventBus()
        bus.emit(Event(event_type=TASK_CREATED, task_id="task-1", data={"goal": "ship"}))
        event = bus.recent_events(limit=1)[0]
        payload = event.data
        assert payload["task_id"] == "task-1"
        assert payload["timestamp"] == event.timestamp
        assert payload["schema_version"] == 1
        assert payload["source_component"] == "unknown"
        assert str(payload.get("event_id", "")).strip()
        assert str(payload.get("correlation_id", "")).startswith("task:")
        assert int(payload.get("sequence", 0)) == 1

    def test_emit_redacts_sensitive_fields(self):
        bus = EventBus()
        bus.emit(Event(
            event_type=TASK_CREATED,
            task_id="task-2",
            data={
                "goal": "ship",
                "auth_token": "secret-token",
                "nested": {"api_key": "xyz"},
            },
        ))
        event = bus.recent_events(limit=1)[0]
        assert event.data["auth_token"] == "[REDACTED]"
        nested = event.data.get("nested", {})
        assert isinstance(nested, dict)
        assert nested.get("api_key") == "[REDACTED]"

    def test_emit_assigns_monotonic_task_sequence(self):
        bus = EventBus()
        bus.emit(Event(event_type=TASK_CREATED, task_id="task-seq", data={"goal": "a"}))
        bus.emit(Event(event_type=TASK_CREATED, task_id="task-seq", data={"goal": "b"}))
        events = bus.recent_events(limit=2)
        assert int(events[0].data.get("sequence", 0)) == 1
        assert int(events[1].data.get("sequence", 0)) == 2

    def test_async_handler_called(self):
        """Test that async handlers are invoked when an event loop is running."""
        bus = EventBus()
        received = []

        async def async_handler(event: Event):
            received.append(event)

        bus.subscribe("test", async_handler)

        async def run():
            bus.emit(Event(event_type="test", task_id="t1"))
            await bus.drain(timeout=1.0)

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
            await bus.drain(timeout=1.0)

        asyncio.run(run())
        assert len(sync_received) == 1
        assert len(async_received) == 1

    def test_async_callable_object_handler(self):
        """A callable with async __call__ should be dispatched as coroutine."""
        bus = EventBus()
        received = []

        class AsyncCallable:
            async def __call__(self, event: Event):
                received.append(event)

        handler = AsyncCallable()
        bus.subscribe("test", handler)

        async def run():
            bus.emit(Event(event_type="test", task_id="t1"))
            await bus.drain(timeout=1.0)

        asyncio.run(run())
        assert len(received) == 1

    def test_drain_waits_for_async_handlers(self):
        bus = EventBus()
        received = []

        async def async_handler(event: Event):
            await asyncio.sleep(0.01)
            received.append(event.task_id)

        bus.subscribe("test", async_handler)

        async def run():
            for i in range(5):
                bus.emit(Event(event_type="test", task_id=f"t{i}"))
            await bus.drain(timeout=1.0)

        asyncio.run(run())
        assert received == ["t0", "t1", "t2", "t3", "t4"]

    def test_event_persister_preserves_emitted_timestamp_and_metadata(self):
        db = AsyncMock()
        persister = EventPersister(db)
        event = Event(
            event_type=TASK_CREATED,
            task_id="task-persist",
            timestamp="2026-03-06T00:00:00+00:00",
            data={
                "goal": "persist",
                "correlation_id": "run:abc123",
                "run_id": "abc123",
                "event_id": "evt-1",
                "sequence": 7,
                "source_component": "api",
                "schema_version": 1,
            },
        )

        async def run():
            await persister.handle(event)

        asyncio.run(run())
        db.insert_event.assert_awaited_once()
        kwargs = db.insert_event.await_args.kwargs
        assert kwargs["correlation_id"] == "run:abc123"
        assert kwargs["timestamp"] == "2026-03-06T00:00:00+00:00"
        assert kwargs["run_id"] == "abc123"
        assert kwargs["event_id"] == "evt-1"
        assert kwargs["sequence"] == 7

    def test_payload_contract_validation_reports_missing_required_keys(self):
        errors = validate_payload_shape(TASK_CREATED, {"task_id": "task-1"})
        assert "missing required key: goal" in errors

        errors = validate_payload_shape(SUBTASK_BLOCKED, {"subtask_id": "s1"})
        assert "missing required key: reasons" in errors

        errors = validate_payload_shape(
            SUBTASK_POLICY_RECONCILED,
            {"reconciled_subtasks": [], "reconciled_count": 0},
        )
        assert errors == []

        errors = validate_payload_shape(
            VERIFICATION_PASSED,
            {"subtask_id": "s1", "tier": 1, "outcome": "pass", "reason_code": "verified"},
        )
        assert errors == []

    def test_unknown_event_type_emits_rate_limited_diagnostic(self):
        bus = EventBus()
        bus.emit(Event(event_type="unknown_event", task_id="task-1", data={"x": 1}))
        history_types = [evt.event_type for evt in bus.recent_events(limit=4)]
        assert "unknown_event" in history_types
        assert TELEMETRY_DIAGNOSTIC in history_types
        diagnostic = [
            evt for evt in bus.recent_events(limit=4)
            if evt.event_type == TELEMETRY_DIAGNOSTIC
        ][-1]
        assert diagnostic.data.get("diagnostic_type") == "unknown_event_type"

    def test_payload_contract_violation_emits_diagnostic(self):
        bus = EventBus()
        bus.emit(Event(event_type=TASK_CREATED, task_id="task-2", data={}))
        diagnostic = [
            evt for evt in bus.recent_events(limit=6)
            if evt.event_type == TELEMETRY_DIAGNOSTIC
        ][-1]
        assert diagnostic.data.get("diagnostic_type") == "payload_contract_violation"
        assert "missing required key: goal" in str(diagnostic.data.get("errors", ""))

    def test_diagnostic_rate_limit_suppresses_excess_events(self):
        bus = EventBus(
            debug_diagnostics_rate_per_minute=1,
            debug_diagnostics_burst=1,
        )
        for idx in range(3):
            bus.emit(Event(event_type=f"unknown_event_{idx}", task_id="task-rate"))
        diagnostics = [
            evt
            for evt in bus.recent_events(limit=12)
            if evt.event_type == TELEMETRY_DIAGNOSTIC
        ]
        assert len(diagnostics) == 1
        assert diagnostics[0].data.get("diagnostic_type") == "unknown_event_type"

    def test_telemetry_diagnostic_event_does_not_recurse(self):
        bus = EventBus()
        bus.emit(
            Event(
                event_type=TELEMETRY_DIAGNOSTIC,
                task_id="task-diag",
                data={"diagnostic_type": "manual"},
            ),
        )
        diagnostics = [
            evt
            for evt in bus.recent_events(limit=4)
            if evt.event_type == TELEMETRY_DIAGNOSTIC
        ]
        assert len(diagnostics) == 1
        assert diagnostics[0].data.get("diagnostic_type") == "manual"

    def test_event_persister_tracks_failed_persistence_attempts(self):
        class FailingDB:
            async def insert_event(self, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("db unavailable")

        persister = EventPersister(FailingDB())
        event = Event(event_type=TASK_CREATED, task_id="task-fail", data={"goal": "x"})

        async def run():
            await persister.handle(event)
            await persister.handle(event)

        asyncio.run(run())
        assert persister._failed_persist_count == 2

    def test_event_persister_emits_persistence_failure_diagnostic(self):
        class FailingDB:
            async def insert_event(self, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("db unavailable")

        bus = EventBus()
        persister = EventPersister(FailingDB())
        persister.attach(bus)
        event = Event(event_type=TASK_CREATED, task_id="task-fail", data={"goal": "x"})

        async def run():
            await persister.handle(event)

        asyncio.run(run())
        diagnostics = [
            evt
            for evt in bus.recent_events(limit=4)
            if evt.event_type == TELEMETRY_DIAGNOSTIC
        ]
        assert diagnostics
        assert diagnostics[-1].data.get("diagnostic_type") == "persistence_failure_count_snapshot"
