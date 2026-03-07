"""Event bus for Loom.

In-process pub/sub for decoupling components. Events are emitted by the
orchestrator and consumed by the API (SSE), logger, and other listeners.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loom.events.contracts import redact_payload, validate_payload_shape
from loom.events.types import TELEMETRY_DIAGNOSTIC, TelemetryMode, is_known_event_type
from loom.events.verbosity import (
    DEFAULT_TELEMETRY_MODE,
    normalize_telemetry_mode,
    should_persist_compliance,
)

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Base event structure."""

    event_type: str
    task_id: str
    data: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not isinstance(self.data, dict):
            self.data = {}
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


# Callback type: async function that takes an Event
EventHandler = Callable[[Event], Any]


@dataclass
class _DiagnosticBucket:
    tokens: float
    last_refill: float


class EventBus:
    """In-process async event bus.

    Supports:
    - subscribe(event_type, handler) for type-specific listening
    - subscribe_all(handler) for global listening (SSE, logging)
    - emit(event) dispatches to all matching handlers
    """

    def __init__(
        self,
        *,
        debug_diagnostics_rate_per_minute: int = 120,
        debug_diagnostics_burst: int = 30,
    ) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []
        self._history: list[Event] = []
        self._max_history = 1000
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._task_sequences: dict[str, int] = {}
        self._operator_mode_provider: Callable[[], TelemetryMode] | None = None
        self._diagnostic_burst = max(1, int(debug_diagnostics_burst))
        self._diagnostic_rate_per_minute = max(1, int(debug_diagnostics_rate_per_minute))
        self._diagnostic_buckets: dict[str, _DiagnosticBucket] = {}
        self._diagnostic_emit_depth = 0

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to a specific event type."""
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a handler from a specific event type."""
        handlers = self._handlers.get(event_type, [])
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    def unsubscribe_all(self, handler: EventHandler) -> None:
        """Remove a handler from the global handlers list."""
        try:
            self._global_handlers.remove(handler)
        except ValueError:
            pass

    def set_operator_mode_provider(
        self,
        provider: Callable[[], TelemetryMode] | None,
    ) -> None:
        """Set process-local provider used by operator sinks for mode lookups."""
        self._operator_mode_provider = provider

    def operator_mode(self) -> TelemetryMode:
        """Return effective process-local operator telemetry mode."""
        provider = self._operator_mode_provider
        if callable(provider):
            try:
                return normalize_telemetry_mode(
                    provider(),
                    default=DEFAULT_TELEMETRY_MODE,
                ).mode
            except Exception:
                logger.debug("Operator mode provider failed; defaulting to active.", exc_info=True)
        return DEFAULT_TELEMETRY_MODE

    def emit(self, event: Event) -> None:
        """Emit an event to all matching handlers.

        Handlers are called as fire-and-forget tasks on the event loop.
        """
        self._normalize_event_payload(event)
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        all_handlers = list(self._global_handlers) + list(self._handlers.get(event.event_type, []))

        for handler in all_handlers:
            is_async = inspect.iscoroutinefunction(handler) or (
                callable(handler)
                and inspect.iscoroutinefunction(getattr(handler, "__call__", None))
            )
            if is_async:
                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(handler(event))
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._on_task_done)
                except RuntimeError:
                    logger.debug(
                        "Skipped async handler %s: no running event loop",
                        getattr(handler, "__name__", handler),
                    )
            else:
                try:
                    handler(event)
                except Exception as e:
                    logger.warning(
                        "Event handler %s failed for %s: %s",
                        getattr(handler, '__name__', handler), event.event_type, e,
                    )

    def _normalize_event_payload(self, event: Event) -> None:
        payload = redact_payload(event.data if isinstance(event.data, dict) else {})
        if "task_id" not in payload:
            payload["task_id"] = event.task_id
        if "timestamp" not in payload:
            payload["timestamp"] = event.timestamp
        payload["schema_version"] = int(payload.get("schema_version", 1) or 1)
        source_component = str(payload.get("source_component", "") or "").strip()
        payload["source_component"] = source_component or "unknown"
        run_id = str(payload.get("run_id", "") or "").strip()
        correlation_id = str(payload.get("correlation_id", "") or "").strip()
        if not correlation_id:
            correlation_id = f"run:{run_id}" if run_id else f"task:{event.task_id}"
        payload["correlation_id"] = correlation_id
        payload["event_id"] = str(payload.get("event_id", "") or "").strip() or uuid.uuid4().hex

        raw_sequence = payload.get("sequence")
        sequence = 0
        if isinstance(raw_sequence, int):
            sequence = raw_sequence
        elif isinstance(raw_sequence, str) and raw_sequence.strip().isdigit():
            sequence = int(raw_sequence.strip())
        if sequence <= 0:
            sequence = int(self._task_sequences.get(event.task_id, 0)) + 1
        current_sequence = int(self._task_sequences.get(event.task_id, 0))
        self._task_sequences[event.task_id] = max(sequence, current_sequence)
        payload["sequence"] = sequence
        event.data = payload

        if event.event_type == TELEMETRY_DIAGNOSTIC:
            return

        if not is_known_event_type(event.event_type):
            logger.warning("Unknown event type emitted: %s", event.event_type)
            self._emit_diagnostic(
                task_id=event.task_id,
                diagnostic_type="unknown_event_type",
                details={"unknown_event_type": str(event.event_type or "").strip()},
            )
            return
        errors = validate_payload_shape(event.event_type, payload)
        if errors:
            joined = "; ".join(errors)
            logger.warning("Invalid payload for %s: %s", event.event_type, joined)
            self._emit_diagnostic(
                task_id=event.task_id,
                diagnostic_type="payload_contract_violation",
                details={
                    "event_type": str(event.event_type or "").strip(),
                    "errors": joined,
                },
            )

    def _emit_diagnostic(
        self,
        *,
        task_id: str,
        diagnostic_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self._diagnostic_emit_depth > 0:
            return
        normalized_type = str(diagnostic_type or "").strip().lower()
        if not normalized_type:
            return
        if not self._diagnostic_budget_available(normalized_type):
            return
        payload: dict[str, Any] = {
            "diagnostic_type": normalized_type,
            "source_component": "event_bus",
            "task_id": str(task_id or "").strip(),
        }
        raw_details = details if isinstance(details, dict) else {}
        for key, value in raw_details.items():
            clean_key = str(key or "").strip()
            if not clean_key:
                continue
            if isinstance(value, bool | int | float):
                payload[clean_key] = value
                continue
            payload[clean_key] = str(value or "").strip()[:240]

        self._diagnostic_emit_depth += 1
        try:
            self.emit(
                Event(
                    event_type=TELEMETRY_DIAGNOSTIC,
                    task_id=str(task_id or "").strip() or "system",
                    data=payload,
                ),
            )
        finally:
            self._diagnostic_emit_depth -= 1

    def _diagnostic_budget_available(self, diagnostic_type: str) -> bool:
        now = time.monotonic()
        bucket = self._diagnostic_buckets.get(diagnostic_type)
        if bucket is None:
            bucket = _DiagnosticBucket(
                tokens=float(self._diagnostic_burst),
                last_refill=now,
            )
            self._diagnostic_buckets[diagnostic_type] = bucket
        elapsed = max(0.0, now - float(bucket.last_refill))
        refill_rate = float(self._diagnostic_rate_per_minute) / 60.0
        bucket.tokens = min(
            float(self._diagnostic_burst),
            float(bucket.tokens) + (elapsed * refill_rate),
        )
        bucket.last_refill = now
        if bucket.tokens < 1.0:
            return False
        bucket.tokens -= 1.0
        return True

    def recent_events(self, limit: int = 50) -> list[Event]:
        """Return recent events."""
        return self._history[-limit:]

    def clear(self) -> None:
        """Clear all handlers and history."""
        self._handlers.clear()
        self._global_handlers.clear()
        self._history.clear()
        for task in list(self._pending_tasks):
            task.cancel()
        self._pending_tasks.clear()

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for in-flight async handler tasks to complete.

        This is primarily useful in tests and shutdown paths where callers need
        deterministic completion of fire-and-forget handlers.
        """
        pending = list(self._pending_tasks)
        if not pending:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(
                "Timed out draining %d pending event handler task(s).",
                len(self._pending_tasks),
            )

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        """Track and log async handler task completion."""
        self._pending_tasks.discard(task)
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.warning("Async event handler failed: %s", exc)


class EventPersister:
    """Subscribes to all events and persists them to the database.

    Uses fire-and-forget async persistence so it doesn't slow the event bus.
    """

    def __init__(self, database: Any) -> None:
        self._db = database
        self._failed_persist_count = 0
        self._event_bus: EventBus | None = None

    async def handle(self, event: Event) -> None:
        """Persist a single event to the database."""
        if not should_persist_compliance(event.event_type):
            return
        try:
            payload = event.data if isinstance(event.data, dict) else {}
            correlation_id = str(payload.get("correlation_id", "") or "").strip()
            if not correlation_id:
                correlation_id = f"task:{event.task_id}"
            try:
                sequence = int(payload.get("sequence", 0) or 0)
            except (TypeError, ValueError):
                sequence = 0
            try:
                schema_version = int(payload.get("schema_version", 1) or 1)
            except (TypeError, ValueError):
                schema_version = 1
            await self._db.insert_event(
                task_id=event.task_id,
                correlation_id=correlation_id,
                event_type=event.event_type,
                data=payload,
                timestamp=event.timestamp,
                run_id=str(payload.get("run_id", "") or "").strip(),
                event_id=str(payload.get("event_id", "") or "").strip(),
                sequence=sequence,
                source_component=str(payload.get("source_component", "") or "").strip(),
                schema_version=schema_version,
            )
        except Exception as e:
            self._failed_persist_count += 1
            logger.warning(
                "Event persistence failed for %s (count=%d): %s",
                event.event_type,
                self._failed_persist_count,
                e,
            )
            if (
                self._event_bus is not None
                and event.event_type != TELEMETRY_DIAGNOSTIC
            ):
                self._event_bus.emit(
                    Event(
                        event_type=TELEMETRY_DIAGNOSTIC,
                        task_id=event.task_id,
                        data={
                            "diagnostic_type": "persistence_failure_count_snapshot",
                            "event_type": str(event.event_type or "").strip(),
                            "failed_persist_count": int(self._failed_persist_count),
                            "error_type": type(e).__name__,
                            "error": str(e)[:240],
                            "source_component": "event_persister",
                        },
                    ),
                )

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to all events on the given bus."""
        self._event_bus = event_bus
        event_bus.subscribe_all(self.handle)
