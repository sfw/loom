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
from collections import defaultdict, deque
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


class _AsyncHandlerWorker:
    """Serialize async event handlers behind a bounded queue."""

    def __init__(self, handler: EventHandler, *, max_queue_size: int) -> None:
        self._handler = handler
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._task: asyncio.Task[Any] | None = None
        self._dropped_events = 0

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    def enqueue(self, event: Event) -> bool:
        self._ensure_started()
        if self._task is None:
            return False
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self._dropped_events += 1
            return False

    async def drain(self) -> None:
        await self._queue.join()

    def close(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()

    def _ensure_started(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._task = None
            return
        self._task = loop.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                event = await self._queue.get()
                try:
                    await self._handler(event)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Async event handler failed: %s", exc)
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            return


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
        async_handler_queue_size: int = 1024,
    ) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []
        self._history: list[Event] = []
        self._max_history = 1000
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._async_handler_workers: dict[int, _AsyncHandlerWorker] = {}
        self._drain_hooks: list[Callable[..., Any]] = []
        self._task_sequences: dict[str, int] = {}
        self._operator_mode_provider: Callable[[], TelemetryMode] | None = None
        self._diagnostic_burst = max(1, int(debug_diagnostics_burst))
        self._diagnostic_rate_per_minute = max(1, int(debug_diagnostics_rate_per_minute))
        self._async_handler_queue_size = max(8, int(async_handler_queue_size))
        self._diagnostic_buckets: dict[str, _DiagnosticBucket] = {}
        self._diagnostic_emit_depth = 0

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to a specific event type."""
        self._handlers[event_type].append(handler)
        self._ensure_async_worker(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)
        self._ensure_async_worker(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a handler from a specific event type."""
        handlers = self._handlers.get(event_type, [])
        try:
            handlers.remove(handler)
        except ValueError:
            pass
        self._cleanup_async_worker(handler)

    def unsubscribe_all(self, handler: EventHandler) -> None:
        """Remove a handler from the global handlers list."""
        try:
            self._global_handlers.remove(handler)
        except ValueError:
            pass
        self._cleanup_async_worker(handler)

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
                worker = self._ensure_async_worker(handler)
                if worker is None or not worker.enqueue(event):
                    logger.debug(
                        "Skipped async handler %s: queue unavailable",
                        getattr(handler, "__name__", handler),
                    )
                    self._emit_diagnostic(
                        task_id=event.task_id,
                        diagnostic_type="async_handler_backpressure",
                        details={
                            "event_type": event.event_type,
                            "handler": getattr(handler, "__name__", handler),
                        },
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
        self._drain_hooks.clear()
        for worker in self._async_handler_workers.values():
            worker.close()
        self._async_handler_workers.clear()
        for task in list(self._pending_tasks):
            task.cancel()
        self._pending_tasks.clear()

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for in-flight async handler tasks to complete.

        This is primarily useful in tests and shutdown paths where callers need
        deterministic completion of fire-and-forget handlers.
        """
        pending = list(self._pending_tasks)
        workers = list(self._async_handler_workers.values())
        if not pending and not workers and not self._drain_hooks:
            return
        try:
            coroutines = [worker.drain() for worker in workers]
            for hook in self._drain_hooks:
                try:
                    coroutines.append(hook(timeout=timeout))
                except TypeError:
                    coroutines.append(hook())
            if pending:
                coroutines.append(asyncio.gather(*pending, return_exceptions=True))
            await asyncio.wait_for(
                asyncio.gather(*coroutines, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(
                "Timed out draining event handlers (tasks=%d, workers=%d).",
                len(self._pending_tasks),
                len(workers),
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

    def stats_snapshot(self) -> dict[str, int]:
        """Return lightweight handler queue diagnostics."""
        return {
            "pending_async_tasks": len(self._pending_tasks),
            "async_handler_queue_depth": sum(
                worker.queue_size for worker in self._async_handler_workers.values()
            ),
            "async_handler_drop_count": sum(
                worker.dropped_events for worker in self._async_handler_workers.values()
            ),
        }

    def register_drain_hook(self, hook: Callable[..., Any]) -> None:
        """Register an awaitable drain hook for shutdown/test synchronization."""
        if hook not in self._drain_hooks:
            self._drain_hooks.append(hook)

    def unregister_drain_hook(self, hook: Callable[..., Any]) -> None:
        """Remove a previously registered drain hook."""
        try:
            self._drain_hooks.remove(hook)
        except ValueError:
            pass

    def _ensure_async_worker(self, handler: EventHandler) -> _AsyncHandlerWorker | None:
        is_async = inspect.iscoroutinefunction(handler) or (
            callable(handler)
            and inspect.iscoroutinefunction(getattr(handler, "__call__", None))
        )
        if not is_async:
            return None
        key = id(handler)
        worker = self._async_handler_workers.get(key)
        if worker is None:
            worker = _AsyncHandlerWorker(
                handler,
                max_queue_size=self._async_handler_queue_size,
            )
            self._async_handler_workers[key] = worker
        return worker

    def _cleanup_async_worker(self, handler: EventHandler) -> None:
        key = id(handler)
        if any(existing is handler for existing in self._global_handlers):
            return
        for handlers in self._handlers.values():
            if any(existing is handler for existing in handlers):
                return
        worker = self._async_handler_workers.pop(key, None)
        if worker is not None:
            worker.close()


class EventPersister:
    """Subscribes to all events and persists them to the database.

    Uses a dedicated in-process queue and batched SQLite inserts.

    Durability takes priority over best-effort async-handler dropping, so the
    persister is attached as a synchronous bus subscriber and manages its own
    queue/worker lifecycle.
    """

    def __init__(
        self,
        database: Any,
        *,
        batch_size: int = 64,
        flush_interval_seconds: float = 0.025,
        max_queue_size: int = 4096,
    ) -> None:
        self._db = database
        self._failed_persist_count = 0
        self._event_bus: EventBus | None = None
        self._batch_size = max(1, int(batch_size))
        self._flush_interval_seconds = max(0.0, float(flush_interval_seconds))
        self._queue_limit = max(1, int(max_queue_size))
        self._pending: deque[Event] = deque()
        self._queue_event: asyncio.Event | None = None
        self._drained_event: asyncio.Event | None = None
        self._worker_task: asyncio.Task[Any] | None = None
        self._unfinished_events = 0
        self._queue_high_water = 0
        self._queue_alarm_count = 0
        self._queue_over_limit = False
        self._batch_count = 0
        self._batch_row_count = 0
        self._max_batch_observed = 0
        self._blocking_flush_count = 0
        self._blocking_flush_row_count = 0

    async def handle(self, event: Event) -> None:
        """Compatibility async entry point for tests/callers."""
        if not self._supports_async_batch_insert() and hasattr(self._db, "insert_event"):
            await self._persist_batch([event])
            return
        self.enqueue(event)

    def enqueue(self, event: Event) -> None:
        """Queue one event for batched persistence without using async handlers."""
        if not should_persist_compliance(event.event_type):
            return
        self._ensure_worker_started()
        if len(self._pending) >= self._queue_limit:
            self._mark_queue_pressure(event, pending_count=len(self._pending) + 1)
            drained_batch = self._drain_oldest_pending_batch()
            if drained_batch:
                if self._persist_batch_blocking(drained_batch):
                    self._mark_batch_done(len(drained_batch))
                else:
                    while drained_batch:
                        self._pending.appendleft(drained_batch.pop())
        self._pending.append(event)
        self._unfinished_events += 1
        pending_count = len(self._pending)
        self._queue_high_water = max(self._queue_high_water, pending_count)
        if self._drained_event is not None:
            self._drained_event.clear()
        if self._queue_event is not None:
            self._queue_event.set()
        if pending_count > self._queue_limit:
            self._mark_queue_pressure(event, pending_count=pending_count)

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for queued event persistence to flush."""
        self._ensure_worker_started()
        if self._unfinished_events <= 0:
            return
        drained_event = self._drained_event
        if drained_event is None:
            return
        wait_coro = drained_event.wait()
        if timeout is None:
            await wait_coro
            return
        await asyncio.wait_for(wait_coro, timeout=timeout)

    def stats_snapshot(self) -> dict[str, int]:
        """Return queue/batch counters for perf diagnostics."""
        return {
            "failed_persist_count": self._failed_persist_count,
            "queue_depth": len(self._pending),
            "queue_high_water": self._queue_high_water,
            "queue_alarm_count": self._queue_alarm_count,
            "batch_count": self._batch_count,
            "batch_row_count": self._batch_row_count,
            "max_batch_size": self._max_batch_observed,
            "blocking_flush_count": self._blocking_flush_count,
            "blocking_flush_row_count": self._blocking_flush_row_count,
        }

    def _ensure_worker_started(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._worker_task = None
            return
        if self._queue_event is None:
            self._queue_event = asyncio.Event()
        if self._drained_event is None:
            self._drained_event = asyncio.Event()
        if self._unfinished_events <= 0:
            self._drained_event.set()
        if self._pending:
            self._queue_event.set()
        self._worker_task = loop.create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                first = await self._get_next_pending_event()
                batch = [first]
                while len(batch) < self._batch_size:
                    timeout = (
                        self._flush_interval_seconds
                        if self._flush_interval_seconds > 0.0
                        else 0.0
                    )
                    next_event = await self._get_next_pending_event(
                        timeout=timeout,
                    )
                    if next_event is None:
                        break
                    batch.append(next_event)
                try:
                    await self._persist_batch(batch)
                finally:
                    self._mark_batch_done(len(batch))
        except asyncio.CancelledError:
            return

    async def _get_next_pending_event(self, timeout: float | None = None) -> Event | None:
        event = self._pop_pending_event()
        if event is not None:
            return event
        queue_event = self._queue_event
        if queue_event is None:
            return None
        queue_event.clear()
        event = self._pop_pending_event()
        if event is not None:
            return event
        if timeout is not None and timeout <= 0.0:
            return None
        try:
            if timeout is None:
                await queue_event.wait()
            else:
                await asyncio.wait_for(queue_event.wait(), timeout=timeout)
        except TimeoutError:
            return None
        return self._pop_pending_event()

    def _pop_pending_event(self) -> Event | None:
        if not self._pending:
            return None
        event = self._pending.popleft()
        if self._queue_over_limit and len(self._pending) <= self._queue_limit:
            self._queue_over_limit = False
        return event

    def _drain_oldest_pending_batch(self) -> list[Event]:
        batch_size = min(self._batch_size, len(self._pending))
        if batch_size <= 0:
            return []
        return [self._pending.popleft() for _ in range(batch_size)]

    def _mark_batch_done(self, size: int) -> None:
        self._unfinished_events = max(0, self._unfinished_events - max(0, int(size)))
        if self._unfinished_events == 0 and self._drained_event is not None:
            self._drained_event.set()

    def _mark_queue_pressure(self, event: Event, *, pending_count: int) -> None:
        if self._queue_over_limit:
            return
        self._queue_over_limit = True
        self._queue_alarm_count += 1
        if self._event_bus is not None and event.event_type != TELEMETRY_DIAGNOSTIC:
            self._event_bus.emit(
                Event(
                    event_type=TELEMETRY_DIAGNOSTIC,
                    task_id=event.task_id,
                    data={
                        "diagnostic_type": "event_persister_backpressure",
                        "queue_depth": pending_count,
                        "queue_limit": self._queue_limit,
                        "source_component": "event_persister",
                    },
                ),
            )

    def _record_persisted_batch(self, size: int, *, blocking: bool = False) -> None:
        self._batch_count += 1
        self._batch_row_count += max(0, int(size))
        self._max_batch_observed = max(self._max_batch_observed, size)
        if blocking:
            self._blocking_flush_count += 1
            self._blocking_flush_row_count += size

    def _handle_persist_failure(self, batch: list[Event], error: Exception) -> None:
        self._failed_persist_count += len(batch)
        logger.warning(
            "Event persistence failed for batch of %d event(s) (count=%d): %s",
            len(batch),
            self._failed_persist_count,
            error,
        )
        event = batch[0]
        if self._event_bus is not None and event.event_type != TELEMETRY_DIAGNOSTIC:
            self._event_bus.emit(
                Event(
                    event_type=TELEMETRY_DIAGNOSTIC,
                    task_id=event.task_id,
                    data={
                        "diagnostic_type": "persistence_failure_count_snapshot",
                        "event_type": str(event.event_type or "").strip(),
                        "failed_persist_count": int(self._failed_persist_count),
                        "error_type": type(error).__name__,
                        "error": str(error)[:240],
                        "source_component": "event_persister",
                    },
                ),
            )

    async def _persist_batch(self, batch: list[Event]) -> None:
        try:
            rows = [self._event_row_payload(event) for event in batch]
            if self._supports_async_batch_insert():
                await self._db.insert_events_batch(rows)
            else:
                for row in rows:
                    await self._db.insert_event(  # type: ignore[attr-defined]
                        str(row.get("task_id", "") or ""),
                        str(row.get("correlation_id", "") or ""),
                        str(row.get("event_type", "") or ""),
                        row.get("data") if isinstance(row.get("data"), dict) else {},
                        timestamp=str(row.get("timestamp", "") or ""),
                        run_id=str(row.get("run_id", "") or ""),
                        event_id=str(row.get("event_id", "") or ""),
                        sequence=int(row.get("sequence", 0) or 0),
                        source_component=str(row.get("source_component", "") or ""),
                        schema_version=int(row.get("schema_version", 1) or 1),
                    )
            self._record_persisted_batch(len(batch))
        except Exception as e:
            self._handle_persist_failure(batch, e)

    def _persist_batch_blocking(self, batch: list[Event]) -> bool:
        if not batch or not hasattr(self._db, "insert_events_batch_blocking"):
            return False
        try:
            self._db.insert_events_batch_blocking([
                self._event_row_payload(event) for event in batch
            ])
            self._record_persisted_batch(len(batch), blocking=True)
            return True
        except Exception as e:
            self._handle_persist_failure(batch, e)
            return False

    def _supports_async_batch_insert(self) -> bool:
        method = getattr(self._db, "insert_events_batch", None)
        return bool(
            inspect.iscoroutinefunction(method)
            or (
                callable(method)
                and inspect.iscoroutinefunction(getattr(method, "__call__", None))
            )
        )

    @staticmethod
    def _event_row_payload(event: Event) -> dict[str, object]:
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
        return {
            "task_id": event.task_id,
            "correlation_id": correlation_id,
            "event_type": event.event_type,
            "data": payload,
            "timestamp": event.timestamp,
            "run_id": str(payload.get("run_id", "") or "").strip(),
            "event_id": str(payload.get("event_id", "") or "").strip(),
            "sequence": sequence,
            "source_component": str(payload.get("source_component", "") or "").strip(),
            "schema_version": schema_version,
        }

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to all events on the given bus."""
        self._event_bus = event_bus
        event_bus.subscribe_all(self.enqueue)
        event_bus.register_drain_hook(self.drain)
