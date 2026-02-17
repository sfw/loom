"""Event bus for Loom.

In-process pub/sub for decoupling components. Events are emitted by the
orchestrator and consumed by the API (SSE), logger, and other listeners.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Base event structure."""

    event_type: str
    task_id: str
    data: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# Callback type: async function that takes an Event
EventHandler = Callable[[Event], Any]


class EventBus:
    """In-process async event bus.

    Supports:
    - subscribe(event_type, handler) for type-specific listening
    - subscribe_all(handler) for global listening (SSE, logging)
    - emit(event) dispatches to all matching handlers
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []
        self._history: list[Event] = []
        self._max_history = 1000
        self._pending_tasks: set[asyncio.Task[Any]] = set()

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

    def emit(self, event: Event) -> None:
        """Emit an event to all matching handlers.

        Handlers are called as fire-and-forget tasks on the event loop.
        """
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

    async def handle(self, event: Event) -> None:
        """Persist a single event to the database."""
        try:
            correlation_id = str(uuid.uuid4().hex[:12])
            await self._db.insert_event(
                task_id=event.task_id,
                correlation_id=correlation_id,
                event_type=event.event_type,
                data=event.data,
            )
        except Exception as e:
            logger.warning("Event persistence failed for %s: %s", event.event_type, e)

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to all events on the given bus."""
        event_bus.subscribe_all(self.handle)
