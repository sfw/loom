"""Event bus for Loom.

In-process pub/sub for decoupling components. Events are emitted by the
orchestrator and consumed by the API (SSE), logger, and other listeners.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to a specific event type."""
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events."""
        self._global_handlers.append(handler)

    def emit(self, event: Event) -> None:
        """Emit an event to all matching handlers.

        Handlers are called as fire-and-forget tasks on the event loop.
        """
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        all_handlers = list(self._global_handlers) + list(self._handlers.get(event.event_type, []))

        for handler in all_handlers:
            if asyncio.iscoroutinefunction(handler):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(handler(event))
                except RuntimeError:
                    pass  # No running loop — skip async handlers
            else:
                try:
                    handler(event)
                except Exception:
                    pass  # Don't let handler errors break the emitter

    def recent_events(self, limit: int = 50) -> list[Event]:
        """Return recent events."""
        return self._history[-limit:]

    def clear(self) -> None:
        """Clear all handlers and history."""
        self._handlers.clear()
        self._global_handlers.clear()
        self._history.clear()
