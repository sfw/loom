# Spec 08: Event System

## Overview

Every significant action in Loom emits a structured event. Events are the glue connecting the engine to all consumers: the TUI, web dashboard, SSE streams, webhook callbacks, SQLite log, and the learning system. Events enable full replay of any task execution.

## Event Types

```python
# events/types.py

class EventType(str, Enum):
    # Task lifecycle
    TASK_CREATED = "task_created"
    TASK_PLANNING = "task_planning"
    TASK_PLAN_READY = "task_plan_ready"
    TASK_REPLANNING = "task_replanning"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"

    # Subtask lifecycle
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_PROGRESS = "subtask_progress"
    SUBTASK_COMPLETED = "subtask_completed"
    SUBTASK_FAILED = "subtask_failed"
    SUBTASK_RETRYING = "subtask_retrying"

    # Verification
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"

    # Human-in-the-loop
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    FEEDBACK_RECEIVED = "feedback_received"
    STEER_INSTRUCTION = "steer_instruction"

    # Model calls (observability)
    MODEL_CALL_STARTED = "model_call_started"
    MODEL_CALL_COMPLETED = "model_call_completed"

    # Tool calls
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"

    # Memory
    MEMORY_ENTRY_CREATED = "memory_entry_created"

    # Workspace
    FILE_CHANGED = "file_changed"
```

## Event Structure

```python
@dataclass
class TaskEvent:
    task_id: str
    correlation_id: str          # Groups related events in a chain
    timestamp: datetime
    event_type: EventType
    data: dict                   # Type-specific payload
    subtask_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "subtask_id": self.subtask_id,
            "data": self.data,
        }
```

## Correlation IDs

Every subtask execution gets a unique correlation ID. All events within that execution chain share the same correlation ID, enabling trace reconstruction:

```
correlation: "exec-schema-convert-attempt-2"
├── subtask_started
├── model_call_started
├── tool_call_started (read_file)
├── tool_call_completed (read_file)
├── model_call_completed
├── model_call_started
├── tool_call_started (edit_file)
├── tool_call_completed (edit_file)
├── model_call_completed
├── verification_started
├── verification_passed
└── subtask_completed
```

## Event Bus

```python
class EventBus:
    """
    In-process event bus. Supports multiple subscribers per task.
    Events are delivered to subscribers AND persisted to SQLite.
    """

    def __init__(self, db: Database):
        self._subscribers: dict[str, list[asyncio.Queue]] = {}  # task_id -> queues
        self._global_subscribers: list[asyncio.Queue] = []       # all events
        self._db = db

    def emit(self, event: TaskEvent) -> None:
        """
        Emit an event. Non-blocking.
        1. Persist to SQLite (fire-and-forget background task)
        2. Deliver to all subscribers for this task
        3. Deliver to global subscribers
        """
        asyncio.create_task(self._persist(event))

        for queue in self._subscribers.get(event.task_id, []):
            queue.put_nowait(event)
        for queue in self._global_subscribers:
            queue.put_nowait(event)

    async def subscribe(self, task_id: str) -> AsyncIterator[TaskEvent]:
        """
        Subscribe to events for a specific task.
        Yields events as they arrive. Used by SSE endpoints.
        """
        queue: asyncio.Queue[TaskEvent] = asyncio.Queue()
        self._subscribers.setdefault(task_id, []).append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
                if event.event_type in (
                    EventType.TASK_COMPLETED,
                    EventType.TASK_FAILED,
                    EventType.TASK_CANCELLED,
                ):
                    break  # Terminal events end the subscription
        finally:
            self._subscribers[task_id].remove(queue)

    async def subscribe_all(self) -> AsyncIterator[TaskEvent]:
        """Subscribe to all events across all tasks."""
        queue: asyncio.Queue[TaskEvent] = asyncio.Queue()
        self._global_subscribers.append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._global_subscribers.remove(queue)

    async def replay(self, task_id: str) -> list[TaskEvent]:
        """Replay all events for a task from the database."""
        rows = await self._db.query(
            "SELECT * FROM events WHERE task_id = ? ORDER BY timestamp",
            (task_id,),
        )
        return [self._row_to_event(row) for row in rows]

    async def _persist(self, event: TaskEvent) -> None:
        """Store event in SQLite for replay and debugging."""
        await self._db.execute(
            "INSERT INTO events (task_id, correlation_id, timestamp, event_type, data) VALUES (?, ?, ?, ?, ?)",
            (event.task_id, event.correlation_id, event.timestamp.isoformat(),
             event.event_type.value, json.dumps(event.data)),
        )
```

## Webhook Delivery

For tasks with a `callback_url`, terminal events trigger webhook delivery:

```python
class WebhookDelivery:
    async def deliver(self, callback_url: str, event: TaskEvent) -> None:
        """POST event to callback URL. Retry 3 times with backoff."""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        callback_url,
                        json=event.to_dict(),
                        timeout=10.0,
                    )
                    response.raise_for_status()
                    return
            except Exception:
                await asyncio.sleep(2 ** attempt)
```

## Acceptance Criteria

- [ ] All engine state transitions emit corresponding events
- [ ] Events persist to SQLite for replay
- [ ] SSE subscribers receive events in real-time
- [ ] Correlation IDs correctly chain related events
- [ ] Task replay reconstructs full execution history from database
- [ ] Webhook delivery fires for terminal events when callback_url is set
- [ ] Subscribers are cleaned up when SSE connections close
- [ ] Event emission is non-blocking (doesn't slow the engine)
- [ ] Global subscribers receive events from all tasks
