"""Layer 2: Structured memory archive backed by SQLite.

Memory entries are extracted at write-time from subtask execution.
Retrieval is deterministic SQL (by task, subtask, type, tags).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiosqlite


@dataclass
class MemoryEntry:
    """A single memory entry in the structured archive."""

    task_id: str = ""
    subtask_id: str = ""
    entry_type: str = ""  # decision, error, tool_result, discovery, artifact, context
    summary: str = ""  # 1-2 sentence summary (max 150 chars)
    detail: str = ""  # Full content
    tags: str = ""  # Comma-separated tags
    relevance_to: str = ""  # Comma-separated subtask IDs
    id: int | None = None
    timestamp: str = ""
    created_at: str = ""


VALID_ENTRY_TYPES = frozenset(
    ["decision", "error", "tool_result", "user_instruction", "discovery", "artifact", "context"]
)


def _escape_like(value: str) -> str:
    """Escape LIKE wildcard characters (%, _, \\) for safe SQL LIKE queries."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class Database:
    """Async SQLite database wrapper for Loom.

    All access is non-blocking via aiosqlite.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)

    async def close(self) -> None:
        """Close database connections. No-op for aiosqlite (per-query connections)."""
        pass

    async def initialize(self) -> None:
        """Create tables from schema.sql if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).parent / "schema.sql"
        schema = schema_path.read_text()
        async with aiosqlite.connect(self._db_path) as db:
            # Enable WAL mode for better concurrent read/write performance
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(schema)
            await db.commit()

    async def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a write query."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(sql, params)
            await db.commit()

    async def execute_returning_id(self, sql: str, params: tuple = ()) -> int:
        """Execute an insert and return the lastrowid."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(sql, params)
            await db.commit()
            return cursor.lastrowid

    async def execute_many_returning_ids(self, sql: str, params_list: list[tuple]) -> list[int]:
        """Execute multiple inserts in a single transaction and return lastrowids."""
        async with aiosqlite.connect(self._db_path) as db:
            ids = []
            for params in params_list:
                cursor = await db.execute(sql, params)
                ids.append(cursor.lastrowid)
            await db.commit()
            return ids

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a read query and return results as dicts."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def query_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Execute a read query and return the first result."""
        results = await self.query(sql, params)
        return results[0] if results else None

    # --- Task operations ---

    async def insert_task(
        self,
        task_id: str,
        goal: str,
        workspace_path: str = "",
        status: str = "pending",
        approval_mode: str = "auto",
        context: dict | None = None,
        callback_url: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Insert a new task record."""
        now = datetime.now().isoformat()
        await self.execute(
            """INSERT INTO tasks (id, goal, context, workspace_path, status,
               approval_mode, callback_url, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                goal,
                json.dumps(context) if context else None,
                workspace_path,
                status,
                approval_mode,
                callback_url,
                json.dumps(metadata) if metadata else None,
                now,
                now,
            ),
        )

    async def update_task_status(self, task_id: str, status: str) -> None:
        """Update task status."""
        now = datetime.now().isoformat()
        completed_at = now if status in ("completed", "failed", "cancelled") else None
        if completed_at:
            await self.execute(
                "UPDATE tasks SET status=?, updated_at=?, completed_at=? WHERE id=?",
                (status, now, completed_at, task_id),
            )
        else:
            await self.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE id=?",
                (status, now, task_id),
            )

    async def update_task_plan(self, task_id: str, plan_json: str) -> None:
        """Update task plan."""
        now = datetime.now().isoformat()
        await self.execute(
            "UPDATE tasks SET plan=?, updated_at=? WHERE id=?",
            (plan_json, now, task_id),
        )

    async def get_task(self, task_id: str) -> dict | None:
        """Retrieve a task by ID."""
        return await self.query_one("SELECT * FROM tasks WHERE id=?", (task_id,))

    async def list_tasks(self, status: str | None = None) -> list[dict]:
        """List tasks, optionally filtered by status."""
        if status:
            return await self.query(
                "SELECT * FROM tasks WHERE status=? ORDER BY created_at DESC", (status,)
            )
        return await self.query("SELECT * FROM tasks ORDER BY created_at DESC")

    # --- Memory operations ---

    async def insert_memory_entry(self, entry: MemoryEntry) -> int:
        """Insert a memory entry and return its ID."""
        return await self.execute_returning_id(
            """INSERT INTO memory_entries
               (task_id, subtask_id, timestamp, entry_type, summary, detail, tags, relevance_to)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.task_id,
                entry.subtask_id,
                entry.timestamp or datetime.now().isoformat(),
                entry.entry_type,
                entry.summary,
                entry.detail,
                entry.tags,
                entry.relevance_to,
            ),
        )

    async def query_memory(
        self,
        task_id: str,
        subtask_id: str | None = None,
        entry_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """Query memory entries with flexible filters."""
        conditions = ["task_id = ?"]
        params: list = [task_id]

        if subtask_id:
            conditions.append("subtask_id = ?")
            params.append(subtask_id)

        if entry_type:
            conditions.append("entry_type = ?")
            params.append(entry_type)

        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ? ESCAPE '\\'")
                params.append(f"%{_escape_like(tag)}%")
            conditions.append(f"({' OR '.join(tag_conditions)})")

        where = " AND ".join(conditions)
        sql = f"""SELECT * FROM memory_entries WHERE {where}
                  ORDER BY timestamp DESC LIMIT ?"""
        params.append(limit)

        rows = await self.query(sql, tuple(params))
        return [self._row_to_entry(r) for r in rows]

    async def query_relevant_memory(
        self,
        task_id: str,
        subtask_id: str,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Retrieve memory entries relevant to a specific subtask.

        Returns entries that are:
        - From the same subtask
        - Marked as relevant to this subtask
        - Global decisions/errors/user instructions
        """
        rows = await self.query(
            """SELECT * FROM memory_entries
               WHERE task_id = ?
               AND (
                   subtask_id = ?
                   OR relevance_to LIKE ? ESCAPE '\\'
                   OR entry_type IN ('decision', 'error', 'user_instruction')
               )
               ORDER BY timestamp DESC
               LIMIT ?""",
            (task_id, subtask_id, f"%{_escape_like(subtask_id)}%", limit),
        )
        return [self._row_to_entry(r) for r in rows]

    async def search_memory(self, task_id: str, query: str, limit: int = 20) -> list[MemoryEntry]:
        """Full-text search across memory entries for a task."""
        escaped = _escape_like(query)
        rows = await self.query(
            """SELECT * FROM memory_entries
               WHERE task_id = ?
               AND (summary LIKE ? ESCAPE '\\' OR detail LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\')
               ORDER BY timestamp DESC
               LIMIT ?""",
            (task_id, f"%{escaped}%", f"%{escaped}%", f"%{escaped}%", limit),
        )
        return [self._row_to_entry(r) for r in rows]

    # --- Event operations ---

    async def insert_event(
        self,
        task_id: str,
        correlation_id: str,
        event_type: str,
        data: dict,
    ) -> int:
        """Insert an event log entry."""
        return await self.execute_returning_id(
            """INSERT INTO events (task_id, correlation_id, timestamp, event_type, data)
               VALUES (?, ?, ?, ?, ?)""",
            (task_id, correlation_id, datetime.now().isoformat(), event_type, json.dumps(data)),
        )

    async def query_events(
        self,
        task_id: str,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query events for a task."""
        if event_type:
            return await self.query(
                """SELECT * FROM events WHERE task_id=? AND event_type=?
                   ORDER BY timestamp DESC LIMIT ?""",
                (task_id, event_type, limit),
            )
        return await self.query(
            "SELECT * FROM events WHERE task_id=? ORDER BY timestamp DESC LIMIT ?",
            (task_id, limit),
        )

    def _row_to_entry(self, row: dict) -> MemoryEntry:
        return MemoryEntry(
            id=row.get("id"),
            task_id=row.get("task_id", ""),
            subtask_id=row.get("subtask_id", ""),
            timestamp=row.get("timestamp", ""),
            entry_type=row.get("entry_type", ""),
            summary=row.get("summary", ""),
            detail=row.get("detail", ""),
            tags=row.get("tags", ""),
            relevance_to=row.get("relevance_to", ""),
            created_at=row.get("created_at", ""),
        )


class MemoryManager:
    """High-level memory operations for the orchestration engine.

    Coordinates between the database and prompt assembly for
    memory extraction and retrieval.
    """

    def __init__(self, db: Database):
        self._db = db

    async def store(self, entry: MemoryEntry) -> int:
        """Store a single memory entry."""
        if entry.entry_type not in VALID_ENTRY_TYPES:
            raise ValueError(
                f"Invalid entry_type: {entry.entry_type}. Must be one of {VALID_ENTRY_TYPES}"
            )
        return await self._db.insert_memory_entry(entry)

    async def store_many(self, entries: list[MemoryEntry]) -> list[int]:
        """Store multiple memory entries atomically in a single transaction."""
        for entry in entries:
            if entry.entry_type not in VALID_ENTRY_TYPES:
                raise ValueError(
                    f"Invalid entry_type: {entry.entry_type}. Must be one of {VALID_ENTRY_TYPES}"
                )
        params_list = [
            (
                entry.task_id, entry.subtask_id,
                entry.timestamp or datetime.now().isoformat(),
                entry.entry_type, entry.summary, entry.detail,
                entry.tags, entry.relevance_to,
            )
            for entry in entries
        ]
        return await self._db.execute_many_returning_ids(
            """INSERT INTO memory_entries
               (task_id, subtask_id, timestamp, entry_type, summary, detail, tags, relevance_to)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            params_list,
        )

    async def query_relevant(self, task_id: str, subtask_id: str) -> list[MemoryEntry]:
        """Get memory entries relevant to a subtask."""
        return await self._db.query_relevant_memory(task_id, subtask_id)

    async def query(
        self,
        task_id: str,
        entry_type: str | None = None,
        subtask_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """General-purpose memory query with filters."""
        return await self._db.query_memory(
            task_id=task_id,
            subtask_id=subtask_id,
            entry_type=entry_type,
            tags=tags,
        )

    async def search(self, task_id: str, query: str) -> list[MemoryEntry]:
        """Full-text search across memory entries."""
        return await self._db.search_memory(task_id, query)

    def format_for_prompt(self, entries: list[MemoryEntry]) -> str:
        """Format memory entries for injection into a prompt."""
        if not entries:
            return "No relevant prior context."

        lines = []
        for entry in entries:
            prefix = f"[{entry.entry_type}]"
            if entry.subtask_id:
                prefix += f" (from {entry.subtask_id})"
            lines.append(f"{prefix} {entry.summary}")
            if entry.detail and len(entry.detail) <= 200:
                lines.append(f"  Detail: {entry.detail}")
        return "\n".join(lines)
