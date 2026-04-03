"""Layer 2: Structured memory archive backed by SQLite.

Memory entries are extracted at write-time from subtask execution.
Retrieval is deterministic SQL (by task, subtask, type, tags).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import aiosqlite

from loom.events.types import DB_MIGRATION_FAILED, DB_SCHEMA_READY
from loom.state.migrations import (
    MIGRATIONS,
    MigrationExecutionError,
    apply_pending_migrations,
    ensure_migration_table,
    has_user_tables,
    verify_schema,
)

logger = logging.getLogger(__name__)

_SQLITE_BUSY_TIMEOUT_MS = 5_000
_SQLITE_WAL_AUTOCHECKPOINT_PAGES = 2_000
_SQLITE_CACHE_SIZE_KIB = 16_384
_SQLITE_READ_POOL_SIZE = 2
_EVENT_INSERT_SQL = """INSERT INTO events
               (task_id, run_id, correlation_id, event_id, sequence, timestamp, event_type,
                source_component, schema_version, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


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


def _iter_sql_statements(script: str) -> list[str]:
    """Split SQL script into complete statements while preserving trigger bodies."""
    statements: list[str] = []
    buffer: list[str] = []
    for raw_line in script.splitlines():
        line = str(raw_line or "")
        if not buffer and not line.strip():
            continue
        buffer.append(line)
        candidate = "\n".join(buffer).strip()
        if not candidate:
            buffer.clear()
            continue
        if sqlite3.complete_statement(candidate):
            statement = candidate.strip()
            if statement.endswith(";"):
                statement = statement[:-1].strip()
            if statement:
                statements.append(statement)
            buffer.clear()

    if any(str(part).strip() for part in buffer):
        raise RuntimeError("Incomplete SQL statement detected in schema script.")
    return statements


async def _apply_schema_script(db: aiosqlite.Connection, script: str) -> None:
    for statement in _iter_sql_statements(script):
        await db.execute(statement)


class Database:
    """Async SQLite database wrapper for Loom.

    All access is non-blocking via aiosqlite.
    """

    _open_instances: ClassVar[set[Database]] = set()

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._write_db: aiosqlite.Connection | None = None
        self._read_dbs: list[aiosqlite.Connection] = []
        self._read_index = 0
        self._lifecycle_lock = asyncio.Lock()
        self._write_tx_lock = asyncio.Lock()
        self._stats: dict[str, float | int] = {
            "connection_open_count": 0,
            "read_query_count": 0,
            "read_query_total_ms": 0.0,
            "write_query_count": 0,
            "write_query_total_ms": 0.0,
            "batch_write_count": 0,
            "batch_row_count": 0,
        }
        self._open_instances.add(self)

    async def close(self) -> None:
        """Close database connections."""
        async with self._lifecycle_lock:
            write_db = self._write_db
            self._write_db = None
            read_dbs = list(self._read_dbs)
            self._read_dbs.clear()
            self._read_index = 0

        if write_db is not None:
            await write_db.close()
        for db in read_dbs:
            await db.close()
        self._open_instances.discard(self)

    @classmethod
    async def close_open_instances(cls) -> None:
        """Close every still-open Database instance.

        Tests use this as a safety net to avoid leaking aiosqlite worker
        threads across event-loop teardown when a test forgets to close its DB.
        """
        errors: list[BaseException] = []
        for instance in list(cls._iter_open_instances()):
            try:
                await instance.close()
            except BaseException as exc:  # pragma: no cover - best effort cleanup
                errors.append(exc)
        if errors:
            raise errors[0]

    @classmethod
    def _iter_open_instances(cls) -> Iterable[Database]:
        return tuple(cls._open_instances)

    async def initialize(self) -> None:
        """Initialize database schema and apply ordered migrations."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        state_dir = Path(__file__).parent
        schema_path = state_dir / "schema.sql"
        base_schema_path = state_dir / "schema" / "base.sql"
        schema = schema_path.read_text()
        base_schema = base_schema_path.read_text() if base_schema_path.exists() else schema
        db_path_text = str(self._db_path)

        def _report(event_type: str, payload: dict[str, object]) -> None:
            body = {
                "event_type": event_type,
                "db_path": db_path_text,
                **payload,
            }
            logger.info("db_migration_diagnostic %s", json.dumps(body, sort_keys=True))

        async with aiosqlite.connect(self._db_path) as db:
            await self._configure_connection(db, role="migration")
            await db.execute("BEGIN IMMEDIATE")
            existing_db = False
            applied_now: list[str] = []
            try:
                existing_db = await has_user_tables(db)
                if existing_db:
                    await ensure_migration_table(db)
                    applied_now = await apply_pending_migrations(
                        db,
                        steps=MIGRATIONS,
                        reporter=_report,
                        db_path=db_path_text,
                    )
                    # Create any newer tables/indexes not present in older databases.
                    await _apply_schema_script(db, schema)
                else:
                    await _apply_schema_script(db, base_schema)
                    await ensure_migration_table(db)
                    # Record bootstrap at latest migration state and verify idempotency.
                    applied_now = await apply_pending_migrations(
                        db,
                        steps=MIGRATIONS,
                        reporter=_report,
                        db_path=db_path_text,
                    )
                await verify_schema(db, steps=MIGRATIONS)
                _report(
                    DB_SCHEMA_READY,
                    {
                        "existing_db": bool(existing_db),
                        "applied_migration_ids": applied_now,
                    },
                )
                await db.commit()
            except Exception as e:
                await db.rollback()
                _report(
                    DB_MIGRATION_FAILED,
                    {
                        "migration_id": (
                            e.migration_id
                            if isinstance(e, MigrationExecutionError)
                            else ""
                        ),
                        "phase": (
                            e.phase
                            if isinstance(e, MigrationExecutionError)
                            else "startup"
                        ),
                        "error_class": e.__class__.__name__,
                        "error_message": str(e),
                        "actionable_suggestion_key": "loom_db_doctor_then_migrate",
                    },
                )
                raise
        await self._open_runtime_connections()

    async def _open_runtime_connections(self) -> None:
        async with self._lifecycle_lock:
            if self._write_db is not None and len(self._read_dbs) == _SQLITE_READ_POOL_SIZE:
                return
            await self._close_connections_locked()

            self._write_db = await self._open_connection(role="write")
            self._read_dbs = [
                await self._open_connection(role=f"read-{index}")
                for index in range(_SQLITE_READ_POOL_SIZE)
            ]
            self._read_index = 0

    async def _close_connections_locked(self) -> None:
        write_db = self._write_db
        read_dbs = list(self._read_dbs)
        self._write_db = None
        self._read_dbs.clear()
        self._read_index = 0

        if write_db is not None:
            await write_db.close()
        for db in read_dbs:
            await db.close()

    async def _open_connection(self, *, role: str) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self._db_path)
        await self._configure_connection(db, role=role)
        self._stats["connection_open_count"] = int(self._stats["connection_open_count"]) + 1
        return db

    async def _configure_connection(
        self,
        db: aiosqlite.Connection,
        *,
        role: str,
    ) -> None:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.execute(f"PRAGMA wal_autocheckpoint={_SQLITE_WAL_AUTOCHECKPOINT_PAGES}")
        await db.execute("PRAGMA temp_store=MEMORY")
        await db.execute(f"PRAGMA cache_size=-{_SQLITE_CACHE_SIZE_KIB}")
        logger.debug("Configured SQLite connection for %s", role)

    def _configure_sync_connection(
        self,
        db: sqlite3.Connection,
        *,
        role: str,
    ) -> None:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute(f"PRAGMA wal_autocheckpoint={_SQLITE_WAL_AUTOCHECKPOINT_PAGES}")
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute(f"PRAGMA cache_size=-{_SQLITE_CACHE_SIZE_KIB}")
        logger.debug("Configured SQLite sync connection for %s", role)

    def _record_write_stats(
        self,
        *,
        row_count: int,
        elapsed_ms: float,
        batch: bool = False,
    ) -> None:
        clean_row_count = max(0, int(row_count))
        self._stats["write_query_count"] = int(self._stats["write_query_count"]) + clean_row_count
        self._stats["write_query_total_ms"] = float(self._stats["write_query_total_ms"]) + max(
            0.0,
            float(elapsed_ms),
        )
        if batch:
            self._stats["batch_write_count"] = int(self._stats["batch_write_count"]) + 1
            self._stats["batch_row_count"] = int(self._stats["batch_row_count"]) + clean_row_count

    async def _get_write_db(self) -> aiosqlite.Connection:
        if self._write_db is None:
            await self._open_runtime_connections()
        if self._write_db is None:
            raise RuntimeError("SQLite write connection is not initialized.")
        return self._write_db

    async def _get_read_db(self) -> aiosqlite.Connection:
        if not self._read_dbs:
            await self._open_runtime_connections()
        if not self._read_dbs:
            raise RuntimeError("SQLite read connection pool is not initialized.")
        db = self._read_dbs[self._read_index % len(self._read_dbs)]
        self._read_index = (self._read_index + 1) % len(self._read_dbs)
        return db

    def stats_snapshot(self) -> dict[str, float | int]:
        """Return lightweight connection/query counters for perf diagnostics."""
        return dict(self._stats)

    async def execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a write query."""
        db = await self._get_write_db()
        started = time.perf_counter()
        await db.execute(sql, params)
        await db.commit()
        self._record_write_stats(
            row_count=1,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def execute_returning_id(self, sql: str, params: tuple = ()) -> int:
        """Execute an insert and return the lastrowid."""
        db = await self._get_write_db()
        started = time.perf_counter()
        cursor = await db.execute(sql, params)
        await db.commit()
        self._record_write_stats(
            row_count=1,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return cursor.lastrowid

    async def execute_rowcount(self, sql: str, params: tuple = ()) -> int:
        """Execute a write query and return the affected row count."""
        db = await self._get_write_db()
        started = time.perf_counter()
        cursor = await db.execute(sql, params)
        await db.commit()
        self._record_write_stats(
            row_count=1,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        return int(cursor.rowcount or 0)

    async def execute_many_returning_ids(self, sql: str, params_list: list[tuple]) -> list[int]:
        """Execute multiple inserts in a single transaction and return lastrowids."""
        db = await self._get_write_db()
        started = time.perf_counter()
        ids = []
        for params in params_list:
            cursor = await db.execute(sql, params)
            ids.append(cursor.lastrowid)
        await db.commit()
        self._record_write_stats(
            row_count=len(params_list),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            batch=True,
        )
        return ids

    async def execute_many(self, sql: str, params_list: list[tuple]) -> None:
        """Execute multiple writes in a single committed transaction."""
        if not params_list:
            return
        db = await self._get_write_db()
        started = time.perf_counter()
        await db.executemany(sql, params_list)
        await db.commit()
        self._record_write_stats(
            row_count=len(params_list),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            batch=True,
        )

    async def run_write_transaction(self, callback) -> object:
        """Run a callback inside one write-connection transaction."""
        db = await self._get_write_db()
        started = time.perf_counter()
        async with self._write_tx_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                result = await callback(db)
            except Exception:
                await db.rollback()
                raise
            await db.commit()
        self._record_write_stats(
            row_count=1,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            batch=True,
        )
        return result

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a read query and return results as dicts."""
        db = await self._get_read_db()
        started = time.perf_counter()
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        self._stats["read_query_count"] = int(self._stats["read_query_count"]) + 1
        self._stats["read_query_total_ms"] = float(self._stats["read_query_total_ms"]) + (
            (time.perf_counter() - started) * 1000.0
        )
        return [dict(row) for row in rows]

    async def query_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Execute a read query and return the first result."""
        db = await self._get_read_db()
        started = time.perf_counter()
        cursor = await db.execute(sql, params)
        row = await cursor.fetchone()
        self._stats["read_query_count"] = int(self._stats["read_query_count"]) + 1
        self._stats["read_query_total_ms"] = float(self._stats["read_query_total_ms"]) + (
            (time.perf_counter() - started) * 1000.0
        )
        return dict(row) if row is not None else None

    async def query_one_write(self, sql: str, params: tuple = ()) -> dict | None:
        """Execute a read query on the write connection.

        Use this sparingly for read-after-write paths that must observe the
        latest committed row without depending on the pooled readers.
        """
        db = await self._get_write_db()
        started = time.perf_counter()
        cursor = await db.execute(sql, params)
        row = await cursor.fetchone()
        self._stats["read_query_count"] = int(self._stats["read_query_count"]) + 1
        self._stats["read_query_total_ms"] = float(self._stats["read_query_total_ms"]) + (
            (time.perf_counter() - started) * 1000.0
        )
        return dict(row) if row is not None else None

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

    async def update_task_metadata(self, task_id: str, metadata: dict | None) -> None:
        """Update task metadata JSON blob."""
        now = datetime.now().isoformat()
        await self.execute(
            "UPDATE tasks SET metadata=?, updated_at=? WHERE id=?",
            (json.dumps(metadata) if metadata is not None else None, now, task_id),
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

    async def list_tasks_for_workspace(self, workspace_path: str) -> list[dict]:
        """List tasks scoped to a workspace or source workspace root."""
        return await self.query(
            """
            SELECT *
            FROM tasks
            WHERE COALESCE(
                    NULLIF(json_extract(COALESCE(metadata, '{}'), '$.source_workspace_root'), ''),
                    workspace_path
                ) = ?
            ORDER BY created_at DESC
            """,
            (workspace_path,),
        )

    async def get_tasks_by_ids(self, task_ids: list[str]) -> dict[str, dict]:
        """Return task rows keyed by task ID."""
        clean_ids = sorted({
            str(task_id or "").strip()
            for task_id in task_ids
            if str(task_id or "").strip()
        })
        if not clean_ids:
            return {}
        placeholders = ", ".join("?" for _ in clean_ids)
        rows = await self.query(
            f"SELECT * FROM tasks WHERE id IN ({placeholders})",
            tuple(clean_ids),
        )
        return {
            str(row.get("id", "") or "").strip(): dict(row)
            for row in rows
            if str(row.get("id", "") or "").strip()
        }

    # --- Task run operations ---

    async def insert_task_run(
        self,
        *,
        run_id: str,
        task_id: str,
        status: str = "queued",
        process_name: str = "",
        attempt: int = 1,
        metadata: dict | None = None,
    ) -> None:
        now = datetime.now().isoformat()
        await self.execute(
            """INSERT INTO task_runs
               (run_id, task_id, status, process_name, attempt, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET
                   task_id=excluded.task_id,
                   process_name=CASE
                       WHEN excluded.process_name <> '' THEN excluded.process_name
                       ELSE task_runs.process_name
                   END,
                   metadata=COALESCE(excluded.metadata, task_runs.metadata),
                   updated_at=excluded.updated_at""",
            (
                run_id,
                task_id,
                status,
                process_name,
                max(1, int(attempt)),
                json.dumps(metadata) if metadata is not None else None,
                now,
                now,
            ),
        )

    async def get_task_run(self, run_id: str) -> dict | None:
        return await self.query_one("SELECT * FROM task_runs WHERE run_id=?", (run_id,))

    # --- Task question operations ---

    async def get_task_question(self, task_id: str, question_id: str) -> dict | None:
        row = await self.query_one(
            "SELECT * FROM task_questions WHERE task_id=? AND question_id=?",
            (task_id, question_id),
        )
        if row is None:
            return None
        return self._decode_task_question_row(row)

    async def upsert_pending_task_question(
        self,
        *,
        question_id: str,
        task_id: str,
        subtask_id: str,
        request_payload: dict | None = None,
        timeout_at: str = "",
    ) -> dict:
        now = datetime.now().isoformat()
        request_json = json.dumps(
            request_payload or {},
            ensure_ascii=False,
            sort_keys=True,
        )
        clean_question_id = str(question_id or "").strip()
        clean_task_id = str(task_id or "").strip()
        clean_subtask_id = str(subtask_id or "").strip()
        clean_timeout = str(timeout_at or "").strip()

        existing = await self.get_task_question(clean_task_id, clean_question_id)
        if existing is not None and str(existing.get("status", "")).strip().lower() != "pending":
            return existing

        if clean_subtask_id:
            scope_pending = await self.query_one(
                """SELECT * FROM task_questions
                   WHERE task_id=? AND subtask_id=? AND status='pending'
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (clean_task_id, clean_subtask_id),
            )
            if (
                scope_pending
                and str(scope_pending.get("question_id", "")).strip() != clean_question_id
            ):
                return self._decode_task_question_row(scope_pending)

        await self.execute(
            """INSERT INTO task_questions
               (question_id, task_id, subtask_id, status, request_payload, answer_payload,
                created_at, updated_at, resolved_at, timeout_at)
               VALUES (?, ?, ?, 'pending', ?, NULL, ?, ?, NULL, ?)
               ON CONFLICT(question_id) DO UPDATE SET
                   status='pending',
                   request_payload=excluded.request_payload,
                   updated_at=excluded.updated_at,
                   timeout_at=excluded.timeout_at""",
            (
                clean_question_id,
                clean_task_id,
                clean_subtask_id,
                request_json,
                now,
                now,
                clean_timeout,
            ),
        )
        stored = await self.get_task_question(clean_task_id, clean_question_id)
        return stored or {
            "question_id": clean_question_id,
            "task_id": clean_task_id,
            "subtask_id": clean_subtask_id,
            "status": "pending",
            "request_payload": request_payload or {},
            "answer_payload": {},
            "created_at": now,
            "updated_at": now,
            "resolved_at": "",
            "timeout_at": clean_timeout,
        }

    async def resolve_task_question(
        self,
        *,
        task_id: str,
        question_id: str,
        status: str,
        answer_payload: dict | None = None,
        resolved_at: str = "",
    ) -> dict | None:
        clean_task_id = str(task_id or "").strip()
        clean_question_id = str(question_id or "").strip()
        existing = await self.get_task_question(clean_task_id, clean_question_id)
        if existing is None:
            return None
        if str(existing.get("status", "")).strip().lower() != "pending":
            return existing

        now = datetime.now().isoformat()
        resolved = str(resolved_at or "").strip() or now
        await self.execute(
            """UPDATE task_questions
               SET status=?, answer_payload=?, updated_at=?, resolved_at=?
               WHERE task_id=? AND question_id=? AND status='pending'""",
            (
                str(status or "").strip().lower() or "answered",
                json.dumps(answer_payload or {}, ensure_ascii=False, sort_keys=True),
                now,
                resolved,
                clean_task_id,
                clean_question_id,
            ),
        )
        return await self.get_task_question(clean_task_id, clean_question_id)

    async def list_pending_task_questions(self, task_id: str) -> list[dict]:
        rows = await self.query(
            """SELECT * FROM task_questions
               WHERE task_id=? AND status='pending'
               ORDER BY created_at ASC""",
            (task_id,),
        )
        return [self._decode_task_question_row(row) for row in rows]

    async def list_task_questions(self, task_id: str) -> list[dict]:
        rows = await self.query(
            """SELECT * FROM task_questions
               WHERE task_id=?
               ORDER BY created_at ASC""",
            (task_id,),
        )
        return [self._decode_task_question_row(row) for row in rows]

    @staticmethod
    def _decode_task_question_row(row: dict) -> dict:
        parsed = dict(row)
        for key in ("request_payload", "answer_payload"):
            raw = parsed.get(key)
            if not isinstance(raw, str):
                parsed[key] = raw if isinstance(raw, dict) else {}
                continue
            text = raw.strip()
            if not text:
                parsed[key] = {}
                continue
            try:
                value = json.loads(text)
            except Exception:
                value = {}
            parsed[key] = value if isinstance(value, dict) else {}
        return parsed

    async def list_recoverable_task_runs(
        self,
        *,
        statuses: tuple[str, ...] = ("queued", "running"),
        limit: int = 100,
    ) -> list[dict]:
        requested = {str(item).strip().lower() for item in statuses if str(item).strip()}
        clauses: list[str] = []
        params: list[object] = []
        if "queued" in requested:
            clauses.append("status='queued'")
        if "running" in requested:
            clauses.append(
                "(status='running' AND "
                "(lease_expires_at IS NULL OR lease_expires_at < datetime('now')))",
            )
        if not clauses:
            return []
        sql = (
            "SELECT * FROM task_runs "
            f"WHERE ({' OR '.join(clauses)}) "
            "ORDER BY created_at ASC LIMIT ?"
        )
        params.append(max(1, int(limit)))
        return await self.query(sql, tuple(params))

    async def acquire_task_run_lease(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
    ) -> bool:
        now = datetime.now().isoformat()
        ttl = max(5, int(lease_seconds))
        # SQLite datetime() expects "YYYY-MM-DD HH:MM:SS"; replace "T" for compatibility.
        expires_expr = "datetime('now', ?)"
        expires_delta = f"+{ttl} seconds"
        rowcount = await self.execute_rowcount(
            f"""UPDATE task_runs
                SET status='running',
                    lease_owner=?,
                    lease_expires_at={expires_expr},
                    heartbeat_at=?,
                    started_at=COALESCE(started_at, ?),
                    updated_at=?
                WHERE run_id=?
                  AND (
                    status='queued'
                    OR lease_owner=?
                    OR lease_expires_at IS NULL
                    OR lease_expires_at < datetime('now')
                  )""",
            (
                lease_owner,
                expires_delta,
                now,
                now,
                now,
                run_id,
                lease_owner,
            ),
        )
        return bool(rowcount)

    async def heartbeat_task_run(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
    ) -> bool:
        now = datetime.now().isoformat()
        ttl = max(5, int(lease_seconds))
        expires_expr = "datetime('now', ?)"
        expires_delta = f"+{ttl} seconds"
        rowcount = await self.execute_rowcount(
            f"""UPDATE task_runs
                SET heartbeat_at=?,
                    lease_expires_at={expires_expr},
                    updated_at=?
                WHERE run_id=? AND lease_owner=?""",
            (now, expires_delta, now, run_id, lease_owner),
        )
        return bool(rowcount)

    async def complete_task_run(
        self,
        *,
        run_id: str,
        status: str,
        last_error: str = "",
    ) -> None:
        now = datetime.now().isoformat()
        await self.execute(
            """UPDATE task_runs
               SET status=?, ended_at=?, lease_owner='', lease_expires_at=NULL,
                   heartbeat_at=?, last_error=?, updated_at=?
               WHERE run_id=?""",
            (status, now, now, last_error, now, run_id),
        )

    async def requeue_task_run(self, *, run_id: str) -> None:
        now = datetime.now().isoformat()
        await self.execute(
            """UPDATE task_runs
               SET status='queued',
                   lease_owner='',
                   lease_expires_at=NULL,
                   heartbeat_at=NULL,
                   updated_at=?
               WHERE run_id=?""",
            (now, run_id),
        )

    async def get_latest_task_run_for_task(self, task_id: str) -> dict | None:
        return await self.query_one(
            """SELECT * FROM task_runs
               WHERE task_id=?
               ORDER BY created_at DESC
               LIMIT 1""",
            (task_id,),
        )

    async def get_latest_task_runs_for_tasks(self, task_ids: list[str]) -> dict[str, dict]:
        """Return the most recent task_run row for each task ID."""
        clean_ids = sorted({
            str(task_id or "").strip()
            for task_id in task_ids
            if str(task_id or "").strip()
        })
        if not clean_ids:
            return {}
        placeholders = ", ".join("?" for _ in clean_ids)
        rows = await self.query(
            f"""
            WITH ranked AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY task_id
                        ORDER BY created_at DESC, run_id DESC
                    ) AS row_rank
                FROM task_runs
                WHERE task_id IN ({placeholders})
            )
            SELECT *
            FROM ranked
            WHERE row_rank = 1
            """,
            tuple(clean_ids),
        )
        return {
            str(row.get("task_id", "") or "").strip(): dict(row)
            for row in rows
            if str(row.get("task_id", "") or "").strip()
        }

    # --- Retry/remediation persistence ---

    async def insert_subtask_attempt(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        attempt: int,
        tier: int,
        retry_strategy: str,
        reason_code: str = "",
        feedback: str = "",
        error: str = "",
        missing_targets: list[str] | None = None,
        error_category: str = "",
        metadata: dict | None = None,
    ) -> int:
        return await self.execute_returning_id(
            """INSERT INTO subtask_attempts
               (task_id, run_id, subtask_id, attempt, tier, retry_strategy, reason_code,
                feedback, error, missing_targets, error_category, metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                run_id,
                subtask_id,
                max(1, int(attempt)),
                int(tier),
                retry_strategy,
                reason_code,
                feedback,
                error,
                json.dumps(missing_targets or []),
                error_category,
                json.dumps(metadata) if metadata is not None else None,
                datetime.now().isoformat(),
            ),
        )

    async def upsert_remediation_item(self, item: dict) -> None:
        now = datetime.now().isoformat()
        payload = dict(item)
        payload.setdefault("updated_at", now)
        payload.setdefault("created_at", now)
        payload.setdefault("run_id", "")
        payload.setdefault("task_id", "")
        payload.setdefault("subtask_id", "")
        payload.setdefault("strategy", "")
        payload.setdefault("reason_code", "")
        payload.setdefault("verification_outcome", "")
        payload.setdefault("feedback", "")
        payload.setdefault("blocking", False)
        payload.setdefault("critical_path", False)
        payload.setdefault("state", "queued")
        payload.setdefault("attempt_count", 0)
        payload.setdefault("max_attempts", 3)
        payload.setdefault("base_backoff_seconds", 2.0)
        payload.setdefault("max_backoff_seconds", 30.0)
        payload.setdefault("next_attempt_at", "")
        payload.setdefault("ttl_at", "")
        payload.setdefault("last_error", "")
        payload.setdefault("terminal_reason", "")
        payload.setdefault("missing_targets", [])
        metadata = dict(payload)
        for key in (
            "id",
            "task_id",
            "run_id",
            "subtask_id",
            "strategy",
            "reason_code",
            "verification_outcome",
            "feedback",
            "blocking",
            "critical_path",
            "state",
            "attempt_count",
            "max_attempts",
            "base_backoff_seconds",
            "max_backoff_seconds",
            "next_attempt_at",
            "ttl_at",
            "last_error",
            "terminal_reason",
            "missing_targets",
            "created_at",
            "updated_at",
        ):
            metadata.pop(key, None)
        await self.execute(
            """INSERT INTO remediation_items
               (id, task_id, run_id, subtask_id, strategy, reason_code, verification_outcome,
                feedback, blocking, critical_path, state, attempt_count, max_attempts,
                base_backoff_seconds, max_backoff_seconds, next_attempt_at, ttl_at,
                last_error, terminal_reason, missing_targets, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 run_id=excluded.run_id,
                 reason_code=excluded.reason_code,
                 verification_outcome=excluded.verification_outcome,
                 feedback=excluded.feedback,
                 blocking=excluded.blocking,
                 critical_path=excluded.critical_path,
                 state=excluded.state,
                 attempt_count=excluded.attempt_count,
                 max_attempts=excluded.max_attempts,
                 base_backoff_seconds=excluded.base_backoff_seconds,
                 max_backoff_seconds=excluded.max_backoff_seconds,
                 next_attempt_at=excluded.next_attempt_at,
                 ttl_at=excluded.ttl_at,
                 last_error=excluded.last_error,
                 terminal_reason=excluded.terminal_reason,
                 missing_targets=excluded.missing_targets,
                 metadata=excluded.metadata,
                 updated_at=excluded.updated_at""",
            (
                str(payload.get("id", "")).strip(),
                str(payload.get("task_id", "")).strip(),
                str(payload.get("run_id", "")).strip(),
                str(payload.get("subtask_id", "")).strip(),
                str(payload.get("strategy", "")).strip(),
                str(payload.get("reason_code", "")).strip(),
                str(payload.get("verification_outcome", "")).strip(),
                str(payload.get("feedback", "")),
                1 if bool(payload.get("blocking", False)) else 0,
                1 if bool(payload.get("critical_path", False)) else 0,
                str(payload.get("state", "queued")).strip(),
                int(payload.get("attempt_count", 0) or 0),
                max(1, int(payload.get("max_attempts", 3) or 3)),
                float(payload.get("base_backoff_seconds", 2.0) or 0.0),
                float(payload.get("max_backoff_seconds", 30.0) or 0.0),
                str(payload.get("next_attempt_at", "") or ""),
                str(payload.get("ttl_at", "") or ""),
                str(payload.get("last_error", "") or ""),
                str(payload.get("terminal_reason", "") or ""),
                json.dumps(payload.get("missing_targets", []) or []),
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                str(payload.get("created_at", now) or now),
                str(payload.get("updated_at", now) or now),
            ),
        )

    async def list_remediation_items(self, *, task_id: str) -> list[dict]:
        rows = await self.query(
            """SELECT * FROM remediation_items
               WHERE task_id=?
               ORDER BY created_at ASC""",
            (task_id,),
        )
        normalized: list[dict] = []
        for row in rows:
            item = dict(row)
            item["blocking"] = bool(item.get("blocking", 0))
            item["critical_path"] = bool(item.get("critical_path", 0))
            raw_targets = item.get("missing_targets")
            if isinstance(raw_targets, str):
                try:
                    item["missing_targets"] = json.loads(raw_targets)
                except Exception:
                    item["missing_targets"] = []
            if not isinstance(item.get("missing_targets"), list):
                item["missing_targets"] = []
            raw_meta = item.get("metadata")
            if isinstance(raw_meta, str) and raw_meta.strip():
                try:
                    meta_obj = json.loads(raw_meta)
                    if isinstance(meta_obj, dict):
                        for key, value in meta_obj.items():
                            if key not in item:
                                item[key] = value
                except Exception:
                    pass
            normalized.append(item)
        return normalized

    async def insert_remediation_attempt(
        self,
        *,
        remediation_id: str,
        task_id: str,
        run_id: str,
        subtask_id: str,
        attempt: int,
        max_attempts: int,
        phase: str = "done",
        outcome: str = "",
        retry_strategy: str = "",
        transient: bool = False,
        reason_code: str = "",
        error: str = "",
    ) -> int:
        return await self.execute_returning_id(
            """INSERT INTO remediation_attempts
               (remediation_id, task_id, run_id, subtask_id, attempt, max_attempts,
                phase, outcome, retry_strategy, transient, reason_code, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                remediation_id,
                task_id,
                run_id,
                subtask_id,
                max(1, int(attempt)),
                max(1, int(max_attempts)),
                phase,
                outcome,
                retry_strategy,
                1 if transient else 0,
                reason_code,
                error,
                datetime.now().isoformat(),
            ),
        )

    # --- Iteration loop persistence ---

    async def upsert_iteration_run(
        self,
        *,
        loop_run_id: str,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        policy_snapshot: dict | None = None,
        terminal_reason: str = "",
        attempt_count: int = 0,
        replan_count: int = 0,
        exhaustion_fingerprint: str = "",
        metadata: dict | None = None,
    ) -> None:
        now = datetime.now().isoformat()
        await self.execute(
            """INSERT INTO iteration_runs
               (loop_run_id, task_id, run_id, subtask_id, phase_id, policy_snapshot,
                terminal_reason, attempt_count, replan_count, exhaustion_fingerprint,
                metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(loop_run_id) DO UPDATE SET
                   run_id=excluded.run_id,
                   phase_id=excluded.phase_id,
                   policy_snapshot=excluded.policy_snapshot,
                   terminal_reason=excluded.terminal_reason,
                   attempt_count=excluded.attempt_count,
                   replan_count=excluded.replan_count,
                   exhaustion_fingerprint=excluded.exhaustion_fingerprint,
                   metadata=excluded.metadata,
                   updated_at=excluded.updated_at""",
            (
                loop_run_id,
                task_id,
                run_id,
                subtask_id,
                phase_id,
                json.dumps(policy_snapshot or {}, ensure_ascii=False, sort_keys=True),
                terminal_reason,
                max(0, int(attempt_count)),
                max(0, int(replan_count)),
                exhaustion_fingerprint,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                now,
                now,
            ),
        )

    async def insert_iteration_attempt(
        self,
        *,
        loop_run_id: str,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        attempt_index: int,
        status: str,
        summary: str = "",
        gate_summary: dict | None = None,
        budget_snapshot: dict | None = None,
        metadata: dict | None = None,
    ) -> int:
        return await self.execute_returning_id(
            """INSERT INTO iteration_attempts
               (loop_run_id, task_id, run_id, subtask_id, phase_id, attempt_index,
                status, summary, gate_summary, budget_snapshot, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                loop_run_id,
                task_id,
                run_id,
                subtask_id,
                phase_id,
                max(1, int(attempt_index)),
                status,
                summary,
                json.dumps(gate_summary or {}, ensure_ascii=False, sort_keys=True),
                json.dumps(budget_snapshot or {}, ensure_ascii=False, sort_keys=True),
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                datetime.now().isoformat(),
            ),
        )

    async def insert_iteration_gate_result(
        self,
        *,
        loop_run_id: str,
        attempt_id: int | None,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        attempt_index: int,
        gate_id: str,
        gate_type: str,
        status: str,
        blocking: bool,
        reason_code: str = "",
        measured_value: object = None,
        threshold_value: object = None,
        detail: str = "",
        metadata: dict | None = None,
    ) -> int:
        return await self.execute_returning_id(
            """INSERT INTO iteration_gate_results
               (loop_run_id, attempt_id, task_id, run_id, subtask_id, phase_id,
                attempt_index, gate_id, gate_type, status, blocking, reason_code,
                measured_value, threshold_value, detail, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                loop_run_id,
                attempt_id,
                task_id,
                run_id,
                subtask_id,
                phase_id,
                max(1, int(attempt_index)),
                gate_id,
                gate_type,
                status,
                1 if blocking else 0,
                reason_code,
                json.dumps(measured_value, ensure_ascii=False, sort_keys=True),
                json.dumps(threshold_value, ensure_ascii=False, sort_keys=True),
                detail,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                datetime.now().isoformat(),
            ),
        )

    async def list_iteration_runs(
        self,
        *,
        task_id: str,
        subtask_id: str | None = None,
    ) -> list[dict]:
        if subtask_id:
            rows = await self.query(
                """SELECT * FROM iteration_runs
                   WHERE task_id=? AND subtask_id=?
                   ORDER BY created_at ASC""",
                (task_id, subtask_id),
            )
        else:
            rows = await self.query(
                """SELECT * FROM iteration_runs
                   WHERE task_id=?
                   ORDER BY created_at ASC""",
                (task_id,),
            )
        return [self._decode_json_columns(row, ("policy_snapshot", "metadata")) for row in rows]

    async def list_iteration_attempts(
        self,
        *,
        loop_run_id: str,
    ) -> list[dict]:
        rows = await self.query(
            """SELECT * FROM iteration_attempts
               WHERE loop_run_id=?
               ORDER BY attempt_index ASC, id ASC""",
            (loop_run_id,),
        )
        return [
            self._decode_json_columns(
                row,
                ("gate_summary", "budget_snapshot", "metadata"),
            )
            for row in rows
        ]

    async def list_iteration_gate_results(
        self,
        *,
        loop_run_id: str,
    ) -> list[dict]:
        rows = await self.query(
            """SELECT * FROM iteration_gate_results
               WHERE loop_run_id=?
               ORDER BY attempt_index ASC, id ASC""",
            (loop_run_id,),
        )
        normalized = []
        for row in rows:
            parsed = self._decode_json_columns(
                row,
                ("measured_value", "threshold_value", "metadata"),
            )
            parsed["blocking"] = bool(parsed.get("blocking", 0))
            normalized.append(parsed)
        return normalized

    # --- Claim/evidence validity lineage ---

    async def insert_artifact_claims(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        claims: list[dict] | None = None,
    ) -> list[int]:
        rows = []
        now = datetime.now().isoformat()
        for claim in claims or []:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id", "") or "").strip()
            claim_text = str(claim.get("text", "") or "").strip()
            if not claim_id or not claim_text:
                continue
            rows.append((
                task_id,
                run_id,
                subtask_id,
                phase_id,
                claim_id,
                claim_text,
                str(claim.get("claim_type", "qualitative") or "qualitative"),
                str(claim.get("criticality", "important") or "important"),
                str(claim.get("status", "extracted") or "extracted"),
                str(claim.get("reason_code", "") or ""),
                json.dumps(claim.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                now,
            ))
        if not rows:
            return []
        return await self.execute_many_returning_ids(
            """INSERT INTO artifact_claims
               (task_id, run_id, subtask_id, phase_id, claim_id, claim_text, claim_type,
                criticality, lifecycle_state, reason_code, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    async def insert_claim_evidence_links(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        links: list[dict] | None = None,
    ) -> list[int]:
        rows = []
        now = datetime.now().isoformat()
        for link in links or []:
            if not isinstance(link, dict):
                continue
            claim_id = str(link.get("claim_id", "") or "").strip()
            evidence_id = str(link.get("evidence_id", "") or "").strip()
            if not claim_id or not evidence_id:
                continue
            rows.append((
                task_id,
                run_id,
                subtask_id,
                claim_id,
                evidence_id,
                str(link.get("link_type", "supporting") or "supporting"),
                float(link.get("score", 0.0) or 0.0),
                json.dumps(link.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                now,
            ))
        if not rows:
            return []
        return await self.execute_many_returning_ids(
            """INSERT INTO claim_evidence_links
               (task_id, run_id, subtask_id, claim_id, evidence_id, link_type, score,
                metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    async def insert_claim_verification_results(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        results: list[dict] | None = None,
    ) -> list[int]:
        rows = []
        now = datetime.now().isoformat()
        for result in results or []:
            if not isinstance(result, dict):
                continue
            claim_id = str(result.get("claim_id", "") or "").strip()
            status = str(result.get("status", "") or "").strip()
            if not claim_id or not status:
                continue
            rows.append((
                task_id,
                run_id,
                subtask_id,
                phase_id,
                claim_id,
                status,
                str(result.get("reason_code", "") or ""),
                str(result.get("verifier", "") or ""),
                float(result.get("confidence", 0.0) or 0.0),
                json.dumps(result.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                now,
            ))
        if not rows:
            return []
        return await self.execute_many_returning_ids(
            """INSERT INTO claim_verification_results
               (task_id, run_id, subtask_id, phase_id, claim_id, status, reason_code,
                verifier, confidence, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    async def insert_artifact_validity_summary(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        extracted_count: int = 0,
        supported_count: int = 0,
        contradicted_count: int = 0,
        insufficient_evidence_count: int = 0,
        pruned_count: int = 0,
        supported_ratio: float = 0.0,
        gate_decision: str = "",
        reason_code: str = "",
        metadata: dict | None = None,
    ) -> int:
        return await self.execute_returning_id(
            """INSERT INTO artifact_validity_summaries
               (task_id, run_id, subtask_id, phase_id, extracted_count, supported_count,
                contradicted_count, insufficient_evidence_count, pruned_count, supported_ratio,
                gate_decision, reason_code, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                run_id,
                subtask_id,
                phase_id,
                max(0, int(extracted_count)),
                max(0, int(supported_count)),
                max(0, int(contradicted_count)),
                max(0, int(insufficient_evidence_count)),
                max(0, int(pruned_count)),
                max(0.0, min(1.0, float(supported_ratio))),
                str(gate_decision or ""),
                str(reason_code or ""),
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                datetime.now().isoformat(),
            ),
        )

    @staticmethod
    def _decode_json_columns(row: dict, columns: tuple[str, ...]) -> dict:
        parsed = dict(row)
        for col in columns:
            raw = parsed.get(col)
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            if not text:
                parsed[col] = {} if col.endswith("metadata") or col.endswith("snapshot") else ""
                continue
            try:
                parsed[col] = json.loads(text)
            except Exception:
                continue
        return parsed

    # --- Mutating tool idempotency ledger ---

    async def get_mutation_ledger_entry(self, idempotency_key: str) -> dict | None:
        return await self.query_one(
            "SELECT * FROM tool_mutation_ledger WHERE idempotency_key=?",
            (idempotency_key,),
        )

    async def upsert_mutation_ledger_entry(
        self,
        *,
        idempotency_key: str,
        task_id: str,
        run_id: str,
        subtask_id: str,
        tool_name: str,
        args_hash: str,
        status: str,
        result_json: str,
    ) -> None:
        now = datetime.now().isoformat()
        await self.execute(
            """INSERT INTO tool_mutation_ledger
               (idempotency_key, task_id, run_id, subtask_id, tool_name, args_hash,
                status, result_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(idempotency_key) DO UPDATE SET
                   status=excluded.status,
                   result_json=excluded.result_json,
                   updated_at=excluded.updated_at""",
            (
                idempotency_key,
                task_id,
                run_id,
                subtask_id,
                tool_name,
                args_hash,
                status,
                result_json,
                now,
                now,
            ),
        )

    async def compute_slo_snapshot(self) -> dict[str, object]:
        """Compute lightweight SLO metrics from task/task_run/event tables."""
        counts = await self.query_one(
            """SELECT
                   COUNT(*) AS total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                   SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                   SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled
               FROM tasks""",
        ) or {}
        duration_rows = await self.query(
            """SELECT
                   CAST(
                       (julianday(completed_at) - julianday(created_at)) * 86400.0
                       AS REAL
                   ) AS seconds
               FROM tasks
               WHERE completed_at IS NOT NULL
                 AND created_at IS NOT NULL
               ORDER BY seconds ASC""",
        )
        durations = [
            float(row.get("seconds", 0.0))
            for row in duration_rows
            if row.get("seconds") is not None
        ]
        p50 = 0.0
        p95 = 0.0
        if durations:
            mid = int((len(durations) - 1) * 0.50)
            hi = int((len(durations) - 1) * 0.95)
            p50 = float(durations[mid])
            p95 = float(durations[hi])

        event_counts = await self.query(
            """SELECT event_type, COUNT(*) AS count
               FROM events
               GROUP BY event_type""",
        )
        counts_by_event = {
            str(row.get("event_type", "")): int(row.get("count", 0) or 0)
            for row in event_counts
        }
        total = int(counts.get("total", 0) or 0)
        completed = int(counts.get("completed", 0) or 0)
        success_rate = (completed / total) if total > 0 else 0.0
        return {
            "task_counts": {
                "total": total,
                "completed": completed,
                "failed": int(counts.get("failed", 0) or 0),
                "cancelled": int(counts.get("cancelled", 0) or 0),
            },
            "task_success_rate": round(success_rate, 4),
            "task_duration_seconds": {
                "p50": round(p50, 3),
                "p95": round(p95, 3),
            },
            "event_counts": counts_by_event,
            "planner_degradation_rate": self._ratio(
                counts_by_event.get("task_plan_degraded", 0),
                max(1, counts_by_event.get("task_planning", 0)),
            ),
            "budget_exhaustion_rate": self._ratio(
                counts_by_event.get("task_budget_exhausted", 0),
                max(1, total),
            ),
            "resume_recovery_success_rate": self._ratio(
                counts_by_event.get("task_run_recovered", 0),
                max(1, counts_by_event.get("task_run_acquired", 0)),
            ),
            "mutating_call_dedupe_rate": self._ratio(
                counts_by_event.get("tool_call_deduplicated", 0),
                max(
                    1,
                    counts_by_event.get("tool_call_completed", 0)
                    + counts_by_event.get("tool_call_deduplicated", 0),
                ),
            ),
            "generated_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        num = max(0, int(numerator or 0))
        den = max(1, int(denominator or 1))
        return round(num / den, 4)

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
               AND (summary LIKE ? ESCAPE '\\'
                    OR detail LIKE ? ESCAPE '\\'
                    OR tags LIKE ? ESCAPE '\\')
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
        *,
        timestamp: str = "",
        run_id: str = "",
        event_id: str = "",
        sequence: int = 0,
        source_component: str = "",
        schema_version: int = 1,
    ) -> int:
        """Insert an event log entry."""
        emitted_timestamp = str(timestamp or "").strip() or datetime.now(UTC).isoformat()
        payload = data if isinstance(data, dict) else {}
        return await self.execute_returning_id(
            _EVENT_INSERT_SQL,
            (
                task_id,
                run_id,
                correlation_id,
                event_id,
                max(0, int(sequence)),
                emitted_timestamp,
                event_type,
                source_component,
                max(1, int(schema_version)),
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )

    @staticmethod
    def _event_insert_params(rows: list[dict[str, object]]) -> list[tuple]:
        params_list: list[tuple] = []
        for row in rows:
            payload = row.get("data") if isinstance(row.get("data"), dict) else {}
            emitted_timestamp = (
                str(row.get("timestamp", "") or "").strip()
                or datetime.now(UTC).isoformat()
            )
            params_list.append(
                (
                    str(row.get("task_id", "") or ""),
                    str(row.get("run_id", "") or ""),
                    str(row.get("correlation_id", "") or ""),
                    str(row.get("event_id", "") or ""),
                    max(0, int(row.get("sequence", 0) or 0)),
                    emitted_timestamp,
                    str(row.get("event_type", "") or ""),
                    str(row.get("source_component", "") or ""),
                    max(1, int(row.get("schema_version", 1) or 1)),
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
        return params_list

    async def insert_events_batch(self, rows: list[dict[str, object]]) -> None:
        """Insert multiple event log rows in one transaction."""
        if not rows:
            return
        params_list = self._event_insert_params(rows)
        await self.execute_many(
            _EVENT_INSERT_SQL,
            params_list,
        )

    def insert_events_batch_blocking(self, rows: list[dict[str, object]]) -> None:
        """Insert event log rows synchronously for overload backpressure handling."""
        if not rows:
            return
        params_list = self._event_insert_params(rows)
        started = time.perf_counter()
        with sqlite3.connect(
            self._db_path,
            timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000.0,
        ) as db:
            self._configure_sync_connection(db, role="blocking-overflow-write")
            self._stats["connection_open_count"] = int(self._stats["connection_open_count"]) + 1
            db.executemany(_EVENT_INSERT_SQL, params_list)
            db.commit()
        self._record_write_stats(
            row_count=len(params_list),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            batch=True,
        )

    async def query_events(
        self,
        task_id: str,
        event_type: str | None = None,
        limit: int = 100,
        *,
        after_id: int = 0,
        after_sequence: int = 0,
        ascending: bool = False,
    ) -> list[dict]:
        """Query events for a task."""
        normalized_after_id = max(0, int(after_id))
        normalized_after_sequence = max(0, int(after_sequence))
        order = "ASC" if ascending else "DESC"
        if event_type:
            return await self.query(
                f"""SELECT * FROM events
                    WHERE task_id=? AND event_type=? AND id>? AND sequence>?
                    ORDER BY sequence {order}, id {order} LIMIT ?""",
                (task_id, event_type, normalized_after_id, normalized_after_sequence, limit),
            )
        return await self.query(
            f"""SELECT * FROM events
                WHERE task_id=? AND id>? AND sequence>?
                ORDER BY sequence {order}, id {order} LIMIT ?""",
            (task_id, normalized_after_id, normalized_after_sequence, limit),
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

    async def update_task_metadata(self, task_id: str, metadata: dict | None) -> None:
        await self._db.update_task_metadata(task_id, metadata)

    async def insert_task_run(
        self,
        *,
        run_id: str,
        task_id: str,
        status: str = "queued",
        process_name: str = "",
        attempt: int = 1,
        metadata: dict | None = None,
    ) -> None:
        await self._db.insert_task_run(
            run_id=run_id,
            task_id=task_id,
            status=status,
            process_name=process_name,
            attempt=attempt,
            metadata=metadata,
        )

    async def get_task_run(self, run_id: str) -> dict | None:
        return await self._db.get_task_run(run_id)

    async def get_task_question(self, task_id: str, question_id: str) -> dict | None:
        return await self._db.get_task_question(task_id, question_id)

    async def upsert_pending_task_question(
        self,
        *,
        question_id: str,
        task_id: str,
        subtask_id: str,
        request_payload: dict | None = None,
        timeout_at: str = "",
    ) -> dict:
        return await self._db.upsert_pending_task_question(
            question_id=question_id,
            task_id=task_id,
            subtask_id=subtask_id,
            request_payload=request_payload,
            timeout_at=timeout_at,
        )

    async def resolve_task_question(
        self,
        *,
        task_id: str,
        question_id: str,
        status: str,
        answer_payload: dict | None = None,
        resolved_at: str = "",
    ) -> dict | None:
        return await self._db.resolve_task_question(
            task_id=task_id,
            question_id=question_id,
            status=status,
            answer_payload=answer_payload,
            resolved_at=resolved_at,
        )

    async def list_pending_task_questions(self, task_id: str) -> list[dict]:
        return await self._db.list_pending_task_questions(task_id)

    async def list_task_questions(self, task_id: str) -> list[dict]:
        return await self._db.list_task_questions(task_id)

    async def list_recoverable_task_runs(
        self,
        *,
        statuses: tuple[str, ...] = ("queued", "running"),
        limit: int = 100,
    ) -> list[dict]:
        return await self._db.list_recoverable_task_runs(statuses=statuses, limit=limit)

    async def acquire_task_run_lease(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
    ) -> bool:
        return await self._db.acquire_task_run_lease(
            run_id=run_id,
            lease_owner=lease_owner,
            lease_seconds=lease_seconds,
        )

    async def heartbeat_task_run(
        self,
        *,
        run_id: str,
        lease_owner: str,
        lease_seconds: int,
    ) -> bool:
        return await self._db.heartbeat_task_run(
            run_id=run_id,
            lease_owner=lease_owner,
            lease_seconds=lease_seconds,
        )

    async def complete_task_run(
        self,
        *,
        run_id: str,
        status: str,
        last_error: str = "",
    ) -> None:
        await self._db.complete_task_run(
            run_id=run_id,
            status=status,
            last_error=last_error,
        )

    async def requeue_task_run(self, *, run_id: str) -> None:
        await self._db.requeue_task_run(run_id=run_id)

    async def get_latest_task_run_for_task(self, task_id: str) -> dict | None:
        return await self._db.get_latest_task_run_for_task(task_id)

    async def get_latest_task_runs_for_tasks(self, task_ids: list[str]) -> dict[str, dict]:
        return await self._db.get_latest_task_runs_for_tasks(task_ids)

    async def get_tasks_by_ids(self, task_ids: list[str]) -> dict[str, dict]:
        return await self._db.get_tasks_by_ids(task_ids)

    async def insert_subtask_attempt(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        attempt: int,
        tier: int,
        retry_strategy: str,
        reason_code: str = "",
        feedback: str = "",
        error: str = "",
        missing_targets: list[str] | None = None,
        error_category: str = "",
        metadata: dict | None = None,
    ) -> int:
        return await self._db.insert_subtask_attempt(
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            attempt=attempt,
            tier=tier,
            retry_strategy=retry_strategy,
            reason_code=reason_code,
            feedback=feedback,
            error=error,
            missing_targets=missing_targets,
            error_category=error_category,
            metadata=metadata,
        )

    async def upsert_remediation_item(self, item: dict) -> None:
        await self._db.upsert_remediation_item(item)

    async def list_remediation_items(self, *, task_id: str) -> list[dict]:
        return await self._db.list_remediation_items(task_id=task_id)

    async def insert_remediation_attempt(
        self,
        *,
        remediation_id: str,
        task_id: str,
        run_id: str,
        subtask_id: str,
        attempt: int,
        max_attempts: int,
        phase: str = "done",
        outcome: str = "",
        retry_strategy: str = "",
        transient: bool = False,
        reason_code: str = "",
        error: str = "",
    ) -> int:
        return await self._db.insert_remediation_attempt(
            remediation_id=remediation_id,
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            attempt=attempt,
            max_attempts=max_attempts,
            phase=phase,
            outcome=outcome,
            retry_strategy=retry_strategy,
            transient=transient,
            reason_code=reason_code,
            error=error,
        )

    async def upsert_iteration_run(
        self,
        *,
        loop_run_id: str,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        policy_snapshot: dict | None = None,
        terminal_reason: str = "",
        attempt_count: int = 0,
        replan_count: int = 0,
        exhaustion_fingerprint: str = "",
        metadata: dict | None = None,
    ) -> None:
        await self._db.upsert_iteration_run(
            loop_run_id=loop_run_id,
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            phase_id=phase_id,
            policy_snapshot=policy_snapshot,
            terminal_reason=terminal_reason,
            attempt_count=attempt_count,
            replan_count=replan_count,
            exhaustion_fingerprint=exhaustion_fingerprint,
            metadata=metadata,
        )

    async def insert_iteration_attempt(
        self,
        *,
        loop_run_id: str,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        attempt_index: int,
        status: str,
        summary: str = "",
        gate_summary: dict | None = None,
        budget_snapshot: dict | None = None,
        metadata: dict | None = None,
    ) -> int:
        return await self._db.insert_iteration_attempt(
            loop_run_id=loop_run_id,
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            phase_id=phase_id,
            attempt_index=attempt_index,
            status=status,
            summary=summary,
            gate_summary=gate_summary,
            budget_snapshot=budget_snapshot,
            metadata=metadata,
        )

    async def insert_iteration_gate_result(
        self,
        *,
        loop_run_id: str,
        attempt_id: int | None,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        attempt_index: int,
        gate_id: str,
        gate_type: str,
        status: str,
        blocking: bool,
        reason_code: str = "",
        measured_value: object = None,
        threshold_value: object = None,
        detail: str = "",
        metadata: dict | None = None,
    ) -> int:
        return await self._db.insert_iteration_gate_result(
            loop_run_id=loop_run_id,
            attempt_id=attempt_id,
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            phase_id=phase_id,
            attempt_index=attempt_index,
            gate_id=gate_id,
            gate_type=gate_type,
            status=status,
            blocking=blocking,
            reason_code=reason_code,
            measured_value=measured_value,
            threshold_value=threshold_value,
            detail=detail,
            metadata=metadata,
        )

    async def list_iteration_runs(
        self,
        *,
        task_id: str,
        subtask_id: str | None = None,
    ) -> list[dict]:
        return await self._db.list_iteration_runs(task_id=task_id, subtask_id=subtask_id)

    async def list_iteration_attempts(self, *, loop_run_id: str) -> list[dict]:
        return await self._db.list_iteration_attempts(loop_run_id=loop_run_id)

    async def list_iteration_gate_results(self, *, loop_run_id: str) -> list[dict]:
        return await self._db.list_iteration_gate_results(loop_run_id=loop_run_id)

    async def insert_artifact_claims(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        claims: list[dict] | None = None,
    ) -> list[int]:
        return await self._db.insert_artifact_claims(
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            phase_id=phase_id,
            claims=claims,
        )

    async def insert_claim_evidence_links(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        links: list[dict] | None = None,
    ) -> list[int]:
        return await self._db.insert_claim_evidence_links(
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            links=links,
        )

    async def insert_claim_verification_results(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        results: list[dict] | None = None,
    ) -> list[int]:
        return await self._db.insert_claim_verification_results(
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            phase_id=phase_id,
            results=results,
        )

    async def insert_artifact_validity_summary(
        self,
        *,
        task_id: str,
        run_id: str,
        subtask_id: str,
        phase_id: str = "",
        extracted_count: int = 0,
        supported_count: int = 0,
        contradicted_count: int = 0,
        insufficient_evidence_count: int = 0,
        pruned_count: int = 0,
        supported_ratio: float = 0.0,
        gate_decision: str = "",
        reason_code: str = "",
        metadata: dict | None = None,
    ) -> int:
        return await self._db.insert_artifact_validity_summary(
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            phase_id=phase_id,
            extracted_count=extracted_count,
            supported_count=supported_count,
            contradicted_count=contradicted_count,
            insufficient_evidence_count=insufficient_evidence_count,
            pruned_count=pruned_count,
            supported_ratio=supported_ratio,
            gate_decision=gate_decision,
            reason_code=reason_code,
            metadata=metadata,
        )

    async def get_mutation_ledger_entry(self, idempotency_key: str) -> dict | None:
        return await self._db.get_mutation_ledger_entry(idempotency_key)

    async def upsert_mutation_ledger_entry(
        self,
        *,
        idempotency_key: str,
        task_id: str,
        run_id: str,
        subtask_id: str,
        tool_name: str,
        args_hash: str,
        status: str,
        result_json: str,
    ) -> None:
        await self._db.upsert_mutation_ledger_entry(
            idempotency_key=idempotency_key,
            task_id=task_id,
            run_id=run_id,
            subtask_id=subtask_id,
            tool_name=tool_name,
            args_hash=args_hash,
            status=status,
            result_json=result_json,
        )

    async def compute_slo_snapshot(self) -> dict[str, object]:
        return await self._db.compute_slo_snapshot()

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
