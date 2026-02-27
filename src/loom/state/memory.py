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
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
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
            await db.commit()
            return bool(cursor.rowcount)

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
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                f"""UPDATE task_runs
                    SET heartbeat_at=?,
                        lease_expires_at={expires_expr},
                        updated_at=?
                    WHERE run_id=? AND lease_owner=?""",
                (now, expires_delta, now, run_id, lease_owner),
            )
            await db.commit()
            return bool(cursor.rowcount)

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
