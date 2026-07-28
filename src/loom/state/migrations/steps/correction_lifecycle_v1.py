"""Migration step for the unified durable correction lifecycle."""

from __future__ import annotations

from loom.state.migrations.runner import index_exists, table_exists

_REQUIRED_TABLES = (
    "correction_cycles",
    "correction_attempts",
    "correction_actions",
)
_REQUIRED_INDEXES = (
    "idx_correction_cycles_task_state",
    "idx_correction_cycles_subtask",
    "idx_correction_cycles_run",
    "idx_correction_attempts_cycle",
    "idx_correction_attempts_task",
    "idx_correction_actions_cycle",
)


async def apply(conn) -> None:
    await conn.execute(
        """CREATE TABLE IF NOT EXISTS correction_cycles (
               id TEXT PRIMARY KEY,
               task_id TEXT NOT NULL,
               run_id TEXT DEFAULT '',
               subtask_id TEXT NOT NULL,
               blocker_fingerprint TEXT NOT NULL,
               state TEXT NOT NULL DEFAULT 'detected',
               blocking INTEGER NOT NULL DEFAULT 1,
               repairability TEXT NOT NULL,
               handler TEXT NOT NULL,
               reason_code TEXT DEFAULT '',
               blocker_snapshot TEXT NOT NULL,
               baseline_progress TEXT NOT NULL,
               latest_progress TEXT NOT NULL,
               attempt_count INTEGER NOT NULL DEFAULT 0,
               no_progress_count INTEGER NOT NULL DEFAULT 0,
               max_attempts INTEGER NOT NULL DEFAULT 3,
               terminal_reason TEXT DEFAULT '',
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               updated_at TEXT NOT NULL DEFAULT (datetime('now')),
               resolved_at TEXT,
               FOREIGN KEY (task_id) REFERENCES tasks(id)
           )""",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_correction_cycles_task_state "
        "ON correction_cycles(task_id, state, created_at)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_correction_cycles_subtask "
        "ON correction_cycles(task_id, subtask_id, updated_at)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_correction_cycles_run ON correction_cycles(run_id)",
    )
    await conn.execute(
        """CREATE TABLE IF NOT EXISTS correction_attempts (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               correction_id TEXT NOT NULL,
               task_id TEXT NOT NULL,
               run_id TEXT DEFAULT '',
               subtask_id TEXT NOT NULL,
               attempt INTEGER NOT NULL,
               state TEXT NOT NULL,
               plan_json TEXT NOT NULL,
               before_progress TEXT NOT NULL,
               after_progress TEXT NOT NULL,
               progress_made INTEGER NOT NULL DEFAULT 0,
               outcome TEXT DEFAULT '',
               error TEXT DEFAULT '',
               metadata TEXT,
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               FOREIGN KEY (correction_id) REFERENCES correction_cycles(id),
               FOREIGN KEY (task_id) REFERENCES tasks(id)
           )""",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_correction_attempts_cycle "
        "ON correction_attempts(correction_id, attempt)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_correction_attempts_task "
        "ON correction_attempts(task_id, subtask_id, created_at)",
    )
    await conn.execute(
        """CREATE TABLE IF NOT EXISTS correction_actions (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               correction_attempt_id INTEGER NOT NULL,
               correction_id TEXT NOT NULL,
               sequence INTEGER NOT NULL,
               action_type TEXT NOT NULL,
               handler TEXT NOT NULL,
               args_json TEXT NOT NULL,
               idempotency_key TEXT NOT NULL UNIQUE,
               state TEXT NOT NULL DEFAULT 'planned',
               result_json TEXT,
               error TEXT DEFAULT '',
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               updated_at TEXT NOT NULL DEFAULT (datetime('now')),
               FOREIGN KEY (correction_attempt_id) REFERENCES correction_attempts(id),
               FOREIGN KEY (correction_id) REFERENCES correction_cycles(id)
           )""",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_correction_actions_cycle "
        "ON correction_actions(correction_id, correction_attempt_id, sequence)",
    )


async def verify(conn) -> None:
    missing_tables = [name for name in _REQUIRED_TABLES if not await table_exists(conn, name)]
    if missing_tables:
        raise RuntimeError(
            "correction lifecycle migration incomplete; missing tables: "
            + ", ".join(missing_tables)
        )
    missing_indexes = [name for name in _REQUIRED_INDEXES if not await index_exists(conn, name)]
    if missing_indexes:
        raise RuntimeError(
            "correction lifecycle migration incomplete; missing indexes: "
            + ", ".join(missing_indexes)
        )
