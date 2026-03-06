"""Migration step for durable ask-user question state."""

from __future__ import annotations

from loom.state.migrations.runner import index_exists, table_exists

_TASK_QUESTIONS_REQUIRED_INDEXES = (
    "idx_task_questions_task_status",
    "idx_task_questions_task_subtask",
    "idx_task_questions_active_scope",
)


async def apply(conn) -> None:
    await conn.execute(
        """CREATE TABLE IF NOT EXISTS task_questions (
               question_id TEXT PRIMARY KEY,
               task_id TEXT NOT NULL,
               subtask_id TEXT NOT NULL DEFAULT '',
               status TEXT NOT NULL DEFAULT 'pending',
               request_payload TEXT NOT NULL,
               answer_payload TEXT,
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               updated_at TEXT NOT NULL DEFAULT (datetime('now')),
               resolved_at TEXT,
               timeout_at TEXT,
               FOREIGN KEY (task_id) REFERENCES tasks(id)
           )""",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_questions_task_status "
        "ON task_questions(task_id, status, created_at)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_task_questions_task_subtask "
        "ON task_questions(task_id, subtask_id, status, created_at)",
    )
    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_task_questions_active_scope "
        "ON task_questions(task_id, subtask_id) "
        "WHERE status = 'pending'",
    )


async def verify(conn) -> None:
    if not await table_exists(conn, "task_questions"):
        raise RuntimeError("task_questions table missing after migration.")
    missing_indexes = [
        name
        for name in _TASK_QUESTIONS_REQUIRED_INDEXES
        if not await index_exists(conn, name)
    ]
    if missing_indexes:
        raise RuntimeError(
            "task_questions migration incomplete; missing indexes: "
            + ", ".join(missing_indexes)
        )
