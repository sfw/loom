"""Migration step for claim/evidence validity lineage tables."""

from __future__ import annotations

from loom.state.migrations.runner import index_exists, table_exists

_REQUIRED_TABLES = (
    "artifact_claims",
    "claim_evidence_links",
    "claim_verification_results",
    "artifact_validity_summaries",
)
_REQUIRED_INDEXES = (
    "idx_artifact_claims_task_subtask",
    "idx_artifact_claims_claim_id",
    "idx_claim_evidence_links_task_subtask",
    "idx_claim_evidence_links_claim",
    "idx_claim_verification_results_task_subtask",
    "idx_claim_verification_results_claim",
    "idx_artifact_validity_summaries_task_subtask",
)


async def apply(conn) -> None:
    await conn.execute(
        """CREATE TABLE IF NOT EXISTS artifact_claims (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               task_id TEXT NOT NULL,
               run_id TEXT DEFAULT '',
               subtask_id TEXT NOT NULL,
               phase_id TEXT DEFAULT '',
               claim_id TEXT NOT NULL,
               claim_text TEXT NOT NULL,
               claim_type TEXT DEFAULT 'qualitative',
               criticality TEXT DEFAULT 'important',
               lifecycle_state TEXT NOT NULL,
               reason_code TEXT DEFAULT '',
               metadata TEXT,
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               FOREIGN KEY (task_id) REFERENCES tasks(id)
           )""",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifact_claims_task_subtask "
        "ON artifact_claims(task_id, subtask_id, created_at)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifact_claims_claim_id "
        "ON artifact_claims(claim_id)",
    )

    await conn.execute(
        """CREATE TABLE IF NOT EXISTS claim_evidence_links (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               task_id TEXT NOT NULL,
               run_id TEXT DEFAULT '',
               subtask_id TEXT NOT NULL,
               claim_id TEXT NOT NULL,
               evidence_id TEXT NOT NULL,
               link_type TEXT DEFAULT 'supporting',
               score REAL DEFAULT 0.0,
               metadata TEXT,
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               FOREIGN KEY (task_id) REFERENCES tasks(id)
           )""",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_claim_evidence_links_task_subtask "
        "ON claim_evidence_links(task_id, subtask_id, created_at)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_claim_evidence_links_claim "
        "ON claim_evidence_links(claim_id)",
    )

    await conn.execute(
        """CREATE TABLE IF NOT EXISTS claim_verification_results (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               task_id TEXT NOT NULL,
               run_id TEXT DEFAULT '',
               subtask_id TEXT NOT NULL,
               phase_id TEXT DEFAULT '',
               claim_id TEXT NOT NULL,
               status TEXT NOT NULL,
               reason_code TEXT DEFAULT '',
               verifier TEXT DEFAULT '',
               confidence REAL DEFAULT 0.0,
               metadata TEXT,
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               FOREIGN KEY (task_id) REFERENCES tasks(id)
           )""",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_claim_verification_results_task_subtask "
        "ON claim_verification_results(task_id, subtask_id, created_at)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_claim_verification_results_claim "
        "ON claim_verification_results(claim_id)",
    )

    await conn.execute(
        """CREATE TABLE IF NOT EXISTS artifact_validity_summaries (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               task_id TEXT NOT NULL,
               run_id TEXT DEFAULT '',
               subtask_id TEXT NOT NULL,
               phase_id TEXT DEFAULT '',
               extracted_count INTEGER NOT NULL DEFAULT 0,
               supported_count INTEGER NOT NULL DEFAULT 0,
               contradicted_count INTEGER NOT NULL DEFAULT 0,
               insufficient_evidence_count INTEGER NOT NULL DEFAULT 0,
               pruned_count INTEGER NOT NULL DEFAULT 0,
               supported_ratio REAL NOT NULL DEFAULT 0.0,
               gate_decision TEXT DEFAULT '',
               reason_code TEXT DEFAULT '',
               metadata TEXT,
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               FOREIGN KEY (task_id) REFERENCES tasks(id)
           )""",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_artifact_validity_summaries_task_subtask "
        "ON artifact_validity_summaries(task_id, subtask_id, created_at)",
    )


async def verify(conn) -> None:
    missing_tables = [name for name in _REQUIRED_TABLES if not await table_exists(conn, name)]
    if missing_tables:
        raise RuntimeError(
            "validity lineage migration incomplete; missing tables: "
            + ", ".join(missing_tables)
        )
    missing_indexes = [name for name in _REQUIRED_INDEXES if not await index_exists(conn, name)]
    if missing_indexes:
        raise RuntimeError(
            "validity lineage migration incomplete; missing indexes: "
            + ", ".join(missing_indexes)
        )
