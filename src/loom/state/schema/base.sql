-- Loom SQLite Schema
-- All tables for task management, memory, events, and learning.

-- Task metadata
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    context TEXT,                                -- JSON
    workspace_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    plan TEXT,                                   -- JSON serialized plan
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    approval_mode TEXT NOT NULL DEFAULT 'auto',
    callback_url TEXT,
    metadata TEXT                                -- JSON
);

-- Memory entries extracted from task execution
CREATE TABLE IF NOT EXISTS memory_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    subtask_id TEXT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    entry_type TEXT NOT NULL,                    -- decision, error, tool_result, etc.
    summary TEXT NOT NULL,                       -- 1-2 sentence summary (max 150 chars)
    detail TEXT,                                 -- Full content
    tags TEXT,                                   -- Comma-separated tags for retrieval
    relevance_to TEXT,                           -- Comma-separated subtask IDs
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_memory_task ON memory_entries(task_id);
CREATE INDEX IF NOT EXISTS idx_memory_task_subtask ON memory_entries(task_id, subtask_id);
CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(entry_type);
CREATE INDEX IF NOT EXISTS idx_memory_tags ON memory_entries(tags);

-- Event log (for replay and debugging)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    correlation_id TEXT NOT NULL,                -- Groups related events
    event_id TEXT NOT NULL DEFAULT '',           -- Stable unique emitted event id
    sequence INTEGER NOT NULL DEFAULT 0,         -- Monotonic ordering key per task/run scope
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    event_type TEXT NOT NULL,
    source_component TEXT NOT NULL DEFAULT '',
    schema_version INTEGER NOT NULL DEFAULT 1,
    data TEXT NOT NULL,                          -- JSON payload
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_events_task_sequence ON events(task_id, sequence);
CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON events(run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id) WHERE event_id <> '';

-- Learning database (Phase 2)
CREATE TABLE IF NOT EXISTS learned_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,                  -- subtask_success, model_failure, etc.
    pattern_key TEXT NOT NULL,                   -- Searchable pattern identifier
    data TEXT NOT NULL,                          -- JSON with details
    frequency INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT NOT NULL DEFAULT (datetime('now')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_patterns_type ON learned_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_key ON learned_patterns(pattern_key);

-- Cowork sessions
CREATE TABLE IF NOT EXISTS cowork_sessions (
    id TEXT PRIMARY KEY,
    workspace_path TEXT NOT NULL,
    model_name TEXT NOT NULL,
    system_prompt TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at TEXT NOT NULL DEFAULT (datetime('now')),
    total_tokens INTEGER DEFAULT 0,
    turn_count INTEGER DEFAULT 0,
    session_state TEXT,                              -- JSON: structured session state
    is_active INTEGER DEFAULT 1
);

-- Verbatim conversation turns (append-only, never compacted)
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL,                              -- user | assistant | tool | system
    content TEXT,                                    -- message text (verbatim, no truncation)
    tool_calls TEXT,                                 -- JSON array of tool calls (nullable)
    tool_call_id TEXT,                               -- for role=tool, the call this responds to
    tool_name TEXT,                                  -- for role=tool, which tool was called
    token_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES cowork_sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_ct_session ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_ct_session_turn ON conversation_turns(session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_ct_role ON conversation_turns(session_id, role);
CREATE INDEX IF NOT EXISTS idx_ct_tool_name ON conversation_turns(tool_name);

-- Cowork chat replay journal (UI-facing transcript events)
CREATE TABLE IF NOT EXISTS cowork_chat_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,                            -- JSON payload (versioned)
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES cowork_sessions(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cce_session_seq
    ON cowork_chat_events(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_cce_session_created
    ON cowork_chat_events(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cce_session_id
    ON cowork_chat_events(session_id, id);

-- Typed cowork memory index (marker-oriented conversation memory)
CREATE TABLE IF NOT EXISTS cowork_memory_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    entry_type TEXT NOT NULL,                   -- decision|proposal|research|...
    status TEXT NOT NULL DEFAULT 'active',      -- active|superseded|resolved|rejected
    summary TEXT NOT NULL,
    rationale TEXT DEFAULT '',
    topic TEXT DEFAULT '',
    tags_json TEXT DEFAULT '[]',
    tags_text TEXT DEFAULT '',
    entities_json TEXT DEFAULT '[]',
    entities_text TEXT DEFAULT '',
    source_turn_start INTEGER NOT NULL DEFAULT 0,
    source_turn_end INTEGER NOT NULL DEFAULT 0,
    source_roles_json TEXT DEFAULT '[]',
    evidence_excerpt TEXT DEFAULT '',
    supersedes_entry_id INTEGER,
    confidence REAL DEFAULT 0.0,
    fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES cowork_sessions(id),
    FOREIGN KEY (supersedes_entry_id) REFERENCES cowork_memory_entries(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cme_session_fingerprint
    ON cowork_memory_entries(session_id, fingerprint);
CREATE INDEX IF NOT EXISTS idx_cme_session_type_status_updated
    ON cowork_memory_entries(session_id, entry_type, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_cme_session_turn_range
    ON cowork_memory_entries(session_id, source_turn_start, source_turn_end);

CREATE TABLE IF NOT EXISTS cowork_memory_index_state (
    session_id TEXT PRIMARY KEY,
    last_indexed_turn INTEGER NOT NULL DEFAULT 0,
    index_version INTEGER NOT NULL DEFAULT 1,
    index_degraded INTEGER NOT NULL DEFAULT 0,
    last_indexed_at TEXT,
    last_error TEXT DEFAULT '',
    failure_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES cowork_sessions(id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS cowork_memory_fts USING fts5(
    summary,
    rationale,
    topic,
    tags_text,
    entities_text,
    evidence_excerpt,
    content='cowork_memory_entries',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS cowork_memory_entries_ai
AFTER INSERT ON cowork_memory_entries BEGIN
    INSERT INTO cowork_memory_fts(
        rowid,
        summary,
        rationale,
        topic,
        tags_text,
        entities_text,
        evidence_excerpt
    ) VALUES (
        new.id,
        new.summary,
        new.rationale,
        new.topic,
        new.tags_text,
        new.entities_text,
        new.evidence_excerpt
    );
END;

CREATE TRIGGER IF NOT EXISTS cowork_memory_entries_ad
AFTER DELETE ON cowork_memory_entries BEGIN
    INSERT INTO cowork_memory_fts(
        cowork_memory_fts,
        rowid,
        summary,
        rationale,
        topic,
        tags_text,
        entities_text,
        evidence_excerpt
    ) VALUES (
        'delete',
        old.id,
        old.summary,
        old.rationale,
        old.topic,
        old.tags_text,
        old.entities_text,
        old.evidence_excerpt
    );
END;

CREATE TRIGGER IF NOT EXISTS cowork_memory_entries_au
AFTER UPDATE ON cowork_memory_entries BEGIN
    INSERT INTO cowork_memory_fts(
        cowork_memory_fts,
        rowid,
        summary,
        rationale,
        topic,
        tags_text,
        entities_text,
        evidence_excerpt
    ) VALUES (
        'delete',
        old.id,
        old.summary,
        old.rationale,
        old.topic,
        old.tags_text,
        old.entities_text,
        old.evidence_excerpt
    );
    INSERT INTO cowork_memory_fts(
        rowid,
        summary,
        rationale,
        topic,
        tags_text,
        entities_text,
        evidence_excerpt
    ) VALUES (
        new.id,
        new.summary,
        new.rationale,
        new.topic,
        new.tags_text,
        new.entities_text,
        new.evidence_excerpt
    );
END;

-- Durable task run lifecycle (for crash-safe background execution)
CREATE TABLE IF NOT EXISTS task_runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',       -- queued|running|completed|failed|cancelled
    process_name TEXT DEFAULT '',
    lease_owner TEXT DEFAULT '',
    lease_expires_at TEXT,
    heartbeat_at TEXT,
    attempt INTEGER NOT NULL DEFAULT 1,
    started_at TEXT,
    ended_at TEXT,
    last_error TEXT DEFAULT '',
    metadata TEXT,                               -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_task_runs_task ON task_runs(task_id);
CREATE INDEX IF NOT EXISTS idx_task_runs_status ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_task_runs_lease ON task_runs(lease_expires_at);

-- Durable ask-user clarification lifecycle
CREATE TABLE IF NOT EXISTS task_questions (
    question_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    subtask_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',      -- pending|answered|timeout|cancelled
    request_payload TEXT NOT NULL,               -- JSON
    answer_payload TEXT,                         -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT,
    timeout_at TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_task_questions_task_status
    ON task_questions(task_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_task_questions_task_subtask
    ON task_questions(task_id, subtask_id, status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_task_questions_active_scope
    ON task_questions(task_id, subtask_id)
    WHERE status = 'pending';

-- Retry lineage and remediation queue state
CREATE TABLE IF NOT EXISTS subtask_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    subtask_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    tier INTEGER NOT NULL,
    retry_strategy TEXT NOT NULL DEFAULT 'generic',
    reason_code TEXT DEFAULT '',
    feedback TEXT DEFAULT '',
    error TEXT DEFAULT '',
    missing_targets TEXT DEFAULT '',             -- JSON array
    error_category TEXT DEFAULT '',
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT,                               -- JSON
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_subtask_attempts_task_subtask
    ON subtask_attempts(task_id, subtask_id, attempt);
CREATE INDEX IF NOT EXISTS idx_subtask_attempts_run ON subtask_attempts(run_id);

CREATE TABLE IF NOT EXISTS remediation_items (
    id TEXT PRIMARY KEY,                         -- rem-<id>
    task_id TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    subtask_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    reason_code TEXT DEFAULT '',
    verification_outcome TEXT DEFAULT '',
    feedback TEXT DEFAULT '',
    blocking INTEGER NOT NULL DEFAULT 0,
    critical_path INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'queued',        -- queued|running|resolved|failed|expired
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    base_backoff_seconds REAL NOT NULL DEFAULT 2.0,
    max_backoff_seconds REAL NOT NULL DEFAULT 30.0,
    next_attempt_at TEXT,
    ttl_at TEXT,
    last_error TEXT DEFAULT '',
    terminal_reason TEXT DEFAULT '',
    missing_targets TEXT DEFAULT '',             -- JSON array
    metadata TEXT,                               -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_remediation_items_task ON remediation_items(task_id);
CREATE INDEX IF NOT EXISTS idx_remediation_items_state ON remediation_items(state);
CREATE INDEX IF NOT EXISTS idx_remediation_items_due ON remediation_items(next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_remediation_items_subtask ON remediation_items(task_id, subtask_id);

CREATE TABLE IF NOT EXISTS remediation_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    remediation_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    subtask_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    phase TEXT NOT NULL DEFAULT 'done',          -- start|done
    outcome TEXT DEFAULT '',                     -- resolved|failed|...
    retry_strategy TEXT DEFAULT '',
    transient INTEGER NOT NULL DEFAULT 0,
    reason_code TEXT DEFAULT '',
    error TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (remediation_id) REFERENCES remediation_items(id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_remediation_attempts_remediation
    ON remediation_attempts(remediation_id, attempt);
CREATE INDEX IF NOT EXISTS idx_remediation_attempts_task
    ON remediation_attempts(task_id, subtask_id);

-- Mutating-tool idempotency ledger
CREATE TABLE IF NOT EXISTS tool_mutation_ledger (
    idempotency_key TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    subtask_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'success',      -- success|failure
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_tool_mutation_ledger_task
    ON tool_mutation_ledger(task_id, subtask_id, tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_mutation_ledger_run
    ON tool_mutation_ledger(run_id);

-- Phase iteration loop persistence
CREATE TABLE IF NOT EXISTS iteration_runs (
    loop_run_id TEXT PRIMARY KEY,               -- iter-<id>
    task_id TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    subtask_id TEXT NOT NULL,
    phase_id TEXT DEFAULT '',
    policy_snapshot TEXT NOT NULL,              -- JSON
    terminal_reason TEXT DEFAULT '',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    replan_count INTEGER NOT NULL DEFAULT 0,
    exhaustion_fingerprint TEXT DEFAULT '',
    metadata TEXT,                              -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_iteration_runs_task
    ON iteration_runs(task_id, subtask_id, created_at);
CREATE INDEX IF NOT EXISTS idx_iteration_runs_run
    ON iteration_runs(run_id);

CREATE TABLE IF NOT EXISTS iteration_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loop_run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    subtask_id TEXT NOT NULL,
    phase_id TEXT DEFAULT '',
    attempt_index INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',   -- completed|retrying|terminal
    summary TEXT DEFAULT '',
    gate_summary TEXT DEFAULT '',               -- JSON
    budget_snapshot TEXT DEFAULT '',            -- JSON
    metadata TEXT,                              -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (loop_run_id) REFERENCES iteration_runs(loop_run_id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_iteration_attempts_loop
    ON iteration_attempts(loop_run_id, attempt_index);
CREATE INDEX IF NOT EXISTS idx_iteration_attempts_task
    ON iteration_attempts(task_id, subtask_id);

CREATE TABLE IF NOT EXISTS iteration_gate_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    loop_run_id TEXT NOT NULL,
    attempt_id INTEGER,
    task_id TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    subtask_id TEXT NOT NULL,
    phase_id TEXT DEFAULT '',
    attempt_index INTEGER NOT NULL,
    gate_id TEXT NOT NULL,
    gate_type TEXT NOT NULL,
    status TEXT NOT NULL,                       -- pass|fail|unevaluable
    blocking INTEGER NOT NULL DEFAULT 0,
    reason_code TEXT DEFAULT '',
    measured_value TEXT DEFAULT '',             -- JSON-encoded scalar/object
    threshold_value TEXT DEFAULT '',            -- JSON-encoded scalar/object
    detail TEXT DEFAULT '',
    metadata TEXT,                              -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (loop_run_id) REFERENCES iteration_runs(loop_run_id),
    FOREIGN KEY (attempt_id) REFERENCES iteration_attempts(id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_iteration_gate_results_loop
    ON iteration_gate_results(loop_run_id, attempt_index);
CREATE INDEX IF NOT EXISTS idx_iteration_gate_results_task
    ON iteration_gate_results(task_id, subtask_id, gate_id);

-- Claim/evidence validity lineage
CREATE TABLE IF NOT EXISTS artifact_claims (
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
    metadata TEXT,                              -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_artifact_claims_task_subtask
    ON artifact_claims(task_id, subtask_id, created_at);
CREATE INDEX IF NOT EXISTS idx_artifact_claims_claim_id
    ON artifact_claims(claim_id);

CREATE TABLE IF NOT EXISTS claim_evidence_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    run_id TEXT DEFAULT '',
    subtask_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    link_type TEXT DEFAULT 'supporting',
    score REAL DEFAULT 0.0,
    metadata TEXT,                              -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_claim_evidence_links_task_subtask
    ON claim_evidence_links(task_id, subtask_id, created_at);
CREATE INDEX IF NOT EXISTS idx_claim_evidence_links_claim
    ON claim_evidence_links(claim_id);

CREATE TABLE IF NOT EXISTS claim_verification_results (
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
    metadata TEXT,                              -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_claim_verification_results_task_subtask
    ON claim_verification_results(task_id, subtask_id, created_at);
CREATE INDEX IF NOT EXISTS idx_claim_verification_results_claim
    ON claim_verification_results(claim_id);

CREATE TABLE IF NOT EXISTS artifact_validity_summaries (
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
    metadata TEXT,                              -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_artifact_validity_summaries_task_subtask
    ON artifact_validity_summaries(task_id, subtask_id, created_at);
