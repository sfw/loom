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
    correlation_id TEXT NOT NULL,                -- Groups related events
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    event_type TEXT NOT NULL,
    data TEXT NOT NULL,                          -- JSON payload
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id);
CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

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
