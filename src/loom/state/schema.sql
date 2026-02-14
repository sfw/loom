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
