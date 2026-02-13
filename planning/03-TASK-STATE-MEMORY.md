# Spec 03: Task State and Memory Architecture

## Overview

Loom replaces lossy conversation-history compaction with a three-layer structured memory system. Information is never lost. The LLM always knows full project state through a compact YAML representation, and detailed history is retrievable on demand via deterministic SQL queries.

## Three-Layer Architecture

```
┌────────────────────────────────────────────────────────┐
│ LAYER 1: Structured Task State                         │
│ Always in context. ~500-1500 tokens. Updated in place. │
│ Format: YAML                                           │
│ Storage: File on disk (per-task)                       │
│ Never compacted. Never summarized. Always current.     │
├────────────────────────────────────────────────────────┤
│ LAYER 2: Structured Memory Archive                     │
│ Retrieved on demand. Entries extracted at write time.   │
│ Format: Structured rows in SQLite                      │
│ Query: Deterministic SQL by subtask, type, tags        │
│ Full detail preserved. No lossy summarization.         │
├────────────────────────────────────────────────────────┤
│ LAYER 3: Vector Search (Optional, Phase 2)             │
│ Fallback for fuzzy recall when tag-based misses.       │
│ Embed the summary field from Layer 2 entries.          │
│ Only needed if Layer 2 retrieval proves insufficient.  │
└────────────────────────────────────────────────────────┘
```

## Layer 1: Always-In-Context Task State

This YAML object is included in every prompt sent to the model. It is the model's "working memory" and must be kept compact.

### Schema

```yaml
task:
  id: "a1b2c3d4"
  goal: "Migrate database from MySQL to PostgreSQL"
  status: executing
  workspace: "/Users/scott/projects/myapp"

plan:
  version: 3
  last_replanned: "2026-02-13T14:30:00"

subtasks:
  - id: backup
    status: completed
    summary: "Full backup at /artifacts/backup.sql (2.3GB, verified checksum)"
  - id: schema-convert
    status: in_progress
    summary: "Converting 47 tables. 31 done. Blocked on ENUM type mapping."
    active_issue: "MySQL ENUM needs custom type mapping for PostgreSQL"
  - id: data-migrate
    status: pending
    depends_on: [schema-convert]
  - id: verify-data
    status: pending
    depends_on: [data-migrate]
  - id: update-app-config
    status: pending
    depends_on: [verify-data]

decisions_log:
  - "Using pgloader for bulk data transfer (handles type coercion)"
  - "Keeping ENUM as VARCHAR with CHECK constraints (simpler than custom types)"

errors_encountered:
  - subtask: schema-convert
    error: "SPATIAL index not supported in pgloader"
    resolution: "Manual post-migration index creation queued"

workspace_changes:
  files_created: 3
  files_modified: 12
  last_change: "2026-02-13T14:32:00"
```

### Implementation: task_state.py

```python
@dataclass
class TaskState:
    """
    Manages the always-in-context YAML state for a task.
    Updated in place after every subtask completion.
    """

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def create(self, task: Task) -> None:
        """Create initial state file for a new task."""

    def load(self, task_id: str) -> dict:
        """Load state from disk."""

    def save(self, task: Task) -> None:
        """Write current state to disk. Atomic write (write to temp, rename)."""

    def to_yaml(self, task: Task) -> str:
        """Render task state as YAML string for prompt injection."""

    def update_subtask(self, task_id: str, subtask_id: str, **updates) -> None:
        """Update a specific subtask's state fields."""

    def add_decision(self, task_id: str, decision: str) -> None:
        """Append to the decisions log."""

    def add_error(self, task_id: str, subtask_id: str, error: str, resolution: str = None) -> None:
        """Record an error encountered during execution."""
```

### Size Budget

The YAML state must stay under ~1500 tokens to leave room for the rest of the prompt. This means:
- Subtask summaries: max 100 characters each
- Decisions log: max 10 entries, oldest pruned
- Errors: max 5 entries, oldest pruned
- If the plan has many subtasks (>15), only show completed count + current + next 3 pending

## Layer 2: Structured Memory Archive (SQLite)

### Database Schema

```sql
-- schema.sql

-- Memory entries extracted from task execution
CREATE TABLE memory_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    subtask_id TEXT,
    timestamp TEXT NOT NULL,                     -- ISO 8601
    entry_type TEXT NOT NULL,                    -- see Entry Types below
    summary TEXT NOT NULL,                       -- 1-2 sentence summary
    detail TEXT,                                 -- Full content (tool output, model reasoning, etc.)
    tags TEXT,                                   -- Comma-separated tags for retrieval
    relevance_to TEXT,                           -- Comma-separated subtask IDs this matters for
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_memory_task ON memory_entries(task_id);
CREATE INDEX idx_memory_task_subtask ON memory_entries(task_id, subtask_id);
CREATE INDEX idx_memory_type ON memory_entries(entry_type);
CREATE INDEX idx_memory_tags ON memory_entries(tags);

-- Task metadata
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    context TEXT,                                -- JSON
    workspace_path TEXT,
    status TEXT NOT NULL,
    plan TEXT,                                   -- JSON serialized plan
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    approval_mode TEXT NOT NULL DEFAULT 'auto',
    callback_url TEXT,
    metadata TEXT                                -- JSON
);

-- Event log (for replay and debugging)
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    correlation_id TEXT NOT NULL,                -- Groups related events
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    data TEXT NOT NULL,                          -- JSON payload
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE INDEX idx_events_task ON events(task_id);
CREATE INDEX idx_events_correlation ON events(correlation_id);
CREATE INDEX idx_events_type ON events(event_type);

-- Learning database (Phase 2)
CREATE TABLE learned_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,                  -- "subtask_success", "model_failure", "prompt_effective"
    pattern_key TEXT NOT NULL,                   -- Searchable pattern identifier
    data TEXT NOT NULL,                          -- JSON with details
    frequency INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_patterns_type ON learned_patterns(pattern_type);
CREATE INDEX idx_patterns_key ON learned_patterns(pattern_key);
```

### Entry Types

| Type | Description | Example |
|------|-------------|---------|
| `decision` | A decision made during execution | "Using pgloader instead of manual SQL" |
| `error` | An error encountered and how it was handled | "SPATIAL index not supported" |
| `tool_result` | Significant tool output worth remembering | "Grep found 47 references to deprecated API" |
| `user_instruction` | User-provided correction or steering | "Use PostgreSQL CHECK constraints for enums" |
| `discovery` | New information learned during execution | "Database has 3 undocumented tables" |
| `artifact` | A file or output produced | "Generated migration script at /output/migrate.sql" |
| `context` | Background context relevant to the task | "Project uses Express.js with TypeORM" |

### Write-Time Extraction

Memory extraction happens immediately after each subtask completes, while the model still has full context. This is critical — extracting later loses context.

```python
class MemoryManager:
    """
    Manages Layer 2 structured memory. Entries are extracted at write time
    by a dedicated model call, not at query time.
    """

    async def extract_and_store(
        self,
        task_id: str,
        subtask_id: str,
        tool_calls: list[ToolCallRecord],
        model_output: str,
    ) -> list[MemoryEntry]:
        """
        After a subtask completes, extract structured memory entries.
        Uses the utility model (Qwen3 8B) for extraction.
        """
        # Build extraction prompt
        prompt = self._prompt_assembler.build_extractor_prompt(
            subtask_id=subtask_id,
            tool_calls=tool_calls,
            model_output=model_output,
        )

        # Use small/fast model for extraction
        model = self._model_router.select(tier=1, role="extractor")
        response = await model.complete([{"role": "user", "content": prompt}])

        # Parse structured entries from response
        # Response format is JSON array of entries
        entries = self._parse_entries(response.text)

        # Store in SQLite
        for entry in entries:
            entry.task_id = task_id
            entry.subtask_id = subtask_id
            await self._db.insert_memory_entry(entry)

        return entries

    async def query_relevant(
        self,
        task_id: str,
        subtask_id: str,
    ) -> list[MemoryEntry]:
        """
        Retrieve memory entries relevant to a specific subtask.
        Uses deterministic SQL queries, not embeddings.
        """
        return await self._db.query(
            """
            SELECT * FROM memory_entries
            WHERE task_id = ?
            AND (
                subtask_id = ?
                OR relevance_to LIKE ?
                OR entry_type IN ('decision', 'error', 'user_instruction')
            )
            ORDER BY timestamp DESC
            LIMIT 20
            """,
            (task_id, subtask_id, f"%{subtask_id}%"),
        )

    async def query(
        self,
        task_id: str,
        entry_type: str = None,
        subtask_id: str = None,
        tags: list[str] = None,
    ) -> list[MemoryEntry]:
        """General-purpose memory query with filters."""
        # Build query dynamically based on provided filters
        ...

    async def search(self, query: str) -> list[MemoryEntry]:
        """
        Full-text search across all memory entries.
        Uses SQLite FTS5 if available, falls back to LIKE.
        """
        ...
```

### Extraction Prompt Template

The extraction prompt asks the utility model to identify structured entries from subtask execution:

```yaml
# prompts/templates/extractor.yaml
system: |
  You are a memory extraction assistant. Your job is to extract structured
  information from a completed subtask execution.

  Analyze the tool calls and output, then produce a JSON array of memory entries.
  Each entry has:
  - type: one of "decision", "error", "tool_result", "discovery", "artifact", "context"
  - summary: 1-2 sentence summary (max 150 chars)
  - detail: full relevant content
  - tags: comma-separated keywords for retrieval
  - relevance_to: comma-separated subtask IDs this information might matter for

  Only extract entries that would be useful for future subtasks.
  Do NOT extract trivial or redundant information.
  Respond with ONLY a JSON array, no explanation.

user: |
  Subtask: {subtask_id}

  Tool calls and results:
  {tool_calls_formatted}

  Model output:
  {model_output}
```

## Layer 3: Vector Search (Phase 2)

Not implemented in V1. If Layer 2's tag-based retrieval proves insufficient, add:
- Embed the `summary` field of each memory entry using a small embedding model
- Store embeddings in SQLite using `sqlite-vec` extension or ChromaDB
- Query by similarity when deterministic queries miss
- This is a fallback, not the primary retrieval mechanism

## Database Access Pattern

All database access goes through an async wrapper to avoid blocking the event loop:

```python
class Database:
    def __init__(self, db_path: Path):
        self._db_path = db_path

    async def initialize(self):
        """Create tables from schema.sql if they don't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            schema = (Path(__file__).parent / "schema.sql").read_text()
            await db.executescript(schema)
            await db.commit()

    async def insert_memory_entry(self, entry: MemoryEntry) -> int:
        """Insert a memory entry and return its ID."""
        ...

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as dicts."""
        ...
```

## Acceptance Criteria

- [ ] Task state YAML renders correctly and stays under 1500 tokens for typical tasks
- [ ] State file is atomically written (no corruption on crash)
- [ ] SQLite schema creates all tables and indexes on first run
- [ ] Memory extraction produces structured entries from subtask execution data
- [ ] Memory queries correctly filter by task, subtask, type, and tags
- [ ] Write-time extraction uses the utility model, not the primary model
- [ ] Database access is fully async (no blocking calls)
- [ ] Memory entries are persisted across engine restarts
- [ ] State updates are reflected immediately in subsequent prompt assembly
- [ ] Decision and error logs in Layer 1 state are pruned when they exceed limits
