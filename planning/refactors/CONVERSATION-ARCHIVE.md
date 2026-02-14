# Cowork Mode: Non-Lossy History + Task Delegation

## Problem Statement

Two problems, one root cause: cowork mode is isolated from both its own
history and from the task engine.

**Problem 1: Lossy history.** CoworkSession stores conversation history as
an in-memory `list[dict]`. When messages exceed 200, a sliding window drops
old messages permanently. When the process exits, everything is lost. This
is compaction — exactly what the project's core philosophy rejects:

> "Structured state over conversation — No chat history rot. Deterministic
> retrieval from SQLite, not lossy compaction of message logs."

**Problem 2: No task delegation.** Cowork and task mode are completely
separate worlds. A user in cowork who says "migrate this to TypeScript"
gets the model trying to do everything in a flat tool loop — no planning,
no verification, no parallelism. To use task mode, the user has to quit
cowork, switch modes, and lose the conversation.

This is wrong. Look at how Claude Code actually works: the conversation IS
the interface. Within that conversation, complex work happens via subagents
(the Task tool). The user never leaves the conversation. They discuss, they
say "do it," the agent decomposes and executes, results come back, they
discuss the results. One continuous flow.

Loom should work the same way: **cowork is the primary interface, and task
mode is an execution capability that cowork can invoke when work requires
planning, decomposition, verification, or parallelism.**

---

## Design Goals

1. **Zero information loss** — every message persisted verbatim to SQLite
2. **No compaction** — old messages are archived, never summarized or discarded
3. **Lean context window** — system prompt + session state + recent turns +
   retrieved context, all token-budget-aware
4. **Model-driven retrieval** — a tool the model calls to query the archive
   when it needs prior context ("what did the user say about auth?")
5. **Session resumption** — close terminal, reopen, continue where you left off
6. **Cowork delegates to task mode** — the model can submit complex work to
   the orchestrator, get results back, and continue the conversation
7. **Single continuous experience** — no mode switching, no context loss

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                   CoworkSession                           │
│                                                           │
│  Context Window                                           │
│  ┌──────────────┐ ┌───────────┐ ┌───────────────────────┐│
│  │ System Prompt │ │  Session  │ │   Recent N Turns      ││
│  │   + recall   │ │   State   │ │   (token-budgeted)    ││
│  │  guidelines  │ │  (YAML)   │ │                       ││
│  └──────────────┘ └───────────┘ └───────────────────────┘│
│                                                           │
│  Tools                                                    │
│  ┌──────────────────┐  ┌─────────────────────────────┐   │
│  │ conversation_    │  │ delegate_task               │   │
│  │ recall           │  │                             │   │
│  │ (query archive)  │  │ (submit to orchestrator,    │   │
│  │                  │  │  stream progress, return    │   │
│  │                  │  │  results to conversation)   │   │
│  └────────┬─────────┘  └──────────────┬──────────────┘   │
│           │                           │                   │
└───────────┼───────────────────────────┼───────────────────┘
            │                           │
       reads│                     submits task
            │                           │
            ▼                           ▼
  ┌───────────────────┐   ┌──────────────────────────────┐
  │  SQLite Database  │   │  Orchestrator (in-process)   │
  │                   │   │                              │
  │ conversation_turns│   │  plan → execute → verify     │
  │ cowork_sessions   │   │  parallel subtasks           │
  │ memory_entries    │   │  retry / escalation          │
  │ events            │   │  learning                    │
  └───────────────────┘   └──────────────────────────────┘
```

The key insight: **cowork mode is not a separate mode — it's the primary
interface.** Task mode's orchestrator is an execution engine that cowork
can invoke, like how Claude Code's Task tool spawns subagents. The user
never leaves the conversation.

---

## Component 1: Conversation Store

### New Table: `conversation_turns`

```sql
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,          -- groups turns into sessions
    turn_number INTEGER NOT NULL,      -- monotonic within session
    role TEXT NOT NULL,                 -- user | assistant | tool | system
    content TEXT,                       -- message text (verbatim, no truncation)
    tool_calls TEXT,                    -- JSON array of tool calls (nullable)
    tool_call_id TEXT,                  -- for role=tool, the call this responds to
    tool_name TEXT,                     -- for role=tool, which tool was called
    token_count INTEGER DEFAULT 0,     -- estimated tokens for this message
    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    UNIQUE(session_id, turn_number)
);

CREATE INDEX IF NOT EXISTS idx_ct_session ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_ct_session_turn ON conversation_turns(session_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_ct_role ON conversation_turns(session_id, role);
CREATE INDEX IF NOT EXISTS idx_ct_tool_name ON conversation_turns(tool_name);
```

**Why a separate table, not `memory_entries`?**
- `memory_entries` is designed for extracted summaries with `entry_type`,
  `tags`, `relevance_to` — metadata that doesn't map to raw conversation turns.
- Conversation turns need `role`, `tool_calls`, `tool_call_id` — fields that
  don't exist on `memory_entries`.
- Keeping them separate means task mode's memory system is untouched.
- The conversation store is append-only; `memory_entries` has complex
  query patterns (relevance, tags). Mixing them would complicate both.

### New Table: `cowork_sessions`

```sql
CREATE TABLE IF NOT EXISTS cowork_sessions (
    id TEXT PRIMARY KEY,               -- UUID
    workspace_path TEXT NOT NULL,
    model_name TEXT NOT NULL,
    system_prompt TEXT,                 -- the full system prompt used
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at TEXT NOT NULL DEFAULT (datetime('now')),
    total_tokens INTEGER DEFAULT 0,
    turn_count INTEGER DEFAULT 0,
    session_state TEXT,                 -- JSON: running structured state
    is_active INTEGER DEFAULT 1
);
```

### ConversationStore Class

New file: `src/loom/state/conversation_store.py`

```python
class ConversationStore:
    """Append-only persistence for cowork conversation history.

    Every message sent or received in a CoworkSession is written here
    synchronously (write-through). The in-memory message list in
    CoworkSession becomes a cache of the most recent N turns; the
    store holds everything.
    """

    async def initialize(self, db_path: Path) -> None
    async def create_session(self, workspace, model, system_prompt) -> str
    async def append_turn(self, session_id, role, content, ...) -> int
    async def get_turns(self, session_id, offset=0, limit=50) -> list[dict]
    async def search_turns(self, session_id, query: str) -> list[dict]
    async def get_turn_count(self, session_id) -> int
    async def get_session(self, session_id) -> dict | None
    async def list_sessions(self, workspace: str | None) -> list[dict]
    async def update_session_state(self, session_id, state: dict) -> None
    async def resume_session(self, session_id) -> list[dict]  # load recent turns
```

### Write-Through Pattern

Every `self._messages.append(...)` in `CoworkSession` gets a corresponding
`await self._store.append_turn(...)`. The in-memory list is the hot cache;
SQLite is the durable store. No batch writes, no async background flushes —
synchronous write-through so nothing is ever lost.

```python
# Before (current):
self._messages.append({"role": "user", "content": user_message})

# After:
self._messages.append({"role": "user", "content": user_message})
await self._store.append_turn(
    session_id=self._session_id,
    role="user",
    content=user_message,
)
```

---

## Component 2: Session State (Layer 1 for Cowork)

Task mode has Layer 1 YAML state that's always in context. Cowork mode
needs an equivalent — a lightweight, structured summary of the session
that stays in every prompt. This is NOT a lossy summary of conversation;
it's structured metadata about what has happened.

### Session State Schema

```yaml
session:
  id: "abc123"
  workspace: /home/user/project
  model: claude-sonnet-4-5-20250929
  turn_count: 47
  total_tokens: 124500

files_touched:
  - path: src/auth.py
    action: edited
    turn: 12
  - path: tests/test_auth.py
    action: created
    turn: 15

key_decisions:
  - "Using JWT for authentication (turn 5)"
  - "PostgreSQL over SQLite for prod (turn 8)"

current_focus: "Implementing refresh token rotation"

errors_resolved:
  - "ImportError: jwt module — fixed by adding PyJWT to requirements (turn 14)"
```

**Size budget**: ~500-1000 tokens. Pruned like task mode's Layer 1:
- `files_touched`: last 20 entries
- `key_decisions`: last 10 entries
- `errors_resolved`: last 5 entries

### How Session State Gets Updated

The model updates session state via a lightweight tool (`session_state`)
or we extract it automatically. Automatic extraction is simpler and more
reliable:

**Automatic extraction approach:**
- After each turn completes, scan the turn's tool calls:
  - `write_file` / `edit_file` → add to `files_touched`
  - `shell_execute` with non-zero exit → add to errors if subsequently fixed
  - `git_command commit` → record in decisions
- `current_focus` updated from the user's most recent message (first line,
  truncated to 100 chars)
- No LLM call needed — purely mechanical extraction from tool results

### Session State Injection

The session state YAML is injected into the system prompt after the static
guidelines, before the conversation messages. It's updated in-place on
each turn (not appended as a new message).

```python
def _build_system_prompt(self) -> str:
    return f"""\
{self._static_system_prompt}

## Session State
{self._session_state.to_yaml()}
"""
```

---

## Component 3: `conversation_recall` Tool + Context Awareness

### The Retrieval Problem

A passive tool alone isn't sufficient. The model has to *decide* to call
`conversation_recall` — but if it's already forgotten the context (because
it fell off the window), how does it know to look? Three scenarios:

1. **Model knows it needs history** — User says "remember when we discussed
   auth?" Model recognizes the reference, calls the tool. Works naturally.

2. **Model doesn't realize it's missing context** — User says "now add tests
   for that" and the relevant turns are gone. Model hallucinates or asks the
   user to repeat themselves.

3. **Model needs context it doesn't know exists** — User stated a constraint
   200 turns ago ("never use ORM, raw SQL only"). Model doesn't know to look.

### Three-Layer Solution

**Layer A: Session state handles scenario 3.** Key decisions, constraints,
and architectural choices are captured in the session state YAML that's
*always* in context. "Never use ORM" gets recorded in `key_decisions` the
turn it's stated and stays there permanently. The model doesn't need to
recall it — it never leaves the prompt. This is not a lossy summary of
conversation; it's a structured register of facts that matter.

**Layer B: Dangling reference detection handles scenario 2.** When the
user's message contains references to prior context that isn't in the
recent window, the system injects a hint before the model responds:

```python
def _maybe_inject_recall_hint(self, user_message: str) -> str | None:
    """If the user references something not in recent context, hint the model."""
    indicators = [
        "like we discussed", "as before", "remember when",
        "go back to", "that file", "that error", "that function",
        "what we did", "earlier", "the previous", "as I said",
        "like I mentioned", "that thing", "we already",
    ]
    if any(phrase in user_message.lower() for phrase in indicators):
        return (
            "[System: The user may be referencing earlier conversation "
            "that is no longer in your context window. Use the "
            "conversation_recall tool to search for relevant context "
            "before proceeding.]"
        )
    return None
```

This is a heuristic, but it fails safe — a spurious nudge costs one
unnecessary tool call. The alternative (an LLM call to detect dangling
references) adds latency to every single turn.

**Layer C: System prompt handles scenario 1 and general awareness.** The
model is instructed: "Your context window contains only recent turns. The
full session history is in the archive. Use `conversation_recall` whenever
you're unsure about prior context or when the user references earlier work."
Good models are highly responsive to this instruction.

### The Tool

`conversation_recall` is the active retrieval mechanism for when the model
knows (or is nudged) that it needs prior context.

### Tool Definition

New file: `src/loom/tools/conversation_recall.py`

```python
class ConversationRecallTool(Tool):
    """Search and retrieve past conversation messages from the session archive.

    Use this when you need to recall:
    - What the user said earlier about a topic
    - The output of a previous tool call
    - A decision or discussion from earlier in the session
    - Code that was read/written earlier

    The full conversation history is preserved in the archive — nothing
    is ever lost. Use this tool to access any part of it.
    """

    name = "conversation_recall"
    description = (
        "Search the conversation archive for past messages. "
        "Use 'search' to find messages by keyword, "
        "'range' to get a specific range of turns, "
        "or 'tool_calls' to find past tool executions by name."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "range", "tool_calls", "summary"],
                "description": (
                    "search: full-text search across all messages. "
                    "range: get turns by number range. "
                    "tool_calls: find past calls to a specific tool. "
                    "summary: get a structured summary of the full session."
                ),
            },
            "query": {
                "type": "string",
                "description": "Search query (for 'search' action).",
            },
            "tool_name": {
                "type": "string",
                "description": "Tool name to filter by (for 'tool_calls' action).",
            },
            "start_turn": {
                "type": "integer",
                "description": "Start of turn range (for 'range' action).",
            },
            "end_turn": {
                "type": "integer",
                "description": "End of turn range, inclusive (for 'range' action).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default 10.",
            },
        },
        "required": ["action"],
    }
```

### Actions

**`search`** — Full-text search via SQL LIKE across `content` field.
Returns matching turns with surrounding context (1 turn before and after).
Results sorted by relevance (exact match > partial > content-only).

```sql
SELECT * FROM conversation_turns
WHERE session_id = ? AND content LIKE ?
ORDER BY turn_number DESC
LIMIT ?
```

**`range`** — Direct retrieval by turn number range. For when the model
knows which turns it wants ("show me turns 5-10").

```sql
SELECT * FROM conversation_turns
WHERE session_id = ? AND turn_number BETWEEN ? AND ?
ORDER BY turn_number ASC
```

**`tool_calls`** — Find previous executions of a specific tool. Useful
for "what happened when we ran the tests?" or "show me the last time we
read that file."

```sql
SELECT * FROM conversation_turns
WHERE session_id = ? AND tool_name = ?
ORDER BY turn_number DESC
LIMIT ?
```

**`summary`** — Returns the session state YAML plus high-level stats
(turn count, tokens used, files touched, decisions made). No LLM call —
purely structured data from the session state.

### Output Format

Results are formatted as compact text, not raw JSON, to be token-efficient:

```
[Turn 12] user:
  Can you add JWT refresh token support?

[Turn 13] assistant:
  I'll add refresh token rotation. Let me read the current auth module first.

[Turn 13] tool:read_file(path="src/auth.py")
  [245 lines — content truncated, use range action for full output]

[Turn 14] assistant:
  I see the current implementation uses simple access tokens...
```

### Output Budget

The tool enforces a **max output size of 8000 tokens** (~32KB). If a
result set would exceed this, it truncates tool result contents first
(they're the largest), then truncates oldest messages. The model can
always follow up with a more specific query.

### Wiring

The tool needs a reference to the `ConversationStore`. Since tools are
instantiated once and registered in the `ToolRegistry`, we pass the store
at construction time:

```python
recall_tool = ConversationRecallTool(store=conversation_store, session_id=session_id)
tools.register(recall_tool)
```

This is the same pattern used by `TaskTrackerTool` (holds mutable state
in the instance).

---

## Component 4: Token-Aware Context Assembly

Replace the current count-based `_context_window()` with token-aware
assembly that respects actual model limits.

### Current State

```python
def _context_window(self) -> list[dict]:
    # Just keeps last N messages by count
    if len(self._messages) <= self._max_context:
        return list(self._messages)
    system = [self._messages[0]] if self._messages[0]["role"] == "system" else []
    rest = self._messages[len(system):]
    trimmed = rest[-(self._max_context - len(system)):]
    return system + trimmed
```

### New Approach

```python
def _context_window(self) -> list[dict]:
    """Assemble context window within token budget.

    Priority order:
    1. System prompt (always included, ~400 tokens)
    2. Session state YAML (always included, ~500-1000 tokens)
    3. Recent conversation turns (as many as fit)
    4. Retrieved archive context (if conversation_recall was used)

    Total budget: model's context window minus reserved output tokens.
    """
    budget = self._max_tokens - self._output_reserve  # e.g., 200K - 8K
    used = 0

    # 1. System prompt (always)
    system_msg = self._build_system_message()
    used += estimate_tokens(system_msg["content"])

    # 2. Recent turns — walk backward from most recent, adding turns
    #    until budget is exhausted
    recent = []
    for msg in reversed(self._messages[1:]):  # skip system
        msg_tokens = estimate_tokens_message(msg)
        if used + msg_tokens > budget:
            break
        recent.insert(0, msg)
        used += msg_tokens

    return [system_msg] + recent
```

### Token Estimation

Use the existing `estimate_tokens()` heuristic (len // 4) for now. This
is good enough for context budgeting — we're not doing precise billing,
just making sure we don't exceed the window. Can upgrade to tiktoken or
anthropic's token counter later if precision matters.

### `_maybe_trim()` Changes

The trim method no longer discards messages — it only manages the
in-memory cache size. Messages beyond the cache are still in SQLite
and retrievable via `conversation_recall`.

```python
def _maybe_trim(self) -> None:
    """Trim in-memory message cache if it's very large.

    This does NOT lose data — all messages are persisted in the
    conversation store. This just keeps RAM usage bounded.
    """
    max_cache = self._max_context * 3  # generous in-memory buffer
    if len(self._messages) <= max_cache:
        return
    system = [self._messages[0]] if self._messages[0]["role"] == "system" else []
    self._messages = system + self._messages[-(self._max_context * 2):]
```

---

## Component 5: Session Resumption

### Resume Flow

```
$ loom cowork -w /path/to/project

  Previous sessions for this workspace:
  [1] 2025-01-15 14:30 — 47 turns, model: claude-sonnet
  [2] Start new session

  > 1

  Resuming session abc123 (47 turns in archive)...
  Loading recent context...

  >
```

### Implementation

1. On startup, query `cowork_sessions` for the workspace
2. If sessions exist, show a selection prompt (or auto-resume most recent)
3. Load the last N messages from `conversation_turns` into `self._messages`
4. Rebuild session state from the stored `session_state` JSON
5. Continue the conversation loop as normal

```python
async def resume(self, session_id: str) -> None:
    """Resume a previous session from the archive."""
    session = await self._store.get_session(session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    self._session_id = session_id
    self._session_state = SessionState.from_json(session["session_state"])
    self._total_tokens = session["total_tokens"]

    # Load recent turns into the in-memory cache
    recent = await self._store.resume_session(session_id)
    self._messages = [self._build_system_message()] + recent
```

---

## Component 6: Task Delegation (`delegate_task` Tool)

This is the bridge between cowork and task mode. When the model in cowork
recognizes that work requires decomposition, verification, or parallelism,
it delegates to the orchestrator — exactly like Claude Code's Task tool
spawns subagents.

### How It Works Today (Claude Code)

```
User: "Migrate this Express app to TypeScript"
Claude Code (in conversation):
  1. Spawns Task subagent: "Research migration approach"
  2. Spawns Task subagent: "Find all .js files and categorize"
  3. Gets results back into conversation
  4. Discusses plan with user
  5. Spawns Task subagent: "Execute migration for src/routes/"
  6. Spawns Task subagent: "Execute migration for src/models/"
  7. Gets results, runs tests, discusses with user
```

The user never leaves the conversation. The subagents are invisible
plumbing. The conversation is continuous.

### How It Should Work (Loom)

```
User: "Migrate this Express app to TypeScript"
Loom cowork model:
  1. Calls delegate_task(goal="Analyze all JS files, categorize by
     migration complexity, identify shared types")
     → Orchestrator plans, executes, verifies, returns results
  2. Model discusses results with user in conversation
  3. User says "looks good, do it"
  4. Calls delegate_task(goal="Migrate src/routes/ to TypeScript,
     preserve all existing tests", context={constraints from discussion})
     → Orchestrator executes with parallel subtasks + verification
  5. Results come back, model reports to user
  6. User says "now run the tests"
  7. Model runs shell_execute directly (simple, no delegation needed)
```

### The Tool

New file: `src/loom/tools/delegate_task.py`

```python
class DelegateTaskTool(Tool):
    """Delegate complex work to Loom's orchestration engine.

    Use this when work requires:
    - Breaking down into multiple steps with dependencies
    - Verification of each step's output
    - Parallel execution of independent steps
    - Structured planning before execution

    For simple operations (read a file, run a command, edit code),
    use the direct tools instead. Delegation adds overhead — only
    use it when the task is genuinely complex.
    """

    name = "delegate_task"
    description = (
        "Submit complex multi-step work to the task orchestrator. "
        "The orchestrator will plan, decompose into subtasks, execute "
        "with verification, and return results. Use for tasks that need "
        "decomposition, parallel execution, or step-by-step verification. "
        "Simple operations should use direct tools instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What needs to be accomplished. Be specific — include "
                    "file paths, constraints, and acceptance criteria."
                ),
            },
            "context": {
                "type": "object",
                "description": (
                    "Additional context from the conversation: constraints, "
                    "decisions, preferences, files already discussed."
                ),
            },
            "wait": {
                "type": "boolean",
                "description": (
                    "If true (default), block until task completes and "
                    "return full results. If false, return task_id for "
                    "async monitoring."
                ),
                "default": True,
            },
        },
        "required": ["goal"],
    }
```

### In-Process vs HTTP

The MCP server communicates with the orchestrator over HTTP (because it's
designed for external agents in a separate process). Cowork doesn't need
this indirection — the orchestrator can be instantiated and called directly
in the same process.

```python
class DelegateTaskTool(Tool):
    def __init__(self, orchestrator_factory):
        """
        orchestrator_factory: callable that returns a configured Orchestrator.
        Lazy creation so we don't spin up the full engine until needed.
        """
        self._factory = orchestrator_factory
        self._orchestrator = None

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if self._orchestrator is None:
            self._orchestrator = await self._factory()

        task = create_task(
            goal=args["goal"],
            workspace=str(ctx.workspace),
            context=args.get("context", {}),
        )

        if args.get("wait", True):
            result = await self._orchestrator.execute_task(task)
            return self._format_result(task, result)
        else:
            # Fire-and-forget, return task_id
            asyncio.create_task(self._orchestrator.execute_task(task))
            return ToolResult.ok(f"Task submitted: {task.id}")
```

### Context Passing

The critical piece: the cowork model needs to pass conversational context
to the orchestrator so subtasks have the benefit of what was discussed.
The `context` parameter carries this:

```python
# Model calls:
delegate_task(
    goal="Add JWT refresh token rotation to src/auth.py",
    context={
        "constraints": [
            "Use PyJWT library (already in requirements)",
            "Refresh tokens expire after 7 days",
            "Store refresh tokens in Redis, not SQLite",
        ],
        "files_relevant": ["src/auth.py", "src/config.py", "tests/test_auth.py"],
        "decisions": ["Using RS256 algorithm per user preference"],
    },
)
```

This context gets injected into the orchestrator's planning prompt, so
subtasks inherit the conversational decisions without needing access to
the full conversation history.

### Result Format

When the task completes, the tool returns a structured summary that fits
cleanly into the conversation:

```
Task completed: "Add JWT refresh token rotation"

Subtasks:
  [x] Analyze current auth implementation (src/auth.py)
  [x] Add refresh token generation and validation
  [x] Add Redis storage for refresh tokens
  [x] Update login endpoint to return refresh + access tokens
  [x] Add /auth/refresh endpoint
  [x] Add tests for refresh token flow

Files changed:
  - src/auth.py (modified)
  - src/auth/refresh.py (created)
  - src/config.py (modified)
  - tests/test_auth_refresh.py (created)

Verification: All subtasks passed tier-1 verification.
4 new tests passing.
```

If the task fails, the result includes what failed and why, so the model
can discuss it with the user and decide how to proceed — retry, adjust
the approach, or do it manually with direct tools.

### When to Delegate vs Direct Execute

The system prompt instructs the model on this:

```
WHEN TO USE delegate_task:
- Multi-file refactoring (>3 files)
- Tasks that benefit from a plan (migrations, new features)
- Work that needs verification (changes with test requirements)
- Independent steps that can run in parallel

WHEN TO USE direct tools:
- Reading/editing a single file
- Running a command
- Quick fixes (typos, small bugs)
- Exploratory work (searching, reading, understanding)
- Anything where the overhead of planning isn't worth it
```

The model makes this judgment naturally — the same way I (Claude Code)
decide whether to use a Task subagent or just do something directly.

### Event Streaming

When `wait=True`, the tool should stream progress events back into the
conversation so the user sees what's happening:

```python
async def execute(self, args, ctx):
    # Subscribe to events for this task
    progress_lines = []
    def on_event(event):
        if event.event_type == SUBTASK_COMPLETED:
            progress_lines.append(f"  [x] {event.data.get('description', event.data.get('subtask_id'))}")
        elif event.event_type == SUBTASK_FAILED:
            progress_lines.append(f"  [!] {event.data.get('subtask_id')}: {event.data.get('error', 'failed')}")

    self._orchestrator._events.subscribe_all(on_event)
    result = await self._orchestrator.execute_task(task)
    # Include progress in output
```

(The exact streaming mechanism depends on how we want to surface progress
in the TUI — this can be refined during implementation.)

---

## Implementation Plan

### Phase 1: Persistence Layer

**Files to create:**
- `src/loom/state/conversation_store.py` — ConversationStore class

**Files to modify:**
- `src/loom/state/schema.sql` — add `conversation_turns` and `cowork_sessions` tables
- `src/loom/state/memory.py` — add schema migration support (run new CREATE TABLE IF NOT EXISTS)

**Tests:**
- `tests/test_conversation_store.py` — CRUD, search, resume, session lifecycle

### Phase 2: Session State

**Files to create:**
- `src/loom/cowork/session_state.py` — SessionState dataclass + auto-extraction

**Files to modify:**
- `src/loom/cowork/session.py` — integrate SessionState, inject into system prompt

**Tests:**
- `tests/test_session_state.py` — extraction from tool calls, YAML rendering, pruning

### Phase 3: Conversation Recall Tool + Context Awareness

**Files to create:**
- `src/loom/tools/conversation_recall.py` — ConversationRecallTool

**Files to modify:**
- `src/loom/tools/__init__.py` — register the tool in `create_default_registry`
- `src/loom/cowork/session.py` — pass store to tool at session creation,
  add `_maybe_inject_recall_hint()` to the message handling path
- `src/loom/cowork/approval.py` — add `conversation_recall` to auto-approved list

**Tests:**
- `tests/test_conversation_recall.py` — all four actions, output formatting, budget limits
- `tests/test_recall_hint.py` — dangling reference detection, false positive rate

### Phase 4: Context Window Rewrite

**Files to modify:**
- `src/loom/cowork/session.py` — replace `_context_window()` and `_maybe_trim()`,
  integrate recall hint injection into `send()` / `send_streaming()`

**Tests:**
- `tests/test_cowork_context.py` — token budgeting, priority order, hint injection

### Phase 5: Task Delegation Tool

**Files to create:**
- `src/loom/tools/delegate_task.py` — DelegateTaskTool

**Files to modify:**
- `src/loom/tools/__init__.py` — register delegate_task (only in cowork context)
- `src/loom/__main__.py` — create orchestrator factory, pass to tool at session setup
- `src/loom/cowork/session.py` — wire up orchestrator factory
- `src/loom/cowork/approval.py` — delegate_task requires explicit approval (not auto)

**Key wiring:**
- The tool takes an `orchestrator_factory` callable (lazy init)
- Factory reuses the same Config, ModelRouter, ToolRegistry, etc. from
  the cowork session — no separate server needed
- Context parameter passes conversational decisions to the orchestrator's
  planning prompt
- Results formatted as structured text for the conversation

**Tests:**
- `tests/test_delegate_task.py` — delegation flow, context passing, result formatting
- `tests/test_delegate_task_failure.py` — partial failure, retry suggestions

### Phase 6: Session Resumption + CLI

**Files to modify:**
- `src/loom/__main__.py` — session selection on startup, `--resume` flag
- `src/loom/cowork/session.py` — `resume()` method
- `src/loom/tui/app.py` — session selection in TUI

**Tests:**
- `tests/test_session_resume.py` — resume flow, state reconstruction

### Phase 7: System Prompt Update

**Files to modify:**
- `src/loom/cowork/session.py` — update `build_cowork_system_prompt()` to instruct
  the model about all new capabilities

The system prompt needs to tell the model:
- It has `conversation_recall` for accessing prior conversation
- The session state block shows structured metadata about the session
- It should use `conversation_recall` when it needs information from
  earlier in the session rather than asking the user to repeat themselves
- When its context window doesn't contain enough information to act
  confidently, it should search before guessing
- It has `delegate_task` for complex multi-step work that benefits from
  planning, decomposition, verification, or parallelism
- When to delegate vs when to use direct tools (the heuristics from
  Component 6)
- The conversation is continuous — delegated task results come back into
  the conversation for discussion

---

## What This Does NOT Change

- **Task mode API** — REST endpoints, MCP server, external agent integration untouched
- **Orchestrator internals** — plan-execute-verify pipeline unchanged;
  `delegate_task` is a new consumer of the existing `Orchestrator.execute_task()`
- **Memory entries** — existing `memory_entries` table and MemoryManager unchanged
- **Event bus** — event system unchanged (delegate_task subscribes to it read-only)
- **Tool execution** — all existing tools work identically
- **Model providers** — no changes to API interaction

---

## Key Design Decisions

### Why not use memory_entries for conversation storage?

The schemas serve different purposes:
- `memory_entries`: extracted knowledge with semantic metadata (tags, relevance_to, entry_type)
- `conversation_turns`: verbatim message log with conversation metadata (role, tool_call_id, turn_number)

Forcing conversation turns into memory_entries would either lose the
conversation structure (role, tool_call_id, ordering) or require adding
fields that make the memory system more complex for task mode.

### Why a tool instead of automatic injection?

The model is better positioned to know when it needs past context. Automatic
injection would either:
- Inject too much (wasting tokens on irrelevant history)
- Inject too little (missing what the model actually needs)
- Require an LLM call to decide what to inject (expensive, slow)

A tool call is explicit, cheap, and lets the model retrieve exactly what it
needs. It also makes the retrieval visible to the user ("I'm looking up what
we discussed about auth...").

### Why write-through instead of periodic flush?

Losing messages is the one failure mode we absolutely cannot accept. Write-through
guarantees that if the process crashes after a message is appended, it's in SQLite.
The latency cost is negligible — SQLite writes to a local file take <1ms, and we're
already doing async I/O for tool execution.

### Why token-aware instead of message-count?

Message count is a terrible proxy for context usage. A message containing a
300-line file read uses vastly more tokens than a 5-word user message. Token
estimation gives us much more efficient context packing — more useful
conversation history in the same budget.

### Why session state instead of conversation summary?

A conversation summary is an LLM-generated lossy compression. Session state is
structured data extracted mechanically from tool results. It's deterministic,
cheap (no LLM call), and can't hallucinate or lose important details. The model
can always use `conversation_recall` to get the actual conversation if the
session state isn't sufficient.

### Why in-process delegation instead of HTTP?

The MCP server uses HTTP to communicate with the orchestrator because it's
designed for external agents (Claude Code, other tools) running in separate
processes. Cowork doesn't need this indirection — it's already in the same
process. In-process delegation via a lazy-initialized Orchestrator is:
- Simpler (no HTTP server needed for `loom cowork`)
- Faster (no serialization/deserialization overhead)
- More reliable (no network failures on localhost)
- Easier to stream events (direct EventBus subscription)

The orchestrator's `execute_task()` interface is the same either way — the
tool is just a new caller of an existing API.

### Why cowork as primary, task as execution engine?

This matches how every successful AI coding tool actually works. Claude Code,
Cursor, Windsurf — the user is always in a conversation. Complex work happens
*within* the conversation via subagents/background tasks. Nobody wants to
switch modes.

Loom's task mode is a genuinely powerful orchestrator (planning, parallel
subtasks, verification, retry, learning). But it shouldn't require leaving
the conversation to use. By making it callable from cowork, users get both:
the natural conversation flow AND the structured execution when they need it.
