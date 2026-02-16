# Loom Foundation Refactor Plan

Comprehensive refactoring plan derived from a full codebase audit of all 89 source
files (~16,000 lines). Organized into six phases, each independently shippable.
Phases are ordered by severity and dependency: fix what's broken first, then
decouple, then optimize.

**Total issues identified: 70+**
**Estimated scope: ~3,500 lines changed across ~40 files**

---

## Phase 1: Error Handling & Silent Failures

**Why first:** Silent failures mask real problems, corrupt data, and make every
other refactor harder to validate. Fix observability before changing behavior.

### 1a. Eliminate bare `except Exception: pass` blocks

Across the codebase, 20+ locations catch `Exception` and silently discard it.
Every one of these needs logging at minimum, and most need specific exception
types.

| File | Lines | Context | Fix |
|------|-------|---------|-----|
| `engine/orchestrator.py` | 509, 517 | Workspace analysis (code + doc scan) | Log warning, continue |
| `engine/orchestrator.py` | 603 | Process workspace scan | Log warning, continue |
| `engine/orchestrator.py` | 738 | Learning after task completion | Log warning, continue |
| `engine/verification.py` | 225-226 | Deliverable file reads | Log warning, mark check as skipped |
| `engine/verification.py` | 266 | Verifier model unavailable | Return tier=0 skip, not fake tier=2 failure |
| `engine/runner.py` | 274-276, 300-301 | Memory extraction | Log, don't lose extraction data |
| `events/bus.py` | 85-96 | Event handler dispatch | Log handler name + exception |
| `events/bus.py` | 118-129 | Event persistence | Log persistence failure with event type |
| `models/anthropic_provider.py` | 117 | Health check | Catch specific httpx exceptions |
| `models/router.py` | 107 | Health check aggregation | Log per-provider failure |
| `tui/app.py` | 380-383 | Delegate tool binding | Log exception, notify user |
| `cowork/session.py` | 109-110 | Token estimation JSON parse | Log malformed content |
| `cowork/session.py` | 653-655, 668-670 | Turn/metadata persistence | Log, consider retry |
| `cowork/session_state.py` | 149-159 | Session state restoration | Log corruption, warn user |

**Pattern to follow:**

```python
# BEFORE (throughout codebase)
except Exception:
    pass

# AFTER
except (SpecificError, OtherError) as e:
    logger.warning("Context about what failed: %s", e)
```

**Rules:**
- Every `except` block must name specific exception types OR log with `exc_info=True`
- `pass` after `except` is banned -- minimum action is `logger.warning()`
- Silent failures in persistence paths get retry logic (see 1c)

### 1b. Fix error propagation in task execution

**File:** `engine/orchestrator.py:239-245`

The top-level `execute_task` catch loses exception type and traceback:

```python
# Current: str(e) loses everything
task.add_error("orchestrator", str(e))

# Fix: preserve type, log traceback, protect save
except Exception as e:
    logger.exception("Fatal error in task %s", task.id)
    task.add_error("orchestrator", f"{type(e).__name__}: {e}")
    try:
        self._state.save(task)
    except Exception as save_err:
        logger.error("Failed to save after fatal: %s", save_err)
    self._emit(TASK_FAILED, task.id, {
        "error": str(e),
        "error_type": type(e).__name__,
    })
```

Same pattern needed in `_dispatch_subtask` (lines 200-217) where
`asyncio.gather` results lose exception context.

### 1c. Fix silent JSON parse failures in tool call parsing

**Files:** `models/ollama_provider.py:112-116`, `openai_provider.py:116-120`,
`anthropic_provider.py:148-152`

All three providers silently discard malformed tool call arguments:

```python
# Current: tool gets empty dict, fails later with cryptic error
except json.JSONDecodeError:
    args = {}

# Fix: log the bad JSON, include it in tool result
except json.JSONDecodeError as e:
    logger.warning("Malformed tool args for %s: %s", tool_name, args_str[:200])
    args = {}  # still degrade gracefully, but now visible
```

### 1d. Fix verification tier semantic confusion

**File:** `engine/verification.py:264-271`

When verifier model is missing, code returns `tier=2, passed=False` -- claiming
tier 2 ran when it didn't. Downstream can't distinguish config error from
real verification failure.

```python
# Fix: return tier=0 skip instead of fake tier=2 failure
return VerificationResult(
    tier=0, passed=True, confidence=0.5,
    feedback="Verification skipped: verifier model not configured",
)
```

**Tests:** Add tests asserting no bare `except: pass` in source (grep-based).

---

## Phase 2: State Safety & Data Integrity

**Why second:** Race conditions and data corruption can produce bugs that are
impossible to reproduce. Fix state safety before adding features.

### 2a. Protect shared mutable state in CoworkSession

**File:** `cowork/session.py:171, 220-531`

`_messages`, `_turn_counter`, and `_message_counter` are modified by concurrent
async coroutines with no synchronization:

```python
# Fix: add locks for mutable state
def __init__(self, ...):
    self._messages_lock = asyncio.Lock()
    self._counter_lock = asyncio.Lock()

async def send(self, user_message: str):
    async with self._counter_lock:
        self._turn_counter += 1
        turn = self._turn_counter
    async with self._messages_lock:
        self._messages.append({"role": "user", "content": user_message})
```

### 2b. Make persistence writes atomic

**File:** `cowork/session.py:643-670`

Session metadata and turn persistence are separate writes. If metadata
succeeds but turn fails, session state is inconsistent.

Fix: batch both writes in a single SQLite transaction, or at minimum retry
turn persistence with backoff before giving up.

### 2c. Protect monotonic counter for turn persistence

**File:** `cowork/session.py:150-151, 643-652`

`_message_counter` increment + persist can race. Two concurrent
`_persist_turn` calls can produce duplicate turn numbers.

Fix: use the same `_counter_lock` from 2a around the increment-and-persist
sequence.

### 2d. Add input validation to ConversationStore queries

**File:** `state/conversation_store.py:158-172`

`limit` parameter has no upper bound -- callers can pass `limit=999999`.

```python
MAX_QUERY_LIMIT = 1000

async def get_turns(self, session_id: str, offset: int = 0, limit: int = 50):
    limit = min(limit, self.MAX_QUERY_LIMIT)
```

Also: parameterize the `is_active = 1` condition (line 71) instead of
hardcoding it in the SQL string.

**Tests:** Concurrent send() test, persistence failure + retry test, limit
enforcement test.

---

## Phase 3: Interface Consistency & Type Safety

**Why third:** Inconsistent interfaces cause subtle bugs and make the code hard
to extend. Unify interfaces before refactoring the components that use them.

### 3a. Add SubtaskResultStatus enum

**File:** `engine/runner.py:48`

`SubtaskResult.status` is a bare string ("success", "failed", "blocked").
Typos aren't caught, no IDE autocomplete, no exhaustive matching.

```python
from enum import StrEnum

class SubtaskResultStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class SubtaskResult:
    status: SubtaskResultStatus = SubtaskResultStatus.SUCCESS
    ...
```

Update all callers (orchestrator.py, runner.py).

### 3b. Fix VerificationResult tier semantics

**File:** `engine/verification.py:41+`

Current `tier` field is ambiguous -- does tier=2 mean "ran tier 2" or "tried
tier 2 but failed"?

```python
@dataclass
class VerificationResult:
    tier_attempted: int   # what tier was attempted
    tier_completed: int   # highest tier that actually ran
    passed: bool
    confidence: float = 1.0
    checks: list[Check] = field(default_factory=list)
    feedback: str | None = None

    @property
    def skipped(self) -> bool:
        return self.tier_attempted == 0
```

### 3c. Unify token estimation

**Files:** `cowork/session.py:76-80`, `state/conversation_store.py:18-22`,
`prompts/assembler.py:472-474`

Three different implementations of `len(text) // 4` with subtle differences
(one skips `max(1, ...)`, one skips empty-string check).

```python
# New file: src/loom/utils/tokens.py
def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars/token). Always returns >= 1."""
    if not text:
        return 1
    return max(1, len(text) // 4)
```

Replace all three callsites.

### 3d. Unify memory formatting

**Files:** `state/memory.py:381-394`, `prompts/assembler.py:445-455`

Two implementations of memory-to-prompt formatting. The assembler version
omits detail truncation.

Fix: delete the assembler's `_format_memory` and call
`MemoryManager.format_for_prompt()` with a `include_details` parameter.

### 3e. Consolidate tool schema conversion

**Files:** `models/ollama_provider.py:276-289`, `openai_provider.py:299-311`,
`anthropic_provider.py:212-238`

Three providers each implement tool-to-API-format conversion. Ollama and OpenAI
are identical. Anthropic differs only in `input_schema` vs `parameters`.

```python
# New: models/schema_adapter.py
class ToolSchemaAdapter:
    @staticmethod
    def to_openai(tools: list[dict]) -> list[dict]:
        """OpenAI/Ollama format."""
        ...

    @staticmethod
    def to_anthropic(tools: list[dict]) -> list[dict]:
        """Anthropic format (input_schema key)."""
        ...
```

### 3f. Consolidate streaming error handling

**Files:** All three providers' `stream()` methods

Identical try/except blocks for `ConnectError`, `TimeoutException`,
`HTTPStatusError` duplicated across all providers.

```python
# New: models/errors.py
class ModelConnectionError(Exception): ...

async def handle_provider_errors(provider_name: str, base_url: str, fn):
    """Unified error handling wrapper for provider HTTP calls."""
    try:
        return await fn()
    except httpx.ConnectError as e:
        raise ModelConnectionError(f"Cannot connect to {provider_name} at {base_url}: {e}")
    except httpx.TimeoutException as e:
        raise ModelConnectionError(f"{provider_name} timed out: {e}")
```

**Tests:** Enum exhaustiveness test, token estimation consistency test, schema
adapter round-trip tests.

---

## Phase 4: Decoupling & Architecture

**Why fourth:** With errors visible, state safe, and interfaces clean, now
decouple the major architectural violations.

### 4a. Add public API to ToolRegistry

**Files:** `tools/registry.py`, `engine/orchestrator.py:131`,
`tui/app.py:235-236`

Two places directly mutate `registry._tools` (private dict). Add public
methods:

```python
class ToolRegistry:
    def exclude(self, name: str) -> bool:
        """Remove a tool. Returns True if it existed."""
        ...

    def has(self, name: str) -> bool:
        """Check if tool is registered."""
        ...
```

Update orchestrator.py and app.py to use the public API.

### 4b. Unify approval logic

**Files:** `cowork/approval.py`, `recovery/approval.py`

Two separate approval modules with overlapping risk detection, decision enums,
and prompting logic. Extract shared logic:

```python
# New: approval/core.py
ALWAYS_GATE_PATTERNS: frozenset[str]  # single source of truth

class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    APPROVE_ALL = "approve_all"

def is_high_risk(tool_name: str, args: dict) -> bool:
    """Single risk assessment function used by both contexts."""
    ...
```

`cowork/approval.py` and `recovery/approval.py` import from `approval/core.py`.

### 4c. Extract TUI God Object

**File:** `tui/app.py` (1,071 lines, ~179 members)

LoomApp does everything: session management, tool binding, turn execution,
approval handling, data panel updates. Extract:

1. **SessionManager** -- session lifecycle (init, resume, create, persist)
2. **TurnExecutor** -- turn execution and event streaming
3. **ToolBinder** -- tool binding (recall, delegate) with error reporting
4. **UIFacade** -- cached widget references (replace 28 `query_one()` calls)

### 4d. Eliminate `_run_turn` / `_run_followup` duplication

**File:** `tui/app.py:689-891`

Two methods share ~80% code. Merge into one:

```python
async def _run_interaction(self, message: str, *, is_followup: bool = False):
    """Execute a turn. If is_followup, skip adding message to chat."""
    ...
```

### 4e. Move session tool binding out of TUI

**File:** `tui/app.py:336-383`

Tool binding (ConversationRecallTool, DelegateTaskTool) includes inline
orchestrator factory creation with 30+ lines. Extract to a `ToolBinder` class
that both TUI and CLI cowork can share.

### 4f. Fix manual context manager usage in providers

**Files:** All three providers' `stream()` methods

All use `cm.__aenter__()` / `cm.__aexit__()` instead of `async with`. This
is fragile -- errors between enter and the try block leak resources.

```python
# BEFORE
cm = client.stream("POST", url, json=payload)
response = await cm.__aenter__()
try:
    ...
finally:
    await cm.__aexit__(None, None, None)

# AFTER
async with client.stream("POST", url, json=payload) as response:
    ...
```

**Tests:** Registry public API tests, approval core tests, TUI component
isolation tests.

---

## Phase 5: Performance

**Why fifth:** Performance fixes are lower risk once the architecture is clean.

### 5a. Fix O(n^2) context window construction

**File:** `cowork/session.py:475-509`

Uses `list.insert(0, ...)` in a loop, which is O(n^2). For 200 messages this
is measurable.

```python
# BEFORE
selected.insert(0, msg)  # O(n) per insert

# AFTER
selected.append(msg)   # O(1) per append
selected.reverse()     # O(n) once
```

### 5b. Async-ify synchronous workspace scanning

**File:** `engine/orchestrator.py:522-568`

`_scan_workspace_documents()` is synchronous but called from async context,
blocking the event loop during `rglob("*")`.

```python
# Fix: run in thread pool
doc_summary = await asyncio.get_event_loop().run_in_executor(
    None, self._scan_workspace_documents, workspace_path
)
```

### 5c. Parallelize workspace analysis in planning

**File:** `engine/orchestrator.py:453-467`

`list_directory` and `_analyze_workspace` run sequentially but are independent.

```python
listing_result, workspace_analysis = await asyncio.gather(
    self._tools.execute("list_directory", {}, workspace=workspace_path),
    self._analyze_workspace(workspace_path),
)
```

### 5d. Cache ChangeLog per task

**File:** `engine/orchestrator.py:694-701`

`_get_changelog()` creates a new `ChangeLog` instance every dispatch, reading
JSON from disk each time. For 10 parallel subtasks, that's 10 redundant reads.

```python
def _get_changelog(self, task: Task) -> ChangeLog | None:
    if task.id in self._changelog_cache:
        return self._changelog_cache[task.id]
    ...
    self._changelog_cache[task.id] = changelog
    return changelog
```

### 5e. Buffer streaming text in TUI

**File:** `tui/widgets/chat_log.py:57-72`

`add_streaming_text()` calls `str(last.renderable)` and does string
concatenation on every chunk. For long responses this is O(n^2).

Fix: buffer chunks in a list, flush to widget every N chunks or every 100ms.

**Tests:** Benchmark context window construction, streaming buffer flush test.

---

## Phase 6: Dead Code & Cleanup

**Why last:** Dead code removal is safe and mechanical, but less urgent than
the above. Do it when the architecture is stable.

### 6a. Remove dead code

| File | Item | Reason |
|------|------|--------|
| `engine/scheduler.py:11-23` | `Scheduler.next_subtask()` | Never called; only `runnable_subtasks()` is used |
| `models/router.py:118-193` | `ResponseValidator` class | Never instantiated anywhere |
| `models/ollama_provider.py:351-375` | `ThinkingModeManager` class | Defined but never called |
| `prompts/assembler.py:164` | `memory_formatter` parameter | Shadows instance method, creates confusing dual path |

### 6b. Fix incomplete ANSI sanitization

**File:** `cowork/display.py:298-300`

Current sanitizer only strips `\033` but leaves `[31m` fragments:

```python
# BEFORE
text.replace("\033", "")

# AFTER
import re
re.sub(r'\033\[[0-9;]*[a-zA-Z]', '', text)
```

### 6c. Fix confidence score normalization

**File:** `recovery/confidence.py:44-88`

When tier 2 verification doesn't run, `weights_total` is 0.5 instead of 0.7,
biasing scores upward. Fix: renormalize correctly when optional components
are absent.

### 6d. Fix error categorization pattern ordering

**File:** `recovery/errors.py:40-86`

Generic `r"Error|Exception|Traceback|Failed"` pattern matches everything,
potentially shadowing earlier specific patterns. Reorder: most specific
patterns first, generic last with negative lookahead.

### 6e. Remove redundant PDF page range check

**File:** `tools/file_ops.py:141-144`

```python
page_end = min(page_start + MAX_PDF_PAGES_PER_READ, total_pages)
page_end = min(page_end, total_pages)  # redundant
```

### 6f. Fix Anthropic streaming (not actually streaming)

**File:** `models/anthropic_provider.py:451-551`

The stream implementation accumulates all text and yields a single final chunk.
This defeats the purpose of streaming -- clients see nothing until generation
completes.

Fix: yield `StreamChunk(text=text)` on each `content_block_delta` event,
yield tool calls and usage only in the final chunk.

**Tests:** Dead code removal tests (import checks), streaming yield-per-chunk test.

---

## Cross-Cutting Concerns

These apply across all phases:

### Logging standard

Every module should use:

```python
import logging
logger = logging.getLogger(__name__)
```

Not inline `import logging` inside except blocks (currently in 4 places).

### Exception hierarchy

```python
# New: src/loom/exceptions.py
class LoomError(Exception):
    """Base for all Loom exceptions."""

class EngineError(LoomError):
    """Orchestrator, runner, scheduler failures."""

class ModelError(LoomError):
    """Provider connection, timeout, parse failures."""

class ToolError(LoomError):
    """Tool execution failures."""

class StateError(LoomError):
    """Persistence, memory, session state failures."""
```

Providers raise `ModelError` subtypes instead of re-raising `httpx` exceptions.
Tools raise `ToolError`. Engine code catches these specifically.

### Add `__all__` exports

Modules currently export everything. Add `__all__` to public-facing modules
to make the API surface explicit.

---

## Summary

| Phase | Focus | Files | Severity | Effort |
|-------|-------|-------|----------|--------|
| 1. Error Handling | Silent failures, lost exceptions | ~20 | CRITICAL | Medium |
| 2. State Safety | Race conditions, data integrity | ~5 | HIGH | Medium |
| 3. Interface Consistency | Type safety, DRY | ~15 | HIGH | Medium |
| 4. Decoupling | God objects, encapsulation | ~10 | HIGH | High |
| 5. Performance | Blocking I/O, O(n^2) | ~6 | MEDIUM | Medium |
| 6. Dead Code & Cleanup | Unused code, minor bugs | ~10 | LOW | Low |

**Dependency chain:** Phase 1 enables validating all later phases. Phase 2 is
independent. Phases 3-4 build on 1. Phase 5 is independent. Phase 6 is
independent.

**Parallel work:** Phases 2, 5, and 6 can run in parallel with each other and
with phases 1/3/4.
