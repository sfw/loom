# Loom: Up-to-Snuff Plan

Prioritized by impact and dependency order. Each phase is independently shippable.

---

## Phase 1: New Tools (Foundation) ✓ DONE

Everything else depends on having a richer tool set.

All Phase 1 tools are implemented and auto-discovered via `Tool.__init_subclass__`.
No manual registration needed — `create_default_registry()` uses `discover_tools()`
to scan the `loom.tools` package automatically.

### 1a. `git_command` tool (`tools/git.py`) ✓
### 1b. `delete_file` tool (`tools/file_ops.py`) ✓
### 1c. `move_file` tool (`tools/file_ops.py`) ✓
### 1d. Auto-registration via plugin discovery ✓

**Tests:** Unit tests for each tool (safety blocklist, path validation, changelog integration)

---

## Phase 2: Streaming Token Output ✓ DONE

All providers support streaming. Cowork CLI uses streaming by default.

### 2a. Add streaming interface to `ModelProvider` ABC (`models/base.py`)
- New method: `async def stream(messages, tools, ...) -> AsyncGenerator[StreamChunk, None]`
- New dataclass: `StreamChunk(text: str, done: bool, tool_calls: list[ToolCall] | None, usage: TokenUsage | None)`
- Default implementation: call `complete()` and yield single chunk (backward compat)

### 2b. Implement streaming in `OllamaProvider`
- Set `stream: true` in payload
- Parse chunked JSON lines from `/api/chat`
- Buffer tool_calls (only available in final chunk)
- Yield text tokens as they arrive

### 2c. Implement streaming in `OpenAICompatibleProvider`
- Add `stream: true` to payload
- Parse SSE `data:` lines from `/chat/completions`
- Buffer tool_calls deltas until complete
- Yield text tokens from `delta.content`

### 2d. New event type: `TOKEN_STREAMED` (`events/types.py`)
- Emitted by SubtaskRunner during streaming execution
- Data: `{"subtask_id": str, "token": str, "model": str}`

### 2e. Wire streaming into `SubtaskRunner` (`engine/runner.py`)
- When streaming enabled: use `model.stream()` instead of `model.complete()`
- Emit `TOKEN_STREAMED` events as tokens arrive
- Buffer full response for tool call parsing and verification
- Config flag: `enable_streaming: bool = False` in `ExecutionConfig`

### 2f. Add SSE streaming endpoint for tokens (`api/routes.py`)
- `GET /tasks/{id}/tokens` — streams raw tokens for the active subtask
- Bridges EventBus TOKEN_STREAMED events to SSE

**Tests:** Unit tests for streaming providers (mock HTTP responses), integration test for token event emission

---

## Phase 3: TUI Enhancements ✓ DONE

TUI rewritten to use CoworkSession directly (no server needed).
Streaming chat, tool approval modals, ask_user modals, all 16 tools.

### 3a. Enhanced task creation modal (`tui/app.py`)
- Multi-field form: goal (required), workspace path (with autocomplete), approval_mode (select), context (optional textarea)
- Replace current single-input modal with a proper form screen

### 3b. Live SSE event listener
- Background worker that connects to `stream_all_events()` or `stream_task_events()`
- Auto-updates task table, plan tree, and live output
- Triggers approval modals when `approval_requested` events arrive
- Currently the TUI has SSE methods but never calls them

### 3c. Streaming token display
- Route `TOKEN_STREAMED` events to the `RichLog` widget (`#live-output`)
- Show real-time model output as it generates

### 3d. File changes viewer
- New panel/screen showing `changelog.get_summary()` for the selected task
- Show created/modified/deleted files
- Drill into diff view using `DiffGenerator`

### 3e. Memory/event inspector
- New screen: query memory entries for a task (type filter, search)
- Show decision log, discoveries, errors
- Useful for debugging and understanding agent reasoning

### 3f. Feedback submission
- Modal for rating (1-5) + comment after task completion
- Wire to existing `POST /tasks/{id}/feedback` endpoint

**Tests:** Smoke tests for new TUI screens (Textual pilot testing)

---

## Phase 4: Smarter Planning Context

The planner currently gets only a directory listing. Better context = better plans.

### 4a. Tree-sitter code analysis tool (`tools/code_analysis.py`) ✓
- `analyze_code` tool: parse a file and return structure (classes, functions, imports)
- Tree-sitter backend for Python, JS/TS, Go, Rust via `tree-sitter-language-pack`
- Returns structured JSON: `{classes: [...], functions: [...], imports: [...]}`
- Graceful fallback: regex-based extraction if tree-sitter not installed
- Structural matching in `edit_file` anchors fuzzy search to syntax nodes

### 4b. Planner context enhancement (`engine/orchestrator.py`)
- Before planning, auto-run `analyze_code` on key files in focus_dirs
- Inject code structure summary into planner prompt
- Helps planner make better subtask breakdowns

### 4c. `web_fetch` tool (`tools/web.py`)
- Fetch a URL and return content (for documentation, API specs, etc.)
- Safety: URL allowlist/blocklist in config, timeout, max response size
- Parameters: `{"url": str, "extract_text": bool}`
- Useful for reading docs, checking APIs

**Tests:** Unit tests with mocked tree-sitter, mocked HTTP

---

## Phase 5: Error Intelligence

Currently retry just escalates model tiers. Smarter error handling = fewer wasted retries.

### 5a. Error categorizer (`recovery/errors.py`)
- Classify errors into categories: syntax_error, runtime_error, tool_error, model_error, timeout, safety_violation
- Each category gets different recovery strategy
- e.g., syntax_error → retry with "fix the syntax error on line X", not just "try again"

### 5b. Error-aware retry in SubtaskRunner
- Pass categorized error info into retry context
- Inject specific error feedback into the retry prompt
- e.g., "Previous attempt failed with: FileNotFoundError on /path/to/file. The file doesn't exist yet — you need to create it first."

### 5c. Failure pattern learning (`learning/manager.py`)
- Track which error categories occur per tool, per model, per task type
- Feed patterns into planner context: "This model tends to hallucinate file paths — verify existence first"

**Tests:** Unit tests for error categorization, integration test for error-aware retry

---

## Phase 6: Interactive Conversation Mode ✓ DONE

Cowork mode is the primary interactive interface.

### 6a. Conversation endpoint (`api/routes.py`)
- `POST /tasks/{id}/message` — send a message to the running task
- Messages injected into the executor's context as user messages
- Enables back-and-forth clarification during execution

### 6b. Conversation TUI screen
- Chat-like interface in the TUI
- Shows model reasoning + tool calls + results
- User can type messages that get injected

### 6c. Conversation memory
- Store conversation turns in memory system
- Available for future task context

**Tests:** Integration test for mid-task message injection

---

## Phase 7: Process Definition Plugin Architecture ✓ DONE

Domain specialization without code changes. YAML-based process definitions inject
personas, phase blueprints, verification rules, and tool guidance into the engine.

### 7a. Process definition schema and loader ✓
- `ProcessDefinition` dataclass with phases, verification rules, memory types, tool config
- `ProcessLoader` with multi-path discovery (builtin → user-global → workspace-local)
- Comprehensive validation: name format, phase dependencies, cycle detection (DFS),
  duplicate deliverables, regex compilation
- Process packages: directory-based processes that bundle tools + templates

### 7b. PromptAssembler process injection ✓
- Persona overrides role in all prompt types (planner, executor, verifier, extractor)
- Phase blueprint and planner examples injected into planner prompt
- Tool guidance appended to executor constraints
- LLM verification rules injected into verifier prompt
- Memory extraction types injected into extractor prompt

### 7c. Orchestrator + Verifier process integration ✓
- Orchestrator accepts `ProcessDefinition`, passes to assembler and verification gates
- Tool exclusions applied from process config
- Workspace analysis scans for domain-relevant file types
- `DeterministicVerifier` runs process regex rules and deliverables checks

### 7d. CLI and config ✓
- `--process` flag on `run` and `cowork` commands
- `loom processes` command lists all available definitions
- `[process]` config section for default process and search paths

### 7e. Built-in process definitions ✓
- `investment-analysis` — 5-phase strict financial workflow
- `marketing-strategy` — 6-phase guided GTM pipeline
- `research-report` — 4-phase lightweight research pipeline
- `competitive-intel` — 3-phase fast competitive analysis
- `consulting-engagement` — 5-phase McKinsey-style issue tree

### 7f. New tools ✓
- `calculator` — safe AST-based math evaluation with financial functions (NPV, CAGR, WACC, PMT)
- `spreadsheet` — CSV operations (create, read, add rows/columns, update cells, summary)
- `document_write` — structured Markdown generation with sections and metadata

**Tests:** 905 tests passing (206 new for process system + tools)

### 7g. Process package installer ✓
- `processes/installer.py` — install from GitHub repos, shorthands, or local paths
- `loom install` and `loom uninstall` CLI commands
- `dependencies` field in process.yaml for auto-installing pip packages
- Validates structure (process.yaml, name format), tries `uv` then `pip` for deps
- Protects built-in processes from overwrite/uninstall
- 49 tests covering source resolution, validation, deps, copy, uninstall

**Tests:** 954 tests passing (49 new for installer)

---

## Summary: Priority Order

| Phase | Impact | Effort | Status |
|-------|--------|--------|--------|
| 1. New Tools | High | Low | **DONE** |
| 2. Streaming | High | Medium | **DONE** |
| 3. TUI Enhancements | High | Medium | **DONE** |
| 4. Smarter Planning | Medium | Medium | Phase A+B done (tree-sitter), web_fetch done |
| 5. Error Intelligence | Medium | Low | Partial (error categorizer done) |
| 6. Interactive Mode | Medium | High | **DONE** (cowork + TUI) |
| 7. Process Definitions | High | High | **DONE** (plugin architecture + 5 built-in + installer) |

Additionally implemented (from gap analysis):
- Anthropic/Claude provider, per-tool-call approval, web_search, ripgrep_search,
  glob_find, ask_user, task_tracker, PDF/image support. 954 tests passing.
