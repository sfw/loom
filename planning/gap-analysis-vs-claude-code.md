# Gap Analysis: Loom vs Claude Code (Cowork Mode)

## Executive Summary

Loom has a solid foundation — real orchestration loop, real tools, real verification,
real state management. But compared to Claude Code's coworking model, it has
**fundamental architectural gaps** in how it interacts with users, manages
conversation context, and handles the real-time feedback loop that makes pair
programming feel natural.

The core insight: **Claude Code is a conversation-first agent that uses tools.**
Loom is a **task-execution engine that has a conversation bolted on.** Bridging
this gap requires rethinking the interaction model, not just adding features.

---

## 1. Interaction Model

### Claude Code
- **Interactive conversation loop**: user sends message, agent responds with
  text + tool calls, user can interrupt/redirect at any point
- **User is always in the loop**: every tool call can be approved/denied/modified
- **Natural language is the primary interface**: no forms, no structured input
- **Mid-stream correction**: user can type while the agent is working
- **Permission model**: user controls what actions are auto-approved vs gated

### Loom Current State
- **Fire-and-forget task submission**: POST /tasks with a goal string
- Conversation exists but is **memory injection** (stores as MemoryEntry, not
  real-time conversation turns)
- Steering is async: message goes into memory, picked up on next subtask
  prompt assembly — **not mid-tool-loop**
- TUI has a ConversationScreen but it's a modal overlay, not the primary
  interaction mode
- CLI `run` command submits and streams events, but user can't interact

### Gap: CRITICAL
Loom needs a **conversational execution mode** where:
- The user and agent alternate turns naturally
- The agent can ask clarifying questions before/during execution
- Tool calls are visible and interruptible in real time
- The conversation history IS the context (not a side-channel)

---

## 2. Context Management

### Claude Code
- **Full conversation history** as the context window — every message,
  tool call, and result is in the conversation
- **Automatic context compression** when approaching limits
- **File contents read inline** — the model sees the actual file content
  in the conversation, not a summary
- **System prompts** provide role, constraints, environment info
- **No separate "memory" system** — conversation IS the memory

### Loom Current State
- **Dual-layer architecture**: Layer 1 (YAML task state in every prompt) +
  Layer 2 (SQLite memory archive queried per-subtask)
- Memory entries are **summaries** (max 150 chars) — not full content
- Each subtask gets a **fresh prompt** assembled from templates + state +
  memory — no conversation continuity between subtasks
- Anti-amnesia reminders injected as system messages in tool loop
- Token budget: hard 8000 token prompt limit with naive truncation

### Gap: HIGH
- Subtask isolation means the model **loses all context** between subtasks.
  Claude Code maintains full conversation context across its entire session.
- Memory extraction is a separate LLM call that produces lossy summaries.
  This is architecturally wrong for pair programming — the model needs to
  remember what it actually did, not a summary of what it did.
- The 8000 token prompt budget is extremely tight for real coding tasks.
  Claude Code works in 200K+ token context windows.

---

## 3. Tool Ecosystem

### Claude Code
| Tool | Capability |
|------|-----------|
| Read | Read any file, images, PDFs, notebooks |
| Write | Create new files |
| Edit | Exact string replacement with uniqueness check |
| Glob | Fast file pattern matching |
| Grep | Full ripgrep-powered content search |
| Bash | Shell execution with sandboxing |
| WebFetch | Fetch + AI-process web content |
| WebSearch | Web search with results |
| Task | Spawn sub-agents for parallel work |
| TodoWrite | Track task progress for user visibility |
| NotebookEdit | Edit Jupyter notebooks |
| AskUserQuestion | Ask user for clarification |

### Loom Current State
| Tool | Capability |
|------|-----------|
| read_file | Read text files (no images/PDFs/notebooks) |
| write_file | Create/overwrite files |
| edit_file | String replacement (unique match required) |
| search_files | Regex search (pure Python, slow on large codebases) |
| list_directory | Tree listing, 2 levels deep |
| shell_execute | Shell execution with blocked pattern list |
| git_command | Git with whitelist approach (no push!) |
| web_fetch | URL fetch with HTML stripping |
| analyze_code | Regex-based code structure extraction |

### Gaps

| Missing | Impact |
|---------|--------|
| **Glob/find** | No fast file discovery by pattern |
| **Grep (ripgrep)** | search_files uses pure Python os.walk — painfully slow on real codebases |
| **Web search** | Cannot search the internet for docs/solutions |
| **Sub-agent spawning** | Cannot parallelize research tasks |
| **User question tool** | Cannot ask the user for clarification mid-execution |
| **Image/PDF reading** | Cannot process visual content |
| **Notebook editing** | Cannot work with Jupyter notebooks |
| **git push** | Not in the allowed subcommands list |
| **Progress tracking visible to user** | No equivalent of TodoWrite |

### Tool Quality Gaps
- **search_files**: 200 match limit, no context lines, no file type filtering,
  uses os.walk (no gitignore respect). Claude Code's Grep uses ripgrep with
  context, file types, multiline, offsets.
- **list_directory**: 2 levels deep max. Need recursive glob patterns.
- **shell_execute**: 60s timeout, 10KB output limit. Claude Code allows 10min
  timeout and 30KB output.
- **edit_file**: No `replace_all` mode. No line number awareness.
- **git_command**: No `push` in allowlist — agent can commit but not push.

---

## 4. Execution Architecture

### Claude Code
- **Single agent loop**: model generates text + tool calls, tools execute,
  results go back into conversation, repeat
- **No planning phase**: the model decides what to do as it goes
- **No subtask decomposition**: complex tasks are handled by the model's
  own reasoning in a single continuous conversation
- **Sub-agents** for parallel work (research, testing) that return results
  to the main agent
- **No verification layer**: the model self-corrects by reading results
  and trying again if something fails

### Loom Current State
- **Plan → Execute → Verify pipeline**: always decomposes into subtasks first
- Planning uses a separate LLM call to produce JSON subtask list
- Each subtask runs in an isolated tool loop (max 20 iterations)
- 3-tier verification after every subtask
- Retry with escalation (tier 1 → 2 → 3 → human)
- Parallel subtask dispatch for independent subtasks
- Re-planning when retries are exhausted

### Gap: ARCHITECTURAL
Loom's plan-then-execute model adds latency and loses context:
- **Planning overhead**: Every task, even simple ones, goes through a full
  planning phase. Claude Code just starts working.
- **Subtask isolation**: Each subtask is a fresh prompt. The model can't
  reference what it learned in subtask-1 when executing subtask-3, except
  through lossy memory summaries.
- **Verification overhead**: 3-tier verification is great for autonomous
  batch processing, but adds massive latency for interactive coworking.
  When a user is watching, they ARE the verification layer.
- **Re-planning is expensive**: Requires a full LLM call to restructure
  the plan. Claude Code just adapts naturally in conversation.

The plan-execute-verify model is good for **autonomous background tasks**.
For **interactive coworking**, Loom needs a **direct execution mode** that
skips planning and runs a continuous tool loop like Claude Code does.

---

## 5. Streaming & Real-time Feedback

### Claude Code
- Token-by-token streaming of all model output
- Tool calls appear as they happen with descriptions
- Progress indicators (TodoWrite, status line)
- User sees everything in real time and can interrupt

### Loom Current State
- Streaming is **disabled by default** (`enable_streaming: false`)
- SSE event stream for task-level events (subtask started/completed/failed)
- Token streaming works but only emits events — no inline display in CLI
- TUI has a RichLog that shows events, but it's not a conversation view
- CLI `run` command streams SSE events as raw JSON lines

### Gap: HIGH
- Default non-streaming means the user sees nothing until a subtask completes
- Even with streaming, tokens are events, not inline conversation output
- No way to see what the model is "thinking" or doing in real time
  during the tool loop
- No tool-call-level progress (Claude Code shows "Reading file X",
  "Running command Y" as they happen)

---

## 6. Model Provider Support

### Claude Code
- Uses Claude (Anthropic API) — the model it's built for
- Extended thinking / reasoning built into the model
- Native tool use support in the API
- Automatic prompt caching for efficiency

### Loom Current State
- Ollama provider (local models)
- OpenAI-compatible provider (MLX, LM Studio, vLLM, llama.cpp)
- ThinkingModeManager for Qwen3 `/think` / `/no_think`
- Tier-based model selection (1=small, 2=medium, 3=large)
- No Anthropic/Claude provider (!)

### Gap: MEDIUM-HIGH
- **No Claude/Anthropic provider**: Ironic for a tool meant to cowork with
  Claude Code. Adding Claude API support would let Loom use the best
  available models.
- **No prompt caching**: Local models don't benefit from this, but API
  models do. Missing optimization.
- **Tier inference is crude**: Based on model name string matching
  ("70b" → tier 3). Should be configurable per model.
- **No multi-modal support**: Can't send images to models that support them.

---

## 7. Safety & Permissions

### Claude Code
- **User-controlled permission model**: auto-allow, ask-every-time, or rules
- **Every tool call can be denied** by the user in real time
- **Hooks system**: user-defined shell commands that run on events
- **Sandbox mode**: restricts what the agent can do
- Careful about destructive operations (warns, asks for confirmation)
- Never pushes to git without explicit permission

### Loom Current State
- Blocked command patterns (rm -rf /, curl | sh, etc.)
- Git command whitelist (no push, no force operations)
- File operations sandboxed to workspace directory
- Confidence-based approval gating (auto at >0.8, wait at >0.5, abort at <0.2)
- "Always gate" patterns for destructive operations
- Sensitive file detection (.env, credentials, etc.)

### Gap: MEDIUM
- Loom's safety is **static** — hardcoded patterns. Claude Code's is
  **user-configured** and real-time.
- No per-tool-call approval in the interactive case. The approval system
  is designed for subtask-level gating, not tool-call-level.
- No hooks system for user customization.
- The confidence threshold approach is novel but untested in practice —
  the confidence score is computed from verification results that may
  not exist yet (tier 1 only = just syntax checks).

---

## 8. State Persistence & Recovery

### Claude Code
- Conversation history persists across context window compression
- No persistent state beyond the conversation — start fresh each session
- Git is the persistence layer for code changes

### Loom Current State
- Full task state in YAML on disk (atomic writes)
- SQLite database for memory entries, events, learned patterns
- Workspace changelog with file snapshots for undo
- Event log with persistence
- Task can be resumed after crash (state is always on disk)

### Gap: POSITIVE (Loom is ahead here)
Loom's persistence story is actually **better** than Claude Code for:
- Crash recovery
- Audit trails
- Undo capability
- Cross-session learning

This is a genuine strength to preserve.

---

## 9. Testing

### Claude Code
- Extensive test suites (not directly visible, but implied by stability)

### Loom Current State
28 test files covering:
- Orchestrator, scheduler, verification
- All tools, models, prompts
- API routes, TUI, MCP
- Memory, events, webhooks
- Approval, confidence, retry, errors
- Streaming, workspace, code analysis
- Full integration test
- CLI, conversation, learning

### Gap: LOW
Test coverage appears comprehensive. The tests exist for nearly every
component. Quality/depth would need a deeper review but the structure
is solid.

---

## 10. Priority Gap Summary — ALL RESOLVED

| # | Gap | Severity | Status | Implementation |
|---|-----|----------|--------|----------------|
| 1 | **Conversational execution mode** | CRITICAL | **DONE** | `cowork/session.py` — CoworkSession with continuous tool loop |
| 2 | **Full conversation context** | CRITICAL | **DONE** | Full message history maintained, 200-message context window |
| 3 | **User question tool** | HIGH | **DONE** | `tools/ask_user.py` — free-text and multiple-choice |
| 4 | **Real-time tool visibility** | HIGH | **DONE** | `cowork/display.py` — ANSI tool call display; `tui/app.py` — RichLog |
| 5 | **Ripgrep-powered search** | HIGH | **DONE** | `tools/ripgrep.py` — rg subprocess with grep/Python fallbacks |
| 6 | **Glob file finder** | HIGH | **DONE** | `tools/glob_find.py` — sorted by mtime, skips junk dirs |
| 7 | **Web search tool** | HIGH | **DONE** | `tools/web_search.py` — DuckDuckGo HTML, no API key |
| 8 | **Git push support** | MEDIUM | **DONE** | `tools/git.py` — push/remote added, force push blocked |
| 9 | **Anthropic/Claude provider** | MEDIUM | **DONE** | `models/anthropic_provider.py` — Messages API with streaming |
| 10 | **Per-tool-call approval** | MEDIUM | **DONE** | `cowork/approval.py` — auto/approve/always/deny per tool |
| 11 | **Streaming by default** | MEDIUM | **DONE** | `__main__.py` — cowork CLI uses `send_streaming()` by default |
| 12 | **Larger output limits** | LOW | **DONE** | Shell: 120s timeout, 30KB output limit |
| 13 | **Image/PDF reading** | LOW | **DONE** | `tools/file_ops.py` — pypdf text extraction, image metadata |
| 14 | **Progress tracking tool** | LOW | **DONE** | `tools/task_tracker.py` — in-memory add/update/list/clear |

All 14 gaps identified in this analysis have been implemented and tested (612+ tests passing).

---

## Implementation Summary

### Phase 1: Interactive Cowork Mode (Gaps 1-4) — COMPLETE
- `loom cowork -w /path` — CLI REPL with streaming, tool display, ask_user
- `loom tui -w /path` — Textual app with modals, scrollback, same engine
- CoworkSession: conversation-first tool loop, no planning phase
- Full message history as primary context

### Phase 2: Tool Parity (Gaps 5-8) — COMPLETE
- ripgrep_search, glob_find, web_search tools added
- git push with safety checks
- Shell timeout 120s, output limit 30KB

### Phase 3: Model & UX Polish (Gaps 9-14) — COMPLETE
- Anthropic/Claude provider with streaming
- Per-tool-call approval: [y]es / [a]lways / [n]o
- TUI rewritten for cowork (no server needed)
- PDF/image reading, task_tracker tool

### Preserved Strengths
- Plan-execute-verify pipeline (for autonomous/background tasks)
- 3-tier verification (for high-stakes operations)
- Workspace changelog & undo
- Learning system
- Event bus architecture
- SQLite persistence
