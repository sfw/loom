# Spec 09: Interactive TUI (Default Interface)

## Overview

The TUI is Loom's unified interactive interface — the default `loom` command. Built with Python `textual`, it runs CoworkSession directly with full session persistence, conversation recall, task delegation, and process definitions. No server required.

**Launch:**
```bash
loom -w /path/to/project                    # Default command
loom -w /path/to/project -m claude          # Specific model
loom -w /path/to/project --process consulting-engagement  # With process
loom --resume <session-id>                  # Resume previous session
loom cowork -w /path/to/project             # Alias
```

## Layout

```
┌─ Loom ──────────────────────────────────────────────────────────────┐
│ Header (clock)                                                       │
├─────────┬───────────────────────────────────────────────────────────┤
│         │ [Chat]  [Files Changed]  [Events]                          │
│  S I    │                                                            │
│  I D    │  > Fix the authentication bug in src/auth.py               │
│  D E    │                                                            │
│  E B    │    read_file src/auth.py                                   │
│  B A    │    ok 12ms 45 lines                                        │
│  A R    │    ripgrep_search /validate_token/                         │
│  R      │    ok 3ms 2 results                                        │
│         │    edit_file src/auth.py                                    │
│  Tasks  │    ok 8ms                                                  │
│  Files  │                                                            │
│         │  I found and fixed the bug. The token validation was...     │
│         │                                                            │
│         │  [3 tool calls | 1,247 tokens | claude-sonnet-4-5]        │
├─────────┴───────────────────────────────────────────────────────────┤
│  [>] Type a message... (Enter to send)                               │
├─────────────────────────────────────────────────────────────────────┤
│  /path/to/project  |  qwen3:14b  |  3.2k tokens  |  Ready          │
├─────────────────────────────────────────────────────────────────────┤
│  ^B Sidebar  ^L Clear  ^P Commands  ^C Quit                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

### First-Run Setup Wizard

When Loom launches with no configured models, the TUI pushes a `SetupScreen` modal — a five-step guided wizard that collects provider, model details, roles, an optional utility model, and writes `~/.loom/loom.toml`. The session initializes immediately after without restarting. Users can reconfigure at any time via the `/setup` slash command.

**Steps:**
1. **Provider** — Anthropic, OpenAI-compatible, or Ollama (number keys to select)
2. **Details** — base URL, model name/selection, API key (provider-dependent fields)
3. **Roles** — all roles, primary only (planner + executor), or utility only (extractor + verifier)
4. **Utility model?** — if roles are incomplete, offer to add a second model
5. **Confirm** — summary of all models and roles; Enter to save, Esc to go back

### Multi-Panel Layout
- **Sidebar** — workspace file browser, task progress tracker
- **Chat tab** — streaming conversation with rich tool call rendering
- **Files Changed tab** — tracks all file operations with inline diff viewer
- **Events tab** — timestamped event log with token sparkline

### Session Persistence
- Every turn persisted to SQLite via `ConversationStore` (write-through)
- Session survives restarts — resume with `--resume <id>`
- `SessionState` metadata (focus, decisions, files touched) injected into system prompt
- Slash commands: `/sessions`, `/new`, `/session`, `/resume <id>`

### Conversation Recall
- `conversation_recall` tool lets the model search past context
- Full-text search across turns, tool call filtering
- Dangling reference detection nudges model to use recall

### Task Delegation
- `delegate_task` tool spawns autonomous sub-agents via the orchestrator
- Model can offload complex multi-step work while continuing the conversation

### Streaming Text Display
- Model text tokens stream as they arrive
- Tool calls show inline: tool name + args preview on start; ok/err + elapsed + output preview on completion
- Turn summaries show tool count and token usage

### Tool Approval Modal
When the model calls a write/execute tool, a modal appears:

```
┌─ Approve tool call? ──────────────────────┐
│                                            │
│  shell_execute  ls -la                     │
│                                            │
│  [y] Yes  [a] Always allow  [n] No  [Esc] │
└────────────────────────────────────────────┘
```

- **[y] Yes** — approve this one call
- **[a] Always** — approve all future calls to this tool for the session
- **[n] No / Esc** — deny (model sees "denied by user" error)
- Read-only tools (read_file, search, glob, web_search, etc.) are auto-approved

### Ask User Modal
When the model calls `ask_user`, a modal with input field appears:

```
┌─ Question: Which database should we use? ─┐
│                                            │
│   1. PostgreSQL                            │
│   2. SQLite                                │
│   3. MySQL                                 │
│   Enter a number or type your answer       │
│                                            │
│   [>] Your answer...                       │
└────────────────────────────────────────────┘
```

### Slash Commands

| Command | Action |
|---------|--------|
| `/help` | Show available commands and shortcuts |
| `/sessions` | List all saved sessions |
| `/new` | Start a new session |
| `/session` | Show current session info (turns, tokens, focus) |
| `/resume <id>` | Switch to a different session by ID prefix |
| `/model` | Show current model |
| `/tools` | List available tools |
| `/tokens` | Show session token usage |
| `/setup` | Open the configuration wizard |
| `/clear` | Clear the chat display |
| `/quit` | Exit Loom |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+L` | Clear chat |
| `Ctrl+P` | Command palette |
| `Ctrl+1/2/3` | Switch tabs (Chat / Files / Events) |

## Architecture

```python
# tui/app.py
class LoomApp(App):
    """Unified interactive interface with full cowork backend."""

    def __init__(self, model: ModelProvider | None, tools, workspace, *,
                 config=None, db=None, store=None,
                 resume_session=None, process_name=None):
        ...

    async def on_mount(self):
        # If model is None → push SetupScreen modal, return
        # Otherwise → _initialize_session()

    async def _initialize_session(self):
        # Register conversation_recall + delegate_task tools
        # Load process definition
        # Create or resume session (with persistence)
        # Bind session-dependent tools

    async def _approval_callback(self, tool_name, args) -> ApprovalDecision:
        """Shows ToolApprovalScreen modal, waits for response."""

    @work(exclusive=True)
    async def _run_turn(self, user_message: str):
        """Streams events from session.send_streaming() into the chat log."""
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `LoomApp` | `tui/app.py` | Main Textual application with persistence |
| `ChatLog` | `tui/widgets/chat_log.py` | Scrollable chat with rich tool rendering |
| `Sidebar` | `tui/widgets/sidebar.py` | Workspace browser + task progress |
| `FilesChangedPanel` | `tui/widgets/files_changed.py` | File tracking with diff viewer |
| `EventPanel` | `tui/widgets/event_panel.py` | Timestamped event log |
| `StatusBar` | `tui/widgets/status_bar.py` | Workspace, model, tokens, state |
| `ToolApprovalScreen` | `tui/screens/` | Modal for [y]es/[a]lways/[n]o approval |
| `AskUserScreen` | `tui/screens/` | Modal for model questions |
| `SetupScreen` | `tui/screens/` | Multi-step first-run setup wizard |
| `ToolApprover` | `cowork/approval.py` | Tracks auto-approved and always-approved tools |
| `CoworkSession` | `cowork/session.py` | Conversation-first execution engine |
| `ConversationStore` | `state/conversation_store.py` | SQLite session persistence |

## Acceptance Criteria

- [x] `loom` (no args) launches the TUI with full persistence
- [x] `loom cowork` is an alias for the default TUI
- [x] Chat log shows streaming text tokens in real time
- [x] Tool calls display with args preview and completion status
- [x] Approval modal appears for write/execute tools with [y]/[a]/[n] options
- [x] "Always allow" remembers per-tool for the session
- [x] Ask_user modal shows question, options, and input field
- [x] All 21 tools are supported (including conversation_recall, delegate_task)
- [x] Session persistence — all turns saved to SQLite
- [x] Session resumption — `--resume <id>` restores context
- [x] Session management — `/sessions`, `/new`, `/resume` slash commands
- [x] Process definitions load and inject persona/tool guidance
- [x] Multi-panel layout (sidebar, chat, files, events)
- [x] Status bar shows workspace, model, token count, and state
- [x] Keyboard shortcuts (Ctrl+B sidebar, Ctrl+L clear, Ctrl+P palette)
- [x] First-run setup wizard inside TUI when no models configured
- [x] `/setup` slash command for reconfiguration
- [x] `loom setup` CLI fallback for headless environments
