# Spec 09: Terminal UI Client

## Overview

The TUI is Loom's rich terminal interface for interactive cowork sessions. Built with Python `textual`, it runs CoworkSession directly — no server required. It provides a chat-based interface with streaming text, tool call visualization, approval modals, and ask_user modals.

**Two interface options, same engine:**
- `loom cowork` — lightweight CLI REPL with ANSI output
- `loom tui` — Textual app with scrollable chat, modals, and widgets

## Launch

```bash
loom tui -w /path/to/project          # Textual UI with modals
loom tui -w /path/to/project -m claude # Use a specific model
```

The TUI runs the model client-side via CoworkSession. No `loom serve` needed.

## Layout

```
┌─ Loom ─────────────────────────────────────────────────────────────┐
│ Header (clock)                                                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Chat Log (RichLog, scrollable)                                      │
│                                                                      │
│  > Fix the authentication bug in src/auth.py                         │
│                                                                      │
│    read_file src/auth.py                                             │
│    ok 12ms 45 lines                                                  │
│    ripgrep_search /validate_token/                                   │
│    ok 3ms 2 results                                                  │
│    edit_file src/auth.py                                             │
│    ok 8ms                                                            │
│                                                                      │
│  I found and fixed the bug. The token validation was using...        │
│                                                                      │
│  [3 tool calls | 1,247 tokens | claude-sonnet-4-5]                  │
│                                                                      │
├────────────────────────────────────────────────────────────────────┤
│  [>] Type a message... (Enter to send)                               │
├────────────────────────────────────────────────────────────────────┤
│  /path/to/project  |  Ready                                         │
├────────────────────────────────────────────────────────────────────┤
│  Ctrl+C Quit   Ctrl+L Clear                                         │
└────────────────────────────────────────────────────────────────────┘
```

## Features

### Streaming Text Display
- Model text tokens stream into the RichLog as they arrive
- Tool calls show inline: tool name, args preview on start; ok/err, elapsed, output preview on completion
- Turn summaries show tool count and token usage

### Tool Approval Modal
When the model calls a write/execute tool (shell_execute, git_command, edit_file, etc.), a modal appears:

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
When the model calls `ask_user`, a modal with an input field appears:

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

### Commands
- `/quit`, `/exit`, `/q` — exit the TUI
- `/clear` — clear the chat log

## Architecture

```python
# tui/app.py
class LoomApp(App):
    """Uses CoworkSession directly — no server."""

    def __init__(self, model: ModelProvider, tools: ToolRegistry, workspace: Path):
        ...
        # Session created on mount with ToolApprover wired to modal callback

    async def _approval_callback(self, tool_name, args) -> ApprovalDecision:
        """Shows ToolApprovalScreen modal, waits for response."""

    @work(exclusive=True)
    async def _run_turn(self, user_message: str):
        """Streams events from session.send_streaming() into the RichLog."""
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `LoomApp` | `tui/app.py` | Main Textual application |
| `ToolApprovalScreen` | `tui/app.py` | Modal for [y]es/[a]lways/[n]o approval |
| `AskUserScreen` | `tui/app.py` | Modal for model questions with option input |
| `ToolApprover` | `cowork/approval.py` | Tracks auto-approved and always-approved tools |
| `CoworkSession` | `cowork/session.py` | Conversation-first execution engine |
| `LoomAPIClient` | `tui/api_client.py` | Legacy REST client (still available for server mode) |

## Tool Display Previews

Each tool call shows a compact preview in the chat log:

| Tool | Start Preview | Completion Preview |
|------|--------------|-------------------|
| `read_file` | `src/auth.py` | `ok 12ms 45 lines` |
| `shell_execute` | `pytest tests/ -x` | `ok 3200ms 12 passed` |
| `ripgrep_search` | `/TODO/` | `ok 5ms 7 results` |
| `glob_find` | `**/*.py` | `ok 2ms 23 results` |
| `web_search` | `python asyncio tutorial` | `ok 800ms 3 results` |
| `git_command` | `status` | `ok 15ms` |

## Acceptance Criteria

- [x] `loom tui` launches without a running server (uses CoworkSession directly)
- [x] Chat log shows streaming text tokens in real time
- [x] Tool calls display with args preview and completion status
- [x] Approval modal appears for write/execute tools with [y]/[a]/[n] options
- [x] "Always allow" remembers per-tool for the session
- [x] Ask_user modal shows question, options, and input field
- [x] All 16 tools are supported
- [x] `/quit` and `/clear` commands work
- [x] Status bar shows workspace path and current state
- [x] Keyboard shortcuts (Ctrl+C quit, Ctrl+L clear) work
