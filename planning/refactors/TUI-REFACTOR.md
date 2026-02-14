# TUI Refactor: From Chat Box to Command Center

## Problem Statement

The current TUI (`tui/app.py`) is a single-column chat log with an input bar. It works,
but it surfaces only ~20% of Loom's capabilities. There is no file browser, no task tree,
no diff viewer, no event inspector, no progress visualization, and no visual polish. Users
have no situational awareness of what the agent is doing beyond reading the chat stream.

**Current state:** 502 lines, single `RichLog` + `Input` + two modals.
**Target state:** Multi-panel command center with sidebar, tabbed content, rich tool call
rendering, file tree, diff viewer, and a polished dark theme.

---

## Design Principles

1. **Chat is primary, context is ambient.** The conversation stays front and center.
   Sidebar and tabs provide awareness without demanding attention.
2. **Progressive disclosure.** Show the essential at a glance, details on demand.
   Tool calls show a one-line summary; expand for full output.
3. **Keyboard-first.** Every action reachable by keystroke. Mouse is optional.
4. **Non-blocking.** The TUI never freezes. All agent work runs in `@work` workers.
   The user can always type, scroll, switch tabs, or cancel.
5. **Data already exists.** The engine already tracks everything (changelog, events,
   memory, confidence, verification). The TUI just needs to surface it.

---

## Layout

```
┌─ Loom ──────────────────────── claude-sonnet-4-5 ── 3.2k tokens ──┐
├────────────┬───────────────────────────────────────────────────────┤
│            │ [Chat]  [Files Changed]  [Events]                    │
│  SIDEBAR   │                                                      │
│            │  > Fix the auth bug in src/auth.py                   │
│  Workspace │                                                      │
│  ──────    │  ▸ read_file src/auth.py              ok 12ms 45 ln  │
│  ▸ src/    │  ▸ ripgrep_search /validate_token/    ok  3ms 2 hits │
│    auth.py │  ▸ edit_file src/auth.py              ok  8ms        │
│    main.py │                                                      │
│    utils.  │  I found and fixed the bug. The token validation     │
│  ▸ tests/  │  was comparing against an expired timestamp...       │
│    ...     │                                                      │
│            │  [3 tools | 1,247 tokens | claude-sonnet-4-5]        │
│ ────────── │                                                      │
│  Progress  │                                                      │
│  ──────    │                                                      │
│  ✓ Read    │                                                      │
│  ◉ Fix bug │                                                      │
│  ○ Tests   │                                                      │
│            │                                                      │
├────────────┴───────────────────────────────────────────────────────┤
│  [>] Type a message...              Ready | src/myproject | 3.2k  │
├────────────────────────────────────────────────────────────────────┤
│  ^B Sidebar  ^L Clear  ^P Commands  y Approve  n Deny     ^C Quit │
└────────────────────────────────────────────────────────────────────┘
```

### Zones

| Zone | Width | Content |
|------|-------|---------|
| **Header** | full | App title, model name, session token count |
| **Sidebar** | 30 cols, collapsible | `DirectoryTree` (workspace) + task progress list |
| **Main (tabbed)** | `1fr` | Chat tab, Files Changed tab, Events tab |
| **Input bar** | full, docked bottom | `Input` widget + compact status |
| **Footer** | full | Context-sensitive keybindings |

---

## Color Theme: "Loom Dark"

Based on Tokyo Night palette. Optimized for code readability on dark terminals.

```python
LOOM_THEME = {
    "dark": True,
    "primary": "#7dcfff",        # Cyan — borders, active elements
    "secondary": "#bb9af7",      # Lavender — model badges, accents
    "accent": "#ff9e64",         # Orange — rare highlights
    "warning": "#e0af68",        # Amber — approval prompts
    "error": "#f7768e",          # Soft red — errors, denials
    "success": "#9ece6a",        # Green — ok, approved, created
    "foreground": "#c0caf5",     # Off-white — primary text
    "background": "#1a1b26",     # Near-black — main background
    "surface": "#1e2030",        # Dark gray — panel backgrounds
    "panel": "#24283b",          # Slightly lighter — sidebar
}
```

### Color Usage Rules

| Element | Color | Markup |
|---------|-------|--------|
| User messages | Bold green | `[bold #73daca]> message[/]` |
| Model text | Default foreground | plain text |
| Tool name (starting) | Dim | `[dim]tool_name args_preview[/dim]` |
| Tool result ok | Green | `[#9ece6a]ok[/] [dim]12ms 45 lines[/]` |
| Tool result err | Red | `[#f7768e]err[/] [dim]message[/]` |
| Turn summary | Dim | `[dim][3 tools | 1,247 tokens][/]` |
| Status bar text | Muted foreground | `[dim]Ready | path | tokens[/]` |
| Panel borders (inactive) | Surface lighten | `border: solid $surface-lighten-2` |
| Panel borders (focused) | Primary | `border: solid $primary` |

---

## Features by Phase

### Phase 1: Layout & Polish (Foundation)

Restructure the app from single-column to multi-panel. No new data sources yet.

#### 1a. Three-zone layout with Textual CSS

Replace the flat `compose()` with:
```
Screen → Horizontal[
    Sidebar(id="sidebar", dock=left, width=30),
    Vertical(id="main-area")[
        TabbedContent[
            TabPane("Chat", id="tab-chat")    → RichLog
            TabPane("Files",  id="tab-files")  → placeholder
            TabPane("Events", id="tab-events") → placeholder
        ],
        Input(id="user-input", dock=bottom),
        Static(id="status-bar", dock=bottom),
    ],
]
```

CSS: sidebar background `$panel`, main area `$background`, focused panel gets
`$primary` border, inactive gets `$surface-lighten-2`.

#### 1b. Custom dark theme registration

Register `LOOM_THEME` via `App.register_theme()` in `on_mount`. Set as default.

#### 1c. Sidebar — workspace `DirectoryTree`

Use Textual's built-in `DirectoryTree` widget:
- Filter out `.git`, `__pycache__`, `node_modules`, `.venv`, `venv`
- Highlight files touched during the session (cross-reference with tool call events)
- Make collapsible via `Ctrl+B` keybinding (toggle `display: none` on `#sidebar`)
- Show workspace path as the tree root label

#### 1d. Sidebar — task progress panel

Below the file tree, add a small `Static` or `ListView` showing the task_tracker state:
- `✓` completed (green), `◉` in_progress (cyan), `○` pending (dim)
- Pull data from the `task_tracker` tool's in-memory store
- Update whenever a `ToolCallEvent` for `task_tracker` completes
- If no tasks tracked, show `[dim]No tasks tracked[/dim]`

#### 1e. Enhanced status bar

Replace the simple `Static` with a structured status line:
```
Ready | /path/to/workspace | claude-sonnet-4-5 | 3,247 tokens
```
Fields: state (Ready/Thinking.../Running tool_name...), workspace basename,
model name, cumulative session token count.
Track tokens by summing `CoworkTurn.tokens_used` across all turns.

#### 1f. Header with model info

Use `Header(show_clock=False)`. Override the subtitle to show model name and
token count. Or use a custom `Static` bar if Header is too rigid.

#### 1g. Keyboard navigation

| Key | Action |
|-----|--------|
| `Ctrl+B` | Toggle sidebar visibility |
| `Ctrl+1` / `Ctrl+2` / `Ctrl+3` | Switch tabs (Chat / Files / Events) |
| `Ctrl+L` | Clear chat log |
| `Ctrl+P` | Command palette (built-in Textual) |
| `Ctrl+C` | Quit |
| `Tab` / `Shift+Tab` | Cycle focus: input → chat → sidebar |
| `y` / `n` / `a` / `Esc` | Approval modal (unchanged) |

**Tests:** Textual pilot tests for layout rendering, sidebar toggle, tab switching,
keybindings.

---

### Phase 2: Rich Chat Rendering

Make the chat log beautiful and information-dense.

#### 2a. Collapsible tool call blocks

Replace inline tool call text with `Collapsible` widgets inside the chat log.
Since `RichLog` doesn't support embedded widgets, switch the chat area to a
`VerticalScroll` container that dynamically appends `Static` and `Collapsible`
children.

Each tool call renders as:
```
▸ read_file src/auth.py                          ok 12ms 45 lines
```
Expanding shows:
```
▾ read_file src/auth.py                          ok 12ms 45 lines
  ┌────────────────────────────────────────────────────┐
  │ def validate_token(token):                         │
  │     """Validate JWT token."""                       │
  │     ...                                            │
  └────────────────────────────────────────────────────┘
```

For write/edit tools, show a mini diff preview inside the collapsible.

#### 2b. Syntax-highlighted code blocks

When tool output contains code (detected by file extension from tool args or
fenced code blocks in model text), render with Rich `Syntax` widget using the
Loom dark theme colors.

#### 2c. Markdown rendering for model text

Use Rich's `Markdown` renderer for model responses that contain markdown
formatting (headers, lists, code blocks, bold/italic).

#### 2d. User message styling

User messages get a left-border accent:
```
┃ > Fix the authentication bug in src/auth.py
```
Using Rich markup: `[bold #73daca]> {message}[/]` with a vertical bar gutter.

#### 2e. Turn separator and summary

After each complete turn, render a compact summary line:
```
─── 3 tools | 1,247 tokens | claude-sonnet-4-5 | 2.3s ─────────────
```
Using a horizontal rule with embedded stats. Track elapsed time per turn.

#### 2f. Streaming text with cursor indicator

During streaming, append a blinking block cursor `▌` after the last token.
Remove it when the stream completes. Use `set_interval` to toggle visibility.

#### 2g. Auto-scroll with scroll-lock

Auto-scroll the chat to bottom as new content arrives (current behavior).
But if the user manually scrolls up, disable auto-scroll until they scroll
back to the bottom. Show a "↓ New messages" indicator when auto-scroll is off.

**Tests:** Snapshot tests for rendered tool calls, markdown, code blocks.
Pilot tests for collapsible expand/collapse, scroll behavior.

---

### Phase 3: Files Changed Tab

Surface the workspace changelog that already exists in the engine.

#### 3a. Files changed summary table

`DataTable` in the "Files Changed" tab with columns:
```
| Status   | File              | Subtask | Time     |
|----------|-------------------|---------|----------|
| Created  | src/auth.py       | —       | 12:34:05 |
| Modified | src/utils.py      | —       | 12:34:12 |
| Deleted  | tests/old_test.py | —       | 12:34:18 |
```

Status column color-coded: Created=green, Modified=amber, Deleted=red, Renamed=cyan.

Data source: The `ChangeLog` instance from `CoworkSession`. Currently the cowork
session doesn't expose a changelog — we need to create one and pass it to the
`ToolContext` so file tools record changes.

#### 3b. Inline diff viewer

When the user selects a row in the files table (Enter or click), show a unified
diff below the table using Rich `Syntax` with `diff` language highlighting.

Data source: `DiffGenerator.generate(changelog, file_path)` — this already exists
in `tools/workspace.py`.

#### 3c. Revert action

When viewing a file's changes, offer a `Ctrl+Z` keybinding to revert that file.
Show a confirmation modal first. Uses `changelog.revert_entry(entry_id)`.

#### 3d. File change notifications

When a file operation completes in the chat tab, briefly highlight the changed
file in the sidebar's `DirectoryTree` (flash the node). Also increment a badge
counter on the "Files Changed" tab label: `Files (3)`.

**Tests:** Unit tests for changelog integration, pilot tests for table rendering,
diff display, revert confirmation.

---

### Phase 4: Events & Memory Tab

Surface the event stream and memory system.

#### 4a. Live event log

`DataTable` in the "Events" tab streaming events from the session:
```
| Time     | Type              | Detail                          |
|----------|-------------------|---------------------------------|
| 12:34:01 | tool_call_started | read_file src/auth.py           |
| 12:34:01 | tool_call_done    | ok 12ms                         |
| 12:34:02 | tool_call_started | ripgrep_search /validate_token/ |
| 12:34:02 | tool_call_done    | ok 3ms 2 results                |
```

For cowork mode: derive events from `ToolCallEvent` yields.
For task mode (future): subscribe to `EventBus` events via SSE.

#### 4b. Memory entry viewer

If/when memory entries are available (task mode), show them in a filterable
list with entry type badges:
- `decision` (blue), `error` (red), `discovery` (green), `tool_result` (dim),
  `user_instruction` (yellow), `artifact` (cyan), `context` (gray)

Selecting an entry shows its full `detail` field in a detail panel below.

#### 4c. Token usage chart

A compact sparkline or bar showing token usage per turn across the session.
Helps users understand cost and context window consumption.

Use Textual's `Sparkline` widget (built-in) fed with `tokens_used` per turn.

**Tests:** Pilot tests for event log updates, memory filtering.

---

### Phase 5: Enhanced Modals

Upgrade the approval and ask_user modals.

#### 5a. Rich approval modal

Current modal is plain text. Enhance with:
- Tool name in bold cyan
- Full args rendered as syntax-highlighted JSON (collapsed if >5 lines)
- Risk assessment: show which category the tool falls in (write/execute/delete)
- Previous approval history: "You've approved shell_execute 3 times this session"
- Keyboard: `y`/`n`/`a`/`Esc` (unchanged)

#### 5b. Ask-user modal with Rich formatting

- Render the question with markdown support (model may send formatted questions)
- Options as a `RadioSet` or numbered list with keyboard selection
- Input field for free-text answers
- Show context: "The model is asking because..." (from tool args if available)

#### 5c. Error modal

When a tool call fails or the model encounters an error, show a brief error
toast/notification at the bottom of the screen rather than dumping into the chat.
Use Textual's `notify()` method for non-blocking notifications.

**Tests:** Pilot tests for modal rendering, keyboard interactions.

---

### Phase 6: Command Palette & Power Features

#### 6a. Custom command provider

Register commands in Textual's built-in `Ctrl+P` palette:
- "Clear conversation" — clear chat
- "Toggle sidebar" — show/hide sidebar
- "Switch to Chat/Files/Events" — tab navigation
- "Export conversation" — save chat to file
- "Show model info" — display model name, tier, roles
- "List tools" — show all 16 tools with descriptions
- "Reset token counter" — zero the session counter

#### 6b. Slash commands in input

Extend the existing `/quit`, `/clear` with:
- `/model` — show current model info
- `/tools` — list available tools
- `/tokens` — show token usage breakdown
- `/export [path]` — save conversation to file
- `/diff [file]` — show diff for a specific file
- `/help` — show available commands

#### 6c. Notification toasts

Use `self.notify()` for non-critical feedback:
- "File saved: src/auth.py" (on write_file completion)
- "3 files changed" (on turn completion with file changes)
- "Session exported to chat_log.md"

**Tests:** Command palette provider tests, slash command parsing tests.

---

## Data Flow Changes

### Changelog in CoworkSession

The current `CoworkSession` doesn't create a `ChangeLog`. To power the Files
Changed tab, we need to:

1. Create a `ChangeLog(data_dir, workspace)` in `CoworkSession.__init__`
2. Pass it via `ToolContext` to all tool executions
3. Expose it as `session.changelog` property
4. The TUI reads `session.changelog.get_entries()` and
   `DiffGenerator.generate()` to populate the Files tab

### Token Tracking

Add a `total_tokens` accumulator to `CoworkSession`:
- Increment by `CoworkTurn.tokens_used` after each turn
- Expose as `session.total_tokens` property
- TUI reads this for the status bar and header

### Task Tracker Integration

The `task_tracker` tool stores tasks in-memory at the module level.
The TUI needs to read this state for the sidebar progress panel.
Options:
- Import the tool's `_tasks` dict directly (simple, works for single-session)
- Add a `get_tasks()` class method to `TaskTrackerTool`
- Listen for `ToolCallEvent` where `event.name == "task_tracker"` and
  parse the `event.result.data` to extract task state

The third option is cleanest — no coupling to tool internals.

---

## File Structure

```
src/loom/tui/
    __init__.py
    app.py              # LoomApp — main application (rewritten)
    theme.py            # LOOM_THEME definition and registration
    widgets/
        __init__.py
        chat_log.py     # ChatLog — VerticalScroll with message/tool widgets
        tool_call.py    # ToolCallWidget — Collapsible tool call display
        file_panel.py   # FilesChangedPanel — DataTable + diff viewer
        event_panel.py  # EventPanel — DataTable of session events
        sidebar.py      # Sidebar — DirectoryTree + task progress
        status_bar.py   # StatusBar — structured status line
    screens/
        __init__.py
        approval.py     # ToolApprovalScreen (enhanced)
        ask_user.py     # AskUserScreen (enhanced)
    commands.py         # Command palette provider
    api_client.py       # Legacy REST client (unchanged)
```

---

## Implementation Order

| # | Phase | Effort | Depends On | Impact |
|---|-------|--------|------------|--------|
| 1 | Layout & Polish | Medium | — | High — transforms the UX foundation |
| 2 | Rich Chat Rendering | Medium | Phase 1 | High — makes chat beautiful and useful |
| 3 | Files Changed Tab | Medium | Phase 1 + changelog wiring | High — shows what the agent changed |
| 4 | Events & Memory Tab | Low | Phase 1 | Medium — debugging and observability |
| 5 | Enhanced Modals | Low | Phase 1 | Medium — polish on existing feature |
| 6 | Command Palette | Low | Phase 1 | Low — power user convenience |

**Phases 1-2 are the critical path.** They transform the TUI from a chat box into
a proper tool. Phases 3-4 add the data panels. Phases 5-6 are polish.

---

## Migration Strategy

- Rewrite `tui/app.py` from scratch — the current 502-line file is too simple
  to incrementally evolve into the target architecture
- Keep `tui/api_client.py` unchanged (legacy server-mode client)
- Extract new widgets into `tui/widgets/` subpackage
- Extract enhanced modals into `tui/screens/` subpackage
- Move theme to `tui/theme.py`
- All existing tests in `test_tui.py` will need updates for the new widget
  structure, but the helper function tests (`_tool_args_preview`, `_trunc`, etc.)
  can be preserved

---

## Acceptance Criteria

### Must Have (Phases 1-2)
- [ ] Three-zone layout renders correctly (sidebar + tabbed main + input)
- [ ] Custom dark theme with consistent color palette
- [ ] `DirectoryTree` shows workspace files, filters junk directories
- [ ] Task progress panel shows task_tracker state with status icons
- [ ] `Ctrl+B` toggles sidebar visibility
- [ ] Tab switching works with `Ctrl+1/2/3` and mouse clicks
- [ ] Tool calls render as collapsible blocks with expand/collapse
- [ ] Code in tool output is syntax-highlighted
- [ ] Model text renders markdown formatting
- [ ] Turn separators show tool count, tokens, model, elapsed time
- [ ] Status bar shows state, workspace, model, token count
- [ ] Auto-scroll with scroll-lock on manual scroll-up
- [ ] All existing functionality preserved (streaming, approval, ask_user)

### Should Have (Phases 3-4)
- [ ] Files Changed tab shows created/modified/deleted files
- [ ] Selecting a file shows unified diff
- [ ] File revert with confirmation modal
- [ ] Events tab shows live tool call log
- [ ] Token usage sparkline

### Nice to Have (Phases 5-6)
- [ ] Enhanced approval modal with JSON args preview
- [ ] Error notifications via `notify()`
- [ ] Command palette with custom commands
- [ ] Slash commands for model info, tools, export
- [ ] "New messages" indicator on scroll-lock

---

## References

- Textual docs: widgets, layout, themes, command palette, screens
- OpenCode TUI: three-zone layout with sidebar, tabbed content, agent panel
- Toad (Will McGugan): universal UI for agentic coding, Textual-based
- lazygit: stacked sidebar panels, keyboard-first navigation
- Posting: Textual HTTP client with tabbed content, dark theme
- Claude Code: inline tool calls, streaming, status line, progress tracking
- Tokyo Night color scheme: CIELAB-optimized dark palette for code readability
