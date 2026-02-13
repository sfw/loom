# Spec 09: Terminal UI Client

## Overview

The TUI is Loom's primary human interface for V1. Built with Python `textual`, it connects to the API server as a client and provides real-time task monitoring, approval controls, and steering. It is deliberately NOT a chat interface — it's a task monitor, closer to a CI/CD dashboard than a chatbot.

## Launch

```bash
loom tui                    # Connect to running server at configured host:port
loom tui --server http://localhost:9000
```

The TUI is a client of the API. It does not embed the engine. The engine must be running separately via `loom serve`.

## Layout

```
┌─ Loom ─────────────────────────────────────────────────────────────┐
│ Tasks: 3 active │ Models: M2.1 ● Qwen3 ●  │ ↑↓:navigate  q:quit │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─ Task List ─────────────────────────────────────────────────┐   │
│  │ ● [running] Migrate Express to TypeScript    4/7  57%       │   │
│  │ ● [waiting] Add test coverage for auth       0/5  ⏸ approval│   │
│  │ ✓ [done]    Fix CORS headers                 3/3  100%      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─ Plan ──────────────┐  ┌─ Live Output ─────────────────────┐   │
│  │ ✓ Install deps      │  │ [schema-convert] Reading schema   │   │
│  │ ✓ Rename .js → .ts  │  │ from src/models/...               │   │
│  │ ✓ Add tsconfig.json │  │                                   │   │
│  │ → Add type annot.   │  │ Tool: edit_file                   │   │
│  │ ○ Fix type errors   │  │   path: src/models/User.ts        │   │
│  │ ○ Run tests         │  │   +12 lines, -3 lines             │   │
│  │ ○ Final validation  │  │                                   │   │
│  │                     │  │ Tool: read_file                   │   │
│  │                     │  │   path: src/routes/auth.ts        │   │
│  └─────────────────────┘  └───────────────────────────────────┘   │
│                                                                    │
│  ┌─ Files Changed ────────────────────────────────────────────┐   │
│  │ M src/models/User.ts (+12/-3)    A tsconfig.json           │   │
│  │ M src/routes/auth.ts (+23/-8)    M package.json (+5/-1)    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  [d]iff  [r]evert  [a]pprove  [s]teer  [p]ause  [c]ancel        │
└────────────────────────────────────────────────────────────────────┘
```

## Views

### Dashboard View (Task List)
- Shows all tasks with status, progress bar, and summary
- Keyboard navigation: arrow keys to select, Enter to open detail view
- Status indicators: `●` running, `⏸` waiting for approval, `✓` complete, `✗` failed
- Auto-refreshes via SSE subscription to all events

### Task Detail View
- **Left panel: Plan tree** — Subtask list with status icons (✓ complete, → running, ○ pending, ✗ failed)
- **Center panel: Live output** — Streaming model activity, tool calls, and results for the active subtask
- **Bottom panel: Files changed** — Git-style summary (M modified, A added, D deleted) with line counts
- **Status bar: Controls** — Keyboard shortcuts for actions

### Diff View
- Press `d` on a changed file to see the full diff
- Shows before/after with standard unified diff coloring
- Press `Esc` to return to task detail

### Approval Modal
- When the engine requests approval, a modal overlay appears
- Shows: what the engine wants to do, why it needs approval, risk level
- Keys: `y` approve, `n` reject, `r` reject with reason

### Steer Input
- Press `s` to open a text input for mid-task instructions
- Typed instruction is sent via PATCH /tasks/{id}
- Shown as a system event in the live output

## Textual Application Structure

```python
# tui/app.py
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, ListView

class LoomApp(App):
    CSS_PATH = "loom.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "show_diff", "Diff"),
        ("a", "approve", "Approve"),
        ("s", "steer", "Steer"),
        ("p", "pause", "Pause"),
        ("c", "cancel", "Cancel"),
        ("r", "revert", "Revert"),
        ("escape", "back", "Back"),
    ]

    def __init__(self, server_url: str):
        super().__init__()
        self.server_url = server_url
        self.api_client = LoomAPIClient(server_url)

    def compose(self) -> ComposeResult:
        yield Header()
        yield DashboardView()
        yield Footer()

    async def on_mount(self):
        """Start SSE subscription for real-time updates."""
        self.run_worker(self._subscribe_events())

    async def _subscribe_events(self):
        """Background worker consuming SSE stream."""
        async for event in self.api_client.stream_all_events():
            self.post_message(TaskEventMessage(event))
```

## API Client

The TUI uses a thin async client wrapping the REST API:

```python
class LoomAPIClient:
    def __init__(self, base_url: str):
        self._client = httpx.AsyncClient(base_url=base_url)

    async def list_tasks(self) -> list[dict]: ...
    async def get_task(self, task_id: str) -> dict: ...
    async def create_task(self, goal: str, workspace: str = None) -> dict: ...
    async def cancel_task(self, task_id: str) -> dict: ...
    async def approve(self, task_id: str, subtask_id: str) -> dict: ...
    async def steer(self, task_id: str, instruction: str) -> dict: ...
    async def get_diff(self, task_id: str, file_path: str) -> str: ...

    async def stream_all_events(self) -> AsyncIterator[dict]:
        """Subscribe to global SSE stream."""
        ...

    async def stream_task_events(self, task_id: str) -> AsyncIterator[dict]:
        """Subscribe to task-specific SSE stream."""
        ...
```

## Quick Run Mode

For quick tasks, bypass the TUI and run inline in the terminal:

```bash
loom run "Fix all TypeScript errors in src/" --workspace ./my-project
```

This:
1. Starts the engine (if not running)
2. Submits the task
3. Streams progress inline (simple log output, not full TUI)
4. Returns exit code 0 on success, 1 on failure

## Acceptance Criteria

- [ ] `loom tui` launches and connects to running server
- [ ] Dashboard shows all tasks with real-time status updates
- [ ] Task detail view shows plan tree, live output, and files changed
- [ ] Approval modal appears when engine requests approval
- [ ] Steer input sends instruction to running task
- [ ] Diff view shows file changes with color highlighting
- [ ] Cancel stops a running task
- [ ] Keyboard shortcuts work from all views
- [ ] TUI reconnects if SSE connection drops
- [ ] `loom run` works for quick inline task execution
- [ ] TUI does not embed the engine (pure API client)
