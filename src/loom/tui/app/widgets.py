"""Process-run widgets for the Loom TUI app package."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, VerticalScroll
from textual.widgets import Button, Static

from loom.tui.widgets import ChatLog

from .constants import _PROCESS_STATUS_LABEL, _escape_markup_text


class ProcessRunList(VerticalScroll):
    """Scrollable checklist list used by process-run Progress/Outputs panes."""

    def __init__(
        self,
        *,
        empty_message: str,
        auto_follow: bool = True,
        follow_mode: str = "tail",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._auto_follow = bool(auto_follow)
        self._empty_message = str(empty_message or "No items")
        self._follow_mode = (
            str(follow_mode).strip().lower()
            if str(follow_mode).strip().lower() in {"tail", "active"}
            else "tail"
        )
        self._rows: list[dict] = []
        self._pending_rows: list[dict] | None = None
        self._body = Static("", classes="process-run-list-body", expand=True)

    def compose(self) -> ComposeResult:
        yield self._body

    def set_rows(self, rows: list[dict]) -> None:
        """Replace rendered rows and keep the newest rows visible."""
        normalized = [row for row in rows if isinstance(row, dict)]
        if not self.is_attached:
            self._pending_rows = normalized
            return
        self._rows = normalized
        self._body.update(self._render_rows())
        self._scroll_to_latest()

    def on_mount(self) -> None:
        """Flush pre-mount row updates once the widget is attached."""
        if self._pending_rows is not None:
            pending = list(self._pending_rows)
            self._pending_rows = None
            self.set_rows(pending)
            return
        self._body.update(self._render_rows())

    def _render_rows(self) -> Text:
        if not self._rows:
            empty = Text.from_markup(f"[dim]{self._empty_message}[/dim]")
            empty.no_wrap = False
            empty.overflow = "fold"
            return empty

        lines: list[str] = []
        for row in self._rows:
            status = str(row.get("status", "pending")).strip()
            content = _escape_markup_text(row.get("content", "?")).strip() or "?"
            if status == "completed":
                icon = "[#9ece6a]\u2713[/]"
            elif status == "in_progress":
                icon = "[#7dcfff]\u25c9[/]"
            elif status == "failed":
                icon = "[#f7768e]\u2717[/]"
            elif status == "skipped":
                icon = "[dim]-[/dim]"
            else:
                icon = "[dim]\u25cb[/dim]"
            lines.append(f"{icon} {content}")
        rendered = Text.from_markup("\n".join(lines))
        rendered.no_wrap = False
        rendered.overflow = "fold"
        return rendered

    def _scroll_to_latest(self) -> None:
        if not self._auto_follow or not self.is_attached:
            return
        if self._follow_mode == "active":
            focus_index = self._active_focus_index()
            if focus_index is None:
                return

            # Keep the most relevant in-flight / recently-finished rows in view.
            target_line = max(focus_index - 2, 0)

            def _focus() -> None:
                self.scroll_to(y=target_line, animate=False, force=True)

            self.call_after_refresh(_focus)
            return
        self.call_after_refresh(self.scroll_end, animate=False)

    def _active_focus_index(self) -> int | None:
        """Return the output row index that should stay in view for active-follow."""
        active_idx: int | None = None
        complete_idx: int | None = None
        terminal_idx: int | None = None
        for idx, row in enumerate(self._rows):
            status = str(row.get("status", "pending")).strip()
            if status == "in_progress":
                active_idx = idx
            elif status == "completed":
                complete_idx = idx
            elif status in {"failed", "skipped"}:
                terminal_idx = idx
        if active_idx is not None:
            return active_idx
        if complete_idx is not None:
            return complete_idx
        return terminal_idx


class ProcessRunPane(VerticalScroll):
    """A per-run process pane with status, progress rows, and run log."""

    DEFAULT_CSS = """
    ProcessRunPane {
        height: 1fr;
        padding: 0 1;
        overflow: auto auto;
    }
    ProcessRunPane .process-run-header {
        color: $text;
        text-style: bold;
        width: 1fr;
    }
    ProcessRunPane .process-run-meta-row {
        grid-size: 2 1;
        grid-columns: 1fr auto;
        grid-rows: auto;
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }
    ProcessRunPane .process-run-meta {
        color: $text-muted;
        margin: 0;
    }
    ProcessRunPane .process-run-actions {
        margin: 0;
        height: auto;
        width: auto;
    }
    ProcessRunPane .process-run-restart-btn {
        width: auto;
        min-width: 22;
    }
    ProcessRunPane .process-run-control-btn {
        width: 7;
        min-width: 7;
        max-width: 7;
        height: 3;
        min-height: 3;
        max-height: 3;
        margin: 0;
        padding: 0;
        content-align: center middle;
        border: solid $primary-darken-1;
        background: $surface;
        color: $text-muted;
        text-style: bold;
    }
    ProcessRunPane .process-run-toggle-btn {
        background: $surface;
        color: $text;
        border: solid $primary-darken-1;
    }
    ProcessRunPane .process-run-toggle-btn:hover {
        background: $surface;
        color: $primary;
        border: solid $primary;
    }
    ProcessRunPane .process-run-toggle-btn:focus {
        background: $surface;
        color: $primary;
        text-style: bold;
        border: solid $primary;
    }
    ProcessRunPane .process-run-toggle-btn:disabled {
        background: $surface;
        color: $text-muted;
        border: solid $primary-darken-2;
    }
    ProcessRunPane .process-run-stop-btn {
        margin-left: 1;
        background: $surface;
        color: $error;
        border: solid $primary-darken-1;
    }
    ProcessRunPane .process-run-stop-btn:hover {
        background: $surface;
        color: $error;
        border: solid $error;
    }
    ProcessRunPane .process-run-stop-btn:focus {
        background: $surface;
        color: $error;
        text-style: bold;
        border: solid $error;
    }
    ProcessRunPane .process-run-stop-btn:disabled {
        background: $surface;
        color: $text-muted;
        border: solid $primary-darken-2;
    }
    ProcessRunPane .process-run-section {
        color: $text-muted;
        text-style: bold;
        margin: 1 0 0 0;
    }
    ProcessRunPane .process-run-list {
        border: round $surface-lighten-1;
        margin: 0;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    ProcessRunPane .process-run-list-body {
        width: 100%;
    }
    ProcessRunPane #process-run-progress {
        height: 9;
        min-height: 7;
        max-height: 14;
    }
    ProcessRunPane #process-run-outputs {
        height: 7;
        min-height: 5;
        max-height: 12;
    }
    ProcessRunPane ChatLog {
        height: 1fr;
        min-height: 4;
        border: round $surface-lighten-1;
        margin: 0;
    }
    """

    def __init__(self, *, run_id: str, process_name: str, goal: str) -> None:
        super().__init__()
        self._run_id = run_id
        self._process_name = process_name
        self._goal = goal
        self._pending_tasks: list[dict] | None = None
        self._pending_outputs: list[dict] | None = None
        self._pending_activity: list[str] = []
        self._pending_keyed_activity: dict[str, str] = {}
        self._pending_results: list[tuple[str, bool]] = []
        self._header = Static(classes="process-run-header")
        self._meta_row = Grid(classes="process-run-meta-row")
        self._meta = Static(classes="process-run-meta")
        self._actions = Horizontal(classes="process-run-actions")
        self._actions.display = False
        self._toggle_pause_button = Button(
            "\u23f8",
            id=f"process-run-toggle-{run_id}",
            classes="process-run-control-btn process-run-toggle-btn",
        )
        self._stop_button = Button(
            "\u23f9",
            id=f"process-run-stop-{run_id}",
            classes="process-run-control-btn process-run-stop-btn",
        )
        self._restart_button = Button(
            "Restart Failed Run",
            id=f"process-run-restart-{run_id}",
            classes="process-run-restart-btn",
            variant="primary",
        )
        self._toggle_pause_button.display = False
        self._toggle_pause_button.disabled = True
        self._stop_button.display = False
        self._stop_button.disabled = True
        self._restart_button.display = False
        self._restart_button.disabled = True
        self._progress_label = Static("Progress", classes="process-run-section")
        self._progress = ProcessRunList(
            id="process-run-progress",
            classes="process-run-list",
            auto_follow=True,
            follow_mode="active",
            empty_message="No progress yet",
        )
        self._outputs_label = Static("Outputs", classes="process-run-section")
        self._outputs = ProcessRunList(
            id="process-run-outputs",
            classes="process-run-list",
            auto_follow=True,
            follow_mode="active",
            empty_message="No outputs yet",
        )
        self._log_label = Static("Activity", classes="process-run-section")
        self._log = ChatLog()

    def compose(self) -> ComposeResult:
        yield self._header
        with self._meta_row:
            yield self._meta
            with self._actions:
                yield self._toggle_pause_button
                yield self._stop_button
                yield self._restart_button
        yield self._progress_label
        yield self._progress
        yield self._outputs_label
        yield self._outputs
        yield self._log_label
        yield self._log

    def set_status_header(
        self,
        *,
        status: str,
        elapsed: str,
        task_id: str = "",
        working_folder: str = "",
        pending_inject_count: int = 0,
        pending_inject_preview: str = "",
    ) -> None:
        """Update run header metadata."""
        label = _PROCESS_STATUS_LABEL.get(status, status.title())
        self._header.update(
            f"[bold]{self._process_name}[/bold] [dim]#{self._run_id}[/dim]  "
            f"[{self._status_color(status)}]{label}[/]  [dim]{elapsed}[/dim]"
        )
        meta = f"[dim]Goal:[/] {self._goal}"
        if task_id:
            meta += f"\n[dim]Task:[/] {task_id}"
        if str(working_folder or "").strip():
            safe_folder = _escape_markup_text(working_folder)
            meta += f"\n[dim]Working folder:[/] {safe_folder}"
        if int(pending_inject_count or 0) > 0:
            preview = " ".join(str(pending_inject_preview or "").split()).strip()
            if len(preview) > 72:
                preview = f"{preview[:71]}\u2026"
            if preview:
                safe_preview = _escape_markup_text(preview)
                meta += (
                    f"\n[dim]Inject queue:[/] {int(pending_inject_count)} pending"
                    f" \u2022 {safe_preview}"
                )
            else:
                meta += f"\n[dim]Inject queue:[/] {int(pending_inject_count)} pending"
        self._meta.update(meta)
        terminal = status in {"completed", "failed", "cancelled", "force_closed", "cancel_failed"}
        can_pause = status == "running"
        can_play = status == "paused"
        can_stop = status in {"queued", "running", "paused", "cancel_requested"}
        can_restart = status in {"failed", "cancel_failed"}
        show_toggle = can_pause or can_play
        show_stop = not terminal
        self._actions.display = show_toggle or show_stop or can_restart
        self._toggle_pause_button.display = show_toggle
        self._toggle_pause_button.label = "\u25b6" if can_play else "\u23f8"
        self._toggle_pause_button.tooltip = (
            "Resume run" if can_play else "Pause run"
        )
        self._toggle_pause_button.disabled = not (can_pause or can_play)
        self._stop_button.display = show_stop
        self._stop_button.disabled = not can_stop
        self._restart_button.display = can_restart
        self._restart_button.disabled = not can_restart

    def set_tasks(self, tasks: list[dict]) -> None:
        """Replace task rows shown in the progress section."""
        if not self.is_attached:
            self._pending_tasks = list(tasks)
            return
        self._progress.set_rows(tasks)

    def set_outputs(self, outputs: list[dict]) -> None:
        """Replace output rows shown in the outputs section."""
        if not self.is_attached:
            self._pending_outputs = list(outputs)
            return
        self._outputs.set_rows(outputs)

    def add_activity(self, text: str) -> None:
        """Append informational activity text."""
        safe_text = _escape_markup_text(text)
        if not self.is_attached:
            self._pending_activity.append(safe_text)
            return
        self._log.add_info(safe_text)

    def upsert_activity(self, key: str, text: str) -> None:
        """Insert or update one keyed informational activity line."""
        safe_text = _escape_markup_text(text)
        clean_key = str(key or "").strip()
        if not clean_key:
            self.add_activity(text)
            return
        if not self.is_attached:
            self._pending_keyed_activity[clean_key] = safe_text
            return
        self._log.upsert_info_line(clean_key, safe_text)

    def add_result(self, text: str, *, success: bool) -> None:
        """Append final result text."""
        safe_text = _escape_markup_text(text)
        if not self.is_attached:
            self._pending_results.append((safe_text, success))
            return
        if success:
            self._log.add_model_text(safe_text)
            return
        self._log.add_model_text(
            f"[bold #f7768e]Error:[/] {safe_text}",
            markup=True,
        )

    def on_mount(self) -> None:
        """Flush updates queued before the pane was attached to the DOM."""
        if self._pending_tasks is not None:
            self._progress.set_rows(self._pending_tasks)
            self._pending_tasks = None
        if self._pending_outputs is not None:
            self._outputs.set_rows(self._pending_outputs)
            self._pending_outputs = None
        if self._pending_activity:
            for line in self._pending_activity:
                self._log.add_info(line)
            self._pending_activity.clear()
        if self._pending_keyed_activity:
            for key, line in self._pending_keyed_activity.items():
                self._log.upsert_info_line(key, line)
            self._pending_keyed_activity.clear()
        if self._pending_results:
            for text, success in self._pending_results:
                if success:
                    self._log.add_model_text(text)
                else:
                    self._log.add_model_text(
                        f"[bold #f7768e]Error:[/] {text}",
                        markup=True,
                    )
            self._pending_results.clear()

    @staticmethod
    def _status_color(status: str) -> str:
        if status == "completed":
            return "#9ece6a"
        if status == "failed":
            return "#f7768e"
        if status == "cancel_failed":
            return "#f7768e"
        if status == "paused":
            return "#e0af68"
        if status in {"running", "cancel_requested"}:
            return "#7dcfff"
        if status == "force_closed":
            return "#e0af68"
        return "#a9b1d6"
