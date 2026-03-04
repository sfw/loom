"""Process run workspace selection modal."""

from __future__ import annotations

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class ProcessRunWorkspaceScreen(ModalScreen[str | None]):
    """Prompt for process-run working folder selection."""

    _inherit_bindings = False

    BINDINGS = [
        Binding("enter", "use_folder", "Use folder"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ProcessRunWorkspaceScreen {
        align: center middle;
    }
    #process-workspace-dialog {
        width: 82;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #process-workspace-input {
        margin-top: 1;
    }
    #process-workspace-actions {
        width: 100%;
        height: auto;
        align-horizontal: right;
        margin: 1 0 0 0;
    }
    #process-workspace-use-root {
        margin-right: 1;
    }
    """

    def __init__(
        self,
        *,
        process_name: str,
        workspace_root: str,
        suggested_folder: str,
    ) -> None:
        super().__init__()
        self._process_name = str(process_name or "").strip() or "process-run"
        self._workspace_root = str(workspace_root or "").strip()
        self._suggested_folder = str(suggested_folder or "").strip()

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(
                f"[bold #7dcfff]Choose working folder for {self._process_name}[/]",
            ),
            Label(f"[dim]Workspace root:[/] {self._workspace_root}"),
            Label(
                "[dim]Leave blank to use workspace root, "
                "or edit the folder name below.[/dim]",
            ),
            Input(
                value=self._suggested_folder,
                id="process-workspace-input",
                placeholder="Folder name",
            ),
            Horizontal(
                Button("Use Root", id="process-workspace-use-root"),
                Button(
                    "Use Folder",
                    id="process-workspace-use-folder",
                    variant="primary",
                ),
                id="process-workspace-actions",
            ),
            id="process-workspace-dialog",
        )

    def on_mount(self) -> None:
        input_widget = self.query_one("#process-workspace-input", Input)
        input_widget.focus()
        input_widget.cursor_position = len(str(input_widget.value or ""))

    def action_use_folder(self) -> None:
        value = self.query_one("#process-workspace-input", Input).value.strip()
        self.dismiss(value)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#process-workspace-use-root")
    def _on_use_root(self) -> None:
        self.dismiss("")

    @on(Button.Pressed, "#process-workspace-use-folder")
    def _on_use_folder(self) -> None:
        self.action_use_folder()

    @on(Input.Submitted, "#process-workspace-input")
    def _on_input_submitted(self, _event: Input.Submitted) -> None:
        self.action_use_folder()

    def on_key(self, event: events.Key) -> None:
        key = event.key.lower()
        if key == "escape":
            self.dismiss(None)
            event.stop()
            event.prevent_default()
