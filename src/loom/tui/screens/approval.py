"""Enhanced tool approval modal screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static


class ToolApprovalScreen(ModalScreen[str]):
    """Prompt the user to approve a tool call.

    Returns: "approve", "approve_all", or "deny"
    """

    BINDINGS = [
        Binding("y", "approve", "Yes"),
        Binding("a", "approve_all", "Always"),
        Binding("n", "deny", "No"),
        Binding("escape", "deny", "Cancel"),
    ]

    CSS = """
    #approval-dialog {
        align: center middle;
        width: 72;
        height: auto;
        max-height: 20;
        border: solid $warning;
        padding: 1 2;
        background: $surface;
    }
    #approval-args {
        max-height: 8;
        overflow-y: auto;
        margin: 1 0;
        padding: 0 1;
        background: $panel;
    }
    """

    def __init__(self, tool_name: str, args_preview: str) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args_preview = args_preview

    def compose(self) -> ComposeResult:
        # Determine risk category for visual cue
        risk = _tool_risk(self._tool_name)
        risk_label = ""
        if risk == "write":
            risk_label = " [#e0af68][write][/]"
        elif risk == "execute":
            risk_label = " [#f7768e][execute][/]"
        elif risk == "delete":
            risk_label = " [#f7768e][delete][/]"

        yield Vertical(
            Label("[bold #e0af68]Approve tool call?[/]"),
            Label(
                f"[bold #7dcfff]{self._tool_name}[/]{risk_label}"
            ),
            Static(
                f"[dim]{self._args_preview}[/dim]",
                id="approval-args",
            ),
            Label(""),
            Label(
                "[#9ece6a]y[/] Yes  "
                "[#7dcfff]a[/] Always allow  "
                "[#f7768e]n[/] No  "
                "[dim]Esc[/dim] Cancel"
            ),
            id="approval-dialog",
        )

    def action_approve(self) -> None:
        self.dismiss("approve")

    def action_approve_all(self) -> None:
        self.dismiss("approve_all")

    def action_deny(self) -> None:
        self.dismiss("deny")


def _tool_risk(tool_name: str) -> str:
    """Categorize tool risk for visual display."""
    if tool_name in ("write_file", "edit_file"):
        return "write"
    if tool_name in ("shell_execute", "git_command"):
        return "execute"
    if tool_name == "delete_file":
        return "delete"
    return "read"
