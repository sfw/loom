"""Enhanced tool approval modal screen."""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static


def _escape_markup(value: object | None) -> str:
    """Escape rich markup brackets in dynamic modal content."""
    return str(value or "").replace("[", "\\[")


class ToolApprovalScreen(ModalScreen[str]):
    """Prompt the user to approve a tool call.

    Returns: "approve", "approve_all", or "deny"
    """

    _inherit_bindings = False

    BINDINGS = [
        Binding("y", "approve", "Yes"),
        Binding("a", "approve_all", "Always"),
        Binding("n", "deny", "No"),
        Binding("escape", "deny", "Cancel"),
        Binding("ctrl+c", "deny", show=False),
        Binding("ctrl+z", "deny", show=False),
    ]

    CSS = """
    ToolApprovalScreen {
        align: center middle;
    }
    #approval-dialog {
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

    def __init__(
        self,
        tool_name: str,
        args_preview: str,
        risk_info: dict | None = None,
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args_preview = args_preview
        self._risk_info = risk_info if isinstance(risk_info, dict) else None

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

        warning_block: list[Label] = []
        if self._risk_info:
            risk_level = _escape_markup(
                str(self._risk_info.get("risk_level", "high") or "high").upper(),
            )
            action_class = _escape_markup(
                self._risk_info.get("action_class", "destructive action")
                or "destructive action",
            )
            impact_preview = _escape_markup(
                str(self._risk_info.get("impact_preview", "") or "").strip(),
            )
            consequences = _escape_markup(
                str(self._risk_info.get("consequences", "") or "").strip(),
            )
            warning_block = [
                Label(
                    f"[bold #f7768e]HIGH RISK: {risk_level}[/] [#f7768e]{action_class}[/]",
                ),
                Label(
                    (
                        f"[#e0af68]Impact:[/] {impact_preview}"
                        if impact_preview
                        else "[#e0af68]Impact:[/] Potential destructive changes"
                    ),
                ),
                Label(
                    (
                        f"[#f7768e]Consequence:[/] {consequences}"
                        if consequences
                        else (
                            "[#f7768e]Consequence:[/] "
                            "Potential irreversible data loss."
                        )
                    ),
                ),
            ]

        yield Vertical(
            Label("[bold #e0af68]Approve tool call?[/]"),
            Label(
                f"[bold #7dcfff]{_escape_markup(self._tool_name)}[/]{risk_label}"
            ),
            *warning_block,
            Static(
                f"[dim]{_escape_markup(self._args_preview)}[/dim]",
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

    def on_key(self, event: events.Key) -> None:
        """Fallback key handling for terminals where bindings are flaky."""
        key = event.key.lower()
        if key == "y":
            self.dismiss("approve")
            event.stop()
            event.prevent_default()
            return
        if key == "a":
            self.dismiss("approve_all")
            event.stop()
            event.prevent_default()
            return
        if key in {"n", "escape", "ctrl+c", "ctrl+z"}:
            self.dismiss("deny")
            event.stop()
            event.prevent_default()


def _tool_risk(tool_name: str) -> str:
    """Categorize tool risk for visual display."""
    if tool_name in ("write_file", "edit_file"):
        return "write"
    if tool_name in ("shell_execute", "git_command"):
        return "execute"
    if tool_name == "delete_file":
        return "delete"
    return "read"
