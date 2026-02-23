"""MCP configuration management modal."""

from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import replace

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label

from loom.mcp.config import (
    MCPConfigManager,
    MCPConfigManagerError,
    MCPServerView,
    ensure_valid_alias,
    merge_server_edits,
    parse_mcp_server_from_flags,
)

_ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfirmRemoveScreen(ModalScreen[bool]):
    """Small confirmation dialog before destructive MCP deletion."""

    _inherit_bindings = False

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("enter", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ConfirmRemoveScreen {
        align: center middle;
    }
    #mcp-remove-confirm-dialog {
        width: 62;
        height: auto;
        border: solid $warning;
        padding: 1 2;
        background: $surface;
    }
    #mcp-remove-confirm-actions {
        height: auto;
        margin-top: 1;
    }
    #mcp-remove-confirm-actions Button {
        margin-right: 1;
    }
    """

    def __init__(self, alias: str) -> None:
        super().__init__()
        self._alias = alias

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-remove-confirm-dialog"):
            yield Label("[bold #e0af68]Delete MCP server?[/]")
            yield Label(f"Alias: [bold]{self._alias}[/bold]")
            yield Label("This cannot be undone.")
            with Horizontal(id="mcp-remove-confirm-actions"):
                yield Button("Delete", id="mcp-remove-confirm-yes", variant="error")
                yield Button("Cancel", id="mcp-remove-confirm-no")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#mcp-remove-confirm-yes")
    def _confirm_button(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#mcp-remove-confirm-no")
    def _cancel_button(self) -> None:
        self.dismiss(False)

    def on_key(self, event: events.Key) -> None:
        key = event.key.lower()
        if key in ("y", "enter"):
            self.dismiss(True)
            event.stop()
            event.prevent_default()
            return
        if key in ("n", "escape"):
            self.dismiss(False)
            event.stop()
            event.prevent_default()


class ConfirmAliasSwitchScreen(ModalScreen[str]):
    """Confirm loading another alias when there are unsaved edits."""

    _inherit_bindings = False

    BINDINGS = [
        Binding("s", "save", "Save"),
        Binding("d", "discard", "Discard"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ConfirmAliasSwitchScreen {
        align: center middle;
    }
    #mcp-switch-confirm-dialog {
        width: 76;
        height: auto;
        border: solid $warning;
        padding: 1 2;
        background: $surface;
    }
    #mcp-switch-confirm-actions {
        height: auto;
        margin-top: 1;
    }
    #mcp-switch-confirm-actions Button {
        margin-right: 1;
    }
    """

    def __init__(self, *, current_alias: str, target_alias: str) -> None:
        super().__init__()
        self._current_alias = current_alias or "(new entry)"
        self._target_alias = target_alias

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-switch-confirm-dialog"):
            yield Label("[bold #e0af68]Unsaved MCP changes[/]")
            yield Label(
                "Save changes before switching from "
                f"[bold]{self._current_alias}[/bold] to "
                f"[bold]{self._target_alias}[/bold]?"
            )
            yield Label(
                "Choose Save to keep edits, Discard to drop edits, or Esc to cancel.",
            )
            with Horizontal(id="mcp-switch-confirm-actions"):
                yield Button("Save", id="mcp-switch-confirm-save", variant="primary")
                yield Button("Discard", id="mcp-switch-confirm-discard", variant="warning")
                yield Button("Cancel", id="mcp-switch-confirm-cancel")

    def action_save(self) -> None:
        self.dismiss("save")

    def action_discard(self) -> None:
        self.dismiss("discard")

    def action_cancel(self) -> None:
        self.dismiss("cancel")

    @on(Button.Pressed, "#mcp-switch-confirm-save")
    def _on_save_button(self) -> None:
        self.dismiss("save")

    @on(Button.Pressed, "#mcp-switch-confirm-discard")
    def _on_discard_button(self) -> None:
        self.dismiss("discard")

    @on(Button.Pressed, "#mcp-switch-confirm-cancel")
    def _on_cancel_button(self) -> None:
        self.dismiss("cancel")


class MCPManagerScreen(ModalScreen[dict[str, object] | None]):
    """Modal form for MCP server add/edit/remove/test flows."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("ctrl+r", "refresh", "Refresh"),
    ]

    _FORM_FIELD_IDS = (
        "mcp-alias",
        "mcp-command",
        "mcp-args",
        "mcp-cwd",
        "mcp-timeout",
        "mcp-env",
        "mcp-env-ref",
    )

    CSS = """
    MCPManagerScreen {
        align: center middle;
    }
    #mcp-manager-dialog {
        width: 100;
        height: 90%;
        max-height: 46;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
        overflow: hidden;
    }
    #mcp-manager-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #mcp-manager-summary {
        height: 10;
        max-height: 12;
        margin-bottom: 1;
    }
    #mcp-summary-help {
        margin-bottom: 1;
    }
    #mcp-manager-form {
        height: 1fr;
        overflow-y: auto;
        margin-bottom: 1;
    }
    .mcp-label {
        margin-top: 1;
    }
    .mcp-help {
        color: $text-muted;
        margin-top: 0;
        margin-bottom: 1;
    }
    .mcp-input {
        margin-top: 0;
    }
    .mcp-actions-row {
        height: auto;
        margin-top: 1;
    }
    .mcp-actions-row Button {
        margin-right: 1;
    }
    #mcp-manager-footer {
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(self, manager: MCPConfigManager) -> None:
        super().__init__()
        self._manager = manager
        self._views: list[MCPServerView] = []
        self._summary_aliases: list[str] = []
        self._active_alias = ""
        self._baseline_form_state: dict[str, str] = {}
        self._form_dirty = False
        self._suppress_dirty_tracking = False
        self._changed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-manager-dialog"):
            yield Label(
                "[bold #7dcfff]MCP Server Manager[/bold #7dcfff]",
                id="mcp-manager-title",
            )
            summary_table = DataTable(id="mcp-manager-summary")
            summary_table.cursor_type = "row"
            summary_table.zebra_stripes = True
            summary_table.add_columns("Alias", "Status", "Command", "Source")
            yield summary_table
            yield Label(
                "Select a server row to load it below for editing.",
                classes="mcp-help",
                id="mcp-summary-help",
            )

            with VerticalScroll(id="mcp-manager-form"):
                yield Label("Alias", classes="mcp-label")
                yield Input(
                    id="mcp-alias",
                    classes="mcp-input",
                )
                yield Label(
                    "Unique server id used in commands and auth selectors.",
                    classes="mcp-help",
                )
                yield Label("Command", classes="mcp-label")
                yield Input(
                    id="mcp-command",
                    classes="mcp-input",
                )
                yield Label(
                    "Executable to launch the MCP server (required for new entries).",
                    classes="mcp-help",
                )
                yield Label("Args (shell-style string)", classes="mcp-label")
                yield Input(
                    id="mcp-args",
                    classes="mcp-input",
                )
                yield Label(
                    "Optional command arguments; parsed like a shell command.",
                    classes="mcp-help",
                )
                yield Label("Cwd", classes="mcp-label")
                yield Input(
                    id="mcp-cwd",
                    classes="mcp-input",
                )
                yield Label(
                    "Optional working directory for server startup.",
                    classes="mcp-help",
                )
                yield Label("Timeout seconds", classes="mcp-label")
                yield Input(
                    id="mcp-timeout",
                    classes="mcp-input",
                )
                yield Label(
                    "Request timeout for this MCP server (defaults to 30).",
                    classes="mcp-help",
                )
                yield Label("Env pairs (comma-separated KEY=VALUE)", classes="mcp-label")
                yield Input(
                    id="mcp-env",
                    classes="mcp-input",
                )
                yield Label(
                    "Literal env values written into mcp.toml for this alias.",
                    classes="mcp-help",
                )
                yield Label(
                    "Env refs (comma-separated KEY=ENV_VAR)",
                    classes="mcp-label",
                )
                yield Input(
                    id="mcp-env-ref",
                    classes="mcp-input",
                )
                yield Label(
                    "Runtime env indirection; saved as KEY=${ENV_VAR}.",
                    classes="mcp-help",
                )

            with Horizontal(classes="mcp-actions-row", id="mcp-actions-primary"):
                yield Button("Refresh", id="mcp-btn-refresh")
                yield Button("Load Alias", id="mcp-btn-load")
                yield Button("Save/Add", id="mcp-btn-save", variant="primary")
                yield Button("Test", id="mcp-btn-test")
                yield Button("Close", id="mcp-btn-close")
            with Horizontal(classes="mcp-actions-row", id="mcp-actions-secondary"):
                yield Button("Enable", id="mcp-btn-enable")
                yield Button("Disable", id="mcp-btn-disable")
                yield Button("Remove", id="mcp-btn-remove", variant="error")

            yield Label(
                "[dim]Server list loads automatically on open. "
                "Select a server row to edit it. Save/Add upserts; new aliases start enabled. "
                "Use Enable/Disable buttons to change activation state. "
                "Use separate MCP aliases for multiple accounts.[/dim]",
                id="mcp-manager-footer",
            )

    async def on_mount(self) -> None:
        self._set_form_values(
            alias="",
            command="",
            args="",
            cwd="",
            timeout="30",
            env="",
            env_ref="",
        )
        self._mark_form_clean(active_alias="")
        await self._refresh_summary()
        self.query_one("#mcp-alias", Input).focus()

    def action_close(self) -> None:
        self.dismiss({"changed": self._changed})

    async def action_refresh(self) -> None:
        await self._refresh_summary()

    @on(Button.Pressed)
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "mcp-btn-close":
            self.action_close()
            return
        if button_id == "mcp-btn-refresh":
            await self._refresh_summary()
            return
        if button_id == "mcp-btn-load":
            await self._request_alias_switch(self._current_alias())
            return
        if button_id == "mcp-btn-save":
            await self._save_current_form()
            return
        if button_id == "mcp-btn-enable":
            await self._set_alias_enabled(True)
            return
        if button_id == "mcp-btn-disable":
            await self._set_alias_enabled(False)
            return
        if button_id == "mcp-btn-remove":
            await self._remove_alias()
            return
        if button_id == "mcp-btn-test":
            await self._test_alias()
            return

    @on(Input.Changed)
    def _on_form_input_changed(self, event: Input.Changed) -> None:
        if event.input.id not in self._FORM_FIELD_IDS:
            return
        self._update_form_dirty()

    @on(DataTable.RowSelected, "#mcp-manager-summary")
    async def _on_summary_row_selected(self, event: DataTable.RowSelected) -> None:
        alias = str(getattr(event.row_key, "value", "") or "").strip()
        if not alias and 0 <= event.cursor_row < len(self._summary_aliases):
            alias = self._summary_aliases[event.cursor_row]
        await self._request_alias_switch(alias)

    def _capture_form_state(self) -> dict[str, str]:
        return {
            field_id: self.query_one(f"#{field_id}", Input).value
            for field_id in self._FORM_FIELD_IDS
        }

    def _mark_form_clean(self, *, active_alias: str | None = None) -> None:
        self._baseline_form_state = self._capture_form_state()
        self._form_dirty = False
        if active_alias is not None:
            self._active_alias = active_alias

    def _update_form_dirty(self) -> None:
        if self._suppress_dirty_tracking:
            return
        self._form_dirty = self._capture_form_state() != self._baseline_form_state

    def _set_form_values(
        self,
        *,
        alias: str,
        command: str,
        args: str,
        cwd: str,
        timeout: str,
        env: str,
        env_ref: str,
    ) -> None:
        self._suppress_dirty_tracking = True
        try:
            self.query_one("#mcp-alias", Input).value = alias
            self.query_one("#mcp-command", Input).value = command
            self.query_one("#mcp-args", Input).value = args
            self.query_one("#mcp-cwd", Input).value = cwd
            self.query_one("#mcp-timeout", Input).value = timeout
            self.query_one("#mcp-env", Input).value = env
            self.query_one("#mcp-env-ref", Input).value = env_ref
        finally:
            self._suppress_dirty_tracking = False

    def _set_blank_form(self) -> None:
        self._set_form_values(
            alias="",
            command="",
            args="",
            cwd="",
            timeout="30",
            env="",
            env_ref="",
        )
        self._mark_form_clean(active_alias="")

    async def _refresh_summary(self) -> None:
        try:
            self._views = (await asyncio.to_thread(self._manager.load)).as_views()
        except Exception as e:
            self.notify(f"MCP load failed: {e}", severity="error")
            return
        self._render_summary()

    def _render_summary(self) -> None:
        if not self.is_mounted:
            return

        table = self.query_one("#mcp-manager-summary", DataTable)
        table.clear()
        self._summary_aliases = []

        for view in self._views:
            status = "enabled" if view.server.enabled else "disabled"
            command = " ".join([view.server.command, *view.server.args]).strip()
            source = view.source
            if view.source == "legacy":
                source = "legacy"
            table.add_row(view.alias, status, command, source, key=view.alias)
            self._summary_aliases.append(view.alias)

        selected = self._active_alias or self._current_alias()
        if selected:
            self._select_summary_alias(selected)

    def _select_summary_alias(self, alias: str) -> None:
        if not alias:
            return
        table = self.query_one("#mcp-manager-summary", DataTable)
        for row_index, candidate in enumerate(self._summary_aliases):
            if candidate == alias:
                table.move_cursor(row=row_index, column=0, scroll=True)
                return

    def _current_alias(self) -> str:
        return self.query_one("#mcp-alias", Input).value.strip()

    async def _request_alias_switch(self, alias: str) -> None:
        clean_alias = str(alias or "").strip()
        if not clean_alias:
            self.notify("Enter alias first.", severity="warning")
            return

        if clean_alias == self._active_alias and not self._form_dirty:
            return

        if self._form_dirty and clean_alias != self._active_alias:
            self.app.push_screen(
                ConfirmAliasSwitchScreen(
                    current_alias=self._active_alias,
                    target_alias=clean_alias,
                ),
                callback=lambda decision: self._on_alias_switch_decision(
                    clean_alias,
                    str(decision or "cancel").lower(),
                ),
            )
            return

        await self._load_alias_into_form(alias=clean_alias)

    def _on_alias_switch_decision(self, target_alias: str, decision: str) -> None:
        if decision == "cancel":
            self._select_summary_alias(self._active_alias)
            return
        self.run_worker(
            self._complete_alias_switch(target_alias, decision),
            group="mcp-manager-switch",
            exclusive=True,
        )

    async def _complete_alias_switch(self, target_alias: str, decision: str) -> None:
        if decision == "save":
            saved = await self._save_current_form(notify_success=False)
            if not saved:
                self._select_summary_alias(self._active_alias)
                return
        elif decision != "discard":
            self._select_summary_alias(self._active_alias)
            return

        loaded = await self._load_alias_into_form(alias=target_alias)
        if not loaded:
            self._select_summary_alias(self._active_alias)
            return

        if decision == "save":
            self.notify(f"Saved changes and loaded alias: {target_alias}")

    async def _load_alias_into_form(self, alias: str | None = None) -> bool:
        raw_alias = str(alias or self._current_alias()).strip()
        if not raw_alias:
            self.notify("Enter alias first.", severity="warning")
            return False
        try:
            clean_alias = ensure_valid_alias(raw_alias)
            view = await asyncio.to_thread(self._manager.get_view, clean_alias)
        except Exception as e:
            self.notify(str(e), severity="error")
            return False
        if view is None:
            self.notify(f"Alias not found: {raw_alias}", severity="warning")
            return False

        server = view.server
        env_pairs: list[str] = []
        env_refs: list[str] = []
        for key, value in sorted(server.env.items()):
            match = _ENV_REF_RE.match(str(value).strip())
            if match is not None:
                env_refs.append(f"{key}={match.group(1)}")
            else:
                env_pairs.append(f"{key}={value}")

        self._set_form_values(
            alias=clean_alias,
            command=server.command,
            args=" ".join(shlex.quote(arg) for arg in server.args),
            cwd=server.cwd,
            timeout=str(server.timeout_seconds),
            env=", ".join(env_pairs),
            env_ref=", ".join(env_refs),
        )
        self._mark_form_clean(active_alias=clean_alias)
        self._select_summary_alias(clean_alias)
        self.notify(f"Loaded alias: {clean_alias}")
        return True

    @staticmethod
    def _split_csv_pairs(raw: str) -> tuple[str, ...]:
        value = str(raw or "").strip()
        if not value:
            return ()
        parts = [item.strip() for item in value.split(",")]
        return tuple(item for item in parts if item)

    async def _save_current_form(self, *, notify_success: bool = True) -> bool:
        alias = self._current_alias()
        if not alias:
            self.notify("Alias is required.", severity="error")
            return False
        command = self.query_one("#mcp-command", Input).value.strip()
        args_raw = self.query_one("#mcp-args", Input).value.strip()
        cwd = self.query_one("#mcp-cwd", Input).value.strip()
        timeout_raw = self.query_one("#mcp-timeout", Input).value.strip()
        env_raw = self.query_one("#mcp-env", Input).value
        env_ref_raw = self.query_one("#mcp-env-ref", Input).value

        try:
            clean_alias = ensure_valid_alias(alias)
            args = tuple(shlex.split(args_raw)) if args_raw else ()
            env_pairs = self._split_csv_pairs(env_raw)
            env_refs = self._split_csv_pairs(env_ref_raw)
            timeout = int(timeout_raw) if timeout_raw else 30
        except Exception as e:
            self.notify(f"Invalid MCP form values: {e}", severity="error")
            return False

        try:
            existing = await asyncio.to_thread(self._manager.get_view, clean_alias)
            if existing is None:
                server = parse_mcp_server_from_flags(
                    command=command,
                    args=args,
                    env_pairs=env_pairs,
                    env_refs=env_refs,
                    cwd=cwd,
                    timeout=timeout,
                    disabled=False,
                )
                await asyncio.to_thread(self._manager.add_server, clean_alias, server)
            else:
                new_command = command if command else None
                new_cwd = cwd if cwd else None
                new_timeout = timeout if timeout_raw else None

                def _mutator(current):
                    return merge_server_edits(
                        current=current,
                        command=new_command,
                        args=args,
                        env_pairs=env_pairs,
                        env_refs=env_refs,
                        cwd=new_cwd,
                        timeout=new_timeout,
                        disabled=False,
                    )

                await asyncio.to_thread(self._manager.edit_server, clean_alias, _mutator)
        except MCPConfigManagerError as e:
            self.notify(str(e), severity="error")
            return False
        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error")
            return False

        self._changed = True
        self._active_alias = clean_alias
        self._mark_form_clean(active_alias=clean_alias)
        await self._refresh_summary()
        self._select_summary_alias(clean_alias)
        if notify_success:
            self.notify(f"Saved MCP alias: {clean_alias}")
        return True

    async def _remove_alias(self) -> None:
        alias = self._current_alias()
        if not alias:
            self.notify("Alias is required.", severity="error")
            return
        try:
            clean_alias = ensure_valid_alias(alias)
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        self.app.push_screen(
            ConfirmRemoveScreen(clean_alias),
            callback=lambda confirmed: self._on_remove_confirmed(clean_alias, confirmed),
        )

    def _on_remove_confirmed(self, alias: str, confirmed: bool) -> None:
        if not confirmed:
            return
        self.run_worker(
            self._remove_alias_confirmed(alias),
            group="mcp-manager-remove",
            exclusive=True,
        )

    async def _remove_alias_confirmed(self, alias: str) -> None:
        was_active = alias == self._active_alias
        try:
            await asyncio.to_thread(self._manager.remove_server, alias)
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        self._changed = True
        await self._refresh_summary()
        if was_active:
            self._set_blank_form()
        self.notify(f"Removed MCP alias: {alias}")

    async def _set_alias_enabled(self, enabled: bool) -> None:
        alias = self._current_alias()
        if not alias:
            self.notify("Alias is required.", severity="error")
            return
        try:
            clean_alias = ensure_valid_alias(alias)
            await asyncio.to_thread(
                self._manager.edit_server,
                clean_alias,
                lambda current: replace(current, enabled=enabled),
            )
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        self._changed = True
        await self._refresh_summary()
        self._select_summary_alias(clean_alias)
        self.notify(
            f"MCP alias '{clean_alias}' {'enabled' if enabled else 'disabled'}."
        )

    async def _test_alias(self) -> None:
        alias = self._current_alias()
        if not alias:
            self.notify("Alias is required.", severity="error")
            return
        try:
            clean_alias = ensure_valid_alias(alias)
            view, tools = await asyncio.to_thread(self._manager.probe_server, clean_alias)
        except Exception as e:
            self.notify(f"MCP test failed: {e}", severity="error")
            return
        names = [str(tool.get("name", "")).strip() for tool in tools]
        rendered = ", ".join(name for name in names if name) or "(none)"
        self.notify(
            f"Probe ok for {view.alias}: {len(names)} tool(s): {rendered}",
            timeout=6,
        )
