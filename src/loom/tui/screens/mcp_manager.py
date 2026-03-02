"""MCP configuration management UI."""

from __future__ import annotations

import asyncio
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    DataTable,
    Input,
    Label,
    Select,
    Static,
)

from loom.auth.resources import (
    ResourceDeleteImpact,
    cleanup_deleted_resource,
    rename_resource_key,
    resource_delete_impact,
)
from loom.config import (
    MCP_SERVER_TYPE_LOCAL,
    MCP_SERVER_TYPE_REMOTE,
    MCPConfig,
    MCPOAuthConfig,
    MCPServerConfig,
    validate_mcp_remote_url,
)
from loom.integrations.mcp.oauth import (
    MCPOAuthFlowError,
    MCPOAuthStoreError,
    oauth_state_for_alias,
    remove_mcp_oauth_token,
    resolve_mcp_oauth_provider,
    upsert_mcp_oauth_token,
)
from loom.mcp.config import (
    MCPConfigManager,
    MCPConfigManagerError,
    MCPServerView,
    ensure_valid_alias,
    parse_mcp_server_from_flags,
)
from loom.oauth.engine import OAuthEngine, OAuthEngineError, OAuthProviderConfig

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

    def __init__(
        self,
        alias: str,
        *,
        impact: ResourceDeleteImpact | None = None,
    ) -> None:
        super().__init__()
        self._alias = alias
        self._impact = impact or ResourceDeleteImpact()

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-remove-confirm-dialog"):
            yield Label("[bold #e0af68]Delete MCP server?[/]")
            yield Label(f"Alias: [bold]{self._alias}[/bold]")
            if self._impact.resource_id:
                yield Label(
                    "Auth impact: "
                    f"{len(self._impact.active_profile_ids)} profile(s), "
                    f"{len(self._impact.active_binding_ids)} binding(s), "
                    f"default={'yes' if self._impact.workspace_default_profile_id else 'no'}."
                )
                if self._impact.referencing_processes:
                    yield Label(
                        "Process references: "
                        f"{len(self._impact.referencing_processes)} "
                        "(review process auth requirements before delete)."
                    )
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


class OAuthCodeEntryScreen(ModalScreen[str | None]):
    """Prompt for callback URL/code when auto-callback is unavailable."""

    _inherit_bindings = False

    BINDINGS = [
        Binding("enter", "submit", "Submit"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    OAuthCodeEntryScreen {
        align: center middle;
    }
    #mcp-oauth-code-dialog {
        width: 84;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #mcp-oauth-code-input {
        margin-top: 1;
    }
    #mcp-oauth-code-actions {
        margin-top: 1;
        height: auto;
    }
    #mcp-oauth-code-actions Button {
        margin-right: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-oauth-code-dialog"):
            yield Label("[bold #7dcfff]Enter OAuth Callback[/bold #7dcfff]")
            yield Label("Paste full callback URL or raw authorization code.")
            yield Input(id="mcp-oauth-code-input")
            with Horizontal(id="mcp-oauth-code-actions"):
                yield Button("Submit", id="mcp-oauth-code-submit", variant="primary")
                yield Button("Cancel", id="mcp-oauth-code-cancel")

    def on_mount(self) -> None:
        self.query_one("#mcp-oauth-code-input", Input).focus()

    def action_submit(self) -> None:
        raw = self.query_one("#mcp-oauth-code-input", Input).value.strip()
        self.dismiss(raw or None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#mcp-oauth-code-submit")
    def _on_submit(self) -> None:
        self.action_submit()

    @on(Button.Pressed, "#mcp-oauth-code-cancel")
    def _on_cancel(self) -> None:
        self.action_cancel()


class MCPManagerScreen(Vertical):
    """MCP server add/edit/remove/test flow widget."""

    _TYPE_LOCAL_VALUE = MCP_SERVER_TYPE_LOCAL
    _TYPE_REMOTE_VALUE = MCP_SERVER_TYPE_REMOTE

    BINDINGS = [
        Binding("escape", "request_close", "Close"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+w", "request_close", "Close Tab", show=False, priority=True),
    ]

    _INPUT_FIELD_IDS = (
        "mcp-alias",
        "mcp-command",
        "mcp-args",
        "mcp-url",
        "mcp-fallback-sse-url",
        "mcp-headers",
        "mcp-oauth-scopes",
        "mcp-cwd",
        "mcp-timeout",
        "mcp-env",
        "mcp-env-ref",
    )
    _FORM_SELECT_IDS = ("mcp-type",)
    _FORM_CHECKBOX_IDS = (
        "mcp-oauth-enabled",
        "mcp-allow-insecure-http",
        "mcp-allow-private-network",
    )
    _LOCAL_ONLY_WIDGET_IDS = (
        "mcp-label-command",
        "mcp-command",
        "mcp-help-command",
        "mcp-label-args",
        "mcp-args",
        "mcp-help-args",
        "mcp-label-cwd",
        "mcp-cwd",
        "mcp-help-cwd",
        "mcp-label-env",
        "mcp-env",
        "mcp-help-env",
        "mcp-label-env-ref",
        "mcp-env-ref",
        "mcp-help-env-ref",
    )
    _REMOTE_ONLY_WIDGET_IDS = (
        "mcp-label-url",
        "mcp-url",
        "mcp-help-url",
        "mcp-label-fallback-sse-url",
        "mcp-fallback-sse-url",
        "mcp-help-fallback-sse-url",
        "mcp-allow-insecure-http",
        "mcp-allow-private-network",
        "mcp-manager-remote-advanced",
    )

    DEFAULT_CSS = """
    MCPManagerScreen {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }
    MCPManagerScreen.modal-mode {
        align: center middle;
    }
    MCPManagerScreen.embedded-mode {
        align: left top;
        padding: 0;
    }
    MCPManagerScreen.modal-mode #mcp-manager-dialog {
        width: 100;
        height: 90%;
        max-height: 46;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
        overflow: hidden;
    }
    MCPManagerScreen.embedded-mode #mcp-manager-dialog {
        width: 100%;
        height: 1fr;
        max-height: 100%;
        border: none;
        padding: 0 1;
    }
    #mcp-manager-dialog {
        width: 100%;
        height: 1fr;
        border: none;
        padding: 0 1;
        background: $surface;
    }
    #mcp-manager-title {
        text-style: bold;
        margin-bottom: 0;
    }
    #mcp-header-row {
        height: auto;
        margin-bottom: 1;
    }
    #mcp-manager-summary {
        height: 6;
        max-height: 7;
        margin-bottom: 0;
        border: round $surface-lighten-1;
    }
    #mcp-summary-help {
        margin-bottom: 1;
    }
    #mcp-manager-form {
        height: 1fr;
        overflow-y: auto;
        margin-bottom: 0;
    }
    #mcp-manager-remote-advanced {
        margin-top: 1;
    }
    .mcp-label {
        margin-top: 0;
    }
    .mcp-help {
        color: $text-muted;
        margin-top: 0;
        margin-bottom: 0;
    }
    .mcp-input {
        margin-top: 0;
    }
    .mcp-actions-row {
        height: auto;
        margin-top: 0;
    }
    .mcp-actions-row Button {
        margin-right: 0;
        padding: 0;
        min-width: 0;
        width: 1fr;
    }
    #mcp-actions-oauth {
        margin-bottom: 1;
    }
    #mcp-manager-footer {
        margin-top: 0;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        manager: MCPConfigManager,
        *,
        explicit_auth_path: Path | None = None,
        oauth_browser_login_enabled: bool = True,
        embedded: bool = False,
        on_close: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        super().__init__()
        self._manager = manager
        self._explicit_auth_path = explicit_auth_path
        self._oauth_browser_login_enabled = bool(oauth_browser_login_enabled)
        self._embedded = bool(embedded)
        self._on_close = on_close
        self._views: list[MCPServerView] = []
        self._summary_aliases: list[str] = []
        self._active_alias = ""
        self._baseline_form_state: dict[str, str] = {}
        self._form_dirty = False
        self._suppress_dirty_tracking = False
        self._changed = False
        self._oauth_engine = OAuthEngine()
        self._oauth_pending_alias = ""
        self._oauth_pending_state = ""
        self._oauth_pending_url = ""
        self._oauth_pending_provider: OAuthProviderConfig | None = None
        self._oauth_pending_expires_at: int | None = None
        self._oauth_last_failure_by_alias: dict[str, str] = {}
        if self._embedded:
            self.add_class("embedded-mode")
        else:
            self.add_class("modal-mode")

    def _clear_oauth_pending(self) -> None:
        self._oauth_pending_alias = ""
        self._oauth_pending_state = ""
        self._oauth_pending_url = ""
        self._oauth_pending_provider = None
        self._oauth_pending_expires_at = None

    def _clear_oauth_pending_if_current(self, *, alias: str, state: str) -> None:
        if (
            str(alias or "").strip() == self._oauth_pending_alias
            and str(state or "").strip() == self._oauth_pending_state
        ):
            self._clear_oauth_pending()

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-manager-dialog"):
            with Horizontal(id="mcp-header-row"):
                yield Label(
                    "[bold #7dcfff]MCP Server Manager[/bold #7dcfff]",
                    id="mcp-manager-title",
                )
            with Horizontal(classes="mcp-actions-row", id="mcp-actions-primary"):
                yield Button("Refresh", id="mcp-btn-refresh")
                yield Button("New", id="mcp-btn-new")
                yield Button("Load", id="mcp-btn-load")
                yield Button("Save", id="mcp-btn-save", variant="primary")
                yield Button("Test", id="mcp-btn-test")
                yield Button("Enable", id="mcp-btn-enable")
                yield Button("Disable", id="mcp-btn-disable")
                yield Button("Delete", id="mcp-btn-remove", variant="error")
                yield Button("Close", id="mcp-btn-close")
            with Horizontal(classes="mcp-actions-row", id="mcp-actions-oauth"):
                yield Button("OAuth Login", id="mcp-btn-oauth-login")
                yield Button("Copy URL", id="mcp-btn-oauth-copy-url")
                yield Button("Enter Code", id="mcp-btn-oauth-enter-code")
                yield Button("Logout", id="mcp-btn-oauth-logout")
                yield Button("Import Token", id="mcp-btn-oauth-save")
            summary_table = DataTable(id="mcp-manager-summary")
            summary_table.cursor_type = "row"
            summary_table.zebra_stripes = True
            summary_table.add_columns("Alias", "Type", "Status", "Target", "Source")
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
                yield Label("Server Type", classes="mcp-label")
                yield Select(
                    options=[
                        ("Local (stdio)", self._TYPE_LOCAL_VALUE),
                        ("Remote (HTTP)", self._TYPE_REMOTE_VALUE),
                    ],
                    id="mcp-type",
                    classes="mcp-input",
                    allow_blank=False,
                    value=self._TYPE_LOCAL_VALUE,
                )
                yield Label(
                    "Select local process transport or remote HTTP transport.",
                    classes="mcp-help",
                )
                yield Label("Command (local)", classes="mcp-label", id="mcp-label-command")
                yield Input(
                    id="mcp-command",
                    classes="mcp-input",
                )
                yield Label(
                    "Executable to launch local MCP server (required for local aliases).",
                    classes="mcp-help",
                    id="mcp-help-command",
                )
                yield Label(
                    "Args (local, shell-style string)",
                    classes="mcp-label",
                    id="mcp-label-args",
                )
                yield Input(
                    id="mcp-args",
                    classes="mcp-input",
                )
                yield Label(
                    "Optional local command arguments; parsed like shell args.",
                    classes="mcp-help",
                    id="mcp-help-args",
                )
                yield Label("Cwd (local)", classes="mcp-label", id="mcp-label-cwd")
                yield Input(
                    id="mcp-cwd",
                    classes="mcp-input",
                )
                yield Label(
                    "Optional working directory for local server startup.",
                    classes="mcp-help",
                    id="mcp-help-cwd",
                )
                yield Label(
                    "Env pairs (local KEY=VALUE, comma-separated)",
                    classes="mcp-label",
                    id="mcp-label-env",
                )
                yield Input(
                    id="mcp-env",
                    classes="mcp-input",
                )
                yield Label(
                    "Literal env values for local process transport.",
                    classes="mcp-help",
                    id="mcp-help-env",
                )
                yield Label(
                    "Env refs (local KEY=ENV_VAR, comma-separated)",
                    classes="mcp-label",
                    id="mcp-label-env-ref",
                )
                yield Input(
                    id="mcp-env-ref",
                    classes="mcp-input",
                )
                yield Label(
                    "Runtime env indirection; saved as KEY=${ENV_VAR}.",
                    classes="mcp-help",
                    id="mcp-help-env-ref",
                )
                yield Label("URL (remote)", classes="mcp-label", id="mcp-label-url")
                yield Input(
                    id="mcp-url",
                    classes="mcp-input",
                )
                yield Label(
                    "Remote MCP URL. HTTPS enforced unless insecure override is enabled.",
                    classes="mcp-help",
                    id="mcp-help-url",
                )
                yield Label(
                    "Fallback SSE URL (remote, optional)",
                    classes="mcp-label",
                    id="mcp-label-fallback-sse-url",
                )
                yield Input(
                    id="mcp-fallback-sse-url",
                    classes="mcp-input",
                )
                yield Label(
                    "Optional SSE fallback URL if the server requires it.",
                    classes="mcp-help",
                    id="mcp-help-fallback-sse-url",
                )
                yield Checkbox(
                    "Allow insecure HTTP (remote)",
                    id="mcp-allow-insecure-http",
                    value=False,
                )
                yield Checkbox(
                    "Allow private-network URL (remote)",
                    id="mcp-allow-private-network",
                    value=False,
                )
                with Collapsible(
                    title="Remote Advanced",
                    id="mcp-manager-remote-advanced",
                    collapsed=True,
                ):
                    yield Label(
                        "Headers (remote KEY=VALUE, comma-separated)",
                        classes="mcp-label",
                    )
                    yield Input(
                        id="mcp-headers",
                        classes="mcp-input",
                    )
                    yield Label(
                        "Remote HTTP headers. Sensitive values are redacted in displays.",
                        classes="mcp-help",
                    )
                    yield Checkbox(
                        "OAuth enabled for remote server",
                        id="mcp-oauth-enabled",
                        value=False,
                    )
                    yield Label(
                        "When enabled, MCP calls require OAuth token readiness.",
                        classes="mcp-help",
                    )
                    yield Label("OAuth scopes (comma-separated)", classes="mcp-label")
                    yield Input(
                        id="mcp-oauth-scopes",
                        classes="mcp-input",
                    )
                    yield Label(
                        "OAuth access token import (not in mcp.toml)",
                        classes="mcp-label",
                    )
                    yield Input(
                        id="mcp-oauth-access-token",
                        classes="mcp-input",
                    )
                    yield Label("OAuth refresh token (optional)", classes="mcp-label")
                    yield Input(
                        id="mcp-oauth-refresh-token",
                        classes="mcp-input",
                    )
                    yield Label("OAuth expires in seconds (optional)", classes="mcp-label")
                    yield Input(
                        id="mcp-oauth-expires-in",
                        classes="mcp-input",
                    )
                    yield Static("-", id="mcp-oauth-status")
                yield Label("Timeout seconds", classes="mcp-label")
                yield Input(
                    id="mcp-timeout",
                    classes="mcp-input",
                )
                yield Label(
                    "Request timeout for this MCP server (defaults to 30).",
                    classes="mcp-help",
                )

            yield Label(
                "[dim]Server list loads automatically on open. "
                "Select local/remote transport and only relevant fields are shown. "
                "OAuth Login is browser-first; use Copy URL + Enter Code for SSH/headless "
                "sessions. Import Token is a manual fallback only.[/dim]",
                id="mcp-manager-footer",
            )

    async def on_mount(self) -> None:
        self._set_form_values(
            alias="",
            server_type=self._TYPE_LOCAL_VALUE,
            command="",
            args="",
            url="",
            fallback_sse_url="",
            headers="",
            oauth_enabled=False,
            oauth_scopes="",
            allow_insecure_http=False,
            allow_private_network=False,
            cwd="",
            timeout="30",
            env="",
            env_ref="",
            oauth_access_token="",
            oauth_refresh_token="",
            oauth_expires_in="",
        )
        self._set_oauth_status_text("-")
        self._sync_transport_fields()
        self._mark_form_clean(active_alias="")
        await self._refresh_summary()
        self.set_interval(5.0, self._poll_oauth_status)
        self.query_one("#mcp-alias", Input).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key.lower() != "ctrl+w":
            return
        self.run_worker(
            self.action_request_close(),
            group="mcp-manager-close-request",
            exclusive=True,
        )
        event.stop()
        event.prevent_default()

    def on_unmount(self) -> None:
        self._clear_oauth_pending()
        self._oauth_engine.shutdown()

    def action_close(self) -> None:
        result = {"changed": self._changed}
        if callable(self._on_close):
            self._on_close(result)

    async def action_request_close(self) -> None:
        await self._request_close()

    async def action_refresh(self) -> None:
        await self._refresh_summary()

    @on(Button.Pressed)
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "mcp-btn-close":
            await self.action_request_close()
            return
        if button_id == "mcp-btn-refresh":
            await self._refresh_summary()
            return
        if button_id == "mcp-btn-new":
            await self._start_new_alias()
            return
        if button_id == "mcp-btn-load":
            selected = self._selected_summary_alias()
            await self._request_alias_switch(selected or self._current_alias())
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
        if button_id == "mcp-btn-oauth-login":
            await self._oauth_start_browser_login()
            return
        if button_id == "mcp-btn-oauth-copy-url":
            self._oauth_copy_login_url()
            return
        if button_id == "mcp-btn-oauth-enter-code":
            self._oauth_prompt_callback_code()
            return
        if button_id == "mcp-btn-oauth-logout":
            await self._oauth_clear_token()
            return
        if button_id == "mcp-btn-oauth-save":
            await self._oauth_save_token()
            return
        if button_id == "mcp-btn-remove":
            await self._remove_alias()
            return
        if button_id == "mcp-btn-test":
            await self._test_alias()
            return

    @on(Input.Changed)
    def _on_form_input_changed(self, event: Input.Changed) -> None:
        if event.input.id not in self._INPUT_FIELD_IDS:
            return
        self._update_form_dirty()

    @on(Select.Changed)
    def _on_form_select_changed(self, event: Select.Changed) -> None:
        if event.select.id not in self._FORM_SELECT_IDS:
            return
        if event.select.id == "mcp-type":
            is_remote = self._selected_server_type() == self._TYPE_REMOTE_VALUE
            if not str(self._active_alias or "").strip():
                self.query_one("#mcp-oauth-enabled", Checkbox).value = bool(is_remote)
            self._sync_transport_fields()
        self._update_form_dirty()

    @on(Checkbox.Changed)
    def _on_form_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id not in self._FORM_CHECKBOX_IDS:
            return
        self._update_form_dirty()

    @on(DataTable.RowSelected, "#mcp-manager-summary")
    async def _on_summary_row_selected(self, event: DataTable.RowSelected) -> None:
        alias = str(getattr(event.row_key, "value", "") or "").strip()
        if not alias and 0 <= event.cursor_row < len(self._summary_aliases):
            alias = self._summary_aliases[event.cursor_row]
        await self._request_alias_switch(alias)

    def _capture_form_state(self) -> dict[str, str]:
        state: dict[str, str] = {}
        for field_id in self._INPUT_FIELD_IDS:
            state[field_id] = self.query_one(f"#{field_id}", Input).value
        for field_id in self._FORM_SELECT_IDS:
            state[field_id] = str(self.query_one(f"#{field_id}", Select).value or "")
        for field_id in self._FORM_CHECKBOX_IDS:
            state[field_id] = (
                "1" if bool(self.query_one(f"#{field_id}", Checkbox).value) else "0"
            )
        return state

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
        server_type: str,
        command: str,
        args: str,
        url: str,
        fallback_sse_url: str,
        headers: str,
        oauth_enabled: bool,
        oauth_scopes: str,
        allow_insecure_http: bool,
        allow_private_network: bool,
        cwd: str,
        timeout: str,
        env: str,
        env_ref: str,
        oauth_access_token: str,
        oauth_refresh_token: str,
        oauth_expires_in: str,
    ) -> None:
        self._suppress_dirty_tracking = True
        try:
            self.query_one("#mcp-alias", Input).value = alias
            self.query_one("#mcp-type", Select).value = (
                self._TYPE_REMOTE_VALUE
                if str(server_type).strip().lower() == self._TYPE_REMOTE_VALUE
                else self._TYPE_LOCAL_VALUE
            )
            self.query_one("#mcp-command", Input).value = command
            self.query_one("#mcp-args", Input).value = args
            self.query_one("#mcp-url", Input).value = url
            self.query_one("#mcp-fallback-sse-url", Input).value = fallback_sse_url
            self.query_one("#mcp-headers", Input).value = headers
            self.query_one("#mcp-oauth-enabled", Checkbox).value = bool(oauth_enabled)
            self.query_one("#mcp-oauth-scopes", Input).value = oauth_scopes
            self.query_one("#mcp-allow-insecure-http", Checkbox).value = bool(
                allow_insecure_http
            )
            self.query_one("#mcp-allow-private-network", Checkbox).value = bool(
                allow_private_network
            )
            self.query_one("#mcp-cwd", Input).value = cwd
            self.query_one("#mcp-timeout", Input).value = timeout
            self.query_one("#mcp-env", Input).value = env
            self.query_one("#mcp-env-ref", Input).value = env_ref
            self.query_one("#mcp-oauth-access-token", Input).value = oauth_access_token
            self.query_one("#mcp-oauth-refresh-token", Input).value = oauth_refresh_token
            self.query_one("#mcp-oauth-expires-in", Input).value = oauth_expires_in
        finally:
            self._suppress_dirty_tracking = False
            self._sync_transport_fields()

    def _set_blank_form(self) -> None:
        self._set_form_values(
            alias="",
            server_type=self._TYPE_LOCAL_VALUE,
            command="",
            args="",
            url="",
            fallback_sse_url="",
            headers="",
            oauth_enabled=False,
            oauth_scopes="",
            allow_insecure_http=False,
            allow_private_network=False,
            cwd="",
            timeout="30",
            env="",
            env_ref="",
            oauth_access_token="",
            oauth_refresh_token="",
            oauth_expires_in="",
        )
        self._set_oauth_status_text("-")
        self._mark_form_clean(active_alias="")

    def _set_oauth_status_text(self, text: str) -> None:
        self.query_one("#mcp-oauth-status", Static).update(text)

    async def _poll_oauth_status(self) -> None:
        alias = self._current_alias()
        if not alias:
            return
        try:
            clean_alias = ensure_valid_alias(alias)
            view = await asyncio.to_thread(self._manager.get_view, clean_alias)
        except Exception:
            return
        if view is None or view.server.type != MCP_SERVER_TYPE_REMOTE:
            return
        await self._oauth_show_status(notify=False, quiet=True)

    def _selected_server_type(self) -> str:
        raw = str(self.query_one("#mcp-type", Select).value or "").strip().lower()
        if raw == self._TYPE_REMOTE_VALUE:
            return self._TYPE_REMOTE_VALUE
        return self._TYPE_LOCAL_VALUE

    def _set_widget_display(self, widget_id: str, visible: bool) -> None:
        widget = self.query_one(f"#{widget_id}")
        widget.display = bool(visible)

    def _sync_transport_fields(self) -> None:
        is_remote = self._selected_server_type() == self._TYPE_REMOTE_VALUE
        oauth_browser_enabled = bool(self._oauth_browser_login_enabled)

        for widget_id in self._LOCAL_ONLY_WIDGET_IDS:
            self._set_widget_display(widget_id, not is_remote)
        for widget_id in self._REMOTE_ONLY_WIDGET_IDS:
            self._set_widget_display(widget_id, is_remote)
        self._set_widget_display(
            "mcp-actions-oauth",
            is_remote,
        )

        local_ids = ("mcp-command", "mcp-args", "mcp-cwd", "mcp-env", "mcp-env-ref")
        for field_id in local_ids:
            self.query_one(f"#{field_id}", Input).disabled = is_remote
        for field_id in ("mcp-url", "mcp-fallback-sse-url", "mcp-headers", "mcp-oauth-scopes"):
            self.query_one(f"#{field_id}", Input).disabled = not is_remote
        for field_id in (
            "mcp-oauth-access-token",
            "mcp-oauth-refresh-token",
            "mcp-oauth-expires-in",
        ):
            self.query_one(f"#{field_id}", Input).disabled = not is_remote
        for field_id in (
            "mcp-oauth-enabled",
            "mcp-allow-insecure-http",
            "mcp-allow-private-network",
        ):
            self.query_one(f"#{field_id}", Checkbox).disabled = not is_remote
        for button_id in (
            "mcp-btn-oauth-login",
            "mcp-btn-oauth-copy-url",
            "mcp-btn-oauth-enter-code",
        ):
            self.query_one(f"#{button_id}", Button).disabled = not (
                is_remote and oauth_browser_enabled
            )
        for button_id in ("mcp-btn-oauth-logout", "mcp-btn-oauth-save"):
            self.query_one(f"#{button_id}", Button).disabled = not is_remote

        if not is_remote and not self._oauth_pending_state:
            self._set_oauth_status_text("-")
        elif is_remote and not oauth_browser_enabled and not self._oauth_pending_state:
            self._set_oauth_status_text(
                "OAuth browser login disabled by config; use Import Token."
            )

    async def _refresh_summary(self) -> None:
        try:
            self._views = (await asyncio.to_thread(self._manager.load)).as_views()
        except Exception as e:
            self.notify(f"MCP load failed: {e}", severity="error")
            return
        self._render_summary()

    def _render_summary(self) -> None:
        table = self.query_one("#mcp-manager-summary", DataTable)
        table.clear()
        self._summary_aliases = []

        for view in self._views:
            status = "enabled" if view.server.enabled else "disabled"
            server_type = str(getattr(view.server, "type", MCP_SERVER_TYPE_LOCAL) or "").strip()
            if server_type == MCP_SERVER_TYPE_REMOTE:
                target = str(getattr(view.server, "url", "") or "").strip() or "-"
            else:
                command = str(getattr(view.server, "command", "") or "").strip()
                args = [
                    str(item)
                    for item in list(getattr(view.server, "args", []) or [])
                    if str(item).strip()
                ]
                target = " ".join([command, *args]).strip() or "-"
            source = view.source
            if view.source == "legacy":
                source = "legacy"
            table.add_row(
                view.alias,
                server_type or "local",
                status,
                target,
                source,
                key=view.alias,
            )
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

    def _selected_summary_alias(self) -> str:
        table = self.query_one("#mcp-manager-summary", DataTable)
        row_index = int(getattr(table, "cursor_row", -1))
        if 0 <= row_index < len(self._summary_aliases):
            return self._summary_aliases[row_index]
        return ""

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

    async def _start_new_alias(self) -> None:
        if self._form_dirty:
            self.app.push_screen(
                ConfirmAliasSwitchScreen(
                    current_alias=self._active_alias,
                    target_alias="new entry",
                ),
                callback=lambda decision: self._on_new_alias_decision(
                    str(decision or "cancel").lower(),
                ),
            )
            return
        self._set_blank_form()
        self.query_one("#mcp-alias", Input).focus()
        self.notify("Ready to add a new MCP alias.")

    def _on_new_alias_decision(self, decision: str) -> None:
        if decision == "cancel":
            self._select_summary_alias(self._active_alias)
            return
        self.run_worker(
            self._complete_new_alias(decision),
            group="mcp-manager-new-alias",
            exclusive=True,
        )

    async def _complete_new_alias(self, decision: str) -> None:
        if decision == "save":
            saved = await self._save_current_form(notify_success=False)
            if not saved:
                self._select_summary_alias(self._active_alias)
                return
        elif decision != "discard":
            self._select_summary_alias(self._active_alias)
            return
        self._set_blank_form()
        self.query_one("#mcp-alias", Input).focus()
        if decision == "save":
            self.notify("Saved current alias. Ready to add a new MCP alias.")
        else:
            self.notify("Ready to add a new MCP alias.")

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

    async def _request_close(self) -> None:
        if not self._form_dirty:
            self.action_close()
            return
        current_alias = self._active_alias or self._current_alias() or "(new alias)"
        self.app.push_screen(
            ConfirmAliasSwitchScreen(
                current_alias=current_alias,
                target_alias="close tab",
            ),
            callback=lambda decision: self._on_close_decision(
                str(decision or "cancel").lower(),
            ),
        )

    def _on_close_decision(self, decision: str) -> None:
        self.run_worker(
            self._complete_close(decision),
            group="mcp-manager-close",
            exclusive=True,
        )

    async def _complete_close(self, decision: str) -> None:
        if decision == "save":
            saved = await self._save_current_form(notify_success=False)
            if not saved:
                return
        elif decision != "discard":
            return
        self.action_close()

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
        headers_pairs = [f"{key}={value}" for key, value in sorted(server.headers.items())]

        self._set_form_values(
            alias=clean_alias,
            server_type=server.type,
            command=server.command,
            args=" ".join(shlex.quote(arg) for arg in server.args),
            url=server.url,
            fallback_sse_url=server.fallback_sse_url,
            headers=", ".join(headers_pairs),
            oauth_enabled=bool(server.oauth.enabled),
            oauth_scopes=", ".join(server.oauth.scopes),
            allow_insecure_http=bool(server.allow_insecure_http),
            allow_private_network=bool(server.allow_private_network),
            cwd=server.cwd,
            timeout=str(server.timeout_seconds),
            env=", ".join(env_pairs),
            env_ref=", ".join(env_refs),
            oauth_access_token="",
            oauth_refresh_token="",
            oauth_expires_in="",
        )
        await self._oauth_show_status(notify=False, quiet=True)
        self._mark_form_clean(active_alias=clean_alias)
        self._select_summary_alias(clean_alias)
        self.notify(f"Loaded alias: {clean_alias}")
        return True

    @staticmethod
    def _split_csv_items(raw: str) -> tuple[str, ...]:
        value = str(raw or "").strip()
        if not value:
            return ()
        parts = [item.strip() for item in value.split(",")]
        return tuple(item for item in parts if item)

    @classmethod
    def _parse_csv_map(
        cls,
        raw: str,
        *,
        option_name: str,
    ) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for item in cls._split_csv_items(raw):
            if "=" not in item:
                raise ValueError(f"{option_name} expects KEY=VALUE entries.")
            key, value = item.split("=", 1)
            clean_key = str(key).strip()
            if not clean_key:
                raise ValueError(f"{option_name} key cannot be empty.")
            parsed[clean_key] = value
        return parsed

    def _manager_workspace(self) -> Path | None:
        workspace = getattr(self._manager, "_workspace", None)
        if workspace is None:
            return None
        try:
            return Path(workspace).expanduser().resolve()
        except Exception:
            return None

    async def _save_current_form(self, *, notify_success: bool = True) -> bool:
        alias = self._current_alias()
        if not alias:
            self.notify("Alias is required.", severity="error")
            return False
        server_type = self._selected_server_type()
        command = self.query_one("#mcp-command", Input).value.strip()
        args_raw = self.query_one("#mcp-args", Input).value.strip()
        url_raw = self.query_one("#mcp-url", Input).value.strip()
        fallback_sse_url = self.query_one("#mcp-fallback-sse-url", Input).value.strip()
        headers_raw = self.query_one("#mcp-headers", Input).value
        oauth_scopes_raw = self.query_one("#mcp-oauth-scopes", Input).value
        oauth_enabled = bool(self.query_one("#mcp-oauth-enabled", Checkbox).value)
        allow_insecure_http = bool(
            self.query_one("#mcp-allow-insecure-http", Checkbox).value
        )
        allow_private_network = bool(
            self.query_one("#mcp-allow-private-network", Checkbox).value
        )
        cwd = self.query_one("#mcp-cwd", Input).value.strip()
        timeout_raw = self.query_one("#mcp-timeout", Input).value.strip()
        env_raw = self.query_one("#mcp-env", Input).value
        env_ref_raw = self.query_one("#mcp-env-ref", Input).value

        try:
            clean_alias = ensure_valid_alias(alias)
            args = tuple(shlex.split(args_raw)) if args_raw else ()
            env_pairs = self._split_csv_items(env_raw)
            env_refs = self._split_csv_items(env_ref_raw)
            headers = self._parse_csv_map(headers_raw, option_name="Headers")
            oauth_scopes = [
                scope
                for scope in self._split_csv_items(oauth_scopes_raw)
                if str(scope).strip()
            ]
            timeout = int(timeout_raw) if timeout_raw else 30
        except Exception as e:
            self.notify(f"Invalid MCP form values: {e}", severity="error")
            return False

        previous_alias = str(self._active_alias or "").strip()
        renaming = bool(previous_alias and previous_alias != clean_alias)

        try:
            existing = await asyncio.to_thread(self._manager.get_view, clean_alias)
            existing_enabled = True
            if existing is not None:
                existing_enabled = bool(existing.server.enabled)
            elif renaming:
                previous = await asyncio.to_thread(self._manager.get_view, previous_alias)
                if previous is not None:
                    existing_enabled = bool(previous.server.enabled)

            if server_type == self._TYPE_REMOTE_VALUE:
                try:
                    validated_url = validate_mcp_remote_url(
                        url_raw,
                        allow_insecure_http=allow_insecure_http,
                        allow_private_network=allow_private_network,
                    )
                except Exception as e:
                    self.notify(f"Invalid remote URL: {e}", severity="error")
                    return False
                server = MCPServerConfig(
                    type=MCP_SERVER_TYPE_REMOTE,
                    command="",
                    args=[],
                    env={},
                    url=validated_url,
                    fallback_sse_url=fallback_sse_url,
                    headers=headers,
                    oauth=MCPOAuthConfig(
                        enabled=oauth_enabled,
                        scopes=oauth_scopes,
                    ),
                    allow_insecure_http=allow_insecure_http,
                    allow_private_network=allow_private_network,
                    cwd="",
                    timeout_seconds=timeout,
                    enabled=existing_enabled,
                )
            else:
                server = parse_mcp_server_from_flags(
                    command=command,
                    args=args,
                    env_pairs=env_pairs,
                    env_refs=env_refs,
                    cwd=cwd,
                    timeout=timeout,
                    disabled=False,
                )
                server = replace(server, enabled=existing_enabled)

            if existing is None:
                await asyncio.to_thread(self._manager.add_server, clean_alias, server)
                if renaming:
                    await asyncio.to_thread(self._manager.remove_server, previous_alias)
            else:
                def _mutator(current):
                    return replace(server, enabled=bool(current.enabled))

                await asyncio.to_thread(self._manager.edit_server, clean_alias, _mutator)
        except MCPConfigManagerError as e:
            self.notify(str(e), severity="error")
            return False
        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error")
            return False

        workspace = self._manager_workspace()
        if renaming and workspace is not None:
            try:
                await asyncio.to_thread(
                    rename_resource_key,
                    workspace=workspace,
                    resource_kind="mcp",
                    old_key=previous_alias,
                    new_key=clean_alias,
                )
            except Exception as e:
                self.notify(
                    f"Auth resource rename warning: {e}",
                    severity="warning",
                )

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
        impact = None
        workspace = self._manager_workspace()
        if workspace is not None:
            try:
                impact = await asyncio.to_thread(
                    resource_delete_impact,
                    workspace=workspace,
                    resource_kind="mcp",
                    resource_key=clean_alias,
                )
            except Exception:
                impact = None
        self.app.push_screen(
            ConfirmRemoveScreen(clean_alias, impact=impact),
            callback=lambda confirmed: self._on_remove_confirmed(
                clean_alias,
                confirmed,
                impact,
            ),
        )

    def _on_remove_confirmed(
        self,
        alias: str,
        confirmed: bool,
        impact: ResourceDeleteImpact | None,
    ) -> None:
        if not confirmed:
            return
        self.run_worker(
            self._remove_alias_confirmed(alias, impact),
            group="mcp-manager-remove",
            exclusive=True,
        )

    async def _remove_alias_confirmed(
        self,
        alias: str,
        impact: ResourceDeleteImpact | None,
    ) -> None:
        was_active = alias == self._active_alias
        if alias == self._oauth_pending_alias and self._oauth_pending_state:
            pending_state = self._oauth_pending_state
            try:
                await asyncio.to_thread(
                    self._oauth_engine.cancel_auth,
                    state=pending_state,
                )
            except Exception:
                pass
            self._clear_oauth_pending_if_current(alias=alias, state=pending_state)
        try:
            await asyncio.to_thread(self._manager.remove_server, alias)
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        workspace = self._manager_workspace()
        if workspace is not None:
            try:
                await asyncio.to_thread(
                    cleanup_deleted_resource,
                    workspace=workspace,
                    explicit_auth_path=self._explicit_auth_path,
                    resource_kind="mcp",
                    resource_key=alias,
                )
            except Exception as e:
                self.notify(f"Auth cleanup warning: {e}", severity="warning")
        self._changed = True
        await self._refresh_summary()
        if was_active:
            self._set_blank_form()
        if impact is not None and impact.resource_id:
            self.notify(
                "Auth cleanup: "
                f"{len(impact.active_profile_ids)} profile(s), "
                f"{len(impact.active_binding_ids)} binding(s), "
                f"default={'yes' if impact.workspace_default_profile_id else 'no'}, "
                f"process_refs={len(impact.referencing_processes)}.",
            )
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

    async def _oauth_save_token(self) -> None:
        alias = self._current_alias()
        if not alias:
            self.notify("Alias is required.", severity="error")
            return
        try:
            clean_alias = ensure_valid_alias(alias)
            view = await asyncio.to_thread(self._manager.get_view, clean_alias)
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        if view is None:
            self.notify(f"Alias not found: {clean_alias}", severity="error")
            return
        if view.server.type != MCP_SERVER_TYPE_REMOTE:
            self.notify(
                "OAuth token actions are only valid for remote MCP aliases.",
                severity="warning",
            )
            return
        access_token = self.query_one("#mcp-oauth-access-token", Input).value.strip()
        refresh_token = self.query_one("#mcp-oauth-refresh-token", Input).value.strip()
        expires_in_raw = self.query_one("#mcp-oauth-expires-in", Input).value.strip()
        scopes_raw = self.query_one("#mcp-oauth-scopes", Input).value
        scopes = [
            scope
            for scope in self._split_csv_items(scopes_raw)
            if str(scope).strip()
        ]

        if not access_token:
            self.notify("OAuth access token is required.", severity="error")
            return

        expires_at_unix: int | None = None
        if expires_in_raw:
            try:
                expires_in = int(expires_in_raw)
                if expires_in <= 0:
                    raise ValueError
                expires_at_unix = int(time.time()) + expires_in
            except ValueError:
                self.notify("OAuth expires-in must be a positive integer.", severity="error")
                return
        try:
            await asyncio.to_thread(
                upsert_mcp_oauth_token,
                alias=clean_alias,
                access_token=access_token,
                refresh_token=refresh_token,
                scopes=scopes,
                expires_at_unix=expires_at_unix,
                obtained_via="manual_import_advanced",
            )
        except MCPOAuthStoreError as e:
            self.notify(f"OAuth token save failed: {e}", severity="error")
            return
        self._changed = True
        self.query_one("#mcp-oauth-access-token", Input).value = ""
        self.query_one("#mcp-oauth-refresh-token", Input).value = ""
        self.query_one("#mcp-oauth-expires-in", Input).value = ""
        await self._oauth_show_status(notify=True, quiet=False)

    async def _oauth_start_browser_login(self) -> None:
        if self._oauth_pending_state:
            pending_alias = self._oauth_pending_alias or "another alias"
            self.notify(
                "OAuth login already in progress for "
                f"{pending_alias!r}. Use Enter Callback Code or OAuth Logout first.",
                severity="warning",
                timeout=7,
            )
            return
        if not self._oauth_browser_login_enabled:
            self.notify(
                "OAuth browser login is disabled by config; use Import Token fallback.",
                severity="warning",
                timeout=7,
            )
            return
        alias = self._current_alias()
        if not alias:
            self.notify("Alias is required.", severity="error")
            return
        try:
            clean_alias = ensure_valid_alias(alias)
            view = await asyncio.to_thread(self._manager.get_view, clean_alias)
            if view is None:
                self.notify(f"Alias not found: {clean_alias}", severity="error")
                return
            if view.server.type != MCP_SERVER_TYPE_REMOTE:
                self.notify(
                    "OAuth login is only available for remote MCP aliases.",
                    severity="warning",
                )
                return
            if not view.server.oauth.enabled:
                self.notify(
                    "Enable oauth for this alias before browser login.",
                    severity="warning",
                )
                return
            provider = resolve_mcp_oauth_provider(
                server_url=view.server.url,
                scopes=list(view.server.oauth.scopes),
                redirect_uris=(
                    "http://127.0.0.1:8765/oauth/callback",
                    "http://localhost:8765/oauth/callback",
                    "urn:ietf:wg:oauth:2.0:oob",
                ),
                client_name=f"Loom MCP ({clean_alias})",
            )
            provider_cfg = OAuthProviderConfig(
                authorization_endpoint=provider.authorization_endpoint,
                token_endpoint=provider.token_endpoint,
                client_id=provider.client_id,
                scopes=provider.scopes,
                authorize_params=dict(provider.authorize_params),
                token_params=dict(provider.token_params),
            )
            started = await asyncio.to_thread(
                self._oauth_engine.start_auth,
                provider=provider_cfg,
                preferred_port=8765,
                open_browser=True,
                allow_manual_fallback=True,
            )
        except (MCPOAuthFlowError, OAuthEngineError) as e:
            self._oauth_last_failure_by_alias[clean_alias] = str(e)
            self.notify(f"OAuth login start failed: {e}", severity="error")
            await self._oauth_show_status(notify=False, quiet=True)
            return
        self._oauth_pending_alias = clean_alias
        self._oauth_pending_state = started.state
        self._oauth_pending_url = started.authorization_url
        self._oauth_pending_provider = provider_cfg
        self._oauth_pending_expires_at = started.expires_at_unix
        self._set_oauth_status_text(
            "OAuth state: auth_in_progress "
            f"expires_at={started.expires_at_unix}"
        )
        if started.callback_mode == "manual":
            self.notify(
                "Loopback callback unavailable. Use Copy Login URL and Enter Callback Code.",
                severity="warning",
                timeout=7,
            )
        elif not started.browser_opened:
            self.notify(
                "Browser did not auto-open. Use Copy Login URL and complete auth manually.",
                severity="warning",
                timeout=7,
            )
        else:
            self.notify("Browser login started. Waiting for callback...")
        self.run_worker(
            self._oauth_complete_pending_login(clean_alias),
            group="mcp-oauth-login",
            exclusive=True,
        )

    def _oauth_copy_login_url(self) -> None:
        if not self._oauth_pending_url:
            self.notify("No OAuth login URL is pending.", severity="warning")
            return
        copier = getattr(self.app, "copy_to_clipboard", None)
        if callable(copier):
            try:
                copier(self._oauth_pending_url)
                self.notify("Copied OAuth login URL to clipboard.")
                return
            except Exception:
                pass
        self.notify(
            f"OAuth login URL: {self._oauth_pending_url}",
            timeout=8,
        )

    def _oauth_prompt_callback_code(self) -> None:
        if not self._oauth_pending_state:
            self.notify("No OAuth login is currently pending.", severity="warning")
            return
        self.app.push_screen(
            OAuthCodeEntryScreen(),
            callback=self._on_oauth_code_entered,
        )

    def _on_oauth_code_entered(self, raw: str | None) -> None:
        value = str(raw or "").strip()
        if not value:
            return
        self.run_worker(
            self._oauth_submit_callback_input(value),
            group="mcp-oauth-submit-code",
            exclusive=True,
        )

    async def _oauth_submit_callback_input(self, raw_input: str) -> None:
        if not self._oauth_pending_state:
            self.notify("No OAuth login is currently pending.", severity="warning")
            return
        try:
            await asyncio.to_thread(
                self._oauth_engine.submit_callback_input,
                state=self._oauth_pending_state,
                raw_input=raw_input,
            )
        except OAuthEngineError as e:
            self.notify(f"Callback input rejected: {e}", severity="error")
            return
        self.notify("Callback input submitted. Finalizing OAuth login...")

    async def _oauth_complete_pending_login(self, alias: str) -> None:
        if alias != self._oauth_pending_alias:
            return
        pending_state = self._oauth_pending_state
        provider = self._oauth_pending_provider
        if not pending_state or provider is None:
            return
        try:
            callback = await asyncio.to_thread(
                self._oauth_engine.await_callback,
                state=pending_state,
                timeout_seconds=180,
            )
            token_payload = await asyncio.to_thread(
                self._oauth_engine.finish_auth,
                provider=provider,
                state=pending_state,
                callback=callback,
                timeout_seconds=180,
            )
        except OAuthEngineError as e:
            self._oauth_last_failure_by_alias[alias] = str(e)
            self.notify(f"OAuth login failed: {e}", severity="error")
            self._clear_oauth_pending_if_current(alias=alias, state=pending_state)
            await self._oauth_show_status(notify=False, quiet=True)
            return

        access_token = str(token_payload.get("access_token", "")).strip()
        if not access_token:
            self._oauth_last_failure_by_alias[alias] = "token payload missing access_token"
            self.notify("OAuth login failed: token payload missing access_token.", severity="error")
            self._clear_oauth_pending_if_current(alias=alias, state=pending_state)
            await self._oauth_show_status(notify=False, quiet=True)
            return
        expires_at_unix: int | None = None
        expires_in = token_payload.get("expires_in")
        if expires_in not in (None, ""):
            try:
                expires_at_unix = int(time.time()) + max(1, int(expires_in))
            except (TypeError, ValueError):
                expires_at_unix = None
        if expires_at_unix is None:
            raw_expires_at = token_payload.get("expires_at")
            if raw_expires_at not in (None, ""):
                try:
                    expires_at_unix = int(raw_expires_at)
                except (TypeError, ValueError):
                    expires_at_unix = None

        token_scopes = str(token_payload.get("scope", "")).strip().split(" ")
        merged_scopes = self._split_csv_items(self.query_one("#mcp-oauth-scopes", Input).value)
        final_scopes = [
            scope
            for scope in [*list(provider.scopes), *token_scopes, *list(merged_scopes)]
            if str(scope).strip()
        ]
        final_scopes = list(
            dict.fromkeys(
                str(scope).strip()
                for scope in final_scopes
                if str(scope).strip()
            )
        )

        try:
            client_secret = str(
                dict(provider.token_params).get("client_secret", "")
            ).strip()
            await asyncio.to_thread(
                upsert_mcp_oauth_token,
                alias=alias,
                access_token=access_token,
                refresh_token=str(token_payload.get("refresh_token", "")).strip(),
                token_type=str(token_payload.get("token_type", "")).strip() or "Bearer",
                scopes=final_scopes,
                expires_at_unix=expires_at_unix,
                token_endpoint=provider.token_endpoint,
                authorization_endpoint=provider.authorization_endpoint,
                client_id=provider.client_id,
                obtained_via="browser_pkce",
                extra_fields={
                    "client_secret": client_secret,
                } if client_secret else None,
            )
        except MCPOAuthStoreError as e:
            self._oauth_last_failure_by_alias[alias] = str(e)
            self.notify(f"OAuth token store write failed: {e}", severity="error")
            self._clear_oauth_pending_if_current(alias=alias, state=pending_state)
            await self._oauth_show_status(notify=False, quiet=True)
            return

        self._changed = True
        self._oauth_last_failure_by_alias.pop(alias, None)
        self._clear_oauth_pending_if_current(alias=alias, state=pending_state)
        self.query_one("#mcp-oauth-access-token", Input).value = ""
        self.query_one("#mcp-oauth-refresh-token", Input).value = ""
        self.query_one("#mcp-oauth-expires-in", Input).value = ""
        await self._oauth_show_status(notify=True, quiet=False)

    async def _oauth_show_status(self, *, notify: bool = False, quiet: bool = False) -> None:
        alias = self._current_alias()
        if not alias:
            if not quiet:
                self.notify("Alias is required.", severity="error")
            return
        try:
            clean_alias = ensure_valid_alias(alias)
            view = await asyncio.to_thread(self._manager.get_view, clean_alias)
            if view is None:
                if not quiet:
                    self.notify(f"Alias not found: {clean_alias}", severity="error")
                return
            state = await asyncio.to_thread(oauth_state_for_alias, clean_alias)
        except Exception as e:
            if not quiet:
                self.notify(f"OAuth status failed: {e}", severity="error")
            return

        if clean_alias == self._oauth_pending_alias and self._oauth_pending_state:
            text = (
                "OAuth state: auth_in_progress "
                f"expires_at={self._oauth_pending_expires_at or '-'}"
            )
            self._set_oauth_status_text(text)
            if notify and not quiet:
                self.notify(text)
            return

        last_failure = (
            str(state.get("last_failure_reason", "") or "")
            or self._oauth_last_failure_by_alias.get(clean_alias, "")
        )
        text = (
            f"OAuth state: {state.get('state', 'unknown')} "
            f"expired={'yes' if state.get('expired') else 'no'} "
            f"expires_at={state.get('expires_at') or '-'}"
        )
        if last_failure:
            text += f" last_failure={last_failure}"
        self._set_oauth_status_text(text)
        if notify and not quiet:
            self.notify(text)

    async def _oauth_clear_token(self) -> None:
        alias = self._current_alias()
        if not alias:
            self.notify("Alias is required.", severity="error")
            return
        try:
            clean_alias = ensure_valid_alias(alias)
            view = await asyncio.to_thread(self._manager.get_view, clean_alias)
            if view is None:
                self.notify(f"Alias not found: {clean_alias}", severity="error")
                return
            if clean_alias == self._oauth_pending_alias and self._oauth_pending_state:
                pending_state = self._oauth_pending_state
                await asyncio.to_thread(
                    self._oauth_engine.cancel_auth,
                    state=pending_state,
                )
                self._clear_oauth_pending_if_current(alias=clean_alias, state=pending_state)
            await asyncio.to_thread(remove_mcp_oauth_token, clean_alias)
        except MCPOAuthStoreError as e:
            self.notify(f"OAuth clear failed: {e}", severity="error")
            return
        self._changed = True
        await self._oauth_show_status(notify=True, quiet=False)

    async def _test_alias(self) -> None:
        alias = self._current_alias()
        if not alias:
            self.notify("Alias is required.", severity="error")
            return
        try:
            clean_alias = ensure_valid_alias(alias)
            view = await asyncio.to_thread(self._manager.get_view, clean_alias)
            if view is None:
                self.notify(f"Alias not found: {clean_alias}", severity="error")
                return
            if view.server.type == MCP_SERVER_TYPE_LOCAL:
                _view, tools = await asyncio.to_thread(self._manager.probe_server, clean_alias)
            else:
                from loom.integrations.mcp_tools import MCPConnectionManager

                runtime = MCPConnectionManager(
                    mcp_config=MCPConfig(servers={clean_alias: view.server}),
                )
                tools = await asyncio.to_thread(
                    runtime.list_tools,
                    alias=clean_alias,
                    server=view.server,
                )
        except Exception as e:
            self.notify(f"MCP test failed: {e}", severity="error")
            return
        names = [str(tool.get("name", "")).strip() for tool in tools]
        rendered = ", ".join(name for name in names if name) or "(none)"
        self.notify(
            f"Probe ok for {view.alias}: {len(names)} tool(s): {rendered}",
            timeout=6,
        )


class MCPManagerModalScreen(ModalScreen[dict[str, object] | None]):
    """Modal wrapper hosting the MCP manager widget."""

    DEFAULT_CSS = """
    MCPManagerModalScreen {
        align: center middle;
    }
    """

    def __init__(
        self,
        manager: MCPConfigManager,
        *,
        explicit_auth_path: Path | None = None,
        oauth_browser_login_enabled: bool = True,
    ) -> None:
        super().__init__()
        self._manager = manager
        self._explicit_auth_path = explicit_auth_path
        self._oauth_browser_login_enabled = bool(oauth_browser_login_enabled)

    def compose(self) -> ComposeResult:
        yield MCPManagerScreen(
            self._manager,
            explicit_auth_path=self._explicit_auth_path,
            oauth_browser_login_enabled=self._oauth_browser_login_enabled,
            embedded=False,
            on_close=self._handle_close,
        )

    def _handle_close(self, result: dict[str, object]) -> None:
        self.dismiss(result)
