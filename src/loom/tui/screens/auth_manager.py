"""Auth profile management UI."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable
from dataclasses import replace

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

from loom.auth.config import (
    AuthProfile,
    default_workspace_auth_defaults_path,
    load_merged_auth_config,
    remove_auth_profile,
    resolve_auth_write_path,
    set_workspace_auth_default,
    upsert_auth_profile,
)
from loom.auth.oauth_profiles import (
    OAuthProfileError,
    login_oauth_profile,
    logout_oauth_profile,
    oauth_state_for_profile,
    refresh_oauth_profile,
)
from loom.auth.resources import (
    bind_resource_to_profile,
    default_workspace_auth_resources_path,
    load_workspace_auth_resources,
    profile_bindings_map,
    remove_profile_from_resource_store,
    set_workspace_resource_default,
    sync_missing_drafts,
)
from loom.mcp.config import MCPConfigManager, ensure_valid_env_key
from loom.tools import create_default_registry
from loom.tui.screens.oauth_code_entry import OAuthCodeEntryScreen


class ConfirmProfileSwitchScreen(ModalScreen[str]):
    """Confirm loading another profile when there are unsaved edits."""

    _inherit_bindings = False

    BINDINGS = [
        Binding("s", "save", "Save"),
        Binding("d", "discard", "Discard"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ConfirmProfileSwitchScreen {
        align: center middle;
    }
    #auth-switch-confirm-dialog {
        width: 76;
        height: auto;
        border: solid $warning;
        padding: 1 2;
        background: $surface;
    }
    #auth-switch-confirm-actions {
        height: auto;
        margin-top: 1;
    }
    #auth-switch-confirm-actions Button {
        margin-right: 1;
    }
    """

    def __init__(self, *, current_profile_id: str, target_profile_id: str) -> None:
        super().__init__()
        self._current_profile_id = current_profile_id or "(new profile)"
        self._target_profile_id = target_profile_id

    def compose(self) -> ComposeResult:
        with Vertical(id="auth-switch-confirm-dialog"):
            yield Label("[bold #e0af68]Unsaved auth changes[/]")
            yield Label(
                "Save changes before switching from "
                f"[bold]{self._current_profile_id}[/bold] to "
                f"[bold]{self._target_profile_id}[/bold]?"
            )
            yield Label(
                "Choose Save to keep edits, Discard to drop edits, or Esc to cancel.",
            )
            with Horizontal(id="auth-switch-confirm-actions"):
                yield Button("Save", id="auth-switch-confirm-save", variant="primary")
                yield Button("Discard", id="auth-switch-confirm-discard", variant="warning")
                yield Button("Cancel", id="auth-switch-confirm-cancel")

    def action_save(self) -> None:
        self.dismiss("save")

    def action_discard(self) -> None:
        self.dismiss("discard")

    def action_cancel(self) -> None:
        self.dismiss("cancel")

    @on(Button.Pressed, "#auth-switch-confirm-save")
    def _on_save_button(self) -> None:
        self.dismiss("save")

    @on(Button.Pressed, "#auth-switch-confirm-discard")
    def _on_discard_button(self) -> None:
        self.dismiss("discard")

    @on(Button.Pressed, "#auth-switch-confirm-cancel")
    def _on_cancel_button(self) -> None:
        self.dismiss("cancel")

    def on_key(self, event: events.Key) -> None:
        key = event.key.lower()
        if key == "s":
            self.dismiss("save")
            event.stop()
            event.prevent_default()
            return
        if key == "d":
            self.dismiss("discard")
            event.stop()
            event.prevent_default()
            return
        if key == "escape":
            self.dismiss("cancel")
            event.stop()
            event.prevent_default()


class AuthManagerScreen(Vertical):
    """Auth profile add/edit/remove widget."""

    _NO_TARGET_VALUE = "__none__"
    _NO_MODE_VALUE = "__mode_unset__"
    _SUPPORTED_AUTH_MODES = (
        "api_key",
        "oauth2_pkce",
        "oauth2_device",
        "cli_passthrough",
        "env_passthrough",
    )
    _OAUTH_MODES = ("oauth2_pkce", "oauth2_device")
    _OAUTH_METADATA_KEYS = (
        "oauth_authorization_endpoint",
        "oauth_token_endpoint",
        "oauth_client_id",
        "oauth_client_secret",
        "oauth_scope",
    )
    _OAUTH_METADATA_FIELDS = (
        ("oauth_authorization_endpoint", "auth-oauth-authorize-url"),
        ("oauth_token_endpoint", "auth-oauth-token-url"),
        ("oauth_client_id", "auth-oauth-client-id"),
        ("oauth_client_secret", "auth-oauth-client-secret"),
        ("oauth_scope", "auth-oauth-scope"),
    )
    _OAUTH_REQUIRED_METADATA_FIELDS = (
        ("oauth_authorization_endpoint", "OAuth Authorization URL"),
        ("oauth_token_endpoint", "OAuth Token URL"),
        ("oauth_client_id", "OAuth Client ID"),
    )
    _SAFE_REF_FRAGMENT_RE = re.compile(r"[^a-z0-9_]+")

    BINDINGS = [
        Binding("escape", "request_close", "Close"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+w", "request_close", "Close Tab", show=False, priority=True),
    ]

    _FORM_FIELD_IDS = (
        "auth-profile-id",
        "auth-mode",
        "auth-default-provider",
        "auth-label",
        "auth-resource-target",
        "auth-secret-ref",
        "auth-token-ref",
        "auth-scopes",
        "auth-env",
        "auth-command",
        "auth-auth-check",
        "auth-oauth-authorize-url",
        "auth-oauth-token-url",
        "auth-oauth-client-id",
        "auth-oauth-client-secret",
        "auth-oauth-scope",
        "auth-meta",
    )

    DEFAULT_CSS = """
    AuthManagerScreen {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }
    AuthManagerScreen.modal-mode {
        align: center middle;
    }
    AuthManagerScreen.embedded-mode {
        align: left top;
        padding: 0;
    }
    AuthManagerScreen.modal-mode #auth-manager-dialog {
        width: 100;
        height: 90%;
        max-height: 46;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
        overflow: hidden;
    }
    AuthManagerScreen.embedded-mode #auth-manager-dialog {
        width: 100%;
        height: 1fr;
        max-height: 100%;
        border: none;
        padding: 0 1;
    }
    #auth-manager-dialog {
        width: 100%;
        height: 1fr;
        border: none;
        padding: 0 1;
        background: $surface;
    }
    #auth-manager-title {
        text-style: bold;
        margin-bottom: 0;
    }
    #auth-manager-form {
        height: 1fr;
        overflow-y: auto;
        margin-bottom: 0;
    }
    #auth-manager-context {
        color: $text-muted;
        margin-top: 0;
        margin-bottom: 0;
    }
    #auth-summary-help {
        margin-bottom: 1;
    }
    #auth-manager-summary {
        height: 6;
        max-height: 7;
        margin-bottom: 0;
        border: round $surface-lighten-1;
    }
    #auth-manager-advanced {
        margin-top: 1;
    }
    .auth-label {
        margin-top: 0;
    }
    .auth-help {
        color: $text-muted;
        margin-top: 0;
        margin-bottom: 0;
        width: 1fr;
        height: auto;
        text-wrap: wrap;
        overflow-x: hidden;
    }
    #auth-oauth-settings {
        display: none;
        margin-top: 1;
        height: auto;
        layout: vertical;
    }
    #auth-secret-ref-section,
    #auth-token-ref-section,
    #auth-scopes-section,
    #auth-env-section,
    #auth-command-section {
        height: auto;
        layout: vertical;
    }
    #auth-token-ref-section {
        display: none;
        margin-bottom: 1;
    }
    #auth-scopes-section,
    #auth-env-section,
    #auth-command-section {
        display: none;
    }
    .auth-input {
        margin-top: 0;
    }
    .auth-select {
        margin-top: 0;
    }
    .auth-checkbox {
        margin-top: 0;
    }
    #auth-actions-primary {
        margin-top: 1;
    }
    #auth-actions-secondary {
        margin-top: 0;
        margin-bottom: 1;
    }
    .auth-actions-row {
        height: auto;
        margin-top: 0;
    }
    .auth-actions-row Button {
        margin-right: 0;
        min-width: 0;
        width: 1fr;
        padding: 0;
    }
    #auth-manager-footer {
        margin-top: 0;
        color: $text-muted;
        width: 1fr;
        height: auto;
        text-wrap: wrap;
        overflow-x: hidden;
    }
    """

    def __init__(
        self,
        *,
        workspace,
        explicit_auth_path=None,
        mcp_manager: MCPConfigManager | None = None,
        process_def: object | None = None,
        process_defs: list[object] | tuple[object, ...] | None = None,
        tool_registry=None,
        embedded: bool = False,
        on_close: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._explicit_auth_path = explicit_auth_path
        self._process_def = process_def
        self._process_defs = [item for item in (process_defs or []) if item is not None]
        self._tool_registry = tool_registry
        self._mcp_manager = mcp_manager or MCPConfigManager(
            config=None,
            workspace=workspace,
        )
        self._mcp_aliases: list[str] = []
        self._workspace_defaults: dict[str, str] = {}
        self._workspace_resource_defaults: dict[str, str] = {}
        self._profiles: dict[str, AuthProfile] = {}
        self._profile_ids: list[str] = []
        self._resources_by_id: dict[str, object] = {}
        self._resource_binding_by_profile: dict[str, str] = {}
        self._active_profile_id = ""
        self._active_provider = ""
        self._discovery_process_names: list[str] = []
        self._baseline_form_state: dict[str, str] = {}
        self._form_dirty = False
        self._suppress_dirty_tracking = False
        self._changed = False
        self._embedded = bool(embedded)
        self._on_close = on_close
        if self._embedded:
            self.add_class("embedded-mode")
        else:
            self.add_class("modal-mode")

    def compose(self) -> ComposeResult:
        with Vertical(id="auth-manager-dialog"):
            yield Label(
                "[bold #7dcfff]Auth Profile Manager[/bold #7dcfff]",
                id="auth-manager-title",
            )
            with Horizontal(classes="auth-actions-row", id="auth-actions-primary"):
                yield Button("Refresh", id="auth-btn-refresh")
                yield Button("Sync", id="auth-btn-sync")
                yield Button("Load", id="auth-btn-load")
                yield Button("Save", id="auth-btn-save", variant="primary")
                yield Button("Duplicate", id="auth-btn-duplicate")
                yield Button("Close", id="auth-btn-close")
            with Horizontal(classes="auth-actions-row", id="auth-actions-secondary"):
                yield Button("Rebind", id="auth-btn-rebind")
                yield Button("Archive", id="auth-btn-archive")
                yield Button("Remove", id="auth-btn-remove", variant="error")
            with Horizontal(classes="auth-actions-row", id="auth-actions-oauth"):
                yield Button("OAuth Login", id="auth-btn-oauth-login")
                yield Button("OAuth Status", id="auth-btn-oauth-status")
                yield Button("OAuth Logout", id="auth-btn-oauth-logout")
                yield Button("OAuth Refresh", id="auth-btn-oauth-refresh")
            yield Static("", id="auth-manager-context")
            summary_table = DataTable(id="auth-manager-summary")
            summary_table.cursor_type = "row"
            summary_table.zebra_stripes = True
            summary_table.add_columns(
                "Profile ID",
                "Target Resource",
                "Provider",
                "Mode",
                "Account Label",
            )
            yield summary_table
            yield Label(
                "Select a profile row to load it below for editing.",
                classes="auth-help",
                id="auth-summary-help",
            )
            with VerticalScroll(id="auth-manager-form"):
                yield Label("Profile ID", classes="auth-label")
                yield Input(
                    id="auth-profile-id",
                    classes="auth-input",
                )
                yield Label(
                    "Unique id for this account profile (required).",
                    classes="auth-help",
                )
                yield Label("Target Resource", classes="auth-label")
                yield Select(
                    options=[("None (provider-wide)", self._NO_TARGET_VALUE)],
                    id="auth-resource-target",
                    classes="auth-select",
                    allow_blank=False,
                    value=self._NO_TARGET_VALUE,
                )
                yield Label(
                    "Select MCP/API/tool auth target. Provider is derived automatically.",
                    classes="auth-help",
                )
                yield Label("Provider (derived)", classes="auth-label")
                yield Static("-", id="auth-provider-derived")
                yield Label("Mode", classes="auth-label")
                yield Select(
                    options=self._mode_options(),
                    id="auth-mode",
                    classes="auth-select",
                    allow_blank=False,
                    value=self._NO_MODE_VALUE,
                )
                yield Label(
                    "Credential flow used at runtime (pick from supported modes).",
                    classes="auth-help",
                )
                yield Checkbox(
                    "Set as workspace default for this provider",
                    id="auth-default-provider",
                    classes="auth-checkbox",
                    value=False,
                )
                yield Label(
                    "When enabled, this profile is used by default for\n"
                    "this provider in this workspace.",
                    classes="auth-help",
                )
                yield Label("Account Label", classes="auth-label")
                yield Input(
                    id="auth-label",
                    classes="auth-input",
                )
                yield Label(
                    "Human-friendly name shown in listings.",
                    classes="auth-help",
                )
                with Vertical(id="auth-secret-ref-section"):
                    yield Label("Secret Ref", classes="auth-label")
                    yield Input(
                        id="auth-secret-ref",
                        classes="auth-input",
                    )
                    yield Label(
                        "Where the base secret comes from (env/keychain/vault reference).",
                        classes="auth-help",
                    )
                with Vertical(id="auth-oauth-settings"):
                    yield Label("OAuth Authorization URL (required)", classes="auth-label")
                    yield Input(
                        id="auth-oauth-authorize-url",
                        classes="auth-input",
                    )
                    yield Label(
                        "OAuth authorization endpoint for browser login.",
                        classes="auth-help",
                    )
                    yield Label("OAuth Token URL (required)", classes="auth-label")
                    yield Input(
                        id="auth-oauth-token-url",
                        classes="auth-input",
                    )
                    yield Label(
                        "OAuth token endpoint for code exchange/refresh.",
                        classes="auth-help",
                    )
                    yield Label("OAuth Client ID (required)", classes="auth-label")
                    yield Input(
                        id="auth-oauth-client-id",
                        classes="auth-input",
                    )
                    yield Label(
                        "OAuth client id used for browser login.",
                        classes="auth-help",
                    )
                    yield Label("OAuth Client Secret (Optional)", classes="auth-label")
                    yield Input(
                        id="auth-oauth-client-secret",
                        classes="auth-input",
                    )
                    yield Label(
                        "Optional secret required for token refresh with some providers.",
                        classes="auth-help",
                    )
                    yield Label("OAuth Scope Hint", classes="auth-label")
                    yield Input(
                        id="auth-oauth-scope",
                        classes="auth-input",
                    )
                    yield Label(
                        "Optional space/comma-separated scope hint metadata.",
                        classes="auth-help",
                    )
                with Collapsible(
                    title="Advanced",
                    id="auth-manager-advanced",
                    collapsed=True,
                ):
                    with Vertical(id="auth-token-ref-section"):
                        yield Label("Token Ref", classes="auth-label")
                        yield Input(
                            id="auth-token-ref",
                            classes="auth-input",
                        )
                        yield Label(
                            "OAuth token storage reference. Must be keychain://... "
                            "and should stay separate from MCP alias token stores. "
                            "Default: keychain://loom/<provider>/<profile>/tokens.",
                            classes="auth-help",
                        )
                    with Vertical(id="auth-scopes-section"):
                        yield Label("Scopes (comma-separated)", classes="auth-label")
                        yield Input(
                            id="auth-scopes",
                            classes="auth-input",
                        )
                        yield Label(
                            "Optional OAuth scopes for this profile.",
                            classes="auth-help",
                        )
                    with Vertical(id="auth-env-section"):
                        yield Label("Env pairs (comma-separated KEY=VALUE)", classes="auth-label")
                        yield Input(
                            id="auth-env",
                            classes="auth-input",
                        )
                        yield Label(
                            "Required for env_passthrough; optional for api_key fallback.",
                            classes="auth-help",
                        )
                    with Vertical(id="auth-command-section"):
                        yield Label("Command (cli_passthrough)", classes="auth-label")
                        yield Input(
                            id="auth-command",
                            classes="auth-input",
                        )
                        yield Label(
                            "Required for cli_passthrough mode.",
                            classes="auth-help",
                        )
                    yield Label("Auth check args (comma-separated)", classes="auth-label")
                    yield Input(
                        id="auth-auth-check",
                        classes="auth-input",
                    )
                    yield Label(
                        "Optional health-check args run for this auth method.",
                        classes="auth-help",
                    )
                    yield Label("Metadata (comma-separated KEY=VALUE)", classes="auth-label")
                    yield Input(
                        id="auth-meta",
                        classes="auth-input",
                    )
                    yield Label(
                        "Optional non-secret tags (team, environment, purpose).",
                        classes="auth-help",
                    )
            yield Label(
                "[dim]Refresh is read-only. Use Sync to discover drafts, then Save to upsert profile and bindings.[/dim]",  # noqa: E501
                id="auth-manager-footer",
            )

    async def on_mount(self) -> None:
        self._set_form_values(
            profile_id="",
            provider="",
            mode="",
            set_default=False,
            label="",
            resource_id="",
            secret_ref="",
            token_ref="",
            scopes="",
            env="",
            command="",
            auth_check="",
            metadata="",
        )
        self._mark_form_clean(active_profile_id="")
        await self._refresh_summary()
        self.query_one("#auth-profile-id", Input).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key.lower() != "ctrl+w":
            return
        self.run_worker(
            self.action_request_close(),
            group="auth-manager-close-request",
            exclusive=True,
        )
        event.stop()
        event.prevent_default()

    def action_close(self) -> None:
        result = {"changed": self._changed}
        if callable(self._on_close):
            self._on_close(result)

    async def action_request_close(self) -> None:
        await self._request_close()

    async def action_refresh(self) -> None:
        await self._refresh_summary()

    async def action_sync(self) -> None:
        await self._sync_missing_drafts()
        await self._refresh_summary()

    @on(Button.Pressed)
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "auth-btn-close":
            await self.action_request_close()
            return
        if button_id == "auth-btn-refresh":
            await self.action_refresh()
            return
        if button_id == "auth-btn-sync":
            await self.action_sync()
            return
        if button_id == "auth-btn-load":
            selected = self._selected_summary_profile_id()
            await self._request_profile_switch(selected or self._profile_id())
            return
        if button_id == "auth-btn-save":
            await self._save_profile()
            return
        if button_id == "auth-btn-duplicate":
            await self._duplicate_profile()
            return
        if button_id == "auth-btn-rebind":
            await self._rebind_profile()
            return
        if button_id == "auth-btn-archive":
            await self._archive_profile()
            return
        if button_id == "auth-btn-remove":
            await self._remove_profile()
            return
        if button_id == "auth-btn-oauth-login":
            await self._oauth_login()
            return
        if button_id == "auth-btn-oauth-status":
            await self._oauth_status()
            return
        if button_id == "auth-btn-oauth-logout":
            await self._oauth_logout()
            return
        if button_id == "auth-btn-oauth-refresh":
            await self._oauth_refresh()
            return

    @on(Input.Changed)
    def _on_form_input_changed(self, event: Input.Changed) -> None:
        if event.input.id not in self._FORM_FIELD_IDS:
            return
        if event.input.id == "auth-profile-id" and not self._suppress_dirty_tracking:
            self._maybe_set_default_oauth_token_ref()
        self._update_form_dirty()

    @on(Select.Changed)
    def _on_form_select_changed(self, event: Select.Changed) -> None:
        if event.select.id not in self._FORM_FIELD_IDS:
            return
        if self._suppress_dirty_tracking:
            return
        if event.select.id == "auth-resource-target":
            self._sync_provider_display()
            self._refresh_mode_select()
            self._refresh_oauth_settings_visibility()
            self._maybe_set_default_oauth_token_ref()
        elif event.select.id == "auth-mode":
            self._refresh_oauth_settings_visibility()
            self._maybe_set_default_oauth_token_ref()
        self._update_form_dirty()

    @on(Checkbox.Changed, "#auth-default-provider")
    def _on_form_checkbox_changed(self, _event: Checkbox.Changed) -> None:
        self._update_form_dirty()

    @on(DataTable.RowSelected, "#auth-manager-summary")
    async def _on_summary_row_selected(self, event: DataTable.RowSelected) -> None:
        profile_id = str(getattr(event.row_key, "value", "") or "").strip()
        if not profile_id and 0 <= event.cursor_row < len(self._profile_ids):
            profile_id = self._profile_ids[event.cursor_row]
        await self._request_profile_switch(profile_id)

    def _capture_form_state(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for field_id in self._FORM_FIELD_IDS:
            if field_id == "auth-mode":
                values[field_id] = self._selected_mode()
                continue
            if field_id == "auth-default-provider":
                values[field_id] = "1" if self._default_provider_selected() else "0"
                continue
            if field_id == "auth-resource-target":
                values[field_id] = self._selected_resource_id()
                continue
            values[field_id] = self.query_one(f"#{field_id}", Input).value
        return values

    def _mark_form_clean(self, *, active_profile_id: str | None = None) -> None:
        self._baseline_form_state = self._capture_form_state()
        self._form_dirty = False
        if active_profile_id is not None:
            self._active_profile_id = active_profile_id

    def _update_form_dirty(self) -> None:
        if self._suppress_dirty_tracking:
            return
        self._form_dirty = self._capture_form_state() != self._baseline_form_state

    @classmethod
    def _encode_mcp_server_value(cls, alias: str) -> str:
        clean = str(alias or "").strip()
        if not clean:
            return cls._NO_TARGET_VALUE
        return clean

    @classmethod
    def _decode_mcp_server_value(cls, value: object) -> str:
        clean = str(value or "").strip()
        if not clean or clean == cls._NO_TARGET_VALUE:
            return ""
        return clean

    def _mcp_target_options(
        self,
        *,
        include_alias: str = "",
    ) -> list[tuple[str, str]]:
        resource_id = str(include_alias or "").strip()
        options: list[tuple[str, str]] = [("None (provider-wide)", self._NO_TARGET_VALUE)]
        if self._resources_by_id:
            for item in sorted(
                self._resources_by_id.items(),
                key=lambda pair: (
                    str(getattr(pair[1], "resource_kind", "")),
                    str(getattr(pair[1], "display_name", "")),
                ),
            ):
                rid, resource = item
                label = (
                    f"{getattr(resource, 'display_name', rid)}"
                    f" [{getattr(resource, 'provider', '-')}]"
                )
                options.append((label, rid))
            if resource_id and resource_id not in self._resources_by_id:
                options.append((f"Unknown resource: {resource_id}", resource_id))
            return options

        aliases = {str(alias).strip() for alias in self._mcp_aliases if str(alias).strip()}
        if resource_id:
            aliases.add(resource_id)
        for alias in sorted(aliases):
            options.append((f"MCP: {alias}", alias))
        return options

    def _mode_options(
        self,
        *,
        include_mode: str = "",
    ) -> list[tuple[str, str]]:
        selected_resource = self._resources_by_id.get(self._selected_resource_id())
        if selected_resource is not None and getattr(selected_resource, "modes", ()):
            modes = {
                str(mode).strip().lower()
                for mode in getattr(selected_resource, "modes", ())
                if str(mode).strip().lower() in self._SUPPORTED_AUTH_MODES
            }
        else:
            modes = set(self._SUPPORTED_AUTH_MODES)
        if include_mode:
            modes.add(include_mode)
        options: list[tuple[str, str]] = [("Select mode", self._NO_MODE_VALUE)]
        for mode in sorted(modes):
            options.append((mode, mode))
        return options

    @classmethod
    def _encode_mode_value(cls, mode: str) -> str:
        clean = str(mode or "").strip().lower()
        if not clean:
            return cls._NO_MODE_VALUE
        return clean

    @classmethod
    def _decode_mode_value(cls, value: object) -> str:
        clean = str(value or "").strip().lower()
        if not clean or clean == cls._NO_MODE_VALUE:
            return ""
        return clean

    def _refresh_mode_select(self, *, include_mode: str = "") -> None:
        select = self.query_one("#auth-mode", Select)
        current_mode = self._decode_mode_value(select.value)
        target_mode = str(include_mode or current_mode).strip().lower()
        options = self._mode_options(include_mode=target_mode)
        select.set_options(options)
        if target_mode:
            encoded = self._encode_mode_value(target_mode)
            if select.value != encoded:
                select.value = encoded

    def _set_mode_select_value(self, mode: str) -> None:
        clean = str(mode or "").strip().lower()
        self._refresh_mode_select(include_mode=clean)
        select = self.query_one("#auth-mode", Select)
        select.value = self._encode_mode_value(clean)

    def _selected_mode(self) -> str:
        select = self.query_one("#auth-mode", Select)
        return self._decode_mode_value(select.value)

    @classmethod
    def _sanitize_ref_fragment(cls, raw: object, *, fallback: str) -> str:
        cleaned = cls._SAFE_REF_FRAGMENT_RE.sub(
            "_",
            str(raw or "").strip().lower(),
        ).strip("_")
        return cleaned or fallback

    def _default_oauth_token_ref(
        self,
        *,
        provider: str = "",
        key_name: str = "",
    ) -> str:
        provider_hint = str(provider or "").strip()
        if not provider_hint:
            provider_hint = self._provider_for_selected_resource()
        if not provider_hint:
            provider_hint = self._active_provider

        key_hint = str(key_name or "").strip()
        if not key_hint:
            key_hint = self._profile_id()
        if not key_hint:
            selected_resource = self._resources_by_id.get(self._selected_resource_id())
            if selected_resource is not None:
                key_hint = str(getattr(selected_resource, "resource_key", "")).strip()
                if not key_hint:
                    key_hint = str(getattr(selected_resource, "resource_id", "")).strip()
        if not key_hint:
            key_hint = provider_hint or "default"

        provider_part = self._sanitize_ref_fragment(
            provider_hint or key_hint,
            fallback="oauth",
        )
        key_part = self._sanitize_ref_fragment(
            key_hint,
            fallback="default",
        )
        return f"keychain://loom/{provider_part}/{key_part}/tokens"

    def _maybe_set_default_oauth_token_ref(
        self,
        *,
        provider: str = "",
        key_name: str = "",
    ) -> None:
        if self._selected_mode() not in self._OAUTH_MODES:
            return
        token_input = self.query_one("#auth-token-ref", Input)
        if str(token_input.value or "").strip():
            return
        token_input.value = self._default_oauth_token_ref(
            provider=provider,
            key_name=key_name,
        )

    def _refresh_oauth_settings_visibility(self) -> None:
        mode = self._selected_mode()
        oauth_mode = mode in self._OAUTH_MODES
        api_key_mode = mode == "api_key"
        env_passthrough_mode = mode == "env_passthrough"
        cli_passthrough_mode = mode == "cli_passthrough"
        self.query_one("#auth-oauth-settings", Vertical).display = oauth_mode
        self.query_one("#auth-secret-ref-section", Vertical).display = api_key_mode
        self.query_one("#auth-token-ref-section", Vertical).display = oauth_mode
        self.query_one("#auth-scopes-section", Vertical).display = oauth_mode
        self.query_one("#auth-env-section", Vertical).display = (
            api_key_mode or env_passthrough_mode
        )
        self.query_one("#auth-command-section", Vertical).display = cli_passthrough_mode

    def _oauth_metadata_from_form(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for key, field_id in self._OAUTH_METADATA_FIELDS:
            value = self.query_one(f"#{field_id}", Input).value.strip()
            if value:
                metadata[key] = value
        return metadata

    @classmethod
    def _missing_required_oauth_metadata(cls, metadata: dict[str, str]) -> tuple[str, ...]:
        missing: list[str] = []
        for key, label in cls._OAUTH_REQUIRED_METADATA_FIELDS:
            if not str(metadata.get(key, "") or "").strip():
                missing.append(label)
        return tuple(missing)

    def _default_provider_selected(self) -> bool:
        return bool(self.query_one("#auth-default-provider", Checkbox).value)

    def _refresh_mcp_target_select(self, *, include_alias: str = "") -> None:
        select = self.query_one("#auth-resource-target", Select)
        current_alias = self._decode_mcp_server_value(select.value)
        options = self._mcp_target_options(include_alias=include_alias or current_alias)
        select.set_options(options)

    def _set_mcp_server_select_value(self, alias: str) -> None:
        clean = str(alias or "").strip()
        self._refresh_mcp_target_select(include_alias=clean)
        select = self.query_one("#auth-resource-target", Select)
        select.value = self._encode_mcp_server_value(clean)
        self._sync_provider_display()

    def _selected_mcp_server_alias(self) -> str:
        try:
            select = self.query_one("#auth-resource-target", Select)
        except Exception:
            return ""
        return self._decode_mcp_server_value(select.value)

    def _selected_resource_id(self) -> str:
        return self._selected_mcp_server_alias()

    def _provider_for_selected_resource(self) -> str:
        resource_id = self._selected_resource_id()
        resource = self._resources_by_id.get(resource_id)
        if resource is None:
            return self._active_provider
        provider = str(getattr(resource, "provider", "")).strip()
        return provider or self._active_provider

    def _sync_provider_display(self) -> None:
        provider = self._provider_for_selected_resource()
        if not provider:
            provider = "-"
        self.query_one("#auth-provider-derived", Static).update(provider)

    def _discovery_scope_text(self) -> str:
        count = len(self._discovery_process_names)
        if count > 0:
            return (
                "discovery scope: "
                f"all workspace processes ({count}) + allowed tool auth + active MCP aliases"
            )
        return (
            "discovery scope: "
            "no process contracts loaded (uses loaded tool auth + active MCP aliases)"
        )

    @staticmethod
    def _compact_path(value: object, *, max_chars: int = 52) -> str:
        raw = str(value or "-")
        if len(raw) <= max_chars:
            return raw
        return f"...{raw[-(max_chars - 3):]}"

    def _resource_id_for_profile(self, profile: AuthProfile) -> str:
        profile_id = str(getattr(profile, "profile_id", "")).strip()
        if profile_id:
            bound = self._resource_binding_by_profile.get(profile_id)
            if bound:
                return bound
        return ""

    def _is_workspace_default(
        self,
        *,
        provider: str,
        profile_id: str,
        resource_id: str = "",
    ) -> bool:
        clean_resource_id = str(resource_id or "").strip()
        if clean_resource_id:
            return self._workspace_resource_defaults.get(clean_resource_id) == profile_id
        clean_provider = str(provider or "").strip()
        clean_profile_id = str(profile_id or "").strip()
        if not clean_provider or not clean_profile_id:
            return False
        return self._workspace_defaults.get(clean_provider) == clean_profile_id

    def _set_form_values(
        self,
        *,
        profile_id: str,
        provider: str,
        mode: str,
        set_default: bool,
        label: str,
        resource_id: str,
        secret_ref: str,
        token_ref: str,
        scopes: str,
        env: str,
        command: str,
        auth_check: str,
        metadata: str,
        oauth_metadata: dict[str, str] | None = None,
    ) -> None:
        self._suppress_dirty_tracking = True
        try:
            self.query_one("#auth-profile-id", Input).value = profile_id
            self._active_provider = str(provider or "").strip()
            self.query_one("#auth-default-provider", Checkbox).value = bool(set_default)
            self.query_one("#auth-label", Input).value = label
            self._set_mcp_server_select_value(resource_id)
            self._set_mode_select_value(mode)
            self.query_one("#auth-secret-ref", Input).value = secret_ref
            self.query_one("#auth-token-ref", Input).value = token_ref
            self.query_one("#auth-scopes", Input).value = scopes
            self.query_one("#auth-env", Input).value = env
            self.query_one("#auth-command", Input).value = command
            self.query_one("#auth-auth-check", Input).value = auth_check
            oauth_values = dict(oauth_metadata or {})
            for key, field_id in self._OAUTH_METADATA_FIELDS:
                self.query_one(f"#{field_id}", Input).value = str(
                    oauth_values.get(key, "") or ""
                ).strip()
            self.query_one("#auth-meta", Input).value = metadata
            self._sync_provider_display()
            self._refresh_oauth_settings_visibility()
        finally:
            self._suppress_dirty_tracking = False

    def _set_blank_form(self) -> None:
        self._set_form_values(
            profile_id="",
            provider="",
            mode="",
            set_default=False,
            label="",
            resource_id="",
            secret_ref="",
            token_ref="",
            scopes="",
            env="",
            command="",
            auth_check="",
            metadata="",
            oauth_metadata={},
        )
        self._mark_form_clean(active_profile_id="")

    async def _sync_missing_drafts(self) -> None:
        registry = self._tool_registry
        if registry is None:
            try:
                registry = create_default_registry()
            except Exception:
                registry = None
        try:
            result = await asyncio.to_thread(
                sync_missing_drafts,
                workspace=self._workspace,
                explicit_auth_path=self._explicit_auth_path,
                process_def=self._process_def,
                process_defs=self._process_defs,
                tool_registry=registry,
                mcp_manager=self._mcp_manager,
                scope="active",
            )
        except Exception as e:
            self.notify(
                f"Auth draft sync warning: {e}",
                severity="warning",
            )
            return
        if result.changed:
            self.notify(
                (
                    "Auth sync: "
                    f"+{result.created_drafts} drafts, "
                    f"+{result.created_bindings} bindings, "
                    f"+{result.created_resources} resources."
                ),
            )
        for warning in result.warnings:
            self.notify(f"Auth sync warning: {warning}", severity="warning")

    async def _refresh_summary(self) -> None:
        try:
            merged = await asyncio.to_thread(
                load_merged_auth_config,
                workspace=self._workspace,
                explicit_path=self._explicit_auth_path,
            )
        except Exception as e:
            self.notify(f"Auth load failed: {e}", severity="error")
            return

        try:
            views = await asyncio.to_thread(self._mcp_manager.list_views)
            self._mcp_aliases = sorted(
                {
                    str(view.alias).strip()
                    for view in views
                    if str(view.alias).strip()
                }
            )
        except Exception as e:
            self._mcp_aliases = []
            self.notify(f"MCP aliases unavailable: {e}", severity="warning")

        resources_path = default_workspace_auth_resources_path(self._workspace.resolve())
        try:
            store = await asyncio.to_thread(load_workspace_auth_resources, resources_path)
            self._resources_by_id = {
                resource_id: resource
                for resource_id, resource in store.resources.items()
                if str(getattr(resource, "status", "")).strip().lower() == "active"
            }
            self._workspace_resource_defaults = dict(store.workspace_defaults)
            self._resource_binding_by_profile = profile_bindings_map(store)
        except Exception as e:
            self._resources_by_id = {}
            self._workspace_resource_defaults = {}
            self._resource_binding_by_profile = {}
            self.notify(f"Auth resources unavailable: {e}", severity="warning")

        process_names = sorted(
            {
                str(getattr(item, "name", "")).strip()
                for item in self._process_defs
                if str(getattr(item, "name", "")).strip()
            }
        )
        self._discovery_process_names = process_names
        self._profiles = dict(merged.config.profiles)
        self._workspace_defaults = dict(merged.workspace_defaults)
        source_label = "explicit" if merged.explicit_path else "user"
        context_text = (
            f"profiles={len(self._profiles)} source={source_label} "
            f"discovery={len(process_names)} process(es), {len(self._mcp_aliases)} mcp alias(es)"
        )
        self.query_one("#auth-manager-context", Static).update(context_text)
        current_alias = self._selected_resource_id()
        self._suppress_dirty_tracking = True
        try:
            self._set_mcp_server_select_value(current_alias)
        finally:
            self._suppress_dirty_tracking = False
        self._render_summary()

    def _render_summary(self) -> None:
        table = self.query_one("#auth-manager-summary", DataTable)
        table.clear()
        self._profile_ids = []

        for profile_id in sorted(self._profiles):
            profile = self._profiles[profile_id]
            label = profile.account_label or "-"
            resource_id = self._resource_id_for_profile(profile)
            resource = self._resources_by_id.get(resource_id)
            resource_label = "-"
            if resource is not None:
                resource_label = str(getattr(resource, "display_name", "")).strip() or (
                    f"{getattr(resource, 'resource_kind', '?')}:"
                    f"{getattr(resource, 'resource_key', '?')}"
                )
            else:
                mcp_alias = str(getattr(profile, "mcp_server", "")).strip()
                if mcp_alias:
                    resource_label = f"Unbound (mcp:{mcp_alias})"
            table.add_row(
                profile.profile_id,
                resource_label,
                profile.provider,
                profile.mode,
                label,
                key=profile.profile_id,
            )
            self._profile_ids.append(profile.profile_id)

        selected = self._active_profile_id or self._profile_id()
        if selected:
            self._select_summary_profile(selected)

    def _select_summary_profile(self, profile_id: str) -> None:
        if not profile_id:
            return
        table = self.query_one("#auth-manager-summary", DataTable)
        for row_index, candidate in enumerate(self._profile_ids):
            if candidate == profile_id:
                table.move_cursor(row=row_index, column=0, scroll=True)
                return

    @staticmethod
    def _split_csv_values(raw: str) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []
        return [item.strip() for item in text.split(",") if item.strip()]

    def _profile_id(self) -> str:
        return self.query_one("#auth-profile-id", Input).value.strip()

    def _selected_summary_profile_id(self) -> str:
        table = self.query_one("#auth-manager-summary", DataTable)
        row_index = int(getattr(table, "cursor_row", -1))
        if 0 <= row_index < len(self._profile_ids):
            return self._profile_ids[row_index]
        return ""

    @staticmethod
    def _parse_kv_pairs(values: list[str], *, env_keys: bool = False) -> dict[str, str]:
        result: dict[str, str] = {}
        for value in values:
            if "=" not in value:
                raise ValueError(f"Expected KEY=VALUE entry, got {value!r}.")
            key, item = value.split("=", 1)
            clean_key = key.strip()
            if env_keys:
                clean_key = ensure_valid_env_key(clean_key)
            if not clean_key:
                raise ValueError(f"Invalid empty key in {value!r}.")
            result[clean_key] = item
        return result

    async def _request_profile_switch(self, profile_id: str) -> None:
        clean_profile_id = str(profile_id or "").strip()
        if not clean_profile_id:
            self.notify("Enter profile id first.", severity="warning")
            return

        if clean_profile_id == self._active_profile_id and not self._form_dirty:
            return

        if self._form_dirty and clean_profile_id != self._active_profile_id:
            self.app.push_screen(
                ConfirmProfileSwitchScreen(
                    current_profile_id=self._active_profile_id,
                    target_profile_id=clean_profile_id,
                ),
                callback=lambda decision: self._on_profile_switch_decision(
                    clean_profile_id,
                    str(decision or "cancel").lower(),
                ),
            )
            return

        await self._load_profile_into_form(profile_id=clean_profile_id)

    def _on_profile_switch_decision(self, target_profile_id: str, decision: str) -> None:
        if decision == "cancel":
            self._select_summary_profile(self._active_profile_id)
            return
        self.run_worker(
            self._complete_profile_switch(target_profile_id, decision),
            group="auth-manager-switch",
            exclusive=True,
        )

    async def _complete_profile_switch(self, target_profile_id: str, decision: str) -> None:
        if decision == "save":
            saved = await self._save_profile(notify_success=False)
            if not saved:
                self._select_summary_profile(self._active_profile_id)
                return
        elif decision != "discard":
            self._select_summary_profile(self._active_profile_id)
            return

        loaded = await self._load_profile_into_form(profile_id=target_profile_id)
        if not loaded:
            self._select_summary_profile(self._active_profile_id)
            return

        if decision == "save":
            self.notify(f"Saved changes and loaded profile: {target_profile_id}")

    async def _request_close(self) -> None:
        if not self._form_dirty:
            self.action_close()
            return
        current_profile_id = self._active_profile_id or self._profile_id() or "(new profile)"
        self.app.push_screen(
            ConfirmProfileSwitchScreen(
                current_profile_id=current_profile_id,
                target_profile_id="close tab",
            ),
            callback=lambda decision: self._on_close_decision(
                str(decision or "cancel").lower(),
            ),
        )

    def _on_close_decision(self, decision: str) -> None:
        self.run_worker(
            self._complete_close(decision),
            group="auth-manager-close",
            exclusive=True,
        )

    async def _complete_close(self, decision: str) -> None:
        if decision == "save":
            saved = await self._save_profile(notify_success=False)
            if not saved:
                return
        elif decision != "discard":
            return
        self.action_close()

    async def _load_profile_into_form(self, profile_id: str | None = None) -> bool:
        raw_profile_id = str(profile_id or self._profile_id()).strip()
        if not raw_profile_id:
            self.notify("Enter profile id first.", severity="warning")
            return False

        profile = self._profiles.get(raw_profile_id)
        if profile is None:
            await self._refresh_summary()
            profile = self._profiles.get(raw_profile_id)
        if profile is None:
            self.notify(f"Profile not found: {raw_profile_id}", severity="warning")
            return False

        resource_id = self._resource_id_for_profile(profile)
        profile_metadata = dict(profile.metadata)
        oauth_metadata = {
            key: str(profile_metadata.pop(key, "") or "").strip()
            for key in self._OAUTH_METADATA_KEYS
        }
        self._set_form_values(
            profile_id=profile.profile_id,
            provider=profile.provider,
            mode=profile.mode,
            set_default=self._is_workspace_default(
                provider=profile.provider,
                profile_id=profile.profile_id,
                resource_id=resource_id,
            ),
            label=profile.account_label,
            resource_id=resource_id,
            secret_ref=profile.secret_ref,
            token_ref=profile.token_ref,
            scopes=", ".join(profile.scopes),
            env=", ".join(f"{key}={value}" for key, value in sorted(profile.env.items())),
            command=profile.command,
            auth_check=", ".join(profile.auth_check),
            metadata=", ".join(
                f"{key}={value}" for key, value in sorted(profile_metadata.items())
            ),
            oauth_metadata=oauth_metadata,
        )
        self._mark_form_clean(active_profile_id=profile.profile_id)
        self._select_summary_profile(profile.profile_id)
        self.notify(f"Loaded profile: {profile.profile_id}")
        return True

    async def _save_profile(self, *, notify_success: bool = True) -> bool:
        profile_id = self._profile_id()
        if not profile_id:
            self.notify("Profile id is required.", severity="error")
            return False

        selected_resource_id = self._selected_resource_id()
        selected_resource = self._resources_by_id.get(selected_resource_id)
        existing = self._profiles.get(profile_id)
        provider = str(getattr(selected_resource, "provider", "")).strip()
        if not provider and existing is not None:
            provider = str(existing.provider or "").strip()
        mode = self._selected_mode()
        if not provider or not mode:
            self.notify(
                "Select a resource and mode (or load an existing profile with provider).",
                severity="error",
            )
            return False

        label = self.query_one("#auth-label", Input).value.strip()
        set_default = self._default_provider_selected()
        mcp_server = ""
        if selected_resource is not None and str(
            getattr(selected_resource, "resource_kind", "")
        ) == "mcp":
            mcp_server = str(getattr(selected_resource, "resource_key", "")).strip()
        secret_ref = self.query_one("#auth-secret-ref", Input).value.strip()
        token_ref = self.query_one("#auth-token-ref", Input).value.strip()
        scopes = self._split_csv_values(self.query_one("#auth-scopes", Input).value)
        env_values = self._split_csv_values(self.query_one("#auth-env", Input).value)
        command = self.query_one("#auth-command", Input).value.strip()
        auth_check = self._split_csv_values(self.query_one("#auth-auth-check", Input).value)
        meta_values = self._split_csv_values(self.query_one("#auth-meta", Input).value)

        try:
            env = self._parse_kv_pairs(env_values, env_keys=True)
            metadata = self._parse_kv_pairs(meta_values, env_keys=False)
            for key in self._OAUTH_METADATA_KEYS:
                metadata.pop(key, None)
            metadata.update(self._oauth_metadata_from_form())
            if mode in self._OAUTH_MODES:
                missing_oauth_fields = self._missing_required_oauth_metadata(metadata)
                if missing_oauth_fields:
                    self.notify(
                        "OAuth mode requires: "
                        + ", ".join(missing_oauth_fields)
                        + ".",
                        severity="error",
                    )
                    return False
            if mode in self._OAUTH_MODES and not token_ref:
                token_ref = self._default_oauth_token_ref(
                    provider=provider,
                    key_name=profile_id,
                )
                self.query_one("#auth-token-ref", Input).value = token_ref
            if mode in self._OAUTH_MODES and not token_ref.lower().startswith("keychain://"):
                self.notify(
                    "OAuth token_ref must use keychain://... storage.",
                    severity="error",
                )
                return False
            status = "ready"
            if existing is not None and str(existing.status or "").strip().lower() == "archived":
                status = "archived"
            profile = AuthProfile(
                profile_id=profile_id,
                provider=provider,
                mode=mode,
                account_label=label,
                mcp_server=mcp_server,
                secret_ref=secret_ref,
                token_ref=token_ref,
                scopes=scopes,
                env=env,
                command=command,
                auth_check=auth_check,
                metadata=metadata,
                status=status,
            )
            target = resolve_auth_write_path(
                explicit_path=self._explicit_auth_path,
            )
            await asyncio.to_thread(
                upsert_auth_profile,
                target,
                profile,
            )
            defaults_path = default_workspace_auth_defaults_path(self._workspace.resolve())
            resources_path = default_workspace_auth_resources_path(self._workspace.resolve())

            if selected_resource_id and selected_resource_id in self._resources_by_id:
                await asyncio.to_thread(
                    bind_resource_to_profile,
                    resources_path,
                    resource_id=selected_resource_id,
                    profile_id=profile_id,
                    generated_from=f"tui:{selected_resource_id}",
                    priority=0,
                )

            selectors_for_profile = [
                selector
                for selector, mapped_profile_id in self._workspace_defaults.items()
                if str(mapped_profile_id).strip() == profile_id
            ]
            for selector in selectors_for_profile:
                if selector == provider and set_default:
                    continue
                await asyncio.to_thread(
                    set_workspace_auth_default,
                    defaults_path,
                    selector=selector,
                    profile_id=None,
                )
                self._workspace_defaults.pop(selector, None)
            if set_default and provider:
                await asyncio.to_thread(
                    set_workspace_auth_default,
                    defaults_path,
                    selector=provider,
                    profile_id=profile_id,
                )
                self._workspace_defaults[provider] = profile_id

            resource_defaults_for_profile = [
                rid
                for rid, mapped_profile_id in self._workspace_resource_defaults.items()
                if str(mapped_profile_id).strip() == profile_id
            ]
            for rid in resource_defaults_for_profile:
                if rid == selected_resource_id and set_default:
                    continue
                await asyncio.to_thread(
                    set_workspace_resource_default,
                    resources_path,
                    resource_id=rid,
                    profile_id=None,
                )
                self._workspace_resource_defaults.pop(rid, None)

            if (
                set_default
                and selected_resource_id
                and selected_resource_id in self._resources_by_id
            ):
                await asyncio.to_thread(
                    set_workspace_resource_default,
                    resources_path,
                    resource_id=selected_resource_id,
                    profile_id=profile_id,
                )
                self._workspace_resource_defaults[selected_resource_id] = profile_id
        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error")
            return False

        self._changed = True
        self._active_profile_id = profile_id
        self._mark_form_clean(active_profile_id=profile_id)
        await self._refresh_summary()
        self._select_summary_profile(profile_id)
        if notify_success:
            self.notify(f"Saved profile: {profile_id}")
        return True

    def _next_duplicate_profile_id(self, profile_id: str) -> str:
        base = str(profile_id or "").strip() or "profile"
        candidate = f"{base}_copy"
        if candidate not in self._profiles:
            return candidate
        index = 2
        while True:
            candidate = f"{base}_copy{index}"
            if candidate not in self._profiles:
                return candidate
            index += 1

    async def _duplicate_profile(self) -> None:
        source_profile_id = self._profile_id() or self._active_profile_id
        if not source_profile_id:
            self.notify("Load a profile first.", severity="warning")
            return
        profile = self._profiles.get(source_profile_id)
        if profile is None:
            await self._refresh_summary()
            profile = self._profiles.get(source_profile_id)
        if profile is None:
            self.notify(f"Profile not found: {source_profile_id}", severity="error")
            return

        duplicate_id = self._next_duplicate_profile_id(source_profile_id)
        duplicate_label = (
            f"{profile.account_label} Copy"
            if str(profile.account_label or "").strip()
            else f"{source_profile_id} copy"
        )
        duplicate = replace(
            profile,
            profile_id=duplicate_id,
            account_label=duplicate_label,
            status="ready"
            if str(profile.status or "").strip().lower() == "archived"
            else profile.status,
        )

        try:
            target = resolve_auth_write_path(
                explicit_path=self._explicit_auth_path,
            )
            await asyncio.to_thread(
                upsert_auth_profile,
                target,
                duplicate,
                False,
            )
            selected_resource_id = self._selected_resource_id() or self._resource_id_for_profile(
                profile
            )
            if selected_resource_id and selected_resource_id in self._resources_by_id:
                resources_path = default_workspace_auth_resources_path(
                    self._workspace.resolve()
                )
                await asyncio.to_thread(
                    bind_resource_to_profile,
                    resources_path,
                    resource_id=selected_resource_id,
                    profile_id=duplicate_id,
                    generated_from=f"tui:duplicate:{selected_resource_id}",
                    priority=0,
                )
        except Exception as e:
            self.notify(f"Duplicate failed: {e}", severity="error")
            return

        self._changed = True
        await self._refresh_summary()
        await self._load_profile_into_form(profile_id=duplicate_id)
        self.notify(f"Duplicated profile: {source_profile_id} -> {duplicate_id}")

    async def _rebind_profile(self) -> None:
        profile_id = self._profile_id()
        if not profile_id:
            self.notify("Load a profile first.", severity="warning")
            return
        profile = self._profiles.get(profile_id)
        if profile is None:
            await self._refresh_summary()
            profile = self._profiles.get(profile_id)
        if profile is None:
            self.notify(f"Profile not found: {profile_id}", severity="error")
            return
        selected_resource_id = self._selected_resource_id()
        if not selected_resource_id or selected_resource_id not in self._resources_by_id:
            self.notify("Select a target resource first.", severity="warning")
            return

        selected_resource = self._resources_by_id[selected_resource_id]
        current_resource_id = self._resource_id_for_profile(profile)
        provider = str(getattr(selected_resource, "provider", "")).strip() or profile.provider
        if profile.provider != provider:
            self.notify(
                (
                    f"Resource provider {provider!r} does not match profile provider "
                    f"{profile.provider!r}."
                ),
                severity="error",
            )
            return

        mcp_server = ""
        if str(getattr(selected_resource, "resource_kind", "")) == "mcp":
            mcp_server = str(getattr(selected_resource, "resource_key", "")).strip()
        updated_profile = replace(profile, mcp_server=mcp_server)
        set_default = self._default_provider_selected()
        resources_path = default_workspace_auth_resources_path(self._workspace.resolve())
        defaults_path = default_workspace_auth_defaults_path(self._workspace.resolve())

        try:
            target = resolve_auth_write_path(
                explicit_path=self._explicit_auth_path,
            )
            await asyncio.to_thread(
                upsert_auth_profile,
                target,
                updated_profile,
                True,
            )
            await asyncio.to_thread(
                bind_resource_to_profile,
                resources_path,
                resource_id=selected_resource_id,
                profile_id=profile_id,
                generated_from=f"tui:rebind:{selected_resource_id}",
                priority=0,
            )

            selectors_for_profile = [
                selector
                for selector, mapped_profile_id in self._workspace_defaults.items()
                if str(mapped_profile_id).strip() == profile_id
            ]
            for selector in selectors_for_profile:
                if selector == provider and set_default:
                    continue
                await asyncio.to_thread(
                    set_workspace_auth_default,
                    defaults_path,
                    selector=selector,
                    profile_id=None,
                )
                self._workspace_defaults.pop(selector, None)
            if set_default and provider:
                await asyncio.to_thread(
                    set_workspace_auth_default,
                    defaults_path,
                    selector=provider,
                    profile_id=profile_id,
                )
                self._workspace_defaults[provider] = profile_id

            if set_default:
                await asyncio.to_thread(
                    set_workspace_resource_default,
                    resources_path,
                    resource_id=selected_resource_id,
                    profile_id=profile_id,
                )
                self._workspace_resource_defaults[selected_resource_id] = profile_id
            elif (
                current_resource_id
                and current_resource_id != selected_resource_id
                and self._workspace_resource_defaults.get(current_resource_id) == profile_id
            ):
                await asyncio.to_thread(
                    set_workspace_resource_default,
                    resources_path,
                    resource_id=current_resource_id,
                    profile_id=None,
                )
                self._workspace_resource_defaults.pop(current_resource_id, None)
        except Exception as e:
            self.notify(f"Rebind failed: {e}", severity="error")
            return

        self._changed = True
        await self._refresh_summary()
        await self._load_profile_into_form(profile_id=profile_id)
        self.notify(
            "Rebound profile "
            f"{profile_id} to "
            f"{getattr(selected_resource, 'display_name', selected_resource_id)}."
        )

    async def _archive_profile(self) -> None:
        profile_id = self._profile_id()
        if not profile_id:
            self.notify("Load a profile first.", severity="warning")
            return
        profile = self._profiles.get(profile_id)
        if profile is None:
            await self._refresh_summary()
            profile = self._profiles.get(profile_id)
        if profile is None:
            self.notify(f"Profile not found: {profile_id}", severity="error")
            return
        if str(profile.status or "").strip().lower() == "archived":
            self.notify(f"Profile already archived: {profile_id}")
            return

        try:
            target = resolve_auth_write_path(
                explicit_path=self._explicit_auth_path,
            )
            await asyncio.to_thread(
                upsert_auth_profile,
                target,
                replace(profile, status="archived"),
                True,
            )
            defaults_path = default_workspace_auth_defaults_path(self._workspace.resolve())
            resources_path = default_workspace_auth_resources_path(self._workspace.resolve())
            selectors_for_profile = [
                selector
                for selector, mapped_profile_id in self._workspace_defaults.items()
                if str(mapped_profile_id).strip() == profile_id
            ]
            for selector in selectors_for_profile:
                await asyncio.to_thread(
                    set_workspace_auth_default,
                    defaults_path,
                    selector=selector,
                    profile_id=None,
                )
                self._workspace_defaults.pop(selector, None)
            resource_defaults_for_profile = [
                resource_id
                for resource_id, mapped_profile_id in self._workspace_resource_defaults.items()
                if str(mapped_profile_id).strip() == profile_id
            ]
            for resource_id in resource_defaults_for_profile:
                await asyncio.to_thread(
                    set_workspace_resource_default,
                    resources_path,
                    resource_id=resource_id,
                    profile_id=None,
                )
                self._workspace_resource_defaults.pop(resource_id, None)
        except Exception as e:
            self.notify(f"Archive failed: {e}", severity="error")
            return

        self._changed = True
        await self._refresh_summary()
        if profile_id == self._active_profile_id:
            self._set_blank_form()
        self.notify(f"Archived profile: {profile_id}")

    async def _remove_profile(self) -> None:
        profile_id = self._profile_id()
        if not profile_id:
            self.notify("Profile id is required.", severity="error")
            return
        try:
            target = resolve_auth_write_path(
                explicit_path=self._explicit_auth_path,
            )
            await asyncio.to_thread(remove_auth_profile, target, profile_id)
            defaults_path = default_workspace_auth_defaults_path(self._workspace.resolve())
            resources_path = default_workspace_auth_resources_path(self._workspace.resolve())
            selectors_for_profile = [
                selector
                for selector, mapped_profile_id in self._workspace_defaults.items()
                if str(mapped_profile_id).strip() == profile_id
            ]
            for selector in selectors_for_profile:
                await asyncio.to_thread(
                    set_workspace_auth_default,
                    defaults_path,
                    selector=selector,
                    profile_id=None,
                )
                self._workspace_defaults.pop(selector, None)
            await asyncio.to_thread(
                remove_profile_from_resource_store,
                resources_path,
                profile_id=profile_id,
            )
            resource_defaults_for_profile = [
                resource_id
                for resource_id, mapped_profile_id in self._workspace_resource_defaults.items()
                if str(mapped_profile_id).strip() == profile_id
            ]
            for resource_id in resource_defaults_for_profile:
                self._workspace_resource_defaults.pop(resource_id, None)
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        self._changed = True
        await self._refresh_summary()
        if profile_id == self._active_profile_id:
            self._set_blank_form()
        self.notify(f"Removed profile: {profile_id}")

    async def _oauth_target_profile(self) -> AuthProfile | None:
        if self._form_dirty:
            saved = await self._save_profile(notify_success=False)
            if not saved:
                self.notify(
                    "OAuth action canceled because profile changes could not be saved.",
                    severity="warning",
                )
                return None

        profile_id = self._profile_id() or self._active_profile_id
        if not profile_id:
            self.notify("Load an OAuth profile first.", severity="warning")
            return None
        profile = self._profiles.get(profile_id)
        if profile is None:
            await self._refresh_summary()
            profile = self._profiles.get(profile_id)
        if profile is None:
            self.notify(f"Profile not found: {profile_id}", severity="error")
            return None
        if str(profile.mode or "").strip().lower() not in self._OAUTH_MODES:
            self.notify(
                f"Profile {profile.profile_id!r} is not an OAuth mode profile.",
                severity="warning",
            )
            return None
        return profile

    def _oauth_callback_prompt(self, prompt_text: str) -> str:
        entered = {"value": ""}
        done = threading.Event()

        def _on_done(raw: str | None) -> None:
            entered["value"] = str(raw or "").strip()
            done.set()

        def _show_prompt() -> None:
            self.app.push_screen(
                OAuthCodeEntryScreen(
                    title_text="Enter OAuth Callback",
                    prompt_text=str(
                        prompt_text or "Paste callback URL or authorization code."
                    ).strip()
                    or "Paste callback URL or authorization code.",
                ),
                callback=_on_done,
            )

        try:
            self.app.call_from_thread(_show_prompt)
        except Exception:
            return ""
        done.wait()
        return entered["value"]

    async def _oauth_login(self) -> None:
        profile = await self._oauth_target_profile()
        if profile is None:
            return
        start_payload: dict[str, str] = {}

        def _on_start(started) -> None:
            start_payload["authorization_url"] = str(
                getattr(started, "authorization_url", "") or ""
            ).strip()
            start_payload["callback_mode"] = str(
                getattr(started, "callback_mode", "") or ""
            ).strip()
            start_payload["browser_warning"] = str(
                getattr(started, "browser_error", "") or ""
            ).strip()

        try:
            result = await asyncio.to_thread(
                login_oauth_profile,
                profile,
                on_start=_on_start,
                callback_prompt=self._oauth_callback_prompt,
            )
        except OAuthProfileError as e:
            auth_url = start_payload.get("authorization_url", "")
            if auth_url:
                self.notify(f"OAuth login URL: {auth_url}")
            self.notify(
                f"OAuth login failed ({e.reason_code}): {e}",
                severity="error",
            )
            if e.reason_code == "callback_missing":
                self.notify(
                    "OAuth callback input was not provided; login canceled.",
                    severity="warning",
                )
            return
        auth_url = start_payload.get("authorization_url", "") or result.authorization_url
        if auth_url:
            self.notify(f"OAuth login URL: {auth_url}")
        browser_warning = (
            start_payload.get("browser_warning", "")
            or str(getattr(result, "browser_warning", "") or "").strip()
        )
        if browser_warning:
            self.notify(f"Browser open warning: {browser_warning}", severity="warning")
        if result.callback_mode == "manual":
            self.notify(
                "OAuth login fell back to manual callback mode; use CLI for manual callback.",
                severity="warning",
            )
        elif result.expires_at is not None:
            self.notify(f"OAuth login complete. expires_at={result.expires_at}")
        else:
            self.notify("OAuth login complete.")

    async def _oauth_status(self) -> None:
        profile = await self._oauth_target_profile()
        if profile is None:
            return
        try:
            state = await asyncio.to_thread(oauth_state_for_profile, profile)
        except Exception as e:
            self.notify(f"OAuth status failed: {e}", severity="error")
            return
        message = (
            f"OAuth state={state.state}, token={'yes' if state.has_token else 'no'}, "
            f"expired={'yes' if state.expired else 'no'}"
        )
        if state.expires_at is not None:
            message = f"{message}, expires_at={state.expires_at}"
        self.notify(message)
        if state.reason:
            self.notify(f"OAuth status detail: {state.reason}", severity="warning")

    async def _oauth_logout(self) -> None:
        profile = await self._oauth_target_profile()
        if profile is None:
            return
        try:
            token_ref = await asyncio.to_thread(logout_oauth_profile, profile)
        except OAuthProfileError as e:
            self.notify(
                f"OAuth logout failed ({e.reason_code}): {e}",
                severity="error",
            )
            return
        self.notify(f"Cleared OAuth token payload at {token_ref}")

    async def _oauth_refresh(self) -> None:
        profile = await self._oauth_target_profile()
        if profile is None:
            return
        try:
            result = await asyncio.to_thread(refresh_oauth_profile, profile)
        except OAuthProfileError as e:
            self.notify(
                f"OAuth refresh failed ({e.reason_code}): {e}",
                severity="error",
            )
            return
        if result.expires_at is not None:
            self.notify(
                "OAuth refresh complete: "
                f"expires_at={result.expires_at}"
            )
            return
        self.notify("OAuth refresh complete.")


class AuthManagerModalScreen(ModalScreen[dict[str, object] | None]):
    """Modal wrapper hosting the auth manager widget."""

    DEFAULT_CSS = """
    AuthManagerModalScreen {
        align: center middle;
    }
    """

    def __init__(
        self,
        *,
        workspace,
        explicit_auth_path=None,
        mcp_manager: MCPConfigManager | None = None,
        process_def: object | None = None,
        process_defs: list[object] | tuple[object, ...] | None = None,
        tool_registry=None,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._explicit_auth_path = explicit_auth_path
        self._mcp_manager = mcp_manager
        self._process_def = process_def
        self._process_defs = process_defs
        self._tool_registry = tool_registry

    def compose(self) -> ComposeResult:
        yield AuthManagerScreen(
            workspace=self._workspace,
            explicit_auth_path=self._explicit_auth_path,
            mcp_manager=self._mcp_manager,
            process_def=self._process_def,
            process_defs=self._process_defs,
            tool_registry=self._tool_registry,
            embedded=False,
            on_close=self._handle_close,
        )

    def _handle_close(self, result: dict[str, object]) -> None:
        self.dismiss(result)
