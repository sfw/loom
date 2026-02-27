"""Auth profile management modal."""

from __future__ import annotations

import asyncio
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


class AuthManagerScreen(ModalScreen[dict[str, object] | None]):
    """Modal form for auth profile add/edit/remove."""

    _NO_TARGET_VALUE = "__none__"
    _NO_MODE_VALUE = "__mode_unset__"
    _SUPPORTED_AUTH_MODES = (
        "api_key",
        "oauth2_pkce",
        "oauth2_device",
        "cli_passthrough",
        "env_passthrough",
    )

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("ctrl+r", "refresh", "Refresh"),
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
        "auth-meta",
    )

    CSS = """
    AuthManagerScreen {
        align: center middle;
    }
    #auth-manager-dialog {
        width: 100;
        height: 90%;
        max-height: 46;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
        overflow: hidden;
    }
    #auth-manager-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #auth-manager-form {
        height: 1fr;
        overflow-y: auto;
        margin-bottom: 1;
    }
    #auth-manager-context {
        color: $text-muted;
        margin-bottom: 1;
    }
    #auth-manager-summary {
        height: 12;
        max-height: 14;
        margin-bottom: 1;
    }
    #auth-manager-advanced {
        margin-top: 1;
    }
    .auth-label {
        margin-top: 1;
    }
    .auth-help {
        color: $text-muted;
        margin-top: 0;
        margin-bottom: 1;
    }
    .auth-input {
        margin-top: 0;
    }
    .auth-select {
        margin-top: 0;
    }
    .auth-checkbox {
        margin-top: 1;
    }
    .auth-actions {
        height: auto;
        margin-top: 1;
    }
    .auth-actions Button {
        margin-right: 1;
    }
    #auth-manager-footer {
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        workspace,
        explicit_auth_path=None,
        mcp_manager: MCPConfigManager | None = None,
        process_def: object | None = None,
        tool_registry=None,
    ) -> None:
        super().__init__()
        self._workspace = workspace
        self._explicit_auth_path = explicit_auth_path
        self._process_def = process_def
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
        self._baseline_form_state: dict[str, str] = {}
        self._form_dirty = False
        self._suppress_dirty_tracking = False
        self._changed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="auth-manager-dialog"):
            yield Label(
                "[bold #7dcfff]Auth Profile Manager[/bold #7dcfff]",
                id="auth-manager-title",
            )
            with VerticalScroll(id="auth-manager-form"):
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
                )

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
                    "When enabled, /run and /process use this profile by default "
                    "for the provider in this workspace.",
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
                yield Label("Secret Ref", classes="auth-label")
                yield Input(
                    id="auth-secret-ref",
                    classes="auth-input",
                )
                yield Label(
                    "Where the base secret comes from (env/keychain/vault reference).",
                    classes="auth-help",
                )
                yield Label("Token Ref", classes="auth-label")
                yield Input(
                    id="auth-token-ref",
                    classes="auth-input",
                )
                yield Label(
                    "Optional token storage reference for refresh/access tokens.",
                    classes="auth-help",
                )
                with Collapsible(
                    title="Advanced",
                    id="auth-manager-advanced",
                    collapsed=True,
                ):
                    yield Label("Scopes (comma-separated)", classes="auth-label")
                    yield Input(
                        id="auth-scopes",
                        classes="auth-input",
                    )
                    yield Label(
                        "Optional OAuth scopes for this profile.",
                        classes="auth-help",
                    )
                    yield Label("Env pairs (comma-separated KEY=VALUE)", classes="auth-label")
                    yield Input(
                        id="auth-env",
                        classes="auth-input",
                    )
                    yield Label(
                        "Extra env vars injected when this profile is applied.",
                        classes="auth-help",
                    )
                    yield Label("Command (cli_passthrough)", classes="auth-label")
                    yield Input(
                        id="auth-command",
                        classes="auth-input",
                    )
                    yield Label(
                        "Command name used for cli_passthrough auth mode.",
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

            with Horizontal(classes="auth-actions"):
                yield Button("Refresh", id="auth-btn-refresh")
                yield Button("Load Profile", id="auth-btn-load")
                yield Button("Save/Add", id="auth-btn-save", variant="primary")
                yield Button("Duplicate", id="auth-btn-duplicate")
                yield Button("Rebind", id="auth-btn-rebind")
                yield Button("Archive", id="auth-btn-archive")
                yield Button("Remove", id="auth-btn-remove", variant="error")
                yield Button("Close", id="auth-btn-close")

            yield Label(
                "[dim]Profile list and missing draft profiles load automatically on open. "
                "Select a row to edit it. Save/Add upserts a profile and keeps resource "
                "bindings/defaults in sync. Duplicate/Rebind/Archive are explicit lifecycle "
                "actions. Open Advanced for optional "
                "scopes/env/command/check/metadata fields.[/dim]",
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
        await self._sync_missing_drafts()
        await self._refresh_summary()
        self.query_one("#auth-profile-id", Input).focus()

    def action_close(self) -> None:
        self.dismiss({"changed": self._changed})

    async def action_refresh(self) -> None:
        await self._sync_missing_drafts()
        await self._refresh_summary()

    @on(Button.Pressed)
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "auth-btn-close":
            self.action_close()
            return
        if button_id == "auth-btn-refresh":
            await self._refresh_summary()
            return
        if button_id == "auth-btn-load":
            await self._request_profile_switch(self._profile_id())
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

    @on(Input.Changed)
    def _on_form_input_changed(self, event: Input.Changed) -> None:
        if event.input.id not in self._FORM_FIELD_IDS:
            return
        self._update_form_dirty()

    @on(Select.Changed)
    def _on_form_select_changed(self, event: Select.Changed) -> None:
        if event.select.id not in self._FORM_FIELD_IDS:
            return
        if event.select.id == "auth-resource-target":
            self._sync_provider_display()
            self._refresh_mode_select()
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
        options = self._mode_options(include_mode=include_mode or current_mode)
        select.set_options(options)

    def _set_mode_select_value(self, mode: str) -> None:
        clean = str(mode or "").strip().lower()
        self._refresh_mode_select(include_mode=clean)
        select = self.query_one("#auth-mode", Select)
        select.value = self._encode_mode_value(clean)

    def _selected_mode(self) -> str:
        select = self.query_one("#auth-mode", Select)
        return self._decode_mode_value(select.value)

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

    def _resource_id_for_profile(self, profile: AuthProfile) -> str:
        profile_id = str(getattr(profile, "profile_id", "")).strip()
        if profile_id:
            bound = self._resource_binding_by_profile.get(profile_id)
            if bound:
                return bound
        mcp_alias = str(getattr(profile, "mcp_server", "")).strip()
        if mcp_alias:
            for resource_id, resource in self._resources_by_id.items():
                if str(getattr(resource, "resource_kind", "")) != "mcp":
                    continue
                if str(getattr(resource, "resource_key", "")) == mcp_alias:
                    return resource_id
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
    ) -> None:
        self._suppress_dirty_tracking = True
        try:
            self.query_one("#auth-profile-id", Input).value = profile_id
            self._active_provider = str(provider or "").strip()
            self._set_mode_select_value(mode)
            self.query_one("#auth-default-provider", Checkbox).value = bool(set_default)
            self.query_one("#auth-label", Input).value = label
            self._set_mcp_server_select_value(resource_id)
            self.query_one("#auth-secret-ref", Input).value = secret_ref
            self.query_one("#auth-token-ref", Input).value = token_ref
            self.query_one("#auth-scopes", Input).value = scopes
            self.query_one("#auth-env", Input).value = env
            self.query_one("#auth-command", Input).value = command
            self.query_one("#auth-auth-check", Input).value = auth_check
            self.query_one("#auth-meta", Input).value = metadata
            self._sync_provider_display()
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

        self._profiles = dict(merged.config.profiles)
        self._workspace_defaults = dict(merged.workspace_defaults)
        context_lines = [
            f"user: {merged.user_path}",
            f"explicit: {merged.explicit_path or '-'}",
            f"workspace defaults: {merged.workspace_defaults_path or '-'}",
            f"resource registry: {resources_path}",
        ]
        self.query_one("#auth-manager-context", Static).update("\n".join(context_lines))
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
                f"{key}={value}" for key, value in sorted(profile.metadata.items())
            ),
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
