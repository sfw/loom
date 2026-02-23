"""Auth profile management modal."""

from __future__ import annotations

import asyncio

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Static

from loom.auth.config import (
    AuthProfile,
    load_merged_auth_config,
    remove_auth_profile,
    resolve_auth_write_path,
    upsert_auth_profile,
)
from loom.mcp.config import ensure_valid_env_key


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

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("ctrl+r", "refresh", "Refresh"),
    ]

    _FORM_FIELD_IDS = (
        "auth-profile-id",
        "auth-provider",
        "auth-mode",
        "auth-label",
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

    def __init__(self, *, workspace, explicit_auth_path=None) -> None:
        super().__init__()
        self._workspace = workspace
        self._explicit_auth_path = explicit_auth_path
        self._profiles: dict[str, AuthProfile] = {}
        self._profile_ids: list[str] = []
        self._active_profile_id = ""
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
                summary_table.add_columns("Profile ID", "Provider", "Mode", "Account Label")
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
                yield Label("Provider", classes="auth-label")
                yield Input(
                    id="auth-provider",
                    classes="auth-input",
                )
                yield Label(
                    "Provider namespace used by auth resolution (required; "
                    "this is not the MCP alias).",
                    classes="auth-help",
                )
                yield Label("Mode", classes="auth-label")
                yield Input(
                    id="auth-mode",
                    classes="auth-input",
                )
                yield Label(
                    "Auth mode, e.g. oauth2_pkce, api_key, cli_passthrough.",
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
                yield Button("Remove", id="auth-btn-remove", variant="error")
                yield Button("Close", id="auth-btn-close")

            yield Label(
                "[dim]Profile list loads automatically on open. "
                "Select a row to edit it. Save/Add upserts a profile. "
                "Use this for provider auth profiles (non-MCP OAuth bridge state).[/dim]",
                id="auth-manager-footer",
            )

    async def on_mount(self) -> None:
        self._set_form_values(
            profile_id="",
            provider="",
            mode="",
            label="",
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

    def action_close(self) -> None:
        self.dismiss({"changed": self._changed})

    async def action_refresh(self) -> None:
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
        if button_id == "auth-btn-remove":
            await self._remove_profile()
            return

    @on(Input.Changed)
    def _on_form_input_changed(self, event: Input.Changed) -> None:
        if event.input.id not in self._FORM_FIELD_IDS:
            return
        self._update_form_dirty()

    @on(DataTable.RowSelected, "#auth-manager-summary")
    async def _on_summary_row_selected(self, event: DataTable.RowSelected) -> None:
        profile_id = str(getattr(event.row_key, "value", "") or "").strip()
        if not profile_id and 0 <= event.cursor_row < len(self._profile_ids):
            profile_id = self._profile_ids[event.cursor_row]
        await self._request_profile_switch(profile_id)

    def _capture_form_state(self) -> dict[str, str]:
        return {
            field_id: self.query_one(f"#{field_id}", Input).value
            for field_id in self._FORM_FIELD_IDS
        }

    def _mark_form_clean(self, *, active_profile_id: str | None = None) -> None:
        self._baseline_form_state = self._capture_form_state()
        self._form_dirty = False
        if active_profile_id is not None:
            self._active_profile_id = active_profile_id

    def _update_form_dirty(self) -> None:
        if self._suppress_dirty_tracking:
            return
        self._form_dirty = self._capture_form_state() != self._baseline_form_state

    def _set_form_values(
        self,
        *,
        profile_id: str,
        provider: str,
        mode: str,
        label: str,
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
            self.query_one("#auth-provider", Input).value = provider
            self.query_one("#auth-mode", Input).value = mode
            self.query_one("#auth-label", Input).value = label
            self.query_one("#auth-secret-ref", Input).value = secret_ref
            self.query_one("#auth-token-ref", Input).value = token_ref
            self.query_one("#auth-scopes", Input).value = scopes
            self.query_one("#auth-env", Input).value = env
            self.query_one("#auth-command", Input).value = command
            self.query_one("#auth-auth-check", Input).value = auth_check
            self.query_one("#auth-meta", Input).value = metadata
        finally:
            self._suppress_dirty_tracking = False

    def _set_blank_form(self) -> None:
        self._set_form_values(
            profile_id="",
            provider="",
            mode="",
            label="",
            secret_ref="",
            token_ref="",
            scopes="",
            env="",
            command="",
            auth_check="",
            metadata="",
        )
        self._mark_form_clean(active_profile_id="")

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

        self._profiles = dict(merged.config.profiles)
        context_lines = [
            f"user: {merged.user_path}",
            f"explicit: {merged.explicit_path or '-'}",
            f"workspace defaults: {merged.workspace_defaults_path or '-'}",
        ]
        self.query_one("#auth-manager-context", Static).update("\n".join(context_lines))
        self._render_summary()

    def _render_summary(self) -> None:
        if not self.is_mounted:
            return

        table = self.query_one("#auth-manager-summary", DataTable)
        table.clear()
        self._profile_ids = []

        for profile_id in sorted(self._profiles):
            profile = self._profiles[profile_id]
            label = profile.account_label or "-"
            table.add_row(
                profile.profile_id,
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

        self._set_form_values(
            profile_id=profile.profile_id,
            provider=profile.provider,
            mode=profile.mode,
            label=profile.account_label,
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

        provider = self.query_one("#auth-provider", Input).value.strip()
        mode = self.query_one("#auth-mode", Input).value.strip()
        if not provider or not mode:
            self.notify("Provider and mode are required.", severity="error")
            return False

        label = self.query_one("#auth-label", Input).value.strip()
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
            profile = AuthProfile(
                profile_id=profile_id,
                provider=provider,
                mode=mode,
                account_label=label,
                secret_ref=secret_ref,
                token_ref=token_ref,
                scopes=scopes,
                env=env,
                command=command,
                auth_check=auth_check,
                metadata=metadata,
            )
            target = resolve_auth_write_path(
                explicit_path=self._explicit_auth_path,
            )
            await asyncio.to_thread(
                upsert_auth_profile,
                target,
                profile,
            )
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
        except Exception as e:
            self.notify(str(e), severity="error")
            return
        self._changed = True
        await self._refresh_summary()
        if profile_id == self._active_profile_id:
            self._set_blank_form()
        self.notify(f"Removed profile: {profile_id}")
