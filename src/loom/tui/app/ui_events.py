"""UI event handler helpers for interactive controls."""

from __future__ import annotations

from textual import events
from textual.widgets import Button, Input


async def on_landing_close_pressed(self, event: events.Click) -> None:
    """Exit startup landing and open the main workspace chat surface."""
    event.stop()
    event.prevent_default()
    await self._enter_workspace_surface(ensure_session=True)


def on_user_input_changed(self, _event: Input.Changed) -> None:
    """Show slash-command hints as the user types."""
    if self._skip_input_history_reset_once:
        self._skip_input_history_reset_once = False
    elif not self._applying_input_history_navigation:
        self._reset_input_history_navigation()
    if self._skip_slash_cycle_reset_once:
        self._skip_slash_cycle_reset_once = False
    elif not self._applying_slash_tab_completion:
        self._reset_slash_tab_cycle()
    # Use the widget's current value rather than event payload to avoid
    # stale-value edge cases that can appear one keypress behind.
    current = self.query_one("#user-input", Input).value
    self._sync_chat_stop_control()
    self._set_slash_hint(self._render_slash_hint(current))


def on_landing_input_changed(self, _event: Input.Changed) -> None:
    """Show slash-command hints while typing on the startup landing input."""
    current = self.query_one("#landing-input", Input).value
    self._set_slash_hint(self._render_slash_hint(current))


def on_process_run_restart_pressed(self, event: Button.Pressed) -> None:
    """Restart a failed process run directly from its tab button."""
    button_id = str(getattr(event.button, "id", "") or "").strip()
    if not button_id.startswith("process-run-restart-"):
        return
    run_id = button_id.removeprefix("process-run-restart-").strip()
    if not run_id:
        return
    event.stop()
    event.prevent_default()
    self.run_worker(
        self._restart_process_run_in_place(run_id),
        name=f"process-run-restart-{run_id}",
        group=f"process-run-restart-{run_id}",
        exclusive=False,
    )


def on_process_run_control_pressed(self, event: Button.Pressed) -> None:
    """Dispatch pause/play toggle + stop controls from process-run panes."""
    button_id = str(getattr(event.button, "id", "") or "").strip()
    control = ""
    run_id = ""
    for prefix, control_name in (
        ("process-run-toggle-", "toggle"),
        ("process-run-stop-", "stop"),
    ):
        if button_id.startswith(prefix):
            control = control_name
            run_id = button_id.removeprefix(prefix).strip()
            break
    if not control or not run_id:
        return
    run = self._process_runs.get(run_id)
    if run is None or run.closed:
        return
    event.stop()
    event.prevent_default()
    if control == "toggle":
        status = str(getattr(run, "status", "")).strip().lower()
        if status == "paused":
            self.run_worker(
                self._play_process_run(run),
                name=f"process-run-play-{run_id}",
                group=f"process-run-control-{run_id}",
                exclusive=False,
            )
            return
        if status != "running":
            return
        self.run_worker(
            self._pause_process_run(run),
            name=f"process-run-pause-{run_id}",
            group=f"process-run-control-{run_id}",
            exclusive=False,
        )
        return
    if control == "stop":
        self.run_worker(
            self._stop_process_run(run, confirm=True),
            name=f"process-run-stop-{run_id}",
            group=f"process-run-control-{run_id}",
            exclusive=False,
        )
        return


def on_chat_stop_pressed(self, event: Button.Pressed) -> None:
    """Stop the active cowork chat turn from input-row button."""
    event.stop()
    event.prevent_default()
    self.action_stop_chat()


def on_chat_inject_pressed(self, event: Button.Pressed) -> None:
    """Queue inject steering from input-row button."""
    event.stop()
    event.prevent_default()
    self.action_inject_chat()


def on_chat_redirect_pressed(self, event: Button.Pressed) -> None:
    """Apply redirect steering from input-row button."""
    event.stop()
    event.prevent_default()
    self.action_redirect_chat()


def on_dynamic_steer_queue_button_pressed(self, event: Button.Pressed) -> None:
    """Dispatch per-item steering queue actions from dynamic row buttons."""
    button_id = str(getattr(event.button, "id", "") or "").strip()
    if button_id.startswith("steer-queue-edit-"):
        directive_id = button_id.removeprefix("steer-queue-edit-").strip()
        if not directive_id:
            return
        event.stop()
        event.prevent_default()
        self.action_steer_queue_edit(directive_id)
        return
    if button_id.startswith("steer-queue-dismiss-"):
        directive_id = button_id.removeprefix("steer-queue-dismiss-").strip()
        if not directive_id:
            return
        event.stop()
        event.prevent_default()
        self.action_steer_queue_dismiss(directive_id)
        return
    if button_id.startswith("steer-queue-redirect-"):
        directive_id = button_id.removeprefix("steer-queue-redirect-").strip()
        if not directive_id:
            return
        event.stop()
        event.prevent_default()
        self.action_steer_queue_redirect(directive_id)


def on_footer_manager_shortcut_pressed(self, event: Button.Pressed) -> None:
    button_id = str(getattr(event.button, "id", "") or "").strip()
    event.stop()
    event.prevent_default()
    if button_id == "footer-auth-shortcut":
        self.action_open_auth_tab()
        return
    if button_id == "footer-mcp-shortcut":
        self.action_open_mcp_tab()
