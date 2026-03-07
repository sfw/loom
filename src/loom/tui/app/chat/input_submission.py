"""Shared chat and landing input submit flow."""

from __future__ import annotations

from textual.widgets import Input


async def submit_user_text(self, text: str, *, source: str) -> None:
    """Shared submit path for chat and landing composer inputs."""
    clean = str(text or "").strip()
    if not clean:
        return
    original_clean = clean

    if source == "chat":
        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
    elif source == "landing":
        try:
            landing_input = self.query_one("#landing-input", Input)
            landing_input.value = ""
        except Exception:
            pass
        token = clean.split(None, 1)[0].lower()
        await self._enter_workspace_surface(ensure_session=(token != "/new"))
        if not original_clean.startswith("/"):
            clean = f"/run {original_clean}"

    self._reset_slash_tab_cycle()
    self._reset_input_history_navigation()
    self._set_slash_hint("")

    if clean.startswith("/"):
        handled = await self._handle_slash_command(clean)
        if handled:
            self._append_input_history(clean)
            await self._persist_process_run_ui_state()
            return

    focused_run = self._current_process_run()
    if (
        focused_run is not None
        and self._is_process_run_active_status(focused_run.status)
    ):
        injected = await self._inject_process_run(
            focused_run,
            clean,
            source="enter",
            queue_if_unavailable=True,
        )
        if injected:
            self._append_input_history(clean)
            await self._persist_process_run_ui_state()
            return
        if (
            focused_run is not None
            and self._is_process_run_active_status(focused_run.status)
        ):
            self._set_user_input_text(clean)
            return

    if self._is_cowork_stop_visible():
        queued = await self._queue_chat_inject_instruction(clean, source="enter")
        if queued:
            self._append_input_history(clean)
            await self._persist_process_run_ui_state()
            return
        if self._is_cowork_stop_visible():
            self._set_user_input_text(clean)
            return

    self._append_input_history(clean)
    self._chat_turn_worker = self._run_turn(clean)
