"""Input history management for chat command entry."""

from __future__ import annotations

from textual.widgets import Input

from ..constants import _MAX_INPUT_HISTORY


def reset_input_history_navigation(self) -> None:
    """Clear active input-history navigation state."""
    self._input_history_nav_index = None
    self._input_history_nav_draft = ""


def clear_input_history(self) -> None:
    """Drop in-memory input history and reset navigation."""
    self._input_history = []
    self._reset_input_history_navigation()


def append_input_history(self, value: str) -> None:
    """Record one executed user input in bounded history."""
    entry = str(value or "").strip()
    if not entry:
        return
    self._input_history.append(entry)
    if len(self._input_history) > _MAX_INPUT_HISTORY:
        del self._input_history[:-_MAX_INPUT_HISTORY]
    self._sync_input_history_into_session_state()
    self._reset_input_history_navigation()


def hydrate_input_history_from_session(self) -> None:
    """Populate input history from the active session's user messages."""
    self._clear_input_history()
    if self._session is None:
        return
    state = getattr(self._session, "session_state", None)
    ui_state = getattr(state, "ui_state", None) if state is not None else None
    if isinstance(ui_state, dict):
        payload = ui_state.get("input_history")
        items: list[object] | None = None
        if isinstance(payload, dict):
            raw_items = payload.get("items")
            if isinstance(raw_items, list):
                items = raw_items
        elif isinstance(payload, list):
            items = payload
        if items is not None:
            self._input_history = [
                str(item).strip() for item in items
                if str(item).strip()
            ][-_MAX_INPUT_HISTORY:]
            return
    restored: list[str] = []
    messages = getattr(self._session, "messages", [])
    if not isinstance(messages, list):
        messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        if role != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        text = content.strip()
        if text:
            restored.append(text)
    if len(restored) > _MAX_INPUT_HISTORY:
        restored = restored[-_MAX_INPUT_HISTORY:]
    self._input_history = restored


def set_user_input_text(
    self,
    value: str,
    *,
    from_history_navigation: bool = False,
) -> None:
    """Update the main input box value and keep the cursor at the end."""
    input_widget = self.query_one("#user-input", Input)
    if from_history_navigation:
        self._applying_input_history_navigation = True
        self._skip_input_history_reset_once = True
    try:
        input_widget.value = value
        input_widget.cursor_position = len(value)
    finally:
        if from_history_navigation:
            self._applying_input_history_navigation = False
    self._sync_chat_stop_control()


def apply_input_history_navigation(self, *, older: bool) -> bool:
    """Move through recorded input history like a shell."""
    if not self._input_history:
        return False
    input_widget = self.query_one("#user-input", Input)
    if self._input_history_nav_index is None:
        self._input_history_nav_index = len(self._input_history)
        self._input_history_nav_draft = input_widget.value
    index = self._input_history_nav_index
    if index is None:
        return False

    if older:
        next_index = max(0, index - 1)
        self._input_history_nav_index = next_index
        text = self._input_history[next_index]
    else:
        if index >= len(self._input_history):
            return False
        next_index = index + 1
        if next_index >= len(self._input_history):
            self._input_history_nav_index = len(self._input_history)
            text = self._input_history_nav_draft
        else:
            self._input_history_nav_index = next_index
            text = self._input_history[next_index]

    self._set_user_input_text(text, from_history_navigation=True)
    return True
