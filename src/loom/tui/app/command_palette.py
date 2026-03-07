"""Command palette prefill and info rendering actions."""

from __future__ import annotations

from textual.widgets import Input

from loom.tui.widgets import ChatLog


def prefill_user_input(self, text: str) -> None:
    """Seed the chat input with a command template and focus it."""
    self._reset_input_history_navigation()
    input_widget = self.query_one("#user-input", Input)
    self._set_user_input_text(text)
    input_widget.focus()
    self._set_slash_hint(self._render_slash_hint(text))


def show_tools(self) -> None:
    chat = self.query_one("#chat-log", ChatLog)
    chat.add_info(self._render_tools_catalog())


def show_model_info(self) -> None:
    chat = self.query_one("#chat-log", ChatLog)
    chat.add_info(self._render_active_model_info())


def show_models_info(self) -> None:
    chat = self.query_one("#chat-log", ChatLog)
    chat.add_info(self._render_models_catalog())


def show_process_info(self) -> None:
    chat = self.query_one("#chat-log", ChatLog)
    chat.add_info(self._render_process_usage())


def show_process_list(self) -> None:
    chat = self.query_one("#chat-log", ChatLog)
    chat.add_info(self._render_process_catalog())


def iter_dynamic_process_palette_entries(self) -> list[tuple[str, str, str]]:
    """Return palette entries for dynamically discovered process commands."""
    self._refresh_process_command_index()
    entries: list[tuple[str, str, str]] = []
    for token, process_name in sorted(self._process_command_map.items()):
        entries.append((
            f"Run {process_name}…",
            f"process_run_prompt:{process_name}",
            f"Prefill {token} for direct process execution",
        ))
    return entries


def show_token_info(self) -> None:
    chat = self.query_one("#chat-log", ChatLog)
    chat.add_info(f"Session tokens: {self._total_tokens:,}")


def show_help(self) -> None:
    chat = self.query_one("#chat-log", ChatLog)
    chat.add_info("\n".join(self._help_lines()))


async def palette_quit(self) -> None:
    await self._request_exit()
