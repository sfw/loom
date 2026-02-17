"""Command palette provider for the Loom TUI."""

from __future__ import annotations

from functools import partial

from textual.command import Hit, Hits, Provider

_COMMANDS = [
    ("Clear conversation", "clear_chat", "Clear the chat log"),
    ("Toggle sidebar", "toggle_sidebar", "Show or hide the sidebar"),
    ("Switch to Chat tab", "tab_chat", "Focus the Chat tab"),
    ("Switch to Files tab", "tab_files", "Focus the Files Changed tab"),
    ("Switch to Events tab", "tab_events", "Focus the Events tab"),
    ("List tools", "list_tools", "Show all available tools"),
    ("Show model info", "model_info", "Display model details"),
    ("Show process info", "process_info", "Display active process"),
    ("List processes", "process_list", "Show available process definitions"),
    ("Disable process", "process_off", "Disable active process"),
    ("Show token usage", "token_info", "Display session token count"),
    ("Show help", "help", "List commands and keyboard shortcuts"),
    ("Quit", "quit", "Save session and exit"),
]


class LoomCommands(Provider):
    """Custom commands for the Ctrl+P command palette."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for label, action, help_text in _COMMANDS:
            score = matcher.match(label)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(label),
                    command=partial(
                        self.app.run_action, f"loom_command('{action}')",
                    ),
                    help=help_text,
                )
