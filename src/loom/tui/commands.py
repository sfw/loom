"""Command palette provider for the Loom TUI."""

from __future__ import annotations

from functools import partial

from textual.command import DiscoveryHit, Hit, Hits, Provider

_COMMANDS = [
    ("Run setup wizard", "setup", "Open the setup flow"),
    ("Clear conversation", "clear_chat", "Clear the chat log"),
    ("Toggle sidebar", "toggle_sidebar", "Show or hide the sidebar"),
    (
        "Reload workspace tree",
        "reload_workspace",
        "Reload workspace files in the sidebar",
    ),
    ("Switch to Chat tab", "tab_chat", "Focus the Chat tab"),
    ("Switch to Files tab", "tab_files", "Focus the Files Changed tab"),
    ("Switch to Events tab", "tab_events", "Focus the Events tab"),
    ("List tools", "list_tools", "Show all available tools"),
    ("Show model info", "model_info", "Display model details"),
    ("Show session info", "session_info", "Display current session details"),
    ("Start new session", "new_session", "Create a fresh session"),
    ("List recent sessions", "sessions_list", "Show recent sessions"),
    ("Resume session…", "resume_prompt", "Prefill /resume for session switch"),
    ("List MCP servers", "mcp_list", "Show configured MCP servers"),
    ("Manage MCP servers…", "mcp_manage", "Open MCP manager screen"),
    ("Add MCP server…", "mcp_add_prompt", "Prefill /mcp add command template"),
    ("Manage auth profiles…", "auth_manage", "Open auth manager screen"),
    ("Show learned patterns", "learned_patterns", "Open learned pattern manager"),
    ("Show process modes", "process_info", "Show ad hoc vs explicit process modes"),
    ("List processes", "process_list", "Show available process definitions"),
    ("Run ad hoc goal…", "run_prompt", "Prefill /run for ad hoc process execution"),
    ("Close tab", "close_process_tab", "Close current tab"),
    ("Show token usage", "token_info", "Display session token count"),
    ("Show help", "help", "List commands and keyboard shortcuts"),
    ("Quit", "quit", "Save session and exit"),
]


class LoomCommands(Provider):
    """Custom commands for the Ctrl+P command palette."""

    def _palette_entries(self) -> list[tuple[str, str, str]]:
        """Return built-in + dynamic command palette entries."""
        entries: list[tuple[str, str, str]] = list(_COMMANDS)
        dynamic_getter = getattr(
            self.app,
            "iter_dynamic_process_palette_entries",
            None,
        )
        if callable(dynamic_getter):
            entries.extend(dynamic_getter())
        return entries

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
        dynamic_getter = getattr(
            self.app,
            "iter_dynamic_process_palette_entries",
            None,
        )
        if callable(dynamic_getter):
            for label, action, help_text in dynamic_getter():
                score = matcher.match(label)
                if score <= 0:
                    continue
                yield Hit(
                    score,
                    matcher.highlight(label),
                    command=partial(
                        self.app.run_action,
                        f"loom_command('{action}')",
                    ),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        """Return full sorted palette list when opened with empty query."""
        for label, action, help_text in sorted(
            self._palette_entries(),
            key=lambda item: item[0].casefold(),
        ):
            yield DiscoveryHit(
                display=label,
                text=label,
                command=partial(
                    self.app.run_action,
                    f"loom_command('{action}')",
                ),
                help=help_text,
            )
