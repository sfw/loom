"""Command palette provider for the Loom TUI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from textual.command import DiscoveryHit, Hit, Hits, Provider


@dataclass(frozen=True)
class PaletteCommand:
    """One command palette row."""

    section: str
    label: str
    action: str
    help_text: str
    shortcut: str = ""


_COMMANDS: tuple[PaletteCommand, ...] = (
    PaletteCommand(
        section="Suggested",
        label="Run ad hoc goal…",
        action="run_prompt",
        help_text="Prefill /run for ad hoc process execution",
    ),
    PaletteCommand(
        section="Session",
        label="Start new session",
        action="new_session",
        help_text="Create a fresh cowork session",
    ),
    PaletteCommand(
        section="Session",
        label="List recent sessions",
        action="sessions_list",
        help_text="Show recent cowork sessions",
    ),
    PaletteCommand(
        section="Session",
        label="Resume session…",
        action="resume_prompt",
        help_text="Prefill /resume to switch to an older cowork session",
    ),
    PaletteCommand(
        section="Session",
        label="Show session info",
        action="session_info",
        help_text="Display current session details",
    ),
    PaletteCommand(
        section="Prompt",
        label="Inject steering…",
        action="inject_prompt",
        help_text="Prefill /inject for queued steering",
    ),
    PaletteCommand(
        section="Prompt",
        label="Redirect cowork…",
        action="redirect_prompt",
        help_text="Prefill /redirect for immediate steering",
    ),
    PaletteCommand(
        section="Cowork",
        label="Pause cowork chat",
        action="pause_chat",
        help_text="Request pause at next safe boundary",
    ),
    PaletteCommand(
        section="Cowork",
        label="Resume cowork chat",
        action="resume_chat",
        help_text="Resume cowork chat after pause",
    ),
    PaletteCommand(
        section="Cowork",
        label="Stop active chat turn",
        action="stop_chat",
        help_text="Request stop for active cowork chat turn",
    ),
    PaletteCommand(
        section="Cowork",
        label="Show steering queue",
        action="steer_queue",
        help_text="Show queued cowork steering state",
    ),
    PaletteCommand(
        section="Cowork",
        label="Clear steering queue",
        action="steer_clear",
        help_text="Clear queued inject and pause state",
    ),
    PaletteCommand(
        section="Workspace",
        label="Clear conversation",
        action="clear_chat",
        help_text="Clear the chat log",
        shortcut="ctrl + l",
    ),
    PaletteCommand(
        section="Workspace",
        label="Toggle sidebar",
        action="toggle_sidebar",
        help_text="Show or hide the sidebar",
        shortcut="ctrl + b",
    ),
    PaletteCommand(
        section="Workspace",
        label="Reload workspace tree",
        action="reload_workspace",
        help_text="Reload workspace files in the sidebar",
        shortcut="ctrl + r",
    ),
    PaletteCommand(
        section="Workspace",
        label="Switch to Chat tab",
        action="tab_chat",
        help_text="Focus the Chat tab",
        shortcut="ctrl + 1",
    ),
    PaletteCommand(
        section="Workspace",
        label="Switch to Files tab",
        action="tab_files",
        help_text="Focus the Files tab",
        shortcut="ctrl + 2",
    ),
    PaletteCommand(
        section="Workspace",
        label="Switch to Events tab",
        action="tab_events",
        help_text="Focus the Events tab",
        shortcut="ctrl + 3",
    ),
    PaletteCommand(
        section="Workspace",
        label="Close tab",
        action="close_process_tab",
        help_text="Close current tab",
        shortcut="ctrl + w",
    ),
    PaletteCommand(
        section="Integrations",
        label="Manage auth profiles…",
        action="auth_manage",
        help_text="Open auth manager tab",
        shortcut="ctrl + a",
    ),
    PaletteCommand(
        section="Integrations",
        label="Manage MCP servers…",
        action="mcp_manage",
        help_text="Open MCP manager tab",
        shortcut="ctrl + m",
    ),
    PaletteCommand(
        section="Integrations",
        label="List MCP servers",
        action="mcp_list",
        help_text="Show configured MCP servers",
    ),
    PaletteCommand(
        section="Integrations",
        label="Add MCP server…",
        action="mcp_add_prompt",
        help_text="Prefill /mcp add command template",
    ),
    PaletteCommand(
        section="System",
        label="Run setup wizard",
        action="setup",
        help_text="Open the setup flow",
    ),
    PaletteCommand(
        section="System",
        label="Show process modes",
        action="process_info",
        help_text="Show ad hoc vs explicit process modes",
    ),
    PaletteCommand(
        section="System",
        label="List processes",
        action="process_list",
        help_text="Show available process definitions",
    ),
    PaletteCommand(
        section="System",
        label="List tools",
        action="list_tools",
        help_text="Show all available tools",
    ),
    PaletteCommand(
        section="System",
        label="Show model info",
        action="model_info",
        help_text="Display active model details",
    ),
    PaletteCommand(
        section="System",
        label="Show models catalog",
        action="models_info",
        help_text="Display configured model details",
    ),
    PaletteCommand(
        section="System",
        label="Show token usage",
        action="token_info",
        help_text="Display session token count",
    ),
    PaletteCommand(
        section="System",
        label="Show learned patterns",
        action="learned_patterns",
        help_text="Open learned pattern manager",
    ),
    PaletteCommand(
        section="System",
        label="Show help",
        action="help",
        help_text="List commands and keyboard shortcuts",
    ),
    PaletteCommand(
        section="System",
        label="Quit",
        action="quit",
        help_text="Save session and exit",
        shortcut="ctrl + c",
    ),
)


def palette_commands_for_app(app: object) -> list[PaletteCommand]:
    """Return static + dynamic palette commands for the given app."""
    entries: list[PaletteCommand] = list(_COMMANDS)
    dynamic_getter = getattr(app, "iter_dynamic_process_palette_entries", None)
    if callable(dynamic_getter):
        for label, action, help_text in dynamic_getter():
            entries.append(
                PaletteCommand(
                    section="Process",
                    label=label,
                    action=action,
                    help_text=help_text,
                )
            )
    return entries


class LoomCommands(Provider):
    """Custom commands for the Ctrl+P command palette."""

    def _palette_entries(self) -> list[PaletteCommand]:
        """Return built-in + dynamic command palette entries."""
        return palette_commands_for_app(self.app)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for entry in self._palette_entries():
            score = max(matcher.match(entry.label), matcher.match(entry.help_text))
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(entry.label),
                    command=partial(
                        self.app.run_action, f"loom_command('{entry.action}')",
                    ),
                    help=entry.help_text,
                )

    async def discover(self) -> Hits:
        """Return full sorted palette list when opened with empty query."""
        for entry in sorted(
            self._palette_entries(),
            key=lambda item: item.label.casefold(),
        ):
            yield DiscoveryHit(
                display=entry.label,
                text=entry.label,
                command=partial(
                    self.app.run_action,
                    f"loom_command('{entry.action}')",
                ),
                help=entry.help_text,
            )
