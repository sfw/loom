"""Custom Loom command palette screen."""

from __future__ import annotations

from collections.abc import Iterable

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from loom.tui.commands import PaletteCommand, palette_commands_for_app


def _safe_markup(value: str) -> str:
    return str(value or "").replace("[", "\\[").replace("]", "\\]")


class LoomCommandPaletteScreen(ModalScreen[None]):
    """Wider, grouped command palette with section headings."""

    # Keep palette-modal key handling isolated from app-level shortcuts.
    _inherit_bindings = False

    BINDINGS = [
        Binding("escape", "close", show=False),
        Binding("up", "cursor_up", show=False),
        Binding("down", "cursor_down", show=False),
        Binding("enter", "activate", show=False),
        # Consume app-level shortcut chords while modal palette is active.
        Binding("ctrl+a", "noop", show=False, priority=True),
        Binding("ctrl+m", "noop", show=False, priority=True),
        Binding("ctrl+w", "noop", show=False, priority=True),
        Binding("ctrl+c", "noop", show=False, priority=True),
        Binding("ctrl+r", "noop", show=False, priority=True),
        Binding("ctrl+p", "close", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    LoomCommandPaletteScreen {
        background: #0b0f1b 62%;
        align: center middle;
    }
    LoomCommandPaletteScreen #loom-command-card {
        width: 140;
        min-width: 92;
        max-width: 96%;
        max-height: 88%;
        background: #1f233b;
        border: none;
        padding: 1 2;
    }
    LoomCommandPaletteScreen #loom-command-header {
        width: 100%;
        height: 1;
        margin: 0 0 1 0;
    }
    LoomCommandPaletteScreen #loom-command-title {
        width: 1fr;
        color: #c9d4ff;
        text-style: bold;
    }
    LoomCommandPaletteScreen #loom-command-dismiss {
        width: auto;
        color: #8fa0d8;
    }
    LoomCommandPaletteScreen #loom-command-input {
        width: 100%;
        height: 3;
        background: #171b2f;
        border: none;
        color: $text;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    LoomCommandPaletteScreen #loom-command-input:focus {
        background: #232949;
        border: none;
    }
    LoomCommandPaletteScreen #loom-command-list {
        height: 1fr;
        max-height: 72vh;
        border: none;
        padding: 0;
        background: transparent;
    }
    LoomCommandPaletteScreen #loom-command-list > .option-list--option {
        padding: 0 1;
        color: #d3dcff;
        text-style: bold;
    }
    LoomCommandPaletteScreen #loom-command-list > .option-list--option-highlighted {
        color: #0b1020;
        background: #83a7ef;
        text-style: bold;
    }
    LoomCommandPaletteScreen #loom-command-list > .option-list--option-disabled {
        color: #f6a77d;
        text-style: bold;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._commands: list[PaletteCommand] = []
        self._option_actions: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="loom-command-card"):
            with Horizontal(id="loom-command-header"):
                yield Static("Commands", id="loom-command-title")
                yield Static("esc", id="loom-command-dismiss")
            yield Input(placeholder="Search commands...", id="loom-command-input")
            yield OptionList(id="loom-command-list")

    def on_mount(self) -> None:
        self._commands = palette_commands_for_app(self.app)
        self._refresh_options("")
        self.query_one("#loom-command-input", Input).focus()

    def _iter_filtered(self, query: str) -> Iterable[PaletteCommand]:
        text = str(query or "").strip().casefold()
        if not text:
            return list(self._commands)
        return [
            item
            for item in self._commands
            if text in item.label.casefold() or text in item.help_text.casefold()
        ]

    @staticmethod
    def _section_heading(section: str) -> str:
        safe = _safe_markup(section)
        return f"[#f6a77d]{safe}[/]"

    @staticmethod
    def _command_text(item: PaletteCommand) -> str:
        title = _safe_markup(item.label)
        shortcut = _safe_markup(item.shortcut)
        if shortcut:
            line_1 = f"[bold]{title}[/]  [dim]{shortcut}[/]"
        else:
            line_1 = f"[bold]{title}[/]"
        line_2 = f"[dim]{_safe_markup(item.help_text)}[/]"
        return f"{line_1}\n{line_2}"

    def _refresh_options(self, query: str) -> None:
        filtered = list(self._iter_filtered(query))
        options: list[Option] = []
        self._option_actions = {}
        seen_sections: set[str] = set()
        command_index = 0
        for item in filtered:
            if item.section not in seen_sections:
                heading_id = f"section-{len(seen_sections)}"
                options.append(
                    Option(self._section_heading(item.section), id=heading_id, disabled=True)
                )
                seen_sections.add(item.section)
            option_id = f"command-{command_index}"
            command_index += 1
            options.append(Option(self._command_text(item), id=option_id))
            self._option_actions[option_id] = item.action

        command_list = self.query_one("#loom-command-list", OptionList)
        command_list.clear_options()
        if not options:
            command_list.add_option(
                Option("[dim]No matching commands[/]", id="no-matches", disabled=True)
            )
            command_list.highlighted = None
            return
        command_list.add_options(options)
        for index in range(command_list.option_count):
            option = command_list.get_option_at_index(index)
            if not option.disabled:
                command_list.highlighted = index
                return
        command_list.highlighted = None

    async def _activate_highlighted(self) -> None:
        command_list = self.query_one("#loom-command-list", OptionList)
        highlighted = command_list.highlighted
        if highlighted is None:
            return
        option = command_list.get_option_at_index(highlighted)
        option_id = str(getattr(option, "id", "") or "")
        action = self._option_actions.get(option_id, "")
        if not action:
            return
        self.dismiss()
        await self.app.run_action(f"loom_command('{action}')")

    def action_close(self) -> None:
        self.dismiss()

    def action_noop(self) -> None:
        """Consume keybinding without side effects."""

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Close when clicking outside the command card."""
        card = self.query_one("#loom-command-card")
        if card.region.contains(event.screen_x, event.screen_y):
            return
        self.dismiss()
        event.stop()
        event.prevent_default()

    def _move_highlight(self, step: int) -> None:
        command_list = self.query_one("#loom-command-list", OptionList)
        count = int(command_list.option_count)
        if count <= 0:
            command_list.highlighted = None
            return
        current = command_list.highlighted
        if current is None:
            current = -1 if step > 0 else count
        index = int(current)
        for _ in range(count):
            index = (index + step) % count
            option = command_list.get_option_at_index(index)
            if not option.disabled:
                command_list.highlighted = index
                return

    def action_cursor_up(self) -> None:
        self._move_highlight(-1)

    def action_cursor_down(self) -> None:
        self._move_highlight(1)

    async def action_activate(self) -> None:
        await self._activate_highlighted()

    @on(Input.Changed, "#loom-command-input")
    def _on_input_changed(self, event: Input.Changed) -> None:
        self._refresh_options(event.value)

    @on(Input.Submitted, "#loom-command-input")
    async def _on_input_submitted(self, _event: Input.Submitted) -> None:
        await self._activate_highlighted()

    @on(OptionList.OptionSelected, "#loom-command-list")
    async def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option_id = str(getattr(event.option, "id", "") or "")
        action = self._option_actions.get(option_id, "")
        if not action:
            return
        self.dismiss()
        await self.app.run_action(f"loom_command('{action}')")
