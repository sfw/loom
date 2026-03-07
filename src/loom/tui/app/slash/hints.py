"""Slash command hint rendering helpers."""

from __future__ import annotations

from textual.containers import Grid, Vertical, VerticalScroll
from textual.widgets import Static

from ..constants import _SLASH_COMMAND_PRIORITY, _SLASH_COMMANDS
from ..models import SlashCommandSpec
from . import registry as slash_registry


def slash_command_catalog(self) -> list[tuple[str, str]]:
    """Return canonical slash commands with optional alias annotations."""
    entries: list[tuple[str, str]] = []
    for spec in self._ordered_slash_specs():
        label = spec.canonical
        if spec.usage:
            label = f"{label} {spec.usage}"
        desc = spec.description
        if spec.aliases:
            desc = f"{desc} ({', '.join(spec.aliases)})"
        entries.append((label, desc))
    for token, process_name in sorted(self._process_command_map.items()):
        entries.append((f"{token} <goal>", f"run goal via process '{process_name}'"))
    return entries


def render_root_slash_hint(self) -> str:
    """Render full root slash catalog with explicit built-in/process sections."""
    lines = ["[bold #7dcfff]Slash commands:[/]"]
    for spec in self._ordered_slash_specs():
        cmd = spec.canonical
        if spec.usage:
            cmd = f"{cmd} {spec.usage}"
        desc = spec.description
        if spec.aliases:
            desc = f"{desc} ({', '.join(spec.aliases)})"
        safe_cmd = self._escape_markup(cmd)
        safe_desc = self._escape_markup(desc)
        lines.append(f"  [#73daca]{safe_cmd:<10}[/] {safe_desc}")
    if self._process_command_map:
        lines.append("")
        lines.append("[bold #7dcfff]Process slash commands:[/]")
        for token, process_name in sorted(self._process_command_map.items()):
            safe_cmd = self._escape_markup(f"{token} <goal>")
            safe_desc = self._escape_markup(
                f"run goal via process '{process_name}'",
            )
            lines.append(f"  [#73daca]{safe_cmd:<10}[/] {safe_desc}")
    return "\n".join(lines)


def slash_spec_sort_key(spec: SlashCommandSpec) -> tuple[int, str]:
    """Return deterministic display/completion ordering key for slash specs."""
    return slash_registry.slash_spec_sort_key(
        spec,
        priority=_SLASH_COMMAND_PRIORITY,
    )


def ordered_slash_specs() -> list[SlashCommandSpec]:
    """Return built-in slash specs in deterministic UX priority order."""
    return slash_registry.ordered_slash_specs(
        _SLASH_COMMANDS,
        priority=_SLASH_COMMAND_PRIORITY,
    )


def slash_match_keys(spec: SlashCommandSpec) -> tuple[str, ...]:
    """Return normalized command tokens used for prefix matching."""
    return slash_registry.slash_match_keys(spec)


def help_lines(self) -> list[str]:
    """Build slash help lines from the shared command registry."""
    lines = ["[bold #7dcfff]Slash Commands[/bold #7dcfff]"]
    for spec in self._ordered_slash_specs():
        label = spec.canonical
        if spec.usage:
            label = f"{label} {spec.usage}"
        if spec.aliases:
            alias_str = ", ".join(spec.aliases)
            label = f"{label} (aliases: {alias_str})"
        lines.append(f"  [bold]{self._escape_markup(label)}[/bold]")
        lines.append(
            self._wrap_info_text(
                self._escape_markup(spec.description),
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )
    if self._process_command_map:
        lines.append("")
        lines.append("[bold #7dcfff]Process Slash Commands[/bold #7dcfff]")
        for token, process_name in sorted(self._process_command_map.items()):
            lines.append(f"  [bold]{self._escape_markup(token)} <goal>[/bold]")
            lines.append(
                self._wrap_info_text(
                    self._escape_markup(
                        f"Run goal via process '{process_name}'"
                    ),
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )
    if self._blocked_process_commands:
        blocked = ", ".join(f"/{name}" for name in self._blocked_process_commands)
        lines.append("")
        lines.append(
            "[#f7768e]Blocked process commands (name collisions): "
            f"{self._escape_markup(blocked)}[/]"
        )
    lines.extend([
        "",
        "[bold #7dcfff]Keys[/bold #7dcfff]",
        self._wrap_info_text(
            "ctrl + b sidebar, ctrl + l clear, ctrl + r reload workspace, "
            "ctrl + w close tab, ctrl + a auth, ctrl + m mcp, "
            "ctrl + p commands, ctrl + 1/2/3 tabs",
            initial_indent="  ",
            subsequent_indent="  ",
        ),
    ])
    return lines


def render_slash_command_usage(self, command: str, usage: str) -> str:
    """Render a usage error block for slash commands."""
    return (
        f"[bold #7dcfff]Usage[/bold #7dcfff]\n"
        f"  [bold]{self._escape_markup(command)}[/bold] {self._escape_markup(usage)}"
    )


def render_tool_slash_hint(self, raw_input: str) -> str | None:
    """Render helper popup text for `/tool` command composition."""
    raw = str(raw_input or "")
    stripped = raw.strip()
    lowered = stripped.lower()
    if not (lowered == "/tool" or lowered.startswith("/tool ")):
        return None

    tool_names = self._tool_name_inventory()
    if not tool_names:
        return (
            "[bold #7dcfff]/tool[/]\n"
            "  [dim]No tools are currently available.[/dim]"
        )

    remainder = stripped[len("/tool"):].strip()
    if not remainder:
        lines = [
            "[bold #7dcfff]/tool[/] [dim]<tool-name> "
            "\\[key=value ... | json-object-args\\][/]",
            "  [dim]KV example:[/] /tool read_file path=README.md",
            "  [dim]JSON example:[/] /tool read_file "
            "{ \"path\": \"README.md\" }",
            "",
            "[bold #7dcfff]Tool names:[/]",
        ]
        for name in tool_names:
            desc = self._tool_description(name)
            safe_name = self._escape_markup(name)
            safe_desc = self._escape_markup(desc) if desc else "no description"
            lines.append(f"  [#73daca]{safe_name}[/] {safe_desc}")
        return "\n".join(lines)

    tool_token, raw_args = self._split_tool_slash_args(remainder)
    prefix = self._strip_wrapping_quotes(tool_token).lower()
    matches = [name for name in tool_names if name.lower().startswith(prefix)]
    if not matches:
        return (
            f"[#f7768e]No tool matches '{self._escape_markup(tool_token)}'[/]\n"
            "  [dim]Use /tools to inspect available tools.[/dim]"
        )

    exact = next((name for name in matches if name.lower() == prefix), "")
    if exact:
        description = self._tool_description(exact)
        required_text, optional_text = self._tool_argument_summary(exact)
        example = self._tool_argument_example(exact)
        lines = [f"[bold #7dcfff]Tool: {self._escape_markup(exact)}[/]"]
        if description:
            lines.append(f"  {self._escape_markup(description)}")
        lines.append(f"  [bold]Required:[/] {self._escape_markup(required_text)}")
        lines.append(f"  [bold]Optional:[/] {self._escape_markup(optional_text)}")
        lines.append(
            "  [bold]Example:[/] "
            f"/tool {self._escape_markup(exact)} "
            f"{self._escape_markup(example)}"
        )
        if "path" in required_text:
            lines.append(
                "  [bold]KV try:[/] "
                f"/tool {self._escape_markup(exact)} path=README.md"
            )
        if raw_args:
            lines.append("  [dim]Press Enter to run with current JSON args.[/dim]")
        return "\n".join(lines)

    lines = [
        f"[bold #7dcfff]Matching tools for {self._escape_markup(tool_token)}:[/]"
    ]
    for name in matches:
        required_text, _optional_text = self._tool_argument_summary(name)
        required_preview = required_text
        if len(required_preview) > 48:
            required_preview = required_preview[:45].rstrip() + "..."
        lines.append(
            "  "
            f"[#73daca]{self._escape_markup(name)}[/] "
            f"[dim](required: {self._escape_markup(required_preview)})[/]"
        )
    return "\n".join(lines)


def matching_slash_commands(
    self,
    raw_input: str,
) -> tuple[str, list[tuple[str, str]]]:
    """Return current slash token and matching commands."""
    text = raw_input.strip()
    if not text.startswith("/"):
        return "", []
    # Keep dynamic process slash commands in sync as the user types.
    self._refresh_process_command_index(background=True)
    token = text.split()[0].lower()
    if token == "/":
        return token, self._slash_command_catalog()
    return slash_registry.matching_slash_commands(
        raw_input,
        ordered_specs=self._ordered_slash_specs(),
        process_command_map=self._process_command_map,
    )


def render_slash_hint(self, raw_input: str) -> str:
    """Build slash-command hint text for the current input."""
    if raw_input.strip() == "/":
        self._refresh_process_command_index(background=True)
        return self._render_root_slash_hint()
    tool_hint = self._render_tool_slash_hint(raw_input)
    if tool_hint is not None:
        return tool_hint
    if " " in raw_input.strip():
        return ""
    token, matches = self._matching_slash_commands(raw_input)
    if not token:
        return ""

    if not matches:
        return (
            f"[#f7768e]No command matches '{token}'[/]  "
            "[dim]Try /help[/]"
        )

    title = "Slash commands:" if token == "/" else f"Matching {token}:"
    lines = [f"[bold #7dcfff]{title}[/]"]
    for cmd, desc in matches:
        safe_cmd = self._escape_markup(cmd)
        safe_desc = self._escape_markup(desc)
        lines.append(f"  [#73daca]{safe_cmd:<10}[/] {safe_desc}")
    return "\n".join(lines)


def set_landing_slash_hint(self, hint_text: str) -> None:
    """Show or hide slash hints anchored under the landing input."""
    try:
        landing_hint = self.query_one("#landing-slash-hint", VerticalScroll)
        landing_hint_body = self.query_one("#landing-slash-hint-body", Static)
    except Exception:
        return
    update_body = getattr(landing_hint_body, "update", None)
    scroll_home = getattr(landing_hint, "scroll_home", None)
    if not callable(update_body) or not callable(scroll_home):
        return
    text = str(hint_text or "")
    if (
        text
        and text == self._last_landing_slash_hint_text
        and bool(getattr(landing_hint, "display", False))
        and bool(getattr(landing_hint_body, "display", False))
    ):
        return
    if text:
        update_body(text)
        landing_hint_body.display = True
        landing_hint.display = True
        scroll_home(animate=False)
        self._last_landing_slash_hint_text = text
        return
    if (
        not self._last_landing_slash_hint_text
        and not bool(getattr(landing_hint, "display", False))
    ):
        return
    landing_hint.display = False
    landing_hint_body.display = False
    update_body("")
    self._last_landing_slash_hint_text = ""


def set_slash_hint(self, hint_text: str) -> None:
    """Show or hide the slash-command hint panel."""
    if self._startup_landing_active:
        self._set_landing_slash_hint(hint_text)
        try:
            hint = self.query_one("#slash-hint", VerticalScroll)
            hint_body = self.query_one("#slash-hint-body", Static)
            hint.display = False
            hint_body.display = False
            hint_body.update("")
            hint.scroll_home(animate=False)
        except Exception:
            pass
        return
    self._set_landing_slash_hint("")
    hint = self.query_one("#slash-hint", VerticalScroll)
    hint_body = self.query_one("#slash-hint-body", Static)
    queue_grid: Grid | None = None
    queue_list: Vertical | None = None
    try:
        queue_grid = self.query_one("#steer-queue-grid", Grid)
    except Exception:
        queue_grid = None
    try:
        queue_list = self.query_one("#steer-queue-list", Vertical)
    except Exception:
        queue_list = None
    styles = getattr(hint, "styles", None)
    queue_grid_styles = getattr(queue_grid, "styles", None) if queue_grid is not None else None
    if self._should_show_steer_queue_popup():
        rows = max(1, self._pending_inject_count()) + 2
        signature = self._steer_queue_signature()
        if styles is not None:
            styles.height = rows
            styles.max_height = rows
            styles.overflow_y = "hidden"
        if queue_grid_styles is not None:
            queue_grid_styles.background = "#3a4465"
        hint_body.update("")
        hint_body.display = False
        hint.display = True
        if queue_grid is not None:
            add_class = getattr(queue_grid, "add_class", None)
            if callable(add_class):
                add_class("queue-mode")
            queue_grid.display = True
        if queue_list is not None:
            should_render_rows = (
                not bool(getattr(queue_list, "display", False))
                or self._last_rendered_steer_queue_signature != signature
            )
            queue_list.display = True
            if should_render_rows:
                self._render_steer_queue_rows()
        hint.scroll_home(animate=False)
        return
    if styles is not None:
        styles.height = "auto"
        styles.max_height = 14
        styles.overflow_y = "scroll"
    if queue_grid_styles is not None:
        queue_grid_styles.background = "transparent"
    hint_body.display = bool(hint_text)
    if queue_grid is not None:
        remove_class = getattr(queue_grid, "remove_class", None)
        if callable(remove_class):
            remove_class("queue-mode")
        queue_grid.display = bool(hint_text)
    if queue_list is not None:
        queue_list.display = False
    self._last_rendered_steer_queue_signature = ()
    if hint_text:
        hint_body.update(hint_text)
        hint.display = True
        hint.scroll_home(animate=False)
    else:
        if queue_grid is not None:
            queue_grid.display = False
        hint.display = False
        hint_body.update("")
        hint.scroll_home(animate=False)
