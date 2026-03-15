"""`/config` slash command helpers."""

from __future__ import annotations

from loom.config_runtime import (
    ConfigPersistConflictError,
    ConfigPersistDisabledError,
    find_entry,
    list_entries,
    search_entries,
)
from loom.config_runtime.registry import allowed_values_text, display_value

from . import parsing as slash_parsing

_CONFIG_SUBCOMMANDS = ("list", "search", "show", "set", "reset")
_CONFIG_SCOPE_VALUES = ("runtime", "persist", "both")


def _application_class_text(application_class: str) -> str:
    return {
        "live": "yes",
        "next_call": "future calls only",
        "next_run": "future /run launches only",
        "restart_required": "no (restart required)",
    }.get(application_class, application_class)


def _scope_help(entry) -> str:
    scopes: list[str] = []
    if entry.supports_runtime:
        scopes.append("runtime")
    if entry.supports_persist:
        scopes.append("persist")
    if entry.supports_runtime and entry.supports_persist:
        scopes.append("both")
    return ", ".join(scopes) or "(none)"


def _render_entry_line(self, entry, *, include_value: bool = False) -> str:
    line = (
        f"  [#73daca]{self._escape_markup(entry.path)}[/] "
        f"{self._escape_markup(entry.description)}"
    )
    if include_value:
        snapshot = self._config_snapshot(entry.path)
        line += (
            " "
            f"[dim](effective: {self._escape_markup(snapshot['effective_display'])})[/dim]"
        )
    return line


def render_config_help(self) -> str:
    source_path = self._config_runtime_store.persist_target_path()
    lines = [
        "[bold #7dcfff]/config[/]",
        "  [dim]/config list[/] [dim]List supported config paths.[/dim]",
        "  [dim]/config search timeout[/] [dim]Search config paths.[/dim]",
        "  [dim]/config show execution.delegate_task_timeout_seconds[/] [dim]Show one key.[/dim]",
        "  [dim]/config set telemetry.mode debug[/] [dim]Set runtime value.[/dim]",
        (
            "  [dim]/config set tui.run_launch_timeout_seconds 90 --scope "
            "persist[/] [dim]Persist a value.[/dim]"
        ),
        "  [dim]/config reset telemetry.mode --scope runtime[/] [dim]Clear one override.[/dim]",
        "",
        f"source: [bold]{self._escape_markup(str(source_path))}[/bold]",
    ]
    return "\n".join(lines)


def render_config_show(self, path: str) -> str:
    entry = find_entry(path)
    if entry is None:
        return f"[bold #f7768e]Unknown config path:[/] {self._escape_markup(path)}"
    snapshot = self._config_snapshot(entry.path)
    allowed = allowed_values_text(entry)
    lines = [
        f"[bold #7dcfff]Config: {self._escape_markup(entry.path)}[/]",
        f"type: [bold]{self._escape_markup(entry.kind)}[/bold]",
        f"description: {self._escape_markup(entry.description)}",
        f"default: [bold]{self._escape_markup(display_value(entry.default))}[/bold]",
        f"configured: [bold]{self._escape_markup(snapshot['configured_display'])}[/bold]",
        f"runtime override: [bold]{self._escape_markup(snapshot['runtime_display'])}[/bold]",
        f"effective: [bold]{self._escape_markup(snapshot['effective_display'])}[/bold]",
        f"supports: [bold]{self._escape_markup(_scope_help(entry))}[/bold]",
        "applies to active runs: "
        f"[bold]{self._escape_markup(_application_class_text(entry.application_class))}[/bold]",
        f"restart required: [bold]{'yes' if entry.requires_restart else 'no'}[/bold]",
    ]
    if allowed:
        lines.append(f"allowed: [dim]{self._escape_markup(allowed)}[/dim]")
    if snapshot["source_path"]:
        lines.append(f"source: [dim]{self._escape_markup(snapshot['source_path'])}[/dim]")
    if snapshot["updated_at"]:
        lines.append(f"updated_at: [dim]{self._escape_markup(snapshot['updated_at'])}[/dim]")
    return "\n".join(lines)


def render_config_list(self) -> str:
    grouped: dict[str, list[str]] = {}
    for entry in list_entries():
        grouped.setdefault(entry.section, []).append(_render_entry_line(self, entry))
    lines = ["[bold #7dcfff]Config Paths[/]"]
    for section in sorted(grouped):
        lines.append(f"[bold]{self._escape_markup(section)}[/bold]")
        lines.extend(grouped[section])
    return "\n".join(lines)


def render_config_search(self, query: str) -> str:
    matches = search_entries(query)
    if not matches:
        return (
            f"[#f7768e]No config keys match '{self._escape_markup(query)}'[/]\n"
            "  [dim]Try /config list[/dim]"
        )
    lines = [f"[bold #7dcfff]Config Matches: {self._escape_markup(query)}[/]"]
    for entry in matches:
        lines.append(_render_entry_line(self, entry, include_value=True))
    return "\n".join(lines)


def _parse_scope(tokens: list[str]) -> str:
    if not tokens:
        return ""
    if len(tokens) != 2 or tokens[0] != "--scope":
        raise ValueError("Only --scope runtime|persist|both is supported.")
    scope = str(tokens[1] or "").strip().lower()
    if scope not in _CONFIG_SCOPE_VALUES:
        raise ValueError("Scope must be runtime, persist, or both.")
    return scope


def _default_scope(entry) -> str:
    if entry.supports_runtime:
        return "runtime"
    if entry.supports_persist:
        return "persist"
    return ""


def _apply_config_set(self, *, path: str, value_token: str, scope: str) -> str:
    entry = find_entry(path)
    if entry is None:
        return f"[bold #f7768e]Unknown config path:[/] {self._escape_markup(path)}"
    if not scope:
        scope = _default_scope(entry)
    lines = []
    warning_code = ""
    try:
        if scope in {"runtime", "both"}:
            _entry, parsed, _snapshot = self._set_runtime_config_value(
                path=entry.path,
                raw_value=value_token,
            )
            warning_code = parsed.warning_code
        if scope in {"persist", "both"}:
            _entry, parsed, _snapshot = self._persist_config_value(
                path=entry.path,
                raw_value=value_token,
            )
            warning_code = warning_code or parsed.warning_code
    except (ConfigPersistConflictError, ConfigPersistDisabledError, ValueError) as e:
        return f"[bold #f7768e]{self._escape_markup(str(e))}[/]"
    except KeyError:
        return f"[bold #f7768e]Unknown config path:[/] {self._escape_markup(path)}"

    self._refresh_runtime_config_bindings()
    lines.append("Config updated.")
    lines.append(f"path: [bold]{self._escape_markup(entry.path)}[/bold]")
    lines.append(f"scope: [bold]{self._escape_markup(scope)}[/bold]")
    if warning_code:
        lines.append(
            f"[dim]Input normalized ({self._escape_markup(warning_code)}).[/dim]",
        )
    lines.append(render_config_show(self, entry.path))
    return "\n".join(lines)


def _apply_config_reset(self, *, path: str, scope: str) -> str:
    entry = find_entry(path)
    if entry is None:
        return f"[bold #f7768e]Unknown config path:[/] {self._escape_markup(path)}"
    if not scope:
        scope = _default_scope(entry)
    try:
        if scope in {"runtime", "both"}:
            self._clear_runtime_config_value(path=entry.path)
        if scope in {"persist", "both"}:
            self._reset_persisted_config_value(path=entry.path)
    except (ConfigPersistConflictError, ConfigPersistDisabledError, ValueError) as e:
        return f"[bold #f7768e]{self._escape_markup(str(e))}[/]"
    except KeyError:
        return f"[bold #f7768e]Unknown config path:[/] {self._escape_markup(path)}"
    self._refresh_runtime_config_bindings()
    return "\n".join([
        "Config reset.",
        f"path: [bold]{self._escape_markup(entry.path)}[/bold]",
        f"scope: [bold]{self._escape_markup(scope)}[/bold]",
        render_config_show(self, entry.path),
    ])


def handle_config_command(self, arg: str) -> str:
    text = str(arg or "").strip()
    if not text:
        return render_config_help(self)
    try:
        tokens = slash_parsing.split_slash_args(text)
    except ValueError as e:
        return f"[bold #f7768e]{self._escape_markup(str(e))}[/]"
    if not tokens:
        return render_config_help(self)
    subcmd = str(tokens[0] or "").strip().lower()
    rest = tokens[1:]
    if subcmd == "list":
        if rest:
            return self._render_slash_command_usage("/config list", "(no arguments)")
        return render_config_list(self)
    if subcmd == "search":
        if not rest:
            return self._render_slash_command_usage("/config search", "<query>")
        return render_config_search(self, " ".join(rest))
    if subcmd == "show":
        if len(rest) != 1:
            return self._render_slash_command_usage("/config show", "<path>")
        return render_config_show(self, rest[0])
    if subcmd == "set":
        if len(rest) < 2:
            return self._render_slash_command_usage(
                "/config set",
                "<path> <value> [--scope runtime|persist|both]",
            )
        path = rest[0]
        value = rest[1]
        scope = _parse_scope(rest[2:]) if len(rest) > 2 else ""
        return _apply_config_set(self, path=path, value_token=value, scope=scope)
    if subcmd == "reset":
        if not rest:
            return self._render_slash_command_usage(
                "/config reset",
                "<path> [--scope runtime|persist|both]",
            )
        path = rest[0]
        scope = _parse_scope(rest[1:]) if len(rest) > 1 else ""
        return _apply_config_reset(self, path=path, scope=scope)
    return self._render_slash_command_usage(
        "/config",
        (
            "[list|search <query>|show <path>|set <path> <value> "
            "[--scope runtime|persist|both]|reset <path> "
            "[--scope runtime|persist|both]]"
        ),
    )


def completion_candidates(self, raw_input: str) -> tuple[str, list[str]] | None:
    current = str(raw_input or "").lstrip()
    lowered = current.lower()
    if not lowered.startswith("/config"):
        return None
    if lowered == "/config":
        return "/config", ["/config "]
    if not lowered.startswith("/config "):
        return None

    remainder = current[len("/config"):].lstrip()
    trailing_space = current.endswith(" ")
    tokens = slash_parsing.split_slash_args_forgiving(remainder)
    if not tokens:
        candidates = [f"/config {subcmd}" for subcmd in _CONFIG_SUBCOMMANDS]
        return "/config ", candidates

    subcmd = str(tokens[0] or "").strip().lower()
    if len(tokens) == 1 and not trailing_space:
        matches = [cmd for cmd in _CONFIG_SUBCOMMANDS if cmd.startswith(subcmd)]
        return f"/config {subcmd}", [f"/config {cmd}" for cmd in matches]
    if len(tokens) == 1 and trailing_space:
        if subcmd not in _CONFIG_SUBCOMMANDS:
            return None
        if subcmd == "list":
            return None
        if subcmd == "search":
            return None
        if subcmd in {"show", "reset", "set"}:
            matches = [entry.path for entry in list_entries()]
            return (
                f"/config {subcmd} ",
                [f"/config {subcmd} {path}" for path in matches],
            )

    if subcmd in {"show", "reset"}:
        if len(tokens) == 2 and not trailing_space:
            prefix = tokens[1]
            matches = [entry.path for entry in list_entries() if entry.path.startswith(prefix)]
            return f"/config {subcmd} {prefix}", [
                f"/config {subcmd} {path}" for path in matches
            ]
        if subcmd == "reset":
            if trailing_space and len(tokens) == 2:
                return f"/config reset {tokens[1]} ", [f"/config reset {tokens[1]} --scope"]
            if len(tokens) == 3 and not trailing_space and "--scope".startswith(tokens[2]):
                return (
                    f"/config reset {tokens[1]} {tokens[2]}",
                    [f"/config reset {tokens[1]} --scope"],
                )
            if len(tokens) == 3 and trailing_space and tokens[2] == "--scope":
                return f"/config reset {tokens[1]} --scope ", [
                    f"/config reset {tokens[1]} --scope {scope}" for scope in _CONFIG_SCOPE_VALUES
                ]
            if len(tokens) == 4 and not trailing_space and tokens[2] == "--scope":
                prefix = tokens[3]
                matches = [scope for scope in _CONFIG_SCOPE_VALUES if scope.startswith(prefix)]
                return f"/config reset {tokens[1]} --scope {prefix}", [
                    f"/config reset {tokens[1]} --scope {scope}" for scope in matches
                ]
        return None

    if subcmd == "set":
        if len(tokens) == 2 and not trailing_space:
            prefix = tokens[1]
            matches = [entry.path for entry in list_entries() if entry.path.startswith(prefix)]
            return f"/config set {prefix}", [f"/config set {path}" for path in matches]
        if len(tokens) == 2 and trailing_space:
            entry = find_entry(tokens[1])
            if entry is None:
                return None
            if entry.kind == "enum":
                return f"/config set {entry.path} ", [
                    f"/config set {entry.path} {value}" for value in entry.enum_values
                ]
            return None
        if len(tokens) == 3 and not trailing_space:
            entry = find_entry(tokens[1])
            if entry is None or entry.kind != "enum":
                return None
            prefix = tokens[2]
            matches = [value for value in entry.enum_values if value.startswith(prefix)]
            return f"/config set {entry.path} {prefix}", [
                f"/config set {entry.path} {value}" for value in matches
            ]
        if len(tokens) == 3 and trailing_space:
            return (
                f"/config set {tokens[1]} {tokens[2]} ",
                [f"/config set {tokens[1]} {tokens[2]} --scope"],
            )
        if len(tokens) == 4 and not trailing_space and "--scope".startswith(tokens[3]):
            return f"/config set {tokens[1]} {tokens[2]} {tokens[3]}", [
                f"/config set {tokens[1]} {tokens[2]} --scope"
            ]
        if len(tokens) == 4 and trailing_space and tokens[3] == "--scope":
            return f"/config set {tokens[1]} {tokens[2]} --scope ", [
                f"/config set {tokens[1]} {tokens[2]} --scope {scope}"
                for scope in _CONFIG_SCOPE_VALUES
            ]
        if len(tokens) == 5 and not trailing_space and tokens[3] == "--scope":
            prefix = tokens[4]
            matches = [scope for scope in _CONFIG_SCOPE_VALUES if scope.startswith(prefix)]
            return f"/config set {tokens[1]} {tokens[2]} --scope {prefix}", [
                f"/config set {tokens[1]} {tokens[2]} --scope {scope}"
                for scope in matches
            ]
    return None


def render_config_hint(self, raw_input: str) -> str | None:
    current = str(raw_input or "").strip()
    lowered = current.lower()
    if not (lowered == "/config" or lowered.startswith("/config ")):
        return None
    remainder = current[len("/config"):].strip()
    if not remainder:
        return render_config_help(self)
    tokens = slash_parsing.split_slash_args_forgiving(remainder)
    if not tokens:
        return render_config_help(self)
    subcmd = str(tokens[0] or "").strip().lower()
    if len(tokens) == 1 and subcmd not in _CONFIG_SUBCOMMANDS:
        matches = [cmd for cmd in _CONFIG_SUBCOMMANDS if cmd.startswith(subcmd)]
        if not matches:
            return None
        lines = [
            (
                "[bold #7dcfff]Matching /config subcommands for "
                f"{self._escape_markup(subcmd)}:[/]"
            )
        ]
        lines.extend(f"  [#73daca]{self._escape_markup(cmd)}[/]" for cmd in matches)
        return "\n".join(lines)
    if subcmd in {"show", "set", "reset"}:
        if len(tokens) == 1:
            return (
                f"[bold #7dcfff]/config {self._escape_markup(subcmd)}[/]\n"
                "  [dim]Start typing a config path...[/dim]"
            )
        path_token = tokens[1]
        entry = find_entry(path_token)
        if entry is not None:
            return render_config_show(self, entry.path)
        matches = [entry for entry in list_entries() if entry.path.startswith(path_token)]
        if not matches:
            return None
        lines = [f"[bold #7dcfff]Matching config paths for {self._escape_markup(path_token)}:[/]"]
        for match in matches:
            lines.append(_render_entry_line(self, match))
        return "\n".join(lines)
    if subcmd == "search":
        return (
            "[bold #7dcfff]/config search[/]\n"
            "  [dim]Search by path, alias, or description.[/dim]"
        )
    if subcmd == "list":
        return (
            "[bold #7dcfff]/config list[/]\n"
            "  [dim]Lists all supported operator-editable keys.[/dim]"
        )
    return None
