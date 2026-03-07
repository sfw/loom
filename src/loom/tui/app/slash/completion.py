"""Slash tab-completion helpers."""

from __future__ import annotations

from textual.app import ScreenStackError
from textual.widgets import Input

from . import parsing as slash_parsing
from . import registry as slash_registry


def tool_name_completion_candidates(
    self,
    raw_input: str,
) -> tuple[str, list[str]] | None:
    """Return seed/candidates for `/tool <name>` tab completion."""
    current = str(raw_input or "").lstrip()
    lowered = current.lower()
    if not lowered.startswith("/tool"):
        return None
    if lowered == "/tool":
        return None
    if not lowered.startswith("/tool "):
        return None

    remainder = current[len("/tool"):].lstrip()
    if not remainder:
        prefix = ""
    else:
        if " " in remainder:
            return None
        prefix = self._strip_wrapping_quotes(remainder)

    names = self._tool_name_inventory()
    matches = [name for name in names if name.lower().startswith(prefix.lower())]
    candidates = [f"/tool {name}" for name in matches]
    seed = f"/tool {prefix}" if prefix else "/tool "
    return seed, candidates


def reset_slash_tab_cycle(self) -> None:
    """Clear slash tab-completion cycle state."""
    self._slash_cycle_seed = ""
    self._slash_cycle_candidates = []


def slash_completion_candidates(self, token: str) -> list[str]:
    """Return slash command completions for a token prefix."""
    self._refresh_process_command_index(background=True)
    return slash_registry.slash_completion_candidates(
        token,
        ordered_specs=self._ordered_slash_specs(),
        process_command_map=self._process_command_map,
    )


def apply_slash_tab_completion(
    self,
    *,
    reverse: bool = False,
    input_widget: Input | None = None,
) -> bool:
    """Apply slash tab completion (forward/backward)."""
    if input_widget is None:
        focused = None
        try:
            focused = self.focused
        except ScreenStackError:
            focused = None
        if isinstance(focused, Input) and focused.id in {"user-input", "landing-input"}:
            input_widget = focused
        else:
            selector = "#landing-input" if self._startup_landing_active else "#user-input"
            input_widget = self.query_one(selector, Input)
    raw_current = str(input_widget.value or "")
    current = raw_current.lstrip()
    if not current.startswith("/"):
        self._reset_slash_tab_cycle()
        return False

    tool_seed_candidates = self._tool_name_completion_candidates(raw_current)
    if tool_seed_candidates is not None:
        token, candidates = tool_seed_candidates
        if not candidates:
            self._reset_slash_tab_cycle()
            return False
        if current in self._slash_cycle_candidates:
            cycle = self._slash_cycle_candidates
            current_index = cycle.index(current)
            next_index = (
                (current_index - 1) if reverse else (current_index + 1)
            ) % len(cycle)
        else:
            self._slash_cycle_seed = token
            self._slash_cycle_candidates = candidates
            cycle = candidates
            next_index = len(cycle) - 1 if reverse else 0
        completion = cycle[next_index]
        self._applying_slash_tab_completion = True
        self._skip_slash_cycle_reset_once = True
        try:
            input_widget.value = completion
            input_widget.cursor_position = len(completion)
        finally:
            self._applying_slash_tab_completion = False
        return True

    token = current.strip()
    if " " in token:
        self._reset_slash_tab_cycle()
        return False

    if token in self._slash_cycle_candidates:
        candidates = self._slash_cycle_candidates
        current_index = candidates.index(token)
        next_index = (
            (current_index - 1) if reverse else (current_index + 1)
        ) % len(candidates)
    else:
        candidates = self._slash_completion_candidates(token)
        if not candidates:
            self._reset_slash_tab_cycle()
            return False
        self._slash_cycle_seed = token
        self._slash_cycle_candidates = candidates
        next_index = len(candidates) - 1 if reverse else 0

    completion = candidates[next_index]
    self._applying_slash_tab_completion = True
    self._skip_slash_cycle_reset_once = True
    try:
        input_widget.value = completion
        input_widget.cursor_position = len(completion)
    finally:
        self._applying_slash_tab_completion = False
    return True


def strip_wrapping_quotes(value: str) -> str:
    """Remove matching wrapping quotes from a command argument."""
    return slash_parsing.strip_wrapping_quotes(value)
