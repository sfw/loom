"""Process definition loading and prompt/tool policy helpers."""

from __future__ import annotations

from pathlib import Path

from loom.cowork.session import build_cowork_system_prompt
from loom.tui.widgets import ChatLog


def create_process_loader(self):
    """Create a process loader for the current workspace/config."""
    from loom.processes.schema import ProcessLoader

    extra: list[Path] = []
    if self._config:
        extra = [Path(p) for p in self._config.process.search_paths]
    return ProcessLoader(
        workspace=self._workspace,
        extra_search_paths=extra,
        require_rule_scope_metadata=bool(
            getattr(
                getattr(self._config, "process", None),
                "require_rule_scope_metadata",
                False,
            ),
        ),
        require_v2_contract=bool(
            getattr(
                getattr(self._config, "process", None),
                "require_v2_contract",
                False,
            ),
        ),
    )


def active_process_name(self) -> str:
    """Return active process display name."""
    if self._process_defn:
        return self._process_defn.name
    return "none"


def apply_process_tool_policy(self, chat: ChatLog) -> None:
    """Apply process tool exclusions to the active registry."""
    if not self._process_defn:
        return
    if self._process_defn.tools.excluded:
        for tool_name in self._process_defn.tools.excluded:
            self._tools.exclude(tool_name)
    if self._process_defn.tools.required:
        missing = [
            tool_name
            for tool_name in self._process_defn.tools.required
            if not self._tools.has(tool_name)
        ]
        if missing:
            joined = ", ".join(sorted(missing))
            chat.add_info(
                f"[bold #f7768e]Process requires missing tool(s): "
                f"{joined}[/]",
            )


def build_system_prompt(self) -> str:
    """Build cowork system prompt with optional process extensions."""
    system_prompt = build_cowork_system_prompt(self._workspace)
    if self._process_defn:
        if self._process_defn.persona:
            system_prompt += (
                f"\n\nDOMAIN ROLE:\n{self._process_defn.persona.strip()}"
            )
        if self._process_defn.tool_guidance:
            system_prompt += (
                f"\n\nDOMAIN TOOL GUIDANCE:\n"
                f"{self._process_defn.tool_guidance.strip()}"
            )
    return system_prompt
