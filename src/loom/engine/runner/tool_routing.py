"""Process-scoped tool routing helpers for runner execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loom.engine.verification_helpers import (
    VerificationHelperContext,
    route_tool_to_verification_helper,
)

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition


def route_tool_call_for_process(
    *,
    tool_name: str,
    tool_args: object,
    process: ProcessDefinition | None,
    workspace: Path | None,
    subtask_id: str = "",
    execution_surface: str = "",
) -> tuple[str, dict[str, Any], dict[str, object]]:
    """Rewrite one tool call when the active process prefers helper routing."""
    normalized_args = dict(tool_args) if isinstance(tool_args, dict) else {}
    if process is None:
        return str(tool_name or "").strip(), normalized_args, {}
    if process.verifier_tool_success_policy() != "development_balanced":
        return str(tool_name or "").strip(), normalized_args, {}

    decision = route_tool_to_verification_helper(
        str(tool_name or "").strip(),
        normalized_args,
        ctx=VerificationHelperContext(
            workspace=workspace,
            metadata={
                "subtask_id": str(subtask_id or "").strip(),
                "execution_surface": str(execution_surface or "").strip(),
            },
        ),
    )
    if decision is None:
        return str(tool_name or "").strip(), normalized_args, {}
    return (
        decision.target_tool,
        dict(decision.arguments),
        {
            "routed_from_tool": decision.source_tool,
            "routing_reason": decision.reason,
            "helper": decision.helper,
        },
    )
