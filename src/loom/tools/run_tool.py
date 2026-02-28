"""Generic delegated tool execution for hybrid cowork mode.

`run_tool` lets the model execute long-tail tools discovered via `list_tools`
without requiring full per-tool schema injection every turn.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from loom.tools.registry import Tool, ToolContext, ToolResult

_DEFAULT_RUN_TOOL_TIMEOUT_SECONDS = 3600


class RunToolTool(Tool):
    """Execute a named tool with JSON arguments via a bound dispatcher."""

    name = "run_tool"
    description = (
        "Execute a tool by name with JSON arguments. Recommended workflow: "
        "list_tools(detail='compact') to discover names, list_tools(detail='schema') "
        "to inspect arguments, then run_tool."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Target tool name to execute.",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments object for the target tool.",
            },
        },
        "required": ["name"],
    }

    def __init__(
        self,
        dispatcher: Callable[[str, dict, ToolContext], Awaitable[ToolResult] | ToolResult]
        | None = None,
    ) -> None:
        self._dispatcher = dispatcher

    @property
    def timeout_seconds(self) -> int:
        # This wrapper can dispatch to long-running tools (for example delegate_task).
        return _DEFAULT_RUN_TOOL_TIMEOUT_SECONDS

    def bind(
        self,
        dispatcher: Callable[[str, dict, ToolContext], Awaitable[ToolResult] | ToolResult],
    ) -> None:
        self._dispatcher = dispatcher

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if self._dispatcher is None:
            return ToolResult.fail("run_tool is unavailable (dispatcher not bound).")

        name = str(args.get("name", "") or "").strip()
        if not name:
            return ToolResult.fail("'name' is required.")
        if name == self.name:
            return ToolResult.fail("run_tool cannot invoke itself.")

        raw_arguments = args.get("arguments", {})
        if raw_arguments is None:
            raw_arguments = {}
        if not isinstance(raw_arguments, dict):
            return ToolResult.fail("'arguments' must be an object.")

        try:
            maybe_result = self._dispatcher(name, raw_arguments, ctx)
            if inspect.isawaitable(maybe_result):
                result = await maybe_result
            else:
                result = maybe_result
        except Exception as e:
            return ToolResult.fail(f"run_tool dispatcher error: {type(e).__name__}: {e}")

        if not isinstance(result, ToolResult):
            return ToolResult.fail("run_tool dispatcher returned an invalid result.")

        # Attach delegated tool metadata when possible without destroying existing data.
        if isinstance(result.data, dict):
            data = dict(result.data)
            data.setdefault("delegated_tool", name)
            result = ToolResult(
                success=result.success,
                output=result.output,
                content_blocks=result.content_blocks,
                data=data,
                files_changed=list(result.files_changed),
                error=result.error,
            )
        return result
