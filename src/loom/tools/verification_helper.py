"""Tool wrapper for executing registered development verification helpers."""

from __future__ import annotations

from pathlib import Path

from loom.engine.verification_helpers import (
    VerificationHelperContext,
    execute_verification_helper,
)
from loom.tools.registry import Tool, ToolContext, ToolResult


class VerificationHelperTool(Tool):
    @property
    def name(self) -> str:
        return "verification_helper"

    @property
    def description(self) -> str:
        return (
            "Execute a registered development verification helper such as "
            "run_test_suite, run_build_check, http_assert, probe_suite, "
            "browser_session, or render_verification_report."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "helper": {
                    "type": "string",
                    "description": "Registered verification helper name.",
                },
                "args": {
                    "type": "object",
                    "description": "Helper-specific argument payload.",
                },
            },
            "required": ["helper"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 180

    @property
    def is_mutating(self) -> bool:
        return True

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        helper_name = str(args.get("helper", "") or "").strip()
        helper_args = args.get("args", {})
        if not isinstance(helper_args, dict):
            helper_args = {}
        try:
            result = await execute_verification_helper(
                helper_name,
                helper_args,
                ctx=VerificationHelperContext(
                    workspace=Path(ctx.workspace) if ctx.workspace else None,
                    metadata={
                        "subtask_id": str(ctx.subtask_id or "").strip(),
                        "execution_surface": str(ctx.execution_surface or "").strip(),
                    },
                ),
            )
        except Exception as exc:
            return ToolResult.fail(
                f"verification helper {helper_name or '<unknown>'} failed: {exc}"
            )

        data = dict(result.data) if isinstance(result.data, dict) else {}
        if helper_name:
            data.setdefault("helper", helper_name)
        if result.capability:
            data.setdefault("helper_capability", result.capability)
        if result.reason_code:
            data.setdefault("helper_reason_code", result.reason_code)
        files_changed: list[str] = []
        output_path = ""
        output_path = str(data.get("output_path", "") or "").strip()
        if output_path:
            files_changed.append(output_path)
        output = str(result.detail or "").strip()
        if not output:
            output = str(data.get("markdown", "") or "").strip()
        if result.success:
            return ToolResult.ok(
                output or f"verification helper {helper_name} completed successfully.",
                data=data,
                files_changed=files_changed,
            )
        error = str(
            result.reason_code
            or result.detail
            or f"verification helper {helper_name} failed"
        ).strip()
        return ToolResult(
            success=False,
            output=output,
            error=error,
            data=data,
            files_changed=files_changed,
        )
