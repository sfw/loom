"""WordPress local environment lifecycle wrapper around @wordpress/env."""

from __future__ import annotations

from pathlib import Path

from loom.tools.registry import (
    Tool,
    ToolAvailabilityReason,
    ToolAvailabilityStatus,
    ToolContext,
    ToolResult,
)
from loom.tools.tooling_common.binary_resolution import (
    configured_binary_override,
    normalize_binary_overrides,
    resolve_binary,
)
from loom.tools.tooling_common.command_runner import constrained_env, run_command


class WpEnvTool(Tool):
    """Manage local wp-env lifecycle in a structured/safe way."""

    name = "wp_env"
    description = (
        "Run @wordpress/env operations (start/stop/destroy/run/logs) for local "
        "WordPress development environments."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "start | stop | destroy | run | logs",
            },
            "cwd": {
                "type": "string",
                "description": "Project directory relative to workspace.",
            },
            "run_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments for operation=run.",
            },
            "confirm_high_risk": {
                "type": "boolean",
                "description": "Required for operation=destroy.",
            },
        },
        "required": ["operation"],
    }

    def __init__(
        self,
        *,
        enabled: bool = True,
        binary_overrides: dict[str, str] | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._binary_overrides = normalize_binary_overrides(binary_overrides or {})

    @property
    def is_mutating(self) -> bool:
        return True

    @property
    def timeout_seconds(self) -> int:
        return 240

    def _resolve_npx_binary(self):
        return resolve_binary(
            "npx",
            override=configured_binary_override(
                self._binary_overrides,
                self.name,
                "npx",
            ),
        )

    def availability(
        self,
        *,
        execution_surface: object = "tui",
    ) -> ToolAvailabilityStatus:
        base = super().availability(execution_surface=execution_surface)
        if not base.runnable:
            return base
        if not self._enabled:
            return ToolAvailabilityStatus(
                state="unavailable",
                reasons=(
                    ToolAvailabilityReason(
                        code="feature_disabled",
                        message="WordPress tools are disabled by configuration.",
                    ),
                ),
            )
        resolution = self._resolve_npx_binary()
        if resolution.found:
            return ToolAvailabilityStatus(
                state="available",
                metadata={
                    "binary": "npx",
                    "binary_path": resolution.path,
                    "binary_source": resolution.source,
                },
            )
        return ToolAvailabilityStatus(
            state="unavailable",
            reasons=(
                ToolAvailabilityReason(
                    code=resolution.error_code or "binary_not_found",
                    message=resolution.message or "Binary not found: npx",
                    metadata=dict(resolution.metadata or {}),
                ),
            ),
            metadata={"binary": "npx"},
        )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if not self._enabled:
            return self._error("feature_disabled", "WordPress tools are disabled by configuration.")
        if ctx.workspace is None:
            return self._error("path_outside_workspace", "No workspace set.")

        operation = str(args.get("operation", "") or "").strip().lower()
        if operation not in {"start", "stop", "destroy", "run", "logs"}:
            return self._error("invalid_arguments", "Unsupported operation.")

        raw_cwd = str(args.get("cwd", "") or "").strip()
        try:
            cwd = self._resolve_path(raw_cwd, ctx.workspace) if raw_cwd else ctx.workspace
        except Exception as e:
            return self._error("path_outside_workspace", str(e))

        if operation == "destroy" and not bool(args.get("confirm_high_risk", False)):
            return self._error(
                "high_risk_confirmation_required",
                "wp_env destroy requires confirm_high_risk=true.",
            )

        resolution = self._resolve_npx_binary()
        npx = resolution.path
        if not npx:
            return self._error(
                resolution.error_code or "binary_not_found",
                resolution.message or "Binary not found: npx",
                extra_data={"reason_code": "tool_runtime_capability_unavailable"},
            )

        argv = [npx, "-y", "@wordpress/env", operation]
        if operation == "run":
            run_args = args.get("run_args", [])
            if not isinstance(run_args, list) or not run_args:
                return self._error("invalid_arguments", "operation=run requires non-empty run_args")
            for item in run_args:
                clean = str(item or "").strip()
                if clean:
                    argv.append(clean)

        try:
            result = await run_command(
                argv,
                cwd=cwd,
                timeout_seconds=200,
                env=constrained_env(),
            )
        except Exception as e:
            return self._error("tool_runtime_error", f"wp_env failed: {e}")

        success = result.exit_code == 0 and not result.timed_out
        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.timed_out:
            output_parts.append("[timeout] command exceeded timeout")
        error_code: str | None = None
        if result.timed_out:
            error_code = "timeout_exceeded"
        elif result.exit_code != 0:
            error_code = "command_failed"

        return ToolResult(
            success=success,
            output="\n".join(output_parts).strip(),
            error=(
                "Command timed out"
                if result.timed_out
                else (f"Exit code: {result.exit_code}" if result.exit_code != 0 else None)
            ),
            data={
                "operation": operation,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "timed_out": result.timed_out,
                "truncated": result.truncated,
                "cwd": str(Path(cwd)),
                "error_code": error_code,
            },
        )

    @staticmethod
    def _error(code: str, message: str, *, extra_data: dict | None = None) -> ToolResult:
        data = {"error_code": code}
        if isinstance(extra_data, dict):
            data.update(extra_data)
        return ToolResult(
            success=False,
            output="",
            error=message,
            data=data,
        )
