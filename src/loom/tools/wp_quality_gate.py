"""Unified WordPress quality check runner."""

from __future__ import annotations

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

_CHECKS = frozenset({"wpcs", "plugin_check", "wp_scripts_lint", "wp_scripts_test"})


class WpQualityGateTool(Tool):
    """Run standardized WordPress quality checks in one tool call."""

    name = "wp_quality_gate"
    description = (
        "Run WordPress quality checks (WPCS/PHPCS, Plugin Check, @wordpress/scripts lint/test) "
        "and return a structured summary."
    )
    parameters = {
        "type": "object",
        "properties": {
            "checks": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Subset of checks: wpcs, plugin_check, "
                    "wp_scripts_lint, wp_scripts_test"
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Project directory relative to workspace.",
            },
            "fail_fast": {
                "type": "boolean",
                "description": "Stop after first failed check.",
            },
            "report_format": {
                "type": "string",
                "description": "text | json (summary formatting only)",
            },
        },
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
        return 600

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

        missing: list[ToolAvailabilityReason] = []
        available_checks: list[str] = []
        for check in sorted(_CHECKS):
            command = self._command_for(check)
            if command is None:
                missing.append(
                    ToolAvailabilityReason(
                        code="binary_not_found",
                        message=f"Missing binary for quality check '{check}'.",
                        metadata={"check": check},
                    ),
                )
                continue
            available_checks.append(check)

        if not available_checks:
            return ToolAvailabilityStatus(
                state="unavailable",
                reasons=tuple(missing),
                metadata={"available_checks": [], "requested_checks": sorted(_CHECKS)},
            )
        if missing:
            return ToolAvailabilityStatus(
                state="degraded",
                reasons=tuple(missing),
                metadata={
                    "available_checks": available_checks,
                    "requested_checks": sorted(_CHECKS),
                },
            )
        return ToolAvailabilityStatus(
            state="available",
            metadata={
                "available_checks": available_checks,
                "requested_checks": sorted(_CHECKS),
            },
        )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if not self._enabled:
            return self._error("feature_disabled", "WordPress tools are disabled by configuration.")
        if ctx.workspace is None:
            return self._error("path_outside_workspace", "No workspace set.")

        raw_cwd = str(args.get("cwd", "") or "").strip()
        try:
            cwd = self._resolve_path(raw_cwd, ctx.workspace) if raw_cwd else ctx.workspace
        except Exception as e:
            return self._error("path_outside_workspace", str(e))

        requested_checks = args.get("checks")
        checks: list[str] = []
        if isinstance(requested_checks, list):
            for item in requested_checks:
                check = str(item or "").strip().lower()
                if check in _CHECKS and check not in checks:
                    checks.append(check)
        if not checks:
            checks = ["wpcs", "plugin_check", "wp_scripts_lint"]

        fail_fast = bool(args.get("fail_fast", False))
        report_format = str(args.get("report_format", "text") or "text").strip().lower()
        if report_format not in {"text", "json"}:
            report_format = "text"

        env = constrained_env()
        summaries: list[dict] = []
        output_parts: list[str] = []

        for check in checks:
            command = self._command_for(check)
            if command is None:
                summaries.append({
                    "check": check,
                    "success": False,
                    "error_code": "binary_not_found",
                    "reason_code": "tool_runtime_capability_unavailable",
                    "message": f"Missing binary for check '{check}'",
                })
                if fail_fast:
                    break
                continue

            result = await run_command(
                command,
                cwd=cwd,
                timeout_seconds=300,
                env=env,
            )
            success = result.exit_code == 0 and not result.timed_out
            error_code: str | None = None
            if result.timed_out:
                error_code = "timeout_exceeded"
            elif result.exit_code != 0:
                error_code = "command_failed"
            summaries.append({
                "check": check,
                "success": success,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration_ms": result.duration_ms,
                "truncated": result.truncated,
                "error_code": error_code,
            })

            if result.stdout:
                output_parts.append(f"[{check}]\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"[{check} stderr]\n{result.stderr}")
            if result.timed_out:
                output_parts.append(f"[{check}] timed out")

            if fail_fast and not success:
                break

        all_success = all(bool(item.get("success")) for item in summaries) if summaries else False
        top_level_error_code: str | None = None
        top_level_reason_code: str | None = None
        if not all_success:
            if any(item.get("error_code") == "timeout_exceeded" for item in summaries):
                top_level_error_code = "timeout_exceeded"
            elif any(item.get("error_code") == "binary_not_found" for item in summaries):
                top_level_error_code = "tool_runtime_capability_unavailable"
                top_level_reason_code = "tool_runtime_capability_unavailable"
            else:
                top_level_error_code = "quality_gate_failed"
        if report_format == "json":
            rendered = ""
        else:
            passed = sum(1 for x in summaries if x.get("success"))
            lines = [
                f"WordPress quality gate: {passed}/{len(summaries)} passed.",
            ]
            for item in summaries:
                status = "ok" if item.get("success") else "fail"
                lines.append(f"- {item.get('check')}: {status}")
            rendered = "\n".join(lines)
            if output_parts:
                rendered = rendered + "\n\n" + "\n\n".join(output_parts)

        return ToolResult(
            success=all_success,
            output=rendered,
            error=None if all_success else "One or more quality checks failed.",
            data={
                "checks": summaries,
                "passed": sum(1 for x in summaries if x.get("success")),
                "total": len(summaries),
                "error_code": top_level_error_code,
                "reason_code": top_level_reason_code or "",
            },
        )

    def _resolve_binary_for(self, binary_name: str, *keys: str) -> str:
        resolution = resolve_binary(
            binary_name,
            override=configured_binary_override(
                self._binary_overrides,
                self.name,
                binary_name,
                *keys,
            ),
        )
        return resolution.path

    def _command_for(self, check: str) -> list[str] | None:
        if check == "wpcs":
            phpcs = self._resolve_binary_for("phpcs", "wpcs")
            if not phpcs:
                return None
            return [phpcs, "--standard=WordPress", "."]
        if check == "plugin_check":
            wp = self._resolve_binary_for("wp", "plugin_check")
            if not wp:
                return None
            return [wp, "plugin", "check", "."]
        if check == "wp_scripts_lint":
            npm = self._resolve_binary_for("npm", "wp_scripts_lint")
            if not npm:
                return None
            return [npm, "run", "lint"]
        if check == "wp_scripts_test":
            npm = self._resolve_binary_for("npm", "wp_scripts_test")
            if not npm:
                return None
            return [npm, "test", "--", "--watch=false"]
        return None

    @staticmethod
    def _error(code: str, message: str) -> ToolResult:
        return ToolResult(
            success=False,
            output="",
            error=message,
            data={"error_code": code},
        )
