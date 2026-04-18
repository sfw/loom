"""WordPress block scaffolding wrapper for @wordpress/create-block."""

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


class WpScaffoldBlockTool(Tool):
    """Scaffold Gutenberg block projects with safe path controls."""

    name = "wp_scaffold_block"
    description = (
        "Scaffold a new Gutenberg block plugin via @wordpress/create-block with "
        "optional wp-env bootstrapping."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Block/project name.",
            },
            "variant": {
                "type": "string",
                "description": "Optional scaffold variant/template.",
            },
            "namespace": {
                "type": "string",
                "description": "Optional namespace.",
            },
            "target_dir": {
                "type": "string",
                "description": "Target directory relative to workspace.",
            },
            "with_wp_env": {
                "type": "boolean",
                "description": "If true, start wp-env after scaffolding.",
            },
            "allow_overwrite": {
                "type": "boolean",
                "description": "Allow existing target directory.",
            },
        },
        "required": ["name"],
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
        return 300

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

        name = str(args.get("name", "") or "").strip()
        if not name:
            return self._error("invalid_arguments", "'name' is required.")

        raw_target = str(args.get("target_dir", "") or "").strip()
        try:
            if raw_target:
                target = self._resolve_path(raw_target, ctx.workspace)
            else:
                target = ctx.workspace / name
            self._verify_within_workspace(target, ctx.workspace)
        except Exception as e:
            return self._error("path_outside_workspace", str(e))

        allow_overwrite = bool(args.get("allow_overwrite", False))
        if target.exists() and not allow_overwrite:
            return self._error(
                "path_exists",
                f"Target path already exists: {target}. Set allow_overwrite=true to proceed.",
            )

        resolution = self._resolve_npx_binary()
        npx = resolution.path
        if not npx:
            return self._error(
                resolution.error_code or "binary_not_found",
                resolution.message or "Binary not found: npx",
                extra_data={"reason_code": "tool_runtime_capability_unavailable"},
            )

        argv = [npx, "-y", "@wordpress/create-block", str(target.name)]
        variant = str(args.get("variant", "") or "").strip()
        namespace = str(args.get("namespace", "") or "").strip()
        if variant:
            argv.extend(["--variant", variant])
        if namespace:
            argv.extend(["--namespace", namespace])
        # Skip interactive prompts by default.
        argv.append("--no-wp-scripts")

        try:
            scaffold_result = await run_command(
                argv,
                cwd=target.parent,
                timeout_seconds=240,
                env=constrained_env(),
            )
        except Exception as e:
            return self._error("tool_runtime_error", f"wp_scaffold_block failed: {e}")

        output_parts: list[str] = []
        if scaffold_result.stdout:
            output_parts.append(scaffold_result.stdout)
        if scaffold_result.stderr:
            output_parts.append(f"[stderr]\n{scaffold_result.stderr}")

        success = scaffold_result.exit_code == 0 and not scaffold_result.timed_out
        error_code: str | None = None
        if success and bool(args.get("with_wp_env", False)):
            env_cmd = [npx, "-y", "@wordpress/env", "start"]
            env_result = await run_command(
                env_cmd,
                cwd=target,
                timeout_seconds=180,
                env=constrained_env(),
            )
            if env_result.stdout:
                output_parts.append("\n[wp_env]\n" + env_result.stdout)
            if env_result.stderr:
                output_parts.append("\n[wp_env stderr]\n" + env_result.stderr)
            success = success and env_result.exit_code == 0 and not env_result.timed_out
            if env_result.timed_out:
                error_code = "timeout_exceeded"
            elif env_result.exit_code != 0:
                error_code = "command_failed"
        if error_code is None:
            if scaffold_result.timed_out:
                error_code = "timeout_exceeded"
            elif scaffold_result.exit_code != 0:
                error_code = "command_failed"

        return ToolResult(
            success=success,
            output="\n".join(output_parts).strip(),
            error=None if success else "Block scaffold failed",
            files_changed=[str(target.relative_to(ctx.workspace))] if target.exists() else [],
            data={
                "target_dir": str(target),
                "exit_code": scaffold_result.exit_code,
                "duration_ms": scaffold_result.duration_ms,
                "timed_out": scaffold_result.timed_out,
                "truncated": scaffold_result.truncated,
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
