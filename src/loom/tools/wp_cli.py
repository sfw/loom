"""Structured WordPress CLI wrapper with high-risk safeguards."""

from __future__ import annotations

import json
import logging
import shlex
import time

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
from loom.tools.tooling_common.wp_policy import assess_wp_cli_risk
from loom.utils.latency import log_latency_event

READ_ONLY_ACTIONS = frozenset({
    ("core", "version"),
    ("core", "check-update"),
    ("plugin", "list"),
    ("theme", "list"),
    ("option", "get"),
    ("user", "list"),
    ("post", "list"),
    ("db", "export"),
})

logger = logging.getLogger(__name__)


class WpCliTool(Tool):
    """Execute common WordPress CLI workflows with typed arguments."""

    name = "wp_cli"
    description = (
        "Run structured WordPress CLI operations "
        "(core/plugin/theme/option/user/post/db/search_replace) "
        "with risk-aware safeguards."
    )
    parameters = {
        "type": "object",
        "properties": {
            "group": {
                "type": "string",
                "description": "core | plugin | theme | option | user | post | db | search_replace",
            },
            "action": {
                "type": "string",
                "description": "Group-specific action (for search_replace use action='run').",
            },
            "args": {
                "type": "object",
                "description": "Typed argument object for the selected group/action.",
            },
            "path": {
                "type": "string",
                "description": "Optional WordPress install path relative to workspace.",
            },
            "confirm_high_risk": {
                "type": "boolean",
                "description": "Required true for high-risk operations.",
            },
        },
        "required": ["group", "action"],
    }

    def __init__(
        self,
        *,
        enabled: bool = True,
        high_risk_requires_confirmation: bool = True,
        binary_overrides: dict[str, str] | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._high_risk_requires_confirmation = bool(high_risk_requires_confirmation)
        self._binary_overrides = normalize_binary_overrides(binary_overrides or {})

    @property
    def is_mutating(self) -> bool:
        return True

    @property
    def timeout_seconds(self) -> int:
        return 180

    @property
    def high_risk_requires_confirmation(self) -> bool:
        return self._high_risk_requires_confirmation

    def _resolve_wp_binary(self):
        return resolve_binary(
            "wp",
            override=configured_binary_override(
                self._binary_overrides,
                self.name,
                "wp",
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
        resolution = self._resolve_wp_binary()
        if resolution.found:
            return ToolAvailabilityStatus(
                state="available",
                metadata={
                    "binary": "wp",
                    "binary_path": resolution.path,
                    "binary_source": resolution.source,
                },
            )
        return ToolAvailabilityStatus(
            state="unavailable",
            reasons=(
                ToolAvailabilityReason(
                    code=resolution.error_code or "binary_not_found",
                    message=resolution.message or "Binary not found: wp",
                    metadata=dict(resolution.metadata or {}),
                ),
            ),
            metadata={"binary": "wp"},
        )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        started = time.monotonic()
        if not self._enabled:
            return self._error("feature_disabled", "WordPress tools are disabled by configuration.")

        if ctx.workspace is None:
            return self._error("path_outside_workspace", "No workspace set.")

        group = str(args.get("group", "") or "").strip().lower()
        action = str(args.get("action", "") or "").strip().lower()
        payload = args.get("args", {})
        if not isinstance(payload, dict):
            return self._error("invalid_arguments", "'args' must be an object.")

        if group == "search_replace" and action == "run":
            # keep explicit action for schema consistency
            pass
        elif not action:
            return self._error("invalid_arguments", "'action' is required.")

        raw_path = str(args.get("path", "") or "").strip()
        try:
            wp_path = self._resolve_path(raw_path, ctx.workspace) if raw_path else ctx.workspace
        except Exception as e:
            return self._error("path_outside_workspace", str(e))

        resolution = self._resolve_wp_binary()
        binary_path = resolution.path
        if not binary_path:
            return self._error(
                resolution.error_code or "binary_not_found",
                resolution.message or "Binary not found: wp",
                extra_data={"reason_code": "tool_runtime_capability_unavailable"},
            )

        risk = assess_wp_cli_risk(group, action, payload)
        if (
            risk
            and self._high_risk_requires_confirmation
            and not bool(args.get("confirm_high_risk", False))
        ):
            log_latency_event(
                logger,
                event="wp_cli_high_risk_blocked",
                duration_seconds=time.monotonic() - started,
                fields={
                    "tool_name": self.name,
                    "group": group,
                    "action": action,
                },
            )
            return ToolResult(
                success=False,
                output="",
                error="High-risk WordPress operation requires confirmation.",
                data={
                    "error_code": "high_risk_confirmation_required",
                    "risk_level": risk.risk_level,
                    "action_class": risk.action_class,
                    "impact_preview": risk.impact_preview,
                    "consequences": risk.consequences,
                },
            )

        try:
            wp_args = self._build_wp_args(group, action, payload)
        except ValueError as e:
            return self._error("invalid_arguments", str(e))

        argv = [binary_path, *wp_args, f"--path={wp_path}"]

        try:
            result = await run_command(
                argv,
                cwd=wp_path,
                timeout_seconds=120,
                env=constrained_env(),
            )
        except Exception as e:
            return self._error("tool_runtime_error", f"wp_cli failed: {e}")

        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")
        if result.timed_out:
            output_parts.append("[timeout] command exceeded timeout")

        parsed: dict | None = None
        if result.stdout:
            stripped = result.stdout.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None

        success = result.exit_code == 0 and not result.timed_out
        error_code: str | None = None
        if result.timed_out:
            error_code = "timeout_exceeded"
        elif result.exit_code != 0:
            error_code = "command_failed"
        log_latency_event(
            logger,
            event="wp_cli_invoked",
            duration_seconds=time.monotonic() - started,
            fields={
                "tool_name": self.name,
                "group": group,
                "action": action,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            },
        )
        return ToolResult(
            success=success,
            output="\n".join(output_parts).strip(),
            error=(
                "Command timed out"
                if result.timed_out
                else (f"Exit code: {result.exit_code}" if result.exit_code != 0 else None)
            ),
            data={
                "group": group,
                "action": action,
                "risk_level": risk.risk_level if risk else "normal",
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "timed_out": result.timed_out,
                "truncated": result.truncated,
                "command": self._redact_command(argv),
                "parsed": parsed,
                "is_read_only": (group, action) in READ_ONLY_ACTIONS,
                "error_code": error_code,
            },
        )

    def _build_wp_args(self, group: str, action: str, args: dict) -> list[str]:
        if group == "core":
            return self._build_core_args(action, args)
        if group == "plugin":
            return self._build_plugin_args(action, args)
        if group == "theme":
            return self._build_theme_args(action, args)
        if group == "option":
            return self._build_option_args(action, args)
        if group == "user":
            return self._build_user_args(action, args)
        if group == "post":
            return self._build_post_args(action, args)
        if group == "db":
            return self._build_db_args(action, args)
        if group == "search_replace":
            return self._build_search_replace_args(action, args)
        raise ValueError(f"Unsupported group: {group}")

    @staticmethod
    def _build_core_args(action: str, args: dict) -> list[str]:
        if action in {"version", "check-update", "update"}:
            return ["core", action]
        if action == "install":
            url = str(args.get("url", "") or "").strip()
            title = str(args.get("title", "") or "").strip()
            admin_user = str(args.get("admin_user", "") or "").strip()
            admin_password = str(args.get("admin_password", "") or "").strip()
            admin_email = str(args.get("admin_email", "") or "").strip()
            if not all([url, title, admin_user, admin_password, admin_email]):
                raise ValueError(
                    (
                        "core install requires args: url, title, admin_user, "
                        "admin_password, admin_email"
                    ),
                )
            return [
                "core",
                "install",
                f"--url={url}",
                f"--title={title}",
                f"--admin_user={admin_user}",
                f"--admin_password={admin_password}",
                f"--admin_email={admin_email}",
            ]
        raise ValueError(f"Unsupported core action: {action}")

    @staticmethod
    def _build_plugin_args(action: str, args: dict) -> list[str]:
        if action == "list":
            return ["plugin", "list", "--format=json"]
        slug = str(args.get("slug", "") or args.get("name", "")).strip()
        if action in {"install", "activate", "deactivate", "delete", "update"}:
            if not slug:
                raise ValueError(f"plugin {action} requires args.slug")
            built = ["plugin", action, slug]
            if action == "install" and bool(args.get("activate", False)):
                built.append("--activate")
            return built
        raise ValueError(f"Unsupported plugin action: {action}")

    @staticmethod
    def _build_theme_args(action: str, args: dict) -> list[str]:
        if action == "list":
            return ["theme", "list", "--format=json"]
        slug = str(args.get("slug", "") or args.get("name", "")).strip()
        if action in {"install", "activate", "delete", "update"}:
            if not slug:
                raise ValueError(f"theme {action} requires args.slug")
            built = ["theme", action, slug]
            if action == "install" and bool(args.get("activate", False)):
                built.append("--activate")
            return built
        raise ValueError(f"Unsupported theme action: {action}")

    @staticmethod
    def _build_option_args(action: str, args: dict) -> list[str]:
        key = str(args.get("key", "") or "").strip()
        if not key:
            raise ValueError(f"option {action} requires args.key")
        if action == "get":
            return ["option", "get", key]
        if action == "update":
            value = str(args.get("value", "") or "")
            return ["option", "update", key, value]
        if action == "delete":
            return ["option", "delete", key]
        raise ValueError(f"Unsupported option action: {action}")

    @staticmethod
    def _build_user_args(action: str, args: dict) -> list[str]:
        if action == "list":
            return ["user", "list", "--format=json"]
        if action == "create":
            user_login = str(args.get("user_login", "") or "").strip()
            user_email = str(args.get("user_email", "") or "").strip()
            role = str(args.get("role", "") or "").strip()
            if not user_login or not user_email:
                raise ValueError("user create requires args.user_login and args.user_email")
            built = ["user", "create", user_login, user_email]
            if role:
                built.append(f"--role={role}")
            return built
        if action == "delete":
            user = str(args.get("id", "") or args.get("user", "")).strip()
            if not user:
                raise ValueError("user delete requires args.id or args.user")
            built = ["user", "delete", user]
            if bool(args.get("reassign", False)):
                reassign = str(args.get("reassign_to", "") or "").strip()
                if reassign:
                    built.append(f"--reassign={reassign}")
            return built
        raise ValueError(f"Unsupported user action: {action}")

    @staticmethod
    def _build_post_args(action: str, args: dict) -> list[str]:
        if action == "list":
            return ["post", "list", "--format=json"]
        if action == "create":
            title = str(args.get("post_title", "") or "").strip()
            content = str(args.get("post_content", "") or "")
            if not title:
                raise ValueError("post create requires args.post_title")
            built = ["post", "create", f"--post_title={title}", f"--post_content={content}"]
            status = str(args.get("post_status", "") or "").strip()
            if status:
                built.append(f"--post_status={status}")
            return built
        if action == "update":
            post_id = str(args.get("id", "") or "").strip()
            if not post_id:
                raise ValueError("post update requires args.id")
            built = ["post", "update", post_id]
            if "post_title" in args:
                built.append(f"--post_title={args.get('post_title', '')}")
            if "post_content" in args:
                built.append(f"--post_content={args.get('post_content', '')}")
            if len(built) == 3:
                raise ValueError(
                    "post update requires at least one field "
                    "(post_title/post_content)"
                )
            return built
        if action == "delete":
            post_id = str(args.get("id", "") or "").strip()
            if not post_id:
                raise ValueError("post delete requires args.id")
            built = ["post", "delete", post_id]
            if bool(args.get("force", False)):
                built.append("--force")
            return built
        raise ValueError(f"Unsupported post action: {action}")

    @staticmethod
    def _build_db_args(action: str, args: dict) -> list[str]:
        if action in {"reset", "drop"}:
            return ["db", action, "--yes"]
        if action == "export":
            filename = str(args.get("file", "") or "").strip()
            return ["db", "export", filename] if filename else ["db", "export"]
        if action == "import":
            filename = str(args.get("file", "") or "").strip()
            if not filename:
                raise ValueError("db import requires args.file")
            return ["db", "import", filename]
        raise ValueError(f"Unsupported db action: {action}")

    @staticmethod
    def _build_search_replace_args(action: str, args: dict) -> list[str]:
        if action != "run":
            raise ValueError("search_replace only supports action='run'")
        old = str(args.get("old", "") or "")
        new = str(args.get("new", "") or "")
        if not old:
            raise ValueError("search_replace requires args.old")
        built = ["search-replace", old, new]
        if bool(args.get("all_tables", False)):
            built.append("--all-tables")
        dry_run = bool(args.get("dry_run", True))
        if dry_run:
            built.append("--dry-run")
        if bool(args.get("precise", False)):
            built.append("--precise")
        if bool(args.get("skip_columns")):
            skip_columns = str(args.get("skip_columns", "") or "").strip()
            if skip_columns:
                built.append(f"--skip-columns={skip_columns}")
        return built

    @staticmethod
    def _redact_command(argv: list[str]) -> str:
        if not argv:
            return ""
        # best effort: redact password-like flags
        redacted: list[str] = []
        for token in argv:
            if "password" in token.lower():
                redacted.append("--admin_password=<redacted>")
            else:
                redacted.append(token)
        return shlex.join(redacted)

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
