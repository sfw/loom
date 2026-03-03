"""Shared provider-runner implementation for coding-agent tools."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

from loom.tools.registry import Tool, ToolContext, ToolResult
from loom.tools.tooling_common.command_runner import constrained_env, run_command
from loom.tools.tooling_common.version_matrix import (
    PROVIDER_SPECS,
    ProviderSpec,
    normalize_provider,
    parse_semver_tuple,
    version_supported,
)
from loom.utils.latency import log_latency_event

logger = logging.getLogger(__name__)

_PROVIDER_SANDBOX_FLAGS = {
    "codex": {
        "read_only": "--sandbox=read-only",
        "workspace_write": "--sandbox=workspace-write",
        "unrestricted": "--sandbox=danger-full-access",
    },
}

_PROVIDER_APPROVAL_FLAGS = {
    "codex": {
        "untrusted": "--ask-for-approval=untrusted",
        "on_failure": "--ask-for-approval=on-failure",
        "on_request": "--ask-for-approval=on-request",
        "never": "--ask-for-approval=never",
    },
}

_PROVIDER_NETWORK_FLAGS = {
    # Keep empty for currently supported provider CLIs.
    # New flags can be added when provider docs confirm support.
}

_PROVIDER_OUTPUT_FLAGS = {
    "codex": {
        "json": "--json",
        "stream": "--json",
    },
    "claude_code": {
        "json": "--output-format=json",
        "stream": "--output-format=stream-json",
    },
    "opencode": {
        "json": "--format=json",
        "stream": "--format=json",
    },
}

_PROVIDER_EXTRA_ARG_ALLOWLIST = {
    "codex": frozenset({
        "--model",
        "--profile",
        "--color",
        "--cd",
        "--add-dir",
        "--skip-git-repo-check",
    }),
    "claude_code": frozenset({
        "--model",
        "--agent",
        "--system-prompt",
        "--append-system-prompt",
        "--allowed-tools",
        "--disallowed-tools",
    }),
    "opencode": frozenset({
        "--model",
        "--agent",
        "--variant",
        "--thinking",
        "--title",
        "--command",
        "--attach",
        "--dir",
    }),
}

_PROVIDER_AUTH_ENV_PREFIXES = (
    "ANTHROPIC_",
    "OPENAI_",
    "AZURE_OPENAI_",
    "OPENROUTER_",
    "OPENCODE_",
    "GEMINI_",
    "GOOGLE_",
    "XAI_",
    "MISTRAL_",
    "DEEPSEEK_",
    "COHERE_",
)

_PROVIDER_RUNTIME_ENV_KEYS = frozenset({
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
})

_PROVIDER_TOOL_NAMES = {
    "codex": "openai_codex",
    "claude_code": "claude_code",
    "opencode": "opencode",
}


class ProviderAgentTool(Tool):
    """Shared implementation for provider-specific coding-agent tools."""

    __loom_register__ = False
    _provider = ""

    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Prompt text passed to the external agent.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory (relative to workspace).",
            },
            "sandbox_mode": {
                "type": "string",
                "description": "read_only | workspace_write | unrestricted",
            },
            "network_mode": {
                "type": "string",
                "description": "on | off",
            },
            "approval_mode": {
                "type": "string",
                "description": "untrusted | on_failure | on_request | never",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Command timeout in seconds.",
            },
            "output_mode": {
                "type": "string",
                "description": "text | json | stream",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional allowlisted passthrough CLI args.",
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional provider alias for compatibility. "
                    "Must match this tool's provider."
                ),
            },
        },
        "required": ["prompt"],
    }

    def __init__(
        self,
        *,
        enabled: bool = True,
        allowed_providers: list[str] | None = None,
        max_timeout_seconds: int = 1800,
        default_network_mode: str = "on",
    ) -> None:
        self._enabled = bool(enabled)
        normalized_allowed: list[str] = []
        for item in allowed_providers or list(PROVIDER_SPECS.keys()):
            provider = normalize_provider(item)
            if provider in PROVIDER_SPECS and provider not in normalized_allowed:
                normalized_allowed.append(provider)
        self._allowed_providers = normalized_allowed or list(PROVIDER_SPECS.keys())
        self._max_timeout_seconds = max(30, int(max_timeout_seconds or 1800))
        mode = str(default_network_mode or "on").strip().lower()
        self._default_network_mode = mode if mode in {"on", "off"} else "on"

    @property
    def is_mutating(self) -> bool:
        return True

    @property
    def timeout_seconds(self) -> int:
        # Per-call timeout parameter is additionally bounded by config.
        return max(30, self._max_timeout_seconds)

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        started = time.monotonic()
        provider = self._provider
        log_latency_event(
            logger,
            event="agent_provider_run_started",
            duration_seconds=0.0,
            fields={
                "tool_name": self.name,
                "provider": provider or "unknown",
            },
        )

        def _complete(
            result: ToolResult,
            *,
            exit_code: int | None = None,
            timed_out: bool | None = None,
        ) -> ToolResult:
            fields: dict[str, object] = {
                "tool_name": self.name,
                "provider": provider or "unknown",
                "success": bool(result.success),
            }
            if exit_code is not None:
                fields["exit_code"] = int(exit_code)
            if timed_out is not None:
                fields["timed_out"] = bool(timed_out)
            log_latency_event(
                logger,
                event="agent_provider_run_completed",
                duration_seconds=time.monotonic() - started,
                fields=fields,
            )
            return result

        if not self._enabled:
            return _complete(
                self._error(
                    "feature_disabled",
                    "Agent tools are disabled by configuration.",
                ),
            )

        compatibility_provider = str(args.get("provider", "") or "").strip()
        if compatibility_provider:
            requested_provider = normalize_provider(compatibility_provider)
            if requested_provider not in PROVIDER_SPECS:
                return _complete(
                    self._error(
                        "unknown_provider",
                        f"Unknown provider: {compatibility_provider}",
                    ),
                )
            if requested_provider != provider:
                target_tool = _PROVIDER_TOOL_NAMES.get(requested_provider, requested_provider)
                return _complete(
                    self._error(
                        "invalid_arguments",
                        (
                            f"Tool '{self.name}' only supports provider '{provider}'. "
                            f"Use '/tool {target_tool} ...' for provider "
                            f"'{requested_provider}'."
                        ),
                    ),
                )

        spec = PROVIDER_SPECS.get(provider)
        if spec is None:
            return _complete(
                self._error("unknown_provider", f"Unknown provider: {provider}"),
            )
        if provider not in self._allowed_providers:
            return _complete(
                self._error(
                    "provider_not_allowed",
                    f"Provider '{provider}' is not allowed by configuration.",
                ),
            )

        prompt = str(args.get("prompt", "") or "").strip()
        if not prompt:
            return _complete(self._error("invalid_arguments", "'prompt' is required."))

        if ctx.workspace is None:
            return _complete(
                self._error("path_outside_workspace", "No workspace set."),
            )

        raw_cwd = str(args.get("cwd", "") or "").strip()
        try:
            cwd = self._resolve_path(raw_cwd, ctx.workspace) if raw_cwd else ctx.workspace
        except Exception as e:
            return _complete(
                self._error("path_outside_workspace", str(e)),
            )

        sandbox_mode = str(
            args.get("sandbox_mode", spec.default_sandbox_mode) or spec.default_sandbox_mode,
        ).strip().lower()
        if sandbox_mode not in spec.supports_sandbox_modes:
            return _complete(
                self._error(
                    "unsupported_mode_combination",
                    f"Provider '{provider}' does not support sandbox_mode='{sandbox_mode}'.",
                ),
            )

        network_mode = str(
            args.get("network_mode", self._default_network_mode) or self._default_network_mode,
        ).strip().lower()
        if network_mode not in {"on", "off"}:
            return _complete(
                self._error(
                    "unsupported_mode_combination",
                    "network_mode must be 'on' or 'off'.",
                ),
            )
        if network_mode not in spec.supports_network_modes:
            code = "network_disabled" if network_mode == "off" else "unsupported_mode_combination"
            return _complete(
                self._error(
                    code,
                    f"Provider '{provider}' does not support network_mode='{network_mode}'.",
                ),
            )

        approval_mode = str(
            args.get("approval_mode", spec.default_approval_mode) or spec.default_approval_mode,
        ).strip().lower()
        if approval_mode not in spec.supports_approval_modes:
            return _complete(
                self._error(
                    "unsupported_mode_combination",
                    f"Provider '{provider}' does not support approval_mode='{approval_mode}'.",
                ),
            )

        output_mode = str(args.get("output_mode", "text") or "text").strip().lower()
        if output_mode not in spec.supports_output_modes:
            return _complete(
                self._error(
                    "unsupported_mode_combination",
                    f"Provider '{provider}' does not support output_mode='{output_mode}'.",
                ),
            )

        timeout_seconds = args.get("timeout_seconds", 300)
        try:
            timeout = int(timeout_seconds)
        except (TypeError, ValueError):
            timeout = 300
        timeout = max(1, min(timeout, self._max_timeout_seconds))

        binary_path = shutil.which(spec.binary)
        if not binary_path:
            return _complete(
                self._error("binary_not_found", f"Binary not found: {spec.binary}"),
            )
        version_gate = await self._enforce_version_gate(provider, binary_path, spec)
        if version_gate is not None:
            return _complete(version_gate)

        argv_prefix = [binary_path]
        if provider == "codex":
            approval_flag = _PROVIDER_APPROVAL_FLAGS.get(provider, {}).get(approval_mode)
            if approval_flag:
                # Codex approval policy is accepted as a global flag, not an exec subcommand flag.
                argv_prefix.append(approval_flag)

        argv = [*argv_prefix, *spec.run_base_args]
        if provider == "claude_code":
            permission_mode = self._claude_permission_mode(
                sandbox_mode=sandbox_mode,
                approval_mode=approval_mode,
            )
            argv.append(f"--permission-mode={permission_mode}")
            if sandbox_mode == "unrestricted":
                argv.append("--dangerously-skip-permissions")
        elif provider == "codex":
            sandbox_flag = _PROVIDER_SANDBOX_FLAGS.get(provider, {}).get(sandbox_mode)
            if sandbox_flag:
                argv.append(sandbox_flag)
            skip_git_repo_check = bool(args.get("skip_git_repo_check", True))
            if skip_git_repo_check and "--skip-git-repo-check" not in argv:
                argv.append("--skip-git-repo-check")

        network_flag = _PROVIDER_NETWORK_FLAGS.get(provider, {}).get(network_mode)
        if network_flag:
            argv.append(network_flag)

        output_flag = _PROVIDER_OUTPUT_FLAGS.get(provider, {}).get(output_mode)
        if output_flag:
            argv.append(output_flag)

        passthrough = args.get("args", [])
        if isinstance(passthrough, list):
            allowed = _PROVIDER_EXTRA_ARG_ALLOWLIST.get(provider, frozenset())
            for raw in passthrough:
                value = str(raw or "").strip()
                if not value:
                    continue
                head = value.split("=", 1)[0]
                if head in allowed:
                    argv.append(value)

        argv.append(prompt)

        try:
            result = await run_command(
                argv,
                cwd=cwd,
                timeout_seconds=timeout,
                env=constrained_env(extra=self._provider_env_overrides()),
            )
        except FileNotFoundError:
            return _complete(
                self._error("binary_not_found", f"Binary not found: {spec.binary}"),
            )
        except Exception as e:
            return _complete(
                self._error("tool_runtime_error", f"Failed to execute '{provider}': {e}"),
            )

        if result.timed_out:
            output_parts: list[str] = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[stderr]\n{result.stderr}")
            output_parts.append("[timeout] command exceeded timeout")
            return _complete(
                ToolResult(
                    success=False,
                    output="\n".join(output_parts).strip(),
                    error="Command timed out",
                    data={
                        "error_code": "timeout_exceeded",
                        "provider": provider,
                        "exit_code": result.exit_code,
                        "duration_ms": result.duration_ms,
                        "timed_out": True,
                        "truncated": result.truncated,
                        "provider_command": self._redacted_command(argv),
                    },
                ),
                exit_code=result.exit_code,
                timed_out=True,
            )

        parsed_payload = None
        if output_mode in {"json", "stream"}:
            parsed_payload, parse_ok = self._parse_json_or_jsonl(result.stdout or "")
            if output_mode == "json" and not parse_ok:
                return _complete(
                    ToolResult(
                        success=False,
                        output=result.stdout,
                        error="Provider output was not valid JSON.",
                        data={
                            "error_code": "output_parse_error",
                            "provider": provider,
                            "exit_code": result.exit_code,
                            "duration_ms": result.duration_ms,
                            "timed_out": result.timed_out,
                            "truncated": result.truncated,
                            "provider_command": self._redacted_command(argv),
                        },
                    ),
                    exit_code=result.exit_code,
                    timed_out=result.timed_out,
                )

        success = result.exit_code == 0 and not result.timed_out
        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")

        data: dict[str, object] = {
            "provider": provider,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
            "provider_command": self._redacted_command(argv),
            "parsed_payload": parsed_payload,
            "cwd": str(Path(cwd)),
            "sandbox_mode": sandbox_mode,
            "network_mode": network_mode,
            "approval_mode": approval_mode,
            "output_mode": output_mode,
        }
        if result.exit_code != 0:
            data["error_code"] = "command_failed"

        return _complete(
            ToolResult(
                success=success,
                output="\n".join(output_parts).strip(),
                error=(
                    self._format_command_failure_error(
                        result.exit_code,
                        result.stderr,
                        result.stdout,
                    )
                    if result.exit_code != 0
                    else None
                ),
                data=data,
            ),
            exit_code=result.exit_code,
            timed_out=result.timed_out,
        )

    async def _enforce_version_gate(
        self,
        provider: str,
        binary_path: str,
        spec: ProviderSpec,
    ) -> ToolResult | None:
        minimum = spec.min_supported_version
        if minimum is None:
            return None
        try:
            version_result = await run_command(
                [binary_path, *spec.version_args],
                timeout_seconds=8,
                env=constrained_env(extra=self._provider_env_overrides()),
            )
        except Exception as e:
            return self._error(
                "unsupported_version",
                f"Could not determine {provider} version: {e}",
            )
        version_text = (
            (version_result.stdout or "")
            + "\n"
            + (version_result.stderr or "")
        ).strip()
        parsed_version = parse_semver_tuple(version_text)
        if version_supported(parsed_version, minimum):
            return None
        expected = ".".join(str(part) for part in minimum)
        actual = version_text or "unknown"
        return self._error(
            "unsupported_version",
            (
                f"Provider '{provider}' version is unsupported. "
                f"Installed: {actual}. Minimum required: {expected}."
            ),
        )

    @staticmethod
    def _provider_env_overrides() -> dict[str, str]:
        """Allow provider auth/runtime env vars through constrained subprocess env."""
        extra: dict[str, str] = {}
        for key, value in os.environ.items():
            if not value:
                continue
            if key in _PROVIDER_RUNTIME_ENV_KEYS:
                extra[key] = value
                continue
            if any(key.startswith(prefix) for prefix in _PROVIDER_AUTH_ENV_PREFIXES):
                extra[key] = value
        return extra

    @staticmethod
    def _parse_json_or_jsonl(raw: str) -> tuple[object | None, bool]:
        """Parse either one JSON value or newline-delimited JSON values."""
        text = str(raw or "").strip()
        if not text:
            return None, True
        try:
            return json.loads(text), True
        except json.JSONDecodeError:
            pass

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None, True

        events: list[object] = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                return None, False
        return events, True

    @staticmethod
    def _redacted_command(argv: list[str]) -> str:
        if not argv:
            return ""
        if len(argv) == 1:
            return argv[0]
        # Redact trailing prompt argument.
        safe = [*argv[:-1], "<prompt>"]
        return " ".join(safe)

    @staticmethod
    def _claude_permission_mode(
        *,
        sandbox_mode: str,
        approval_mode: str,
    ) -> str:
        """Map canonical safety knobs to Claude Code permission modes."""
        if sandbox_mode == "unrestricted":
            return "bypassPermissions"
        if approval_mode == "never":
            return "dontAsk"
        return "default"

    @staticmethod
    def _format_command_failure_error(
        exit_code: int,
        stderr: str,
        stdout: str,
    ) -> str:
        """Return a compact failure summary with first stderr/stdout detail."""
        for stream in (stderr, stdout):
            text = str(stream or "").strip()
            if not text:
                continue
            first = text.splitlines()[0].strip()
            if not first:
                continue
            if len(first) > 180:
                first = f"{first[:177]}..."
            return f"Exit code: {int(exit_code)} ({first})"
        return f"Exit code: {int(exit_code)}"

    @staticmethod
    def _error(
        code: str,
        message: str,
        *,
        extra_data: dict | None = None,
    ) -> ToolResult:
        data = {"error_code": code}
        if isinstance(extra_data, dict):
            data.update(extra_data)
        return ToolResult(
            success=False,
            output="",
            error=message,
            data=data,
        )
