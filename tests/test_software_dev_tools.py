"""Tests for software development tool integrations (agent + WordPress)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from loom.tools.claude_code import ClaudeCodeTool
from loom.tools.openai_codex import OpenAICodexTool
from loom.tools.opencode import OpenCodeTool
from loom.tools.registry import ToolContext
from loom.tools.tooling_common.command_runner import CommandRunResult
from loom.tools.tooling_common.version_matrix import PROVIDER_SPECS
from loom.tools.wp_cli import WpCliTool
from loom.tools.wp_env import WpEnvTool
from loom.tools.wp_quality_gate import WpQualityGateTool
from loom.tools.wp_scaffold_block import WpScaffoldBlockTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "project").mkdir()
    return tmp_path


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace)


class TestProviderAgentTools:
    @pytest.mark.parametrize(
        ("tool_cls", "provider"),
        [
            (OpenAICodexTool, "codex"),
            (ClaudeCodeTool, "claude_code"),
            (OpenCodeTool, "opencode"),
        ],
    )
    async def test_disabled_returns_feature_disabled(
        self,
        tool_cls,
        provider: str,
        ctx: ToolContext,
    ):
        tool = tool_cls(enabled=False)
        result = await tool.execute({"prompt": "hello", "provider": provider}, ctx)
        assert not result.success
        assert result.data["error_code"] == "feature_disabled"

    @pytest.mark.parametrize(
        ("tool_cls", "binary_name"),
        [
            (OpenAICodexTool, "codex"),
            (ClaudeCodeTool, "claude"),
            (OpenCodeTool, "opencode"),
        ],
    )
    async def test_missing_binary_reports_not_found(
        self,
        monkeypatch,
        tool_cls,
        binary_name: str,
        ctx: ToolContext,
    ):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: None,
        )
        tool = tool_cls(enabled=True)
        result = await tool.execute({"prompt": "hello"}, ctx)
        assert not result.success
        assert result.data["error_code"] == "binary_not_found"
        assert binary_name in str(result.error or "")

    async def test_codex_availability_uses_configured_binary_override(
        self,
        monkeypatch,
        ctx: ToolContext,
    ):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda value: "/custom/codex" if value == "custom-codex" else None,
        )
        tool = OpenAICodexTool(
            enabled=True,
            binary_overrides={"codex": "custom-codex"},
        )
        status = tool.availability(execution_surface="cli")
        assert status.runnable is True
        assert status.metadata["binary_path"] == "/custom/codex"

    async def test_codex_execute_uses_configured_binary_override(
        self,
        monkeypatch,
        ctx: ToolContext,
    ):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda value: "/custom/codex" if value == "custom-codex" else None,
        )
        seen_argv: list[list[str]] = []

        async def _fake_run(argv, **_kwargs):
            seen_argv.append(list(argv))
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="codex 1.2.3",
                    stderr="",
                    duration_ms=12,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=42,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = OpenAICodexTool(
            enabled=True,
            binary_overrides={"codex": "custom-codex"},
        )
        result = await tool.execute({"prompt": "Summarize repo"}, ctx)
        assert result.success
        assert seen_argv[-1][0] == "/custom/codex"

    async def test_rejects_unsupported_off_network_mode(self, ctx: ToolContext):
        tool = ClaudeCodeTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "hello",
                "network_mode": "off",
            },
            ctx,
        )
        assert not result.success
        assert result.data["error_code"] == "network_disabled"

    async def test_executes_with_mocked_runner(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/codex",
        )

        async def _fake_run(argv, **_kwargs):
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="codex 1.2.3",
                    stderr="",
                    duration_ms=12,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout='{"ok": true}',
                stderr="",
                duration_ms=42,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = OpenAICodexTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "Summarize repo",
                "output_mode": "json",
            },
            ctx,
        )
        assert result.success
        assert result.data["parsed_payload"] == {"ok": True}

    async def test_codex_uses_global_approval_flag_and_no_network_flag(
        self,
        monkeypatch,
        ctx: ToolContext,
    ):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/codex",
        )
        seen_argv: list[list[str]] = []

        async def _fake_run(argv, **_kwargs):
            seen_argv.append(list(argv))
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="codex-cli 0.98.0",
                    stderr="",
                    duration_ms=12,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=42,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = OpenAICodexTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "Summarize repo",
            },
            ctx,
        )
        assert result.success
        run_argv = seen_argv[-1]
        assert run_argv[:3] == [
            "/usr/bin/codex",
            "--ask-for-approval=on-request",
            "exec",
        ]
        assert "--sandbox=workspace-write" in run_argv
        assert "--skip-git-repo-check" in run_argv
        assert not any(item.startswith("--network-access=") for item in run_argv)

    async def test_codex_can_disable_skip_git_repo_check_flag(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/codex",
        )
        seen_argv: list[list[str]] = []

        async def _fake_run(argv, **_kwargs):
            seen_argv.append(list(argv))
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="codex-cli 0.98.0",
                    stderr="",
                    duration_ms=12,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=42,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = OpenAICodexTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "Summarize repo",
                "skip_git_repo_check": False,
            },
            ctx,
        )
        assert result.success
        run_argv = seen_argv[-1]
        assert "--skip-git-repo-check" not in run_argv

    async def test_opencode_uses_supported_flags_only(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/opencode",
        )
        seen_argv: list[list[str]] = []

        async def _fake_run(argv, **_kwargs):
            seen_argv.append(list(argv))
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="1.2.15",
                    stderr="",
                    duration_ms=12,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout='{"type":"result","ok":true}',
                stderr="",
                duration_ms=42,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = OpenCodeTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "Summarize repo",
                "output_mode": "json",
            },
            ctx,
        )
        assert result.success
        run_argv = seen_argv[-1]
        assert run_argv[:2] == ["/usr/bin/opencode", "run"]
        assert "--format=json" in run_argv
        assert not any(item.startswith("--sandbox=") for item in run_argv)
        assert not any(item.startswith("--ask-for-approval=") for item in run_argv)
        assert not any(item.startswith("--network-access=") for item in run_argv)

    async def test_json_mode_parses_jsonl_payloads(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/opencode",
        )

        async def _fake_run(argv, **_kwargs):
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="1.2.15",
                    stderr="",
                    duration_ms=12,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout='{"type":"start"}\n{"type":"result","ok":true}',
                stderr="",
                duration_ms=42,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = OpenCodeTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "Summarize repo",
                "output_mode": "json",
            },
            ctx,
        )
        assert result.success
        assert result.data["parsed_payload"] == [
            {"type": "start"},
            {"type": "result", "ok": True},
        ]

    async def test_opencode_rejects_unsupported_sandbox_mode(self, ctx: ToolContext):
        tool = OpenCodeTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "hello",
                "sandbox_mode": "unrestricted",
            },
            ctx,
        )
        assert not result.success
        assert result.data["error_code"] == "unsupported_mode_combination"

    async def test_timeout_maps_to_timeout_error_code(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/codex",
        )

        async def _fake_run(argv, **_kwargs):
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="codex 1.2.3",
                    stderr="",
                    duration_ms=10,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=-1,
                stdout="",
                stderr="",
                duration_ms=301000,
                timed_out=True,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = OpenAICodexTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "Summarize repo",
            },
            ctx,
        )
        assert not result.success
        assert result.data["error_code"] == "timeout_exceeded"

    async def test_rejects_unsupported_version(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/codex",
        )
        monkeypatch.setitem(
            PROVIDER_SPECS,
            "codex",
            replace(PROVIDER_SPECS["codex"], min_supported_version=(99, 0, 0)),
        )

        async def _fake_run(argv, **_kwargs):
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="codex 1.2.3",
                    stderr="",
                    duration_ms=8,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=22,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = OpenAICodexTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "hello",
            },
            ctx,
        )
        assert not result.success
        assert result.data["error_code"] == "unsupported_version"

    async def test_claude_code_uses_permission_mode_flags(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/claude",
        )
        seen_argv: list[list[str]] = []

        async def _fake_run(argv, **_kwargs):
            seen_argv.append(list(argv))
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="2.1.63 (Claude Code)",
                    stderr="",
                    duration_ms=7,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=25,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = ClaudeCodeTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "hello",
            },
            ctx,
        )
        assert result.success
        run_argv = seen_argv[-1]
        assert "--permission-mode=default" in run_argv
        assert not any(item.startswith("--sandbox=") for item in run_argv)
        assert not any(item.startswith("--ask-for-approval=") for item in run_argv)

    async def test_claude_code_unrestricted_uses_bypass_permissions(
        self,
        monkeypatch,
        ctx: ToolContext,
    ):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/claude",
        )
        seen_argv: list[list[str]] = []

        async def _fake_run(argv, **_kwargs):
            seen_argv.append(list(argv))
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="2.1.63 (Claude Code)",
                    stderr="",
                    duration_ms=7,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=25,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = ClaudeCodeTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "hello",
                "sandbox_mode": "unrestricted",
                "approval_mode": "never",
            },
            ctx,
        )
        assert result.success
        run_argv = seen_argv[-1]
        assert "--permission-mode=bypassPermissions" in run_argv
        assert "--dangerously-skip-permissions" in run_argv

    async def test_provider_auth_env_vars_are_forwarded(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/claude",
        )
        captured_env: dict[str, str] = {}

        async def _fake_run(argv, **kwargs):
            env = kwargs.get("env") or {}
            captured_env.update(env)
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="2.1.63 (Claude Code)",
                    stderr="",
                    duration_ms=7,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_ms=25,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = ClaudeCodeTool(enabled=True)
        result = await tool.execute({"prompt": "hello"}, ctx)
        assert result.success
        assert captured_env.get("ANTHROPIC_API_KEY") == "test-anthropic"
        assert captured_env.get("OPENAI_API_KEY") == "test-openai"

    async def test_failure_error_includes_stderr_detail(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/codex",
        )

        async def _fake_run(argv, **_kwargs):
            if "--version" in argv:
                return CommandRunResult(
                    exit_code=0,
                    stdout="codex 1.2.3",
                    stderr="",
                    duration_ms=8,
                    timed_out=False,
                    truncated=False,
                )
            return CommandRunResult(
                exit_code=1,
                stdout="",
                stderr="error: unknown option '--sandbox=workspace-write'",
                duration_ms=11,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.tooling_common.provider_agent_tool.run_command", _fake_run)

        tool = OpenAICodexTool(enabled=True)
        result = await tool.execute(
            {
                "prompt": "hello",
            },
            ctx,
        )
        assert not result.success
        assert "unknown option '--sandbox=workspace-write'" in str(result.error or "")

    async def test_provider_arg_mismatch_returns_guidance(self, ctx: ToolContext):
        tool = OpenAICodexTool(enabled=True)
        result = await tool.execute(
            {
                "provider": "claude",
                "prompt": "hello",
            },
            ctx,
        )
        assert not result.success
        assert result.data["error_code"] == "invalid_arguments"
        assert "/tool claude_code" in str(result.error or "")


class TestWpCliTool:
    async def test_high_risk_requires_confirmation(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/wp",
        )
        tool = WpCliTool(enabled=True, high_risk_requires_confirmation=True)
        result = await tool.execute(
            {
                "group": "db",
                "action": "reset",
                "args": {},
            },
            ctx,
        )
        assert not result.success
        assert result.data["error_code"] == "high_risk_confirmation_required"

    async def test_plugin_list_runs(self, monkeypatch, ctx: ToolContext):
        monkeypatch.setattr(
            "loom.tools.tooling_common.binary_resolution.shutil.which",
            lambda _: "/usr/bin/wp",
        )

        async def _fake_run(*_args, **_kwargs):
            return CommandRunResult(
                exit_code=0,
                stdout='[{"name": "akismet", "status": "active"}]',
                stderr="",
                duration_ms=50,
                timed_out=False,
                truncated=False,
            )

        monkeypatch.setattr("loom.tools.wp_cli.run_command", _fake_run)

        tool = WpCliTool(enabled=True)
        result = await tool.execute(
            {"group": "plugin", "action": "list", "args": {}},
            ctx,
        )
        assert result.success
        assert result.data["group"] == "plugin"
        assert result.data["action"] == "list"


class TestWpEnvTool:
    async def test_destroy_requires_confirmation(self, ctx: ToolContext):
        tool = WpEnvTool(enabled=True)
        result = await tool.execute({"operation": "destroy"}, ctx)
        assert not result.success
        assert result.data["error_code"] == "high_risk_confirmation_required"


class TestWpScaffoldBlockTool:
    async def test_rejects_existing_target_without_overwrite(self, ctx: ToolContext):
        target = ctx.workspace / "existing"
        target.mkdir()
        tool = WpScaffoldBlockTool(enabled=True)
        result = await tool.execute(
            {
                "name": "my-block",
                "target_dir": "existing",
                "allow_overwrite": False,
            },
            ctx,
        )
        assert not result.success
        assert result.data["error_code"] == "path_exists"


class TestWpQualityGateTool:
    async def test_disabled_returns_feature_disabled(self, ctx: ToolContext):
        tool = WpQualityGateTool(enabled=False)
        result = await tool.execute({}, ctx)
        assert not result.success
        assert result.data["error_code"] == "feature_disabled"
