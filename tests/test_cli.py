"""Tests for CLI entry point."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

from click.testing import CliRunner

from loom.__main__ import cli
from loom.config import Config, ProcessConfig
from loom.processes.testing import ProcessCaseResult


class TestCLI:
    """Test CLI commands."""

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Loom" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_serve_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--workspace" in result.output

    def test_install_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["install", "--help"])
        assert result.exit_code == 0
        assert "--skip-deps" in result.output
        assert "--isolated-deps" in result.output

    def test_process_test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "test", "--help"])
        assert result.exit_code == 0
        assert "--live" in result.output
        assert "--case" in result.output

    def test_learned_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["learned", "--help"])
        assert result.exit_code == 0
        assert "--all" in result.output
        assert "--type" in result.output

    def test_status_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0

    def test_cancel_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cancel", "--help"])
        assert result.exit_code == 0

    def test_models_no_config(self, tmp_path):
        # Use a config with no models to avoid picking up project loom.toml
        empty_toml = tmp_path / "loom.toml"
        empty_toml.write_text("[server]\nport = 9000\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(empty_toml), "models"])
        assert result.exit_code == 0
        assert "No models configured" in result.output

    def test_models_with_config(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        # If loom.toml exists in cwd, it should show configured models
        # This test is environment-dependent; just verify no crash

    def test_cowork_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cowork", "--help"])
        assert result.exit_code == 0
        assert "--workspace" in result.output
        assert "--model" in result.output
        assert "--resume" in result.output

    def test_default_shows_help_with_workspace_and_model(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "--workspace" in result.output
        assert "--model" in result.output

    def test_launch_tui_uses_default_process(self, monkeypatch):
        """When --process is omitted, _launch_tui should use config default."""
        import loom.__main__ as main_mod

        captured: dict[str, object] = {}

        class DummyApp:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, **kwargs):
                captured["run_kwargs"] = kwargs
                return None

        monkeypatch.setattr(main_mod, "_init_persistence", lambda _cfg: (None, None))
        monkeypatch.setattr(
            "loom.tools.create_default_registry",
            lambda _config=None: MagicMock(),
        )
        monkeypatch.setattr("loom.tui.app.LoomApp", DummyApp)

        cfg = Config(process=ProcessConfig(default="marketing-strategy"))
        main_mod._launch_tui(
            config=cfg,
            workspace=None,
            model_name=None,
            resume_session=None,
            process_name=None,
        )

        assert captured["process_name"] == "marketing-strategy"
        assert captured["run_kwargs"] == {"mouse": True}

    def test_run_uses_default_process(self, tmp_path, monkeypatch):
        """`loom run` should send process.default when flag is omitted."""
        import loom.__main__ as main_mod

        captured: dict[str, object] = {}

        async def fake_run_task(_url, _goal, _ws, process_name=None):
            captured["process_name"] = process_name

        monkeypatch.setattr(main_mod, "_run_task", fake_run_task)

        cfg_path = tmp_path / "loom.toml"
        cfg_path.write_text(
            "[server]\n"
            "host = \"127.0.0.1\"\n"
            "port = 9000\n"
            "\n"
            "[process]\n"
            "default = \"marketing-strategy\"\n"
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--config", str(cfg_path), "run", "demo goal"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert captured["process_name"] == "marketing-strategy"

    def test_process_test_runs_selected_cases(self, tmp_path, monkeypatch):
        async def fake_run_process_tests(*args, **kwargs):
            return [
                ProcessCaseResult(
                    case_id="smoke",
                    mode="deterministic",
                    passed=True,
                    duration_seconds=0.01,
                    message="Passed",
                    task_status="completed",
                )
            ]

        monkeypatch.setattr(
            "loom.processes.testing.run_process_tests",
            fake_run_process_tests,
        )

        cfg_path = tmp_path / "loom.toml"
        cfg_path.write_text("[server]\nport = 9000\n")
        process_yaml = tmp_path / "demo-process.yaml"
        process_yaml.write_text("name: demo-process\nversion: '1.0'\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_path),
                "process",
                "test",
                str(process_yaml),
                "--workspace",
                str(tmp_path),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "[PASS] case=smoke mode=deterministic" in result.output


class TestMCPCli:
    def test_mcp_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "migrate" in result.output

    def test_mcp_add_list_show_enable_disable_remove(self, tmp_path):
        cfg = tmp_path / "loom.toml"
        cfg.write_text("[server]\nport = 9000\n")
        mcp_cfg = tmp_path / "mcp.toml"
        runner = CliRunner()

        add_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "add",
                "demo",
                "--command",
                "python",
                "--arg",
                "-m",
                "--arg",
                "demo.server",
                "--env",
                "API_KEY=super-secret",
                "--env-ref",
                "NOTION_TOKEN=NOTION_TOKEN",
            ],
        )
        assert add_result.exit_code == 0

        list_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "list",
                "--json",
            ],
        )
        assert list_result.exit_code == 0
        listed = json.loads(list_result.output)
        assert listed["legacy_sources_detected"] is False
        assert listed["servers"][0]["alias"] == "demo"
        assert listed["servers"][0]["env"]["API_KEY"] == "<redacted>"
        assert listed["servers"][0]["env"]["NOTION_TOKEN"] == "${NOTION_TOKEN}"

        show_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "show",
                "demo",
                "--json",
            ],
        )
        assert show_result.exit_code == 0
        shown = json.loads(show_result.output)
        assert shown["alias"] == "demo"
        assert shown["command"] == "python"
        assert shown["enabled"] is True

        disable_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "disable",
                "demo",
            ],
        )
        assert disable_result.exit_code == 0

        show_disabled = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "show",
                "demo",
                "--json",
            ],
        )
        assert show_disabled.exit_code == 0
        assert json.loads(show_disabled.output)["enabled"] is False

        enable_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "enable",
                "demo",
            ],
        )
        assert enable_result.exit_code == 0

        remove_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "remove",
                "demo",
            ],
        )
        assert remove_result.exit_code == 0

        list_empty = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "list",
                "--json",
            ],
        )
        assert list_empty.exit_code == 0
        assert json.loads(list_empty.output)["servers"] == []

    def test_mcp_migrate_moves_legacy_section(self, tmp_path):
        cfg = tmp_path / "loom.toml"
        cfg.write_text(
            """
[server]
port = 9000

[mcp.servers.demo]
command = "python"
args = ["-m", "demo"]
enabled = true

[mcp.servers.demo.env]
TOKEN = "secret"
"""
        )
        mcp_cfg = tmp_path / "mcp.toml"
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "migrate",
            ],
        )
        assert result.exit_code == 0
        assert "Migrated 1 MCP server" in result.output
        assert "[mcp.servers.demo]" in mcp_cfg.read_text()
        assert "[mcp" not in cfg.read_text()

    def test_mcp_list_warns_on_legacy_sources(self, tmp_path):
        cfg = tmp_path / "loom.toml"
        cfg.write_text(
            """
[server]
port = 9000

[mcp.servers.legacy_demo]
command = "python"
"""
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["--config", str(cfg), "mcp", "list"])
        assert result.exit_code == 0
        assert "legacy_demo" in result.output
        assert "loom mcp migrate" in result.output

    def test_mcp_test_probe(self, tmp_path):
        cfg = tmp_path / "loom.toml"
        cfg.write_text("[server]\nport = 9000\n")
        mcp_cfg = tmp_path / "mcp.toml"
        fake_server = tmp_path / "fake_mcp.py"
        fake_server.write_text(
            """\
import json
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"capabilities": {"tools": {"listChanged": True}}},
        }), flush=True)
        continue
    if method == "notifications/initialized":
        continue
    if method == "tools/list":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {"name": "echo", "inputSchema": {"type": "object"}},
                    {"name": "ping", "inputSchema": {"type": "object"}},
                ]
            },
        }), flush=True)
"""
        )

        runner = CliRunner()
        add_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "add",
                "demo",
                "--command",
                sys.executable,
                "--arg",
                str(fake_server),
            ],
        )
        assert add_result.exit_code == 0

        probe_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--mcp-config",
                str(mcp_cfg),
                "mcp",
                "test",
                "demo",
                "--json",
            ],
        )
        assert probe_result.exit_code == 0
        payload = json.loads(probe_result.output)
        assert payload["tool_count"] == 2
        assert payload["tools"] == ["echo", "ping"]
