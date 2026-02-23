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

        async def fake_prepare_run_payload(
            *,
            config,
            workspace,
            goal,
            process_name,
            fresh_adhoc,
        ):
            assert config is not None
            assert goal == "demo goal"
            assert process_name == "marketing-strategy"
            assert fresh_adhoc is False
            return process_name, workspace

        async def fake_run_task(
            _url,
            _goal,
            _ws,
            process_name=None,
            metadata=None,
        ):
            captured["process_name"] = process_name
            captured["metadata"] = metadata

        monkeypatch.setattr(main_mod, "_prepare_server_run_payload", fake_prepare_run_payload)
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
        assert captured["metadata"] is None

    def test_run_passes_auth_profile_metadata(self, tmp_path, monkeypatch):
        """`loom run --auth-profile` should send metadata to API payload."""
        import loom.__main__ as main_mod

        captured: dict[str, object] = {}

        async def fake_prepare_run_payload(
            *,
            config,
            workspace,
            goal,
            process_name,
            fresh_adhoc,
        ):
            assert config is not None
            assert process_name is None
            assert goal == "demo goal"
            assert fresh_adhoc is False
            return "/tmp/adhoc-runtime.process.yaml", workspace

        async def fake_run_task(
            _url,
            _goal,
            _ws,
            process_name=None,
            metadata=None,
        ):
            captured["process_name"] = process_name
            captured["metadata"] = metadata

        monkeypatch.setattr(main_mod, "_prepare_server_run_payload", fake_prepare_run_payload)
        monkeypatch.setattr(main_mod, "_run_task", fake_run_task)

        cfg_path = tmp_path / "loom.toml"
        cfg_path.write_text(
            "[server]\n"
            "host = \"127.0.0.1\"\n"
            "port = 9000\n"
        )
        auth_cfg = tmp_path / "auth.toml"
        auth_cfg.write_text(
            """
[auth.profiles.notion_marketing]
provider = "notion"
mode = "oauth2_pkce"
"""
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg_path),
                "--auth-config",
                str(auth_cfg),
                "run",
                "demo goal",
                "--auth-profile",
                "notion=notion_marketing",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert captured["process_name"] == "/tmp/adhoc-runtime.process.yaml"
        metadata = captured["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["auth_profile_overrides"] == {
            "notion": "notion_marketing"
        }
        assert metadata["auth_config_path"] == str(auth_cfg.resolve())

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


class TestAuthCli:
    def test_auth_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "check" in result.output
        assert "route" not in result.output

    def test_auth_list_show_check(self, tmp_path):
        cfg = tmp_path / "loom.toml"
        cfg.write_text("[server]\nport = 9000\n")
        auth_cfg = tmp_path / "auth.toml"
        auth_cfg.write_text(
            """
[auth.defaults]
notion = "notion_marketing"

[auth.mcp_alias_profiles]
notion_local = "notion_marketing"

[auth.profiles.notion_marketing]
provider = "notion"
mode = "oauth2_pkce"
account_label = "Marketing"
scopes = ["read:content"]
"""
        )
        runner = CliRunner()

        list_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "list",
                "--json",
            ],
        )
        assert list_result.exit_code == 0
        listed = json.loads(list_result.output)
        assert listed["defaults"]["notion"] == "notion_marketing"
        assert "mcp_alias_profiles" not in listed
        assert listed["profiles"][0]["id"] == "notion_marketing"

        show_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "show",
                "notion_marketing",
                "--json",
            ],
        )
        assert show_result.exit_code == 0
        shown = json.loads(show_result.output)
        assert shown["provider"] == "notion"
        assert shown["mode"] == "oauth2_pkce"

        check_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "check",
            ],
        )
        assert check_result.exit_code == 0
        assert "Auth config is valid." in check_result.output

    def test_auth_select_sets_and_unsets_workspace_default(self, tmp_path):
        cfg = tmp_path / "loom.toml"
        cfg.write_text("[server]\nport = 9000\n")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        auth_cfg = tmp_path / "auth.toml"
        auth_cfg.write_text(
            """
[auth.profiles.notion_marketing]
provider = "notion"
mode = "oauth2_pkce"
"""
        )
        runner = CliRunner()

        set_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--workspace",
                str(workspace),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "select",
                "notion",
                "notion_marketing",
            ],
        )
        assert set_result.exit_code == 0
        defaults_path = workspace / ".loom" / "auth.defaults.toml"
        assert defaults_path.exists()
        content = defaults_path.read_text()
        assert 'notion = "notion_marketing"' in content

        unset_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--workspace",
                str(workspace),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "select",
                "notion",
                "--unset",
            ],
        )
        assert unset_result.exit_code == 0
        assert 'notion = "notion_marketing"' not in defaults_path.read_text()

    def test_auth_select_rejects_mcp_selector(self, tmp_path):
        cfg = tmp_path / "loom.toml"
        cfg.write_text("[server]\nport = 9000\n")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        auth_cfg = tmp_path / "auth.toml"
        auth_cfg.write_text(
            """
[auth.profiles.notion_marketing]
provider = "notion"
mode = "oauth2_pkce"
"""
        )
        runner = CliRunner()

        set_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--workspace",
                str(workspace),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "select",
                "mcp.notion",
                "notion_marketing",
            ],
        )
        assert set_result.exit_code == 1
        assert "MCP selectors are no longer supported" in set_result.output

    def test_auth_route_subcommand_removed(self, tmp_path):
        cfg = tmp_path / "loom.toml"
        cfg.write_text("[server]\nport = 9000\n")
        runner = CliRunner()

        route_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "auth",
                "route",
                "list",
            ],
        )
        assert route_result.exit_code == 2
        assert "No such command 'route'" in route_result.output

    def test_auth_profile_add_edit_remove(self, tmp_path):
        cfg = tmp_path / "loom.toml"
        cfg.write_text("[server]\nport = 9000\n")
        auth_cfg = tmp_path / "auth.toml"
        runner = CliRunner()

        add_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "profile",
                "add",
                "notion_marketing",
                "--provider",
                "notion",
                "--mode",
                "oauth2_pkce",
                "--label",
                "Marketing",
                "--token-ref",
                "keychain://loom/notion/notion_marketing/tokens",
                "--scope",
                "read:content",
                "--meta",
                "workspace=marketing",
            ],
        )
        assert add_result.exit_code == 0

        show_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "show",
                "notion_marketing",
                "--json",
            ],
        )
        assert show_result.exit_code == 0
        shown = json.loads(show_result.output)
        assert shown["provider"] == "notion"
        assert shown["account_label"] == "Marketing"
        assert shown["metadata"]["workspace"] == "marketing"

        edit_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "profile",
                "edit",
                "notion_marketing",
                "--mode",
                "env_passthrough",
                "--env",
                "NOTION_TOKEN=${NOTION_TOKEN}",
                "--clear-scopes",
            ],
        )
        assert edit_result.exit_code == 0

        show_after_edit = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "show",
                "notion_marketing",
                "--json",
            ],
        )
        assert show_after_edit.exit_code == 0
        edited = json.loads(show_after_edit.output)
        assert edited["mode"] == "env_passthrough"
        assert edited["scopes"] == []
        assert edited["env"]["NOTION_TOKEN"] == "${NOTION_TOKEN}"

        list_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "list",
                "--json",
            ],
        )
        assert list_result.exit_code == 0
        listed = json.loads(list_result.output)
        assert listed["profiles"][0]["id"] == "notion_marketing"

        remove_result = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "profile",
                "remove",
                "notion_marketing",
            ],
        )
        assert remove_result.exit_code == 0

        list_after_remove = runner.invoke(
            cli,
            [
                "--config",
                str(cfg),
                "--auth-config",
                str(auth_cfg),
                "auth",
                "list",
                "--json",
            ],
        )
        assert list_after_remove.exit_code == 0
        listed_after_remove = json.loads(list_after_remove.output)
        assert listed_after_remove["profiles"] == []

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
        aliases = [item["alias"] for item in json.loads(list_empty.output)["servers"]]
        assert "demo" not in aliases

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
