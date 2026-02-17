"""Tests for CLI entry point."""

from __future__ import annotations

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

            def run(self):
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
