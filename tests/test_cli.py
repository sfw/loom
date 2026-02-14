"""Tests for CLI entry point."""

from __future__ import annotations

from click.testing import CliRunner

from loom.__main__ import cli


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

    def test_tui_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tui", "--help"])
        assert result.exit_code == 0
        assert "--workspace" in result.output
        assert "--model" in result.output
