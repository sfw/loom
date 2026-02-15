"""Tests for the interactive setup wizard."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from loom.setup import (
    CONFIG_DIR,
    CONFIG_PATH,
    ROLE_PRESETS,
    _generate_toml,
    needs_setup,
    run_setup,
)


class TestNeedsSetup:
    """Test first-run detection."""

    def test_needs_setup_when_no_config(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("loom.setup.CONFIG_PATH", tmp_path / "nope.toml")
        monkeypatch.chdir(tmp_path)
        assert needs_setup() is True

    def test_no_setup_when_home_config_exists(self, tmp_path: Path, monkeypatch):
        cfg = tmp_path / "loom.toml"
        cfg.write_text("[server]\nport = 9000\n")
        monkeypatch.setattr("loom.setup.CONFIG_PATH", cfg)
        monkeypatch.chdir(tmp_path / "elsewhere" if False else tmp_path)
        assert needs_setup() is False

    def test_no_setup_when_cwd_config_exists(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("loom.setup.CONFIG_PATH", tmp_path / "nope.toml")
        cwd_cfg = tmp_path / "loom.toml"
        cwd_cfg.write_text("[server]\nport = 9000\n")
        monkeypatch.chdir(tmp_path)
        assert needs_setup() is False


class TestGenerateToml:
    """Test TOML generation from model dicts."""

    def test_single_model(self):
        models = [{
            "name": "primary",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen3:14b",
            "api_key": "",
            "roles": ["planner", "executor"],
            "max_tokens": 4096,
            "temperature": 0.1,
        }]
        toml = _generate_toml(models)
        assert "[models.primary]" in toml
        assert 'provider = "ollama"' in toml
        assert 'model = "qwen3:14b"' in toml
        assert "api_key" not in toml  # empty key omitted
        assert 'roles = ["planner", "executor"]' in toml

    def test_anthropic_model_includes_api_key(self):
        models = [{
            "name": "primary",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "model": "claude-sonnet-4-5-20250929",
            "api_key": "sk-ant-test123",
            "roles": ["planner", "executor", "extractor", "verifier"],
            "max_tokens": 4096,
            "temperature": 0.1,
        }]
        toml = _generate_toml(models)
        assert 'api_key = "sk-ant-test123"' in toml
        assert 'provider = "anthropic"' in toml

    def test_two_models(self):
        models = [
            {
                "name": "primary",
                "provider": "openai_compatible",
                "base_url": "http://localhost:1234/v1",
                "model": "mistral-nemo",
                "api_key": "",
                "roles": ["planner", "executor"],
                "max_tokens": 4096,
                "temperature": 0.1,
            },
            {
                "name": "utility",
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen3:8b",
                "api_key": "",
                "roles": ["extractor", "verifier"],
                "max_tokens": 2048,
                "temperature": 0.0,
            },
        ]
        toml = _generate_toml(models)
        assert "[models.primary]" in toml
        assert "[models.utility]" in toml
        assert 'model = "mistral-nemo"' in toml
        assert 'model = "qwen3:8b"' in toml

    def test_generated_toml_is_parseable(self):
        import tomllib

        models = [{
            "name": "test",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "llama3:8b",
            "api_key": "",
            "roles": ["planner", "executor", "extractor", "verifier"],
            "max_tokens": 4096,
            "temperature": 0.1,
        }]
        toml = _generate_toml(models)
        parsed = tomllib.loads(toml)
        assert parsed["models"]["test"]["provider"] == "ollama"
        assert parsed["server"]["port"] == 9000
        assert parsed["memory"]["database_path"] == "~/.loom/loom.db"

    def test_generated_toml_loads_as_config(self):
        """Round-trip: generate TOML, write to disk, load with load_config."""
        import tomllib

        from loom.config import load_config

        models = [{
            "name": "primary",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "model": "claude-sonnet-4-5-20250929",
            "api_key": "sk-ant-fake",
            "roles": ["planner", "executor"],
            "max_tokens": 8192,
            "temperature": 0.1,
        }]
        toml = _generate_toml(models)

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False,
        ) as f:
            f.write(toml)
            f.flush()
            config = load_config(Path(f.name))

        assert "primary" in config.models
        assert config.models["primary"].provider == "anthropic"
        assert config.models["primary"].model == "claude-sonnet-4-5-20250929"
        assert config.models["primary"].api_key == "sk-ant-fake"
        assert config.models["primary"].roles == ["planner", "executor"]
        assert config.server.port == 9000


class TestRunSetup:
    """Test the interactive setup flow end-to-end with mocked input."""

    def test_ollama_single_model_all_roles(self, tmp_path: Path, monkeypatch):
        """Simulate: choose Ollama, enter model name, all roles, confirm write."""
        cfg_dir = tmp_path / ".loom"
        cfg_path = cfg_dir / "loom.toml"
        monkeypatch.setattr("loom.setup.CONFIG_DIR", cfg_dir)
        monkeypatch.setattr("loom.setup.CONFIG_PATH", cfg_path)

        # Simulate user input: provider=3 (Ollama), url=default, model=qwen3:14b,
        # roles=1 (all), confirm write=yes
        inputs = iter([
            3,                             # provider: Ollama
            "http://localhost:11434",       # url (default)
            "qwen3:14b",                   # model name
            1,                             # roles: all
        ])

        def mock_prompt(text, **kwargs):
            try:
                return next(inputs)
            except StopIteration:
                return kwargs.get("default", "")

        def mock_confirm(text, **kwargs):
            return True

        monkeypatch.setattr("click.prompt", mock_prompt)
        monkeypatch.setattr("click.confirm", mock_confirm)

        result = run_setup()

        assert result == cfg_path
        assert cfg_path.exists()
        content = cfg_path.read_text()
        assert 'provider = "ollama"' in content
        assert 'model = "qwen3:14b"' in content
        assert (cfg_dir / "scratch").is_dir()
        assert (cfg_dir / "logs").is_dir()
        assert (cfg_dir / "processes").is_dir()

    def test_anthropic_primary_with_utility(self, tmp_path: Path, monkeypatch):
        """Simulate: Anthropic primary (planner+executor) + Ollama utility."""
        cfg_dir = tmp_path / ".loom"
        cfg_path = cfg_dir / "loom.toml"
        monkeypatch.setattr("loom.setup.CONFIG_DIR", cfg_dir)
        monkeypatch.setattr("loom.setup.CONFIG_PATH", cfg_path)

        inputs = iter([
            # Primary model
            1,                                   # provider: Anthropic
            1,                                   # model: claude-sonnet-4-5
            "sk-ant-test-key",                   # api key
            "https://api.anthropic.com",         # base url (default)
            2,                                   # roles: primary
            # Utility model
            3,                                   # provider: Ollama
            "http://localhost:11434",            # url
            "qwen3:8b",                          # model
        ])

        def mock_prompt(text, **kwargs):
            try:
                val = next(inputs)
                return val
            except StopIteration:
                return kwargs.get("default", "")

        def mock_confirm(text, **kwargs):
            return True  # yes to everything

        monkeypatch.setattr("click.prompt", mock_prompt)
        monkeypatch.setattr("click.confirm", mock_confirm)

        result = run_setup()

        assert cfg_path.exists()
        content = cfg_path.read_text()
        assert "[models.primary]" in content
        assert "[models.utility]" in content
        assert 'provider = "anthropic"' in content
        assert 'provider = "ollama"' in content
        assert 'api_key = "sk-ant-test-key"' in content

    def test_existing_config_cancel(self, tmp_path: Path, monkeypatch):
        """If config exists and user declines overwrite, exit."""
        cfg_dir = tmp_path / ".loom"
        cfg_dir.mkdir()
        cfg_path = cfg_dir / "loom.toml"
        cfg_path.write_text("[server]\nport = 9000\n")
        monkeypatch.setattr("loom.setup.CONFIG_DIR", cfg_dir)
        monkeypatch.setattr("loom.setup.CONFIG_PATH", cfg_path)

        monkeypatch.setattr("click.confirm", lambda *a, **kw: False)

        with pytest.raises(SystemExit):
            run_setup()

        # Original content untouched
        assert cfg_path.read_text() == "[server]\nport = 9000\n"


class TestRolePresets:
    """Verify role preset definitions."""

    def test_all_covers_every_role(self):
        assert set(ROLE_PRESETS["all"]) == {
            "planner", "executor", "extractor", "verifier",
        }

    def test_primary_and_utility_cover_all(self):
        combined = set(ROLE_PRESETS["primary"]) | set(ROLE_PRESETS["utility"])
        assert combined == set(ROLE_PRESETS["all"])


class TestSetupScreen:
    """Test the TUI setup wizard screen."""

    def test_init(self):
        from loom.tui.screens.setup import SetupScreen
        screen = SetupScreen()
        assert screen._provider_key == ""
        assert screen._models == []
        assert screen._adding_utility is False

    def test_save_writes_config(self, tmp_path: Path, monkeypatch):
        """_save_and_dismiss writes config and creates dirs."""
        from loom.tui.screens import setup as setup_mod

        cfg_dir = tmp_path / ".loom"
        cfg_path = cfg_dir / "loom.toml"
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", cfg_path)

        screen = setup_mod.SetupScreen()
        screen._models = [{
            "name": "primary",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen3:14b",
            "api_key": "",
            "roles": ["planner", "executor", "extractor", "verifier"],
            "max_tokens": 4096,
            "temperature": 0.1,
        }]

        # Stub dismiss to capture result
        dismissed = []
        screen.dismiss = lambda val: dismissed.append(val)

        screen._save_and_dismiss()

        assert cfg_path.exists()
        assert len(dismissed) == 1
        assert dismissed[0] == screen._models
        assert (cfg_dir / "scratch").is_dir()
        assert (cfg_dir / "logs").is_dir()
        assert (cfg_dir / "processes").is_dir()

        # Verify generated TOML is valid
        import tomllib
        parsed = tomllib.loads(cfg_path.read_text())
        assert parsed["models"]["primary"]["provider"] == "ollama"

    def test_collect_model_all_roles_skips_utility(self):
        """When all roles are covered, skip the utility prompt."""
        from loom.tui.screens.setup import (
            SetupScreen,
            _STEP_CONFIRM,
        )

        screen = SetupScreen()
        screen._provider_key = "ollama"
        screen._base_url = "http://localhost:11434"
        screen._model_name = "qwen3:14b"
        screen._api_key = ""
        screen._roles = ["planner", "executor", "extractor", "verifier"]

        # Stub _prepare_confirm to track call
        called = []
        screen._prepare_confirm = lambda: called.append(True)

        screen._collect_model()

        assert len(screen._models) == 1
        assert screen._models[0]["name"] == "primary"
        assert called  # went directly to confirm, skipping utility

    def test_collect_model_partial_roles_shows_utility(self):
        """When roles are incomplete, show the utility prompt."""
        from loom.tui.screens.setup import (
            SetupScreen,
            _STEP_UTILITY,
        )
        from unittest.mock import MagicMock

        screen = SetupScreen()
        screen._provider_key = "anthropic"
        screen._base_url = "https://api.anthropic.com"
        screen._model_name = "claude-sonnet-4-5-20250929"
        screen._api_key = "sk-ant-test"
        screen._roles = ["planner", "executor"]

        # Mock query_one for the missing roles label
        mock_label = MagicMock()
        screen.query_one = MagicMock(return_value=mock_label)

        screen._collect_model()

        assert len(screen._models) == 1
        assert screen._step == _STEP_UTILITY


class TestLoomAppNoModel:
    """Test that LoomApp accepts model=None for setup flow."""

    def test_init_with_none_model(self):
        from unittest.mock import MagicMock

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=None,
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        assert app._model is None
