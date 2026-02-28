"""Tests for the interactive setup wizard."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.setup import (
    ROLE_PRESETS,
    _generate_toml,
    discover_models,
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
            "roles": ["planner", "executor", "extractor", "verifier", "compactor"],
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
            "roles": ["planner", "executor", "extractor", "verifier", "compactor"],
            "max_tokens": 4096,
            "temperature": 0.1,
        }]
        toml = _generate_toml(models)
        parsed = tomllib.loads(toml)
        assert parsed["models"]["test"]["provider"] == "ollama"
        assert parsed["server"]["port"] == 9000
        assert parsed["execution"]["delegate_task_timeout_seconds"] == 3600
        assert parsed["execution"]["cowork_tool_exposure_mode"] == "adaptive"
        assert parsed["memory"]["database_path"] == "~/.loom/loom.db"

    def test_generated_toml_loads_as_config(self):
        """Round-trip: generate TOML, write to disk, load with load_config."""
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
        assert config.execution.delegate_task_timeout_seconds == 3600
        assert config.execution.cowork_tool_exposure_mode == "adaptive"


class TestRunSetup:
    """Test the interactive setup flow end-to-end with mocked input."""

    def test_ollama_single_model_all_roles(self, tmp_path: Path, monkeypatch):
        """Simulate: choose Ollama, enter model name, all roles, confirm write."""
        cfg_dir = tmp_path / ".loom"
        cfg_path = cfg_dir / "loom.toml"
        monkeypatch.setattr("loom.setup.CONFIG_DIR", cfg_dir)
        monkeypatch.setattr("loom.setup.CONFIG_PATH", cfg_path)
        monkeypatch.setattr("loom.setup.discover_models", lambda *a, **kw: [])

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
        assert "delegate_task_timeout_seconds = 3600" in content
        assert 'cowork_tool_exposure_mode = "adaptive"' in content
        assert "ingest_artifact_retention_max_age_days = 14" in content
        assert "ingest_artifact_retention_max_files_per_scope = 96" in content
        assert "ingest_artifact_retention_max_bytes_per_scope = 268435456" in content
        assert (cfg_dir / "scratch").is_dir()
        assert (cfg_dir / "logs").is_dir()
        assert (cfg_dir / "processes").is_dir()

    def test_anthropic_primary_with_utility(self, tmp_path: Path, monkeypatch):
        """Simulate: Anthropic primary (planner+executor) + Ollama utility."""
        cfg_dir = tmp_path / ".loom"
        cfg_path = cfg_dir / "loom.toml"
        monkeypatch.setattr("loom.setup.CONFIG_DIR", cfg_dir)
        monkeypatch.setattr("loom.setup.CONFIG_PATH", cfg_path)
        monkeypatch.setattr("loom.setup.discover_models", lambda *a, **kw: [])

        inputs = iter([
            # Primary model
            1,                                   # provider: Anthropic
            "sk-ant-test-key",                   # api key
            "https://api.anthropic.com",         # base url (default)
            "claude-sonnet-4-5-20250929",        # model name
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

        run_setup()

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
            "planner", "executor", "extractor", "verifier", "compactor",
        }

    def test_primary_and_utility_cover_all(self):
        combined = set(ROLE_PRESETS["primary"]) | set(ROLE_PRESETS["utility"])
        assert combined == set(ROLE_PRESETS["all"])


class TestDiscoverModels:
    def test_openai_fallbacks_to_v1_models(self, monkeypatch):
        class FakeResponse:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self, **kwargs):
                self.headers = kwargs.get("headers", {})
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, endpoint):
                self.calls.append(endpoint)
                if endpoint == "/models":
                    return FakeResponse(404, {})
                if endpoint == "/v1/models":
                    return FakeResponse(200, {"data": [{"id": "gpt-4o"}]})
                return FakeResponse(404, {})

        fake_client = None

        def fake_client_factory(**kwargs):
            nonlocal fake_client
            fake_client = FakeClient(**kwargs)
            return fake_client

        monkeypatch.setattr("loom.setup.httpx.Client", fake_client_factory)

        models = discover_models(
            "openai_compatible",
            "http://localhost:1234",
            "sk-test",
        )
        assert models == ["gpt-4o"]
        assert fake_client is not None
        assert fake_client.calls == ["/models", "/v1/models"]
        assert fake_client.headers["authorization"] == "Bearer sk-test"

    def test_ollama_discovery_deduplicates_names(self, monkeypatch):
        class FakeResponse:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self, **kwargs):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, endpoint):
                self.calls.append(endpoint)
                return FakeResponse(200, {
                    "models": [
                        {"name": "qwen3:14b"},
                        {"name": "qwen3:14b"},
                        {"model": "llama3:8b"},
                    ],
                })

        fake_client = FakeClient()
        monkeypatch.setattr("loom.setup.httpx.Client", lambda **kwargs: fake_client)

        models = discover_models(
            "ollama",
            "http://localhost:11434",
        )
        assert models == ["qwen3:14b", "llama3:8b"]
        assert fake_client.calls == ["/api/tags"]


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
        screen._primary_model = {
            "name": "primary",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen3:14b",
            "api_key": "",
            "roles": ["planner", "executor", "extractor", "verifier", "compactor"],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

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

    def test_save_does_not_touch_mcp_toml(self, tmp_path: Path, monkeypatch):
        """Setup writes loom.toml without mutating separate mcp.toml config."""
        from loom.tui.screens import setup as setup_mod

        cfg_dir = tmp_path / ".loom"
        cfg_path = cfg_dir / "loom.toml"
        mcp_path = cfg_dir / "mcp.toml"
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_original = (
            "[mcp.servers.demo]\n"
            "command = \"python\"\n"
            "args = [\"-m\", \"demo\"]\n"
        )
        mcp_path.write_text(mcp_original)

        monkeypatch.setattr(setup_mod, "CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", cfg_path)

        screen = setup_mod.SetupScreen()
        screen._primary_model = {
            "name": "primary",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen3:14b",
            "api_key": "",
            "roles": ["planner", "executor", "extractor", "verifier", "compactor"],
            "max_tokens": 4096,
            "temperature": 0.1,
        }
        screen.dismiss = lambda _val: None

        screen._save_and_dismiss()

        assert cfg_path.exists()
        assert mcp_path.read_text() == mcp_original

    def test_collect_model_all_roles_skips_utility(self):
        """When all roles are covered, skip the utility prompt."""
        from loom.tui.screens.setup import SetupScreen

        screen = SetupScreen()
        screen._provider_key = "ollama"
        screen._base_url = "http://localhost:11434"
        screen._model_name = "qwen3:14b"
        screen._api_key = ""
        screen._roles = ["planner", "executor", "extractor", "verifier", "compactor"]

        # Stub _prepare_confirm to track call
        called = []
        screen._prepare_confirm = lambda: called.append(True)

        screen._collect_model()

        assert len(screen._models) == 1
        assert screen._models[0]["name"] == "primary"
        assert called  # went directly to confirm, skipping utility

    def test_collect_model_partial_roles_shows_utility(self):
        """When roles are incomplete, show the utility prompt."""
        from unittest.mock import MagicMock

        from loom.tui.screens.setup import (
            _STEP_UTILITY,
            SetupScreen,
        )

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

    def test_collect_details_autodiscovers_when_model_missing(self):
        """If model is blank, details collection should use cached discovery."""
        from types import SimpleNamespace

        from loom.tui.screens.setup import (
            _STEP_ROLES,
            SetupScreen,
        )

        screen = SetupScreen()
        screen._provider_key = "openai_compatible"
        screen._discovered_models = ["gpt-4o-mini", "gpt-4o"]

        url_input = SimpleNamespace(value="http://localhost:1234/v1", focus=lambda: None)
        api_input = SimpleNamespace(value="", focus=lambda: None)
        model_input = SimpleNamespace(value="", focus=lambda: None)
        discover_button = SimpleNamespace(disabled=False)
        discovery_panel = SimpleNamespace(update=lambda *_: None)
        selection_feedback = SimpleNamespace(update=lambda *_: None)

        widgets = {
            "#input-url": url_input,
            "#input-apikey": api_input,
            "#input-model": model_input,
            "#btn-discover": discover_button,
            "#discovery-results": discovery_panel,
            "#selection-feedback": selection_feedback,
        }
        screen.query_one = lambda selector, *_args, **_kwargs: widgets[selector]
        screen.notify = lambda *_args, **_kwargs: None

        screen._collect_details()
        assert screen._model_name == "gpt-4o-mini"
        assert model_input.value == "gpt-4o-mini"
        assert screen._step == _STEP_ROLES

    def test_digit_selects_discovered_model(self):
        """Digit key should pick discovered models in details step."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from loom.tui.screens.setup import (
            _STEP_DETAILS,
            SetupScreen,
        )

        screen = SetupScreen()
        screen._show_step = MagicMock()
        screen._step = _STEP_DETAILS
        screen._discovered_models = ["model-a", "model-b"]
        screen.notify = lambda *_args, **_kwargs: None

        model_input = SimpleNamespace(value="", focus=lambda: None)
        selection_feedback = SimpleNamespace(update=lambda *_: None)
        widgets = {
            "#input-model": model_input,
            "#selection-feedback": selection_feedback,
        }
        screen.query_one = lambda selector, *_args, **_kwargs: widgets[selector]

        event = MagicMock()
        event.key = "2"
        screen.on_key(event)

        assert screen._model_name == "model-b"
        assert model_input.value == "model-b"
        event.prevent_default.assert_called_once()
        event.stop.assert_called_once()

    def test_discovery_requires_anthropic_key(self):
        """Anthropic discovery should stay disabled until API key is provided."""
        from types import SimpleNamespace

        from loom.tui.screens.setup import SetupScreen

        screen = SetupScreen()
        screen._provider_key = "anthropic"

        url_input = SimpleNamespace(value="https://api.anthropic.com")
        api_input = SimpleNamespace(value="")
        discover_button = SimpleNamespace(disabled=False)
        panel_updates = []
        discovery_panel = SimpleNamespace(update=lambda text: panel_updates.append(text))

        widgets = {
            "#input-url": url_input,
            "#input-apikey": api_input,
            "#btn-discover": discover_button,
            "#discovery-results": discovery_panel,
        }
        screen.query_one = lambda selector, *_args, **_kwargs: widgets[selector]

        screen._render_discovered_models([])
        assert discover_button.disabled is True
        assert any("API key" in text for text in panel_updates)

        api_input.value = "sk-ant-test"
        screen._render_discovered_models([])
        assert discover_button.disabled is False

    def test_discovery_render_limits_visible_entries(self):
        """Discovery panel should cap visible entries to avoid overflow."""
        from types import SimpleNamespace

        from loom.tui.screens.setup import SetupScreen

        screen = SetupScreen()
        screen._provider_key = "openai_compatible"

        url_input = SimpleNamespace(value="http://localhost:1234/v1")
        api_input = SimpleNamespace(value="")
        discover_button = SimpleNamespace(disabled=False)
        panel_updates = []
        discovery_panel = SimpleNamespace(update=lambda text: panel_updates.append(text))

        widgets = {
            "#input-url": url_input,
            "#input-apikey": api_input,
            "#btn-discover": discover_button,
            "#discovery-results": discovery_panel,
        }
        screen.query_one = lambda selector, *_args, **_kwargs: widgets[selector]

        models = [f"model-{i}" for i in range(1, 11)]
        screen._render_discovered_models(models)

        assert panel_updates
        rendered = panel_updates[-1]
        assert "Press 1-6 to pick" in rendered
        assert "... 4 more" in rendered


class TestSetupScreenConfirmHotkeys:
    """Verify save-step keyboard shortcuts behave as documented."""

    def test_confirm_save_hotkeys(self):
        from unittest.mock import MagicMock

        from loom.tui.screens.setup import (
            _STEP_CONFIRM,
            SetupScreen,
        )

        screen = SetupScreen()
        screen._step = _STEP_CONFIRM

        saved = []
        screen._save_and_dismiss = lambda: saved.append(True)

        for expected_calls, key in enumerate(("enter", "y", "Y", "s", "S"), start=1):
            event = MagicMock()
            event.key = key
            screen.on_key(event)
            assert len(saved) == expected_calls
            event.prevent_default.assert_called_once()
            event.stop.assert_called_once()

        assert len(saved) == 5

    def test_confirm_back_hotkeys(self):
        from unittest.mock import MagicMock

        from loom.tui.screens.setup import (
            _STEP_CONFIRM,
            SetupScreen,
        )

        screen = SetupScreen()
        screen._step = _STEP_CONFIRM

        back_calls = []
        screen.action_back_or_cancel = lambda: back_calls.append(True)

        for expected_calls, key in enumerate(("b", "B"), start=1):
            event = MagicMock()
            event.key = key
            screen.on_key(event)
            assert len(back_calls) == expected_calls
            event.prevent_default.assert_called_once()
            event.stop.assert_called_once()

        assert len(back_calls) == 2


class TestSetupScreenModelDrafts:
    """P0-3: Verify primary/utility draft model replaces rather than appends."""

    def test_models_property_empty(self):
        from loom.tui.screens.setup import SetupScreen

        screen = SetupScreen()
        assert screen._models == []

    def test_models_property_primary_only(self):
        from loom.tui.screens.setup import SetupScreen

        screen = SetupScreen()
        screen._primary_model = {"name": "primary", "roles": ["executor"]}
        assert len(screen._models) == 1
        assert screen._models[0]["name"] == "primary"

    def test_models_property_primary_and_utility(self):
        from loom.tui.screens.setup import SetupScreen

        screen = SetupScreen()
        screen._primary_model = {"name": "primary", "roles": ["executor"]}
        screen._utility_model = {"name": "utility", "roles": ["extractor"]}
        assert len(screen._models) == 2

    def test_collect_model_replaces_primary_on_reselection(self):
        """Re-selecting roles should replace, not duplicate, the primary model."""
        from loom.tui.screens.setup import SetupScreen

        screen = SetupScreen()
        screen._provider_key = "ollama"
        screen._base_url = "http://localhost:11434"
        screen._model_name = "model-v1"
        screen._api_key = ""
        screen._roles = ["planner", "executor", "extractor", "verifier", "compactor"]

        screen._prepare_confirm = lambda: None

        # First collection
        screen._collect_model()
        assert len(screen._models) == 1
        assert screen._models[0]["model"] == "model-v1"

        # Simulate re-selection (back nav then re-choose)
        screen._model_name = "model-v2"
        screen._collect_model()
        # Should still be 1, not 2
        assert len(screen._models) == 1
        assert screen._models[0]["model"] == "model-v2"

    def test_back_from_utility_clears_draft(self):
        """Going back from utility step should not leave stale drafts."""
        from loom.tui.screens.setup import (
            _STEP_ROLES,
            _STEP_UTILITY,
            SetupScreen,
        )

        screen = SetupScreen()
        screen._primary_model = {
            "name": "primary",
            "roles": ["planner", "executor"],
            "model": "test",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "api_key": "",
            "max_tokens": 4096,
            "temperature": 0.1,
        }
        screen._step = _STEP_UTILITY

        screen.action_back_or_cancel()
        assert screen._step == _STEP_ROLES

    def test_back_from_confirm_clears_utility(self):
        """Going back from confirm should clear utility draft."""
        from loom.tui.screens.setup import (
            _STEP_UTILITY,
            SetupScreen,
        )

        screen = SetupScreen()
        screen._primary_model = {
            "name": "primary",
            "roles": ["planner", "executor"],
            "model": "test",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "api_key": "",
            "max_tokens": 4096,
            "temperature": 0.1,
        }
        screen._utility_model = {
            "name": "utility",
            "roles": ["extractor", "verifier", "compactor"],
            "model": "test2",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "api_key": "",
            "max_tokens": 2048,
            "temperature": 0.0,
        }
        screen._step = 4  # _STEP_CONFIRM

        screen.action_back_or_cancel()
        assert screen._utility_model is None
        assert screen._adding_utility is False
        assert screen._step == _STEP_UTILITY


class TestSetupScreenExecutorValidation:
    """P0-4: Validate executor-capable model before save."""

    def test_save_blocked_without_executor(self, tmp_path: Path, monkeypatch):
        """_save_and_dismiss should refuse if no model has executor role."""
        from loom.tui.screens import setup as setup_mod

        cfg_dir = tmp_path / ".loom"
        cfg_path = cfg_dir / "loom.toml"
        monkeypatch.setattr(setup_mod, "CONFIG_DIR", cfg_dir)
        monkeypatch.setattr(setup_mod, "CONFIG_PATH", cfg_path)

        screen = setup_mod.SetupScreen()
        screen._primary_model = {
            "name": "primary",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model": "qwen3:14b",
            "api_key": "",
            "roles": ["extractor", "verifier"],  # no executor!
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        dismissed = []
        screen.dismiss = lambda val: dismissed.append(val)
        notifications = []
        screen.notify = lambda msg, **kw: notifications.append(msg)

        screen._save_and_dismiss()

        # Should not have written config or dismissed
        assert not cfg_path.exists()
        assert len(dismissed) == 0
        assert any("executor" in n for n in notifications)

    def test_prepare_confirm_blocked_without_executor(self):
        """_prepare_confirm should bounce back to roles if no executor."""
        from unittest.mock import MagicMock

        from loom.tui.screens.setup import (
            _STEP_ROLES,
            SetupScreen,
        )

        screen = SetupScreen()
        screen._primary_model = {
            "name": "primary",
            "roles": ["extractor", "verifier"],  # no executor
            "model": "test",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "api_key": "",
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        notifications = []
        screen.notify = lambda msg, **kw: notifications.append(msg)
        screen.query_one = MagicMock()

        screen._prepare_confirm()
        assert screen._step == _STEP_ROLES
        assert any("executor" in n for n in notifications)


class TestFilesChangedPanelAccumulation:
    """P1-6: Files panel should accumulate entries across turns."""

    def test_update_files_accumulates(self):
        from unittest.mock import MagicMock

        from loom.tui.widgets.file_panel import FilesChangedPanel

        panel = FilesChangedPanel()
        panel._refresh_table = MagicMock()

        entries_1 = [{"operation": "create", "path": "a.py", "timestamp": "10:00:00"}]
        entries_2 = [{"operation": "modify", "path": "b.py", "timestamp": "10:01:00"}]

        panel.update_files(entries_1)
        panel.update_files(entries_2)
        assert panel._refresh_table.call_count == 2

        assert len(panel._all_entries) == 2
        assert panel._all_entries[0]["path"] == "a.py"
        assert panel._all_entries[1]["path"] == "b.py"

    def test_clear_files_resets(self):
        from unittest.mock import MagicMock

        from loom.tui.widgets.file_panel import FilesChangedPanel

        panel = FilesChangedPanel()
        panel._refresh_table = MagicMock()
        panel._all_entries = [
            {"operation": "create", "path": "a.py", "timestamp": "10:00:00"},
        ]
        panel.clear_files()
        panel._refresh_table.assert_called_once()
        assert len(panel._all_entries) == 0


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
