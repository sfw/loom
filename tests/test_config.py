"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from loom.config import Config, load_config


class TestDefaultConfig:
    """Test default configuration values."""

    def test_default_server(self):
        config = Config()
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000

    def test_default_models_empty(self):
        config = Config()
        assert config.models == {}

    def test_default_execution(self):
        config = Config()
        assert config.execution.max_subtask_retries == 3
        assert config.execution.max_loop_iterations == 50
        assert config.execution.auto_approve_confidence_threshold == 0.8

    def test_default_verification(self):
        config = Config()
        assert config.verification.tier1_enabled is True
        assert config.verification.tier2_enabled is True
        assert config.verification.tier3_enabled is False
        assert config.verification.tier3_vote_count == 3

    def test_database_path_expands_user(self):
        config = Config()
        assert "~" not in str(config.database_path)

    def test_scratch_path_expands_user(self):
        config = Config()
        assert "~" not in str(config.scratch_path)


class TestLoadConfig:
    """Test TOML configuration loading."""

    def test_load_missing_file_returns_defaults(self):
        config = load_config(Path("/nonexistent/loom.toml"))
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000

    def test_load_none_returns_defaults(self, tmp_path: Path):
        # With no loom.toml in cwd or home, should return defaults
        config = load_config(None)
        assert isinstance(config, Config)

    def test_load_valid_toml(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[server]
host = "0.0.0.0"
port = 8080

[models.test_model]
provider = "ollama"
base_url = "http://localhost:11434"
model = "llama3:8b"
max_tokens = 2048
temperature = 0.0
roles = ["executor", "planner"]

[execution]
max_subtask_retries = 5

[memory]
database_path = "/tmp/test.db"
""")
        config = load_config(toml_file)
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 8080
        assert "test_model" in config.models
        assert config.models["test_model"].provider == "ollama"
        assert config.models["test_model"].model == "llama3:8b"
        assert config.models["test_model"].roles == ["executor", "planner"]
        assert config.execution.max_subtask_retries == 5
        assert config.memory.database_path == "/tmp/test.db"

    def test_load_partial_toml(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[server]
port = 7777
""")
        config = load_config(toml_file)
        assert config.server.port == 7777
        assert config.server.host == "127.0.0.1"  # default
        assert config.models == {}  # default

    def test_model_config_single_role_string(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[models.simple]
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:8b"
roles = "executor"
""")
        config = load_config(toml_file)
        assert config.models["simple"].roles == ["executor"]
