"""Configuration loader for Loom.

Loads from loom.toml with sensible defaults when file is absent.
Configuration is loaded once at startup and passed via dependency injection.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9000


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model provider."""

    provider: str  # "ollama" | "openai_compatible"
    base_url: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.1
    roles: list[str] = field(default_factory=lambda: ["executor"])


@dataclass(frozen=True)
class WorkspaceConfig:
    default_path: str = "~/projects"
    scratch_dir: str = "~/.loom/scratch"


@dataclass(frozen=True)
class ExecutionConfig:
    max_subtask_retries: int = 3
    max_loop_iterations: int = 50
    max_parallel_subtasks: int = 3
    auto_approve_confidence_threshold: float = 0.8
    enable_streaming: bool = False


@dataclass(frozen=True)
class VerificationConfig:
    tier1_enabled: bool = True
    tier2_enabled: bool = True
    tier3_enabled: bool = False
    tier3_vote_count: int = 3


@dataclass(frozen=True)
class MemoryConfig:
    database_path: str = "~/.loom/loom.db"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    event_log_path: str = "~/.loom/logs"


@dataclass(frozen=True)
class Config:
    """Top-level Loom configuration."""

    server: ServerConfig = field(default_factory=ServerConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def database_path(self) -> Path:
        return Path(self.memory.database_path).expanduser()

    @property
    def scratch_path(self) -> Path:
        return Path(self.workspace.scratch_dir).expanduser()

    @property
    def log_path(self) -> Path:
        return Path(self.logging.event_log_path).expanduser()


def _parse_model_config(name: str, data: dict) -> ModelConfig:
    """Parse a single model configuration section."""
    roles = data.get("roles", ["executor"])
    if isinstance(roles, str):
        roles = [roles]
    return ModelConfig(
        provider=data["provider"],
        base_url=data["base_url"],
        model=data["model"],
        max_tokens=data.get("max_tokens", 4096),
        temperature=data.get("temperature", 0.1),
        roles=roles,
    )


def load_config(path: Path | None = None) -> Config:
    """Load configuration from a TOML file.

    If path is None, searches for loom.toml in current directory then ~/.loom/.
    Returns default config if no file is found.
    """
    if path is None:
        candidates = [
            Path.cwd() / "loom.toml",
            Path.home() / ".loom" / "loom.toml",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

    if path is None or not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    server_data = raw.get("server", {})
    server = ServerConfig(
        host=server_data.get("host", "127.0.0.1"),
        port=server_data.get("port", 9000),
    )

    models: dict[str, ModelConfig] = {}
    for name, model_data in raw.get("models", {}).items():
        if isinstance(model_data, dict) and "provider" in model_data:
            models[name] = _parse_model_config(name, model_data)

    workspace_data = raw.get("workspace", {})
    workspace = WorkspaceConfig(
        default_path=workspace_data.get("default_path", "~/projects"),
        scratch_dir=workspace_data.get("scratch_dir", "~/.loom/scratch"),
    )

    exec_data = raw.get("execution", {})
    execution = ExecutionConfig(
        max_subtask_retries=exec_data.get("max_subtask_retries", 3),
        max_loop_iterations=exec_data.get("max_loop_iterations", 50),
        max_parallel_subtasks=exec_data.get("max_parallel_subtasks", 3),
        auto_approve_confidence_threshold=exec_data.get(
            "auto_approve_confidence_threshold", 0.8
        ),
    )

    verif_data = raw.get("verification", {})
    verification = VerificationConfig(
        tier1_enabled=verif_data.get("tier1_enabled", True),
        tier2_enabled=verif_data.get("tier2_enabled", True),
        tier3_enabled=verif_data.get("tier3_enabled", False),
        tier3_vote_count=verif_data.get("tier3_vote_count", 3),
    )

    mem_data = raw.get("memory", {})
    memory = MemoryConfig(
        database_path=mem_data.get("database_path", "~/.loom/loom.db"),
    )

    log_data = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=log_data.get("level", "INFO"),
        event_log_path=log_data.get("event_log_path", "~/.loom/logs"),
    )

    return Config(
        server=server,
        models=models,
        workspace=workspace,
        execution=execution,
        verification=verification,
        memory=memory,
        logging=logging_cfg,
    )
