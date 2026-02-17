"""Configuration loader for Loom.

Loads from loom.toml with sensible defaults when file is absent.
Configuration is loaded once at startup and passed via dependency injection.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9000


@dataclass(frozen=True)
class ModelCapabilities:
    """What content types this model can handle."""

    vision: bool = False
    native_pdf: bool = False
    thinking: bool = False
    citations: bool = False
    audio_input: bool = False
    audio_output: bool = False

    @classmethod
    def auto_detect(cls, provider: str, model: str) -> ModelCapabilities:
        """Infer capabilities from provider and model name."""
        model_lower = model.lower()

        if provider == "anthropic":
            return cls(
                vision=True,
                native_pdf=True,
                thinking="opus" in model_lower or "sonnet" in model_lower,
                citations=True,
            )

        if provider == "ollama":
            vision_models = {
                "llava", "bakllava", "gemma3", "smolvlm",
                "llama3.2-vision", "moondream", "minicpm-v",
            }
            has_vision = any(v in model_lower for v in vision_models)
            return cls(
                vision=has_vision,
                thinking="deepseek" in model_lower or "qwq" in model_lower,
            )

        if provider == "openai_compatible":
            has_vision = any(v in model_lower for v in [
                "gpt-4o", "gpt-4-vision", "gpt-4-turbo",
                "gemini", "pixtral", "internvl",
            ])
            return cls(
                vision=has_vision,
                native_pdf="gpt-4o" in model_lower,
            )

        return cls()


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model provider."""

    provider: str  # "ollama" | "openai_compatible" | "anthropic"
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.1
    roles: list[str] = field(default_factory=lambda: ["executor"])
    api_key: str = ""
    tier: int = 0  # 0 = auto-detect from model name
    capabilities: ModelCapabilities | None = None  # None = auto-detect

    @property
    def resolved_capabilities(self) -> ModelCapabilities:
        if self.capabilities is not None:
            return self.capabilities
        return ModelCapabilities.auto_detect(self.provider, self.model)

    def __repr__(self) -> str:
        key_display = f"***{self.api_key[-4:]}" if self.api_key else ""
        return (
            f"ModelConfig(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key={key_display!r})"
        )


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
class ProcessConfig:
    """Configuration for the process definition system."""

    default: str = ""  # Default process name (empty = no process)
    search_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for one external MCP server."""

    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    timeout_seconds: int = 30
    enabled: bool = True


@dataclass(frozen=True)
class MCPConfig:
    """Configuration for MCP-backed tool discovery."""

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)


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
    process: ProcessConfig = field(default_factory=ProcessConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

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

    capabilities = None
    caps_data = data.get("capabilities")
    if isinstance(caps_data, dict):
        capabilities = ModelCapabilities(
            vision=caps_data.get("vision", False),
            native_pdf=caps_data.get("native_pdf", False),
            thinking=caps_data.get("thinking", False),
            citations=caps_data.get("citations", False),
            audio_input=caps_data.get("audio_input", False),
            audio_output=caps_data.get("audio_output", False),
        )

    return ModelConfig(
        provider=data["provider"],
        base_url=data.get("base_url", ""),
        model=data.get("model", ""),
        max_tokens=data.get("max_tokens", 4096),
        temperature=data.get("temperature", 0.1),
        roles=roles,
        api_key=data.get("api_key", ""),
        tier=data.get("tier", 0),
        capabilities=capabilities,
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

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Cannot read config {path}: {e}") from e

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
        enable_streaming=exec_data.get("enable_streaming", False),
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

    proc_data = raw.get("process", {})
    process = ProcessConfig(
        default=proc_data.get("default", ""),
        search_paths=proc_data.get("search_paths", []),
    )

    mcp_servers: dict[str, MCPServerConfig] = {}
    mcp_data = raw.get("mcp", {})
    servers_data = mcp_data.get("servers", {}) if isinstance(mcp_data, dict) else {}
    if isinstance(servers_data, dict):
        for alias, server_data in servers_data.items():
            if not isinstance(server_data, dict):
                continue

            raw_args = server_data.get("args", [])
            args = [str(a) for a in raw_args] if isinstance(raw_args, list) else []

            raw_env = server_data.get("env", {})
            env: dict[str, str] = {}
            if isinstance(raw_env, dict):
                for key, value in raw_env.items():
                    if isinstance(key, str):
                        env[key] = str(value)

            timeout_raw = server_data.get("timeout_seconds", 30)
            try:
                timeout_seconds = int(timeout_raw)
            except (TypeError, ValueError):
                timeout_seconds = 30
            if timeout_seconds <= 0:
                timeout_seconds = 30

            mcp_servers[str(alias)] = MCPServerConfig(
                command=str(server_data.get("command", "")),
                args=args,
                env=env,
                cwd=str(server_data.get("cwd", "")),
                timeout_seconds=timeout_seconds,
                enabled=bool(server_data.get("enabled", True)),
            )

    return Config(
        server=server,
        models=models,
        workspace=workspace,
        execution=execution,
        verification=verification,
        memory=memory,
        logging=logging_cfg,
        process=process,
        mcp=MCPConfig(servers=mcp_servers),
    )
