"""Configuration loader for Loom.

Loads from loom.toml with sensible defaults when file is absent.
Configuration is loaded once at startup and passed via dependency injection.
"""

from __future__ import annotations

import re
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
    max_tokens: int = 8192
    temperature: float = 0.1
    roles: list[str] = field(default_factory=lambda: ["executor"])
    api_key: str = ""
    reasoning_effort: str = ""
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
    delegate_task_timeout_seconds: int = 3600
    model_call_max_attempts: int = 5
    model_call_retry_base_delay_seconds: float = 0.5
    model_call_retry_max_delay_seconds: float = 8.0
    model_call_retry_jitter_seconds: float = 0.25


@dataclass(frozen=True)
class VerificationConfig:
    tier1_enabled: bool = True
    tier2_enabled: bool = True
    tier3_enabled: bool = False
    tier3_vote_count: int = 3
    policy_engine_enabled: bool = True
    regex_default_advisory: bool = True
    strict_output_protocol: bool = True
    shadow_compare_enabled: bool = False
    phase_scope_default: str = "current_phase"  # "current_phase" | "global"
    allow_partial_verified: bool = True
    unconfirmed_supporting_threshold: float = 0.30
    auto_confirm_prune_critical_path: bool = True
    confirm_or_prune_max_attempts: int = 2
    confirm_or_prune_backoff_seconds: float = 2.0
    confirm_or_prune_retry_on_transient: bool = True
    contradiction_guard_enabled: bool = True
    contradiction_guard_strict_coverage: bool = True
    contradiction_scan_max_files: int = 80
    contradiction_scan_max_total_bytes: int = 2_500_000
    contradiction_scan_max_file_bytes: int = 300_000
    contradiction_scan_allowed_suffixes: list[str] = field(default_factory=lambda: [
        ".md",
        ".txt",
        ".rst",
        ".csv",
        ".tsv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".xml",
        ".html",
        ".htm",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".sql",
        ".sh",
    ])
    contradiction_scan_min_files_for_sufficiency: int = 2
    remediation_queue_max_attempts: int = 3
    remediation_queue_backoff_seconds: float = 2.0
    remediation_queue_max_backoff_seconds: float = 30.0


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
    require_rule_scope_metadata: bool = False
    require_v2_contract: bool = False
    tui_run_scoped_workspace_enabled: bool = True
    llm_run_folder_naming_enabled: bool = True


@dataclass(frozen=True)
class RunnerLimitsConfig:
    """Execution-time sizing and compaction limits for subtask runner."""

    max_tool_iterations: int = 20
    max_subtask_wall_clock_seconds: int = 1200
    max_model_context_tokens: int = 24_000
    max_state_summary_chars: int = 480
    max_verification_summary_chars: int = 8000
    default_tool_result_output_chars: int = 4000
    heavy_tool_result_output_chars: int = 2000
    compact_tool_result_output_chars: int = 500
    compact_text_output_chars: int = 900
    minimal_text_output_chars: int = 260
    tool_call_argument_context_chars: int = 500
    compact_tool_call_argument_chars: int = 220
    runner_compaction_policy_mode: str = "tiered"  # "legacy" | "tiered" | "off"
    enable_filetype_ingest_router: bool = True
    enable_artifact_telemetry_events: bool = True
    artifact_telemetry_max_metadata_chars: int = 1200
    enable_model_overflow_fallback: bool = True
    ingest_artifact_retention_max_age_days: int = 14
    ingest_artifact_retention_max_files_per_scope: int = 96
    ingest_artifact_retention_max_bytes_per_scope: int = 268_435_456
    preserve_recent_critical_messages: int = 6
    compaction_pressure_ratio_soft: float = 0.86
    compaction_pressure_ratio_hard: float = 1.02
    compaction_no_gain_min_delta_chars: int = 24
    compaction_no_gain_attempt_limit: int = 2
    compaction_timeout_guard_seconds: int = 30
    extractor_timeout_guard_seconds: int = 20
    extractor_tool_args_max_chars: int = 260
    extractor_tool_trace_max_chars: int = 3600
    extractor_prompt_max_chars: int = 9000
    compaction_churn_warning_calls: int = 10


@dataclass(frozen=True)
class VerifierLimitsConfig:
    """Sizing limits for LLM verification prompts and excerpts."""

    max_tool_args_chars: int = 360
    max_tool_status_chars: int = 320
    max_tool_calls_tokens: int = 4000
    max_verifier_prompt_tokens: int = 12_000
    max_result_summary_chars: int = 7000
    compact_result_summary_chars: int = 2600
    max_evidence_section_chars: int = 4200
    max_evidence_section_compact_chars: int = 2200
    max_artifact_section_chars: int = 4200
    max_artifact_section_compact_chars: int = 2200
    max_tool_output_excerpt_chars: int = 1100
    max_artifact_file_excerpt_chars: int = 800


@dataclass(frozen=True)
class CompactorLimitsConfig:
    """Internal limits for semantic compactor chunking + response sizing."""

    max_chunk_chars: int = 9000
    max_chunks_per_round: int = 12
    max_reduction_rounds: int = 4
    min_compact_target_chars: int = 140
    response_tokens_floor: int = 256
    response_tokens_ratio: float = 0.75
    response_tokens_buffer: int = 256
    json_headroom_chars_floor: int = 48
    json_headroom_chars_ratio: float = 0.08
    json_headroom_chars_cap: int = 320
    chars_per_token_estimate: float = 3.6
    token_headroom: int = 24
    target_chars_ratio: float = 0.75


@dataclass(frozen=True)
class LimitsConfig:
    """Centralized sizing limits for prompts, compaction, and extraction."""

    planning_response_max_tokens: int = 16_384
    adhoc_repair_source_max_chars: int = 0  # 0 = no truncation
    evidence_context_text_max_chars: int = 4000
    runner: RunnerLimitsConfig = field(default_factory=RunnerLimitsConfig)
    verifier: VerifierLimitsConfig = field(default_factory=VerifierLimitsConfig)
    compactor: CompactorLimitsConfig = field(default_factory=CompactorLimitsConfig)


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
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

    def _resolve_database_path(self) -> Path:
        """Resolve the configured database path with scratch-temp redirect.

        Relative temporary DB names (for example ``.tmp_loom.db``) should not
        write next to the current working directory. Treat them as scratch
        artifacts and place them under ``workspace.scratch_dir``.
        """
        path = Path(self.memory.database_path).expanduser()
        if path.is_absolute():
            return path
        if path.name.startswith(".tmp_"):
            return self.scratch_path / path.name
        return path

    @property
    def database_path(self) -> Path:
        return self._resolve_database_path()

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
        max_tokens=data.get("max_tokens", ModelConfig.max_tokens),
        temperature=data.get("temperature", 0.1),
        roles=roles,
        api_key=data.get("api_key", ""),
        reasoning_effort=str(data.get("reasoning_effort", "") or "").strip(),
        tier=data.get("tier", 0),
        capabilities=capabilities,
    )


def _int_from(
    source: dict,
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse an int from dict with optional clamping."""
    raw = source.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _float_from(
    source: dict,
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a float from dict with optional clamping."""
    raw = source.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _bool_from(source: dict, key: str, default: bool) -> bool:
    """Parse a bool from dict with permissive string/int support."""
    raw = source.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(default)


def _suffix_list_from(source: dict, key: str, default: list[str]) -> list[str]:
    """Parse suffix list from config, supporting list or delimited string."""
    raw = source.get(key, default)
    if isinstance(raw, str):
        values = re.split(r"[,\n;]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = list(default)

    normalized: list[str] = []
    for item in values:
        text = str(item or "").strip().lower()
        if not text:
            continue
        text = text.lstrip("*")
        if not text.startswith("."):
            text = f".{text}"
        if text not in normalized:
            normalized.append(text)
    return normalized or list(default)


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
    delegate_timeout_raw = exec_data.get("delegate_task_timeout_seconds", 3600)
    try:
        delegate_task_timeout_seconds = int(delegate_timeout_raw)
    except (TypeError, ValueError):
        delegate_task_timeout_seconds = 3600
    delegate_task_timeout_seconds = max(1, delegate_task_timeout_seconds)

    max_attempts_raw = exec_data.get("model_call_max_attempts", 5)
    try:
        model_call_max_attempts = int(max_attempts_raw)
    except (TypeError, ValueError):
        model_call_max_attempts = 5
    model_call_max_attempts = max(1, min(10, model_call_max_attempts))

    base_delay_raw = exec_data.get("model_call_retry_base_delay_seconds", 0.5)
    try:
        model_call_retry_base_delay_seconds = float(base_delay_raw)
    except (TypeError, ValueError):
        model_call_retry_base_delay_seconds = 0.5
    model_call_retry_base_delay_seconds = max(0.0, model_call_retry_base_delay_seconds)

    max_delay_raw = exec_data.get("model_call_retry_max_delay_seconds", 8.0)
    try:
        model_call_retry_max_delay_seconds = float(max_delay_raw)
    except (TypeError, ValueError):
        model_call_retry_max_delay_seconds = 8.0
    model_call_retry_max_delay_seconds = max(0.0, model_call_retry_max_delay_seconds)
    if model_call_retry_max_delay_seconds < model_call_retry_base_delay_seconds:
        model_call_retry_max_delay_seconds = model_call_retry_base_delay_seconds

    jitter_raw = exec_data.get("model_call_retry_jitter_seconds", 0.25)
    try:
        model_call_retry_jitter_seconds = float(jitter_raw)
    except (TypeError, ValueError):
        model_call_retry_jitter_seconds = 0.25
    model_call_retry_jitter_seconds = max(0.0, model_call_retry_jitter_seconds)

    execution = ExecutionConfig(
        max_subtask_retries=exec_data.get("max_subtask_retries", 3),
        max_loop_iterations=exec_data.get("max_loop_iterations", 50),
        max_parallel_subtasks=exec_data.get("max_parallel_subtasks", 3),
        auto_approve_confidence_threshold=exec_data.get(
            "auto_approve_confidence_threshold", 0.8
        ),
        enable_streaming=exec_data.get("enable_streaming", False),
        delegate_task_timeout_seconds=delegate_task_timeout_seconds,
        model_call_max_attempts=model_call_max_attempts,
        model_call_retry_base_delay_seconds=model_call_retry_base_delay_seconds,
        model_call_retry_max_delay_seconds=model_call_retry_max_delay_seconds,
        model_call_retry_jitter_seconds=model_call_retry_jitter_seconds,
    )

    verif_data = raw.get("verification", {})
    if not isinstance(verif_data, dict):
        verif_data = {}
    threshold_raw = verif_data.get("unconfirmed_supporting_threshold", 0.30)
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        threshold = 0.30
    threshold = max(0.0, min(1.0, threshold))
    max_attempts_raw = verif_data.get("confirm_or_prune_max_attempts", 2)
    try:
        confirm_or_prune_max_attempts = int(max_attempts_raw)
    except (TypeError, ValueError):
        confirm_or_prune_max_attempts = 2
    confirm_or_prune_max_attempts = max(1, confirm_or_prune_max_attempts)

    backoff_raw = verif_data.get("confirm_or_prune_backoff_seconds", 2.0)
    try:
        confirm_or_prune_backoff_seconds = float(backoff_raw)
    except (TypeError, ValueError):
        confirm_or_prune_backoff_seconds = 2.0
    confirm_or_prune_backoff_seconds = max(0.0, confirm_or_prune_backoff_seconds)

    queue_max_attempts_raw = verif_data.get("remediation_queue_max_attempts", 3)
    try:
        remediation_queue_max_attempts = int(queue_max_attempts_raw)
    except (TypeError, ValueError):
        remediation_queue_max_attempts = 3
    remediation_queue_max_attempts = max(1, remediation_queue_max_attempts)

    queue_backoff_raw = verif_data.get("remediation_queue_backoff_seconds", 2.0)
    try:
        remediation_queue_backoff_seconds = float(queue_backoff_raw)
    except (TypeError, ValueError):
        remediation_queue_backoff_seconds = 2.0
    remediation_queue_backoff_seconds = max(0.0, remediation_queue_backoff_seconds)

    queue_max_backoff_raw = verif_data.get(
        "remediation_queue_max_backoff_seconds",
        30.0,
    )
    try:
        remediation_queue_max_backoff_seconds = float(queue_max_backoff_raw)
    except (TypeError, ValueError):
        remediation_queue_max_backoff_seconds = 30.0
    remediation_queue_max_backoff_seconds = max(
        remediation_queue_backoff_seconds,
        remediation_queue_max_backoff_seconds,
    )
    contradiction_scan_max_files = _int_from(
        verif_data,
        "contradiction_scan_max_files",
        VerificationConfig.contradiction_scan_max_files,
        minimum=1,
        maximum=1000,
    )
    contradiction_scan_max_total_bytes = _int_from(
        verif_data,
        "contradiction_scan_max_total_bytes",
        VerificationConfig.contradiction_scan_max_total_bytes,
        minimum=1_024,
        maximum=50_000_000,
    )
    contradiction_scan_max_file_bytes = _int_from(
        verif_data,
        "contradiction_scan_max_file_bytes",
        VerificationConfig.contradiction_scan_max_file_bytes,
        minimum=1,
        maximum=10_000_000,
    )
    contradiction_scan_max_file_bytes = min(
        contradiction_scan_max_file_bytes,
        contradiction_scan_max_total_bytes,
    )
    contradiction_scan_min_files_for_sufficiency = _int_from(
        verif_data,
        "contradiction_scan_min_files_for_sufficiency",
        VerificationConfig.contradiction_scan_min_files_for_sufficiency,
        minimum=1,
        maximum=100,
    )
    contradiction_scan_allowed_suffixes = _suffix_list_from(
        verif_data,
        "contradiction_scan_allowed_suffixes",
        VerificationConfig().contradiction_scan_allowed_suffixes,
    )

    verification = VerificationConfig(
        tier1_enabled=verif_data.get("tier1_enabled", True),
        tier2_enabled=verif_data.get("tier2_enabled", True),
        tier3_enabled=verif_data.get("tier3_enabled", False),
        tier3_vote_count=verif_data.get("tier3_vote_count", 3),
        policy_engine_enabled=verif_data.get("policy_engine_enabled", True),
        regex_default_advisory=verif_data.get("regex_default_advisory", True),
        strict_output_protocol=verif_data.get("strict_output_protocol", True),
        shadow_compare_enabled=verif_data.get("shadow_compare_enabled", False),
        phase_scope_default=str(
            verif_data.get("phase_scope_default", "current_phase"),
        ),
        allow_partial_verified=verif_data.get("allow_partial_verified", True),
        unconfirmed_supporting_threshold=threshold,
        auto_confirm_prune_critical_path=verif_data.get(
            "auto_confirm_prune_critical_path",
            True,
        ),
        confirm_or_prune_max_attempts=confirm_or_prune_max_attempts,
        confirm_or_prune_backoff_seconds=confirm_or_prune_backoff_seconds,
        confirm_or_prune_retry_on_transient=verif_data.get(
            "confirm_or_prune_retry_on_transient",
            True,
        ),
        contradiction_guard_enabled=_bool_from(
            verif_data,
            "contradiction_guard_enabled",
            True,
        ),
        contradiction_guard_strict_coverage=_bool_from(
            verif_data,
            "contradiction_guard_strict_coverage",
            True,
        ),
        contradiction_scan_max_files=contradiction_scan_max_files,
        contradiction_scan_max_total_bytes=contradiction_scan_max_total_bytes,
        contradiction_scan_max_file_bytes=contradiction_scan_max_file_bytes,
        contradiction_scan_allowed_suffixes=contradiction_scan_allowed_suffixes,
        contradiction_scan_min_files_for_sufficiency=(
            contradiction_scan_min_files_for_sufficiency
        ),
        remediation_queue_max_attempts=remediation_queue_max_attempts,
        remediation_queue_backoff_seconds=remediation_queue_backoff_seconds,
        remediation_queue_max_backoff_seconds=remediation_queue_max_backoff_seconds,
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
        require_rule_scope_metadata=proc_data.get(
            "require_rule_scope_metadata",
            False,
        ),
        require_v2_contract=proc_data.get(
            "require_v2_contract",
            False,
        ),
        tui_run_scoped_workspace_enabled=proc_data.get(
            "tui_run_scoped_workspace_enabled",
            True,
        ),
        llm_run_folder_naming_enabled=proc_data.get(
            "llm_run_folder_naming_enabled",
            True,
        ),
    )

    limits_data = raw.get("limits", {})
    if not isinstance(limits_data, dict):
        limits_data = {}
    runner_limits_data = limits_data.get("runner", {})
    if not isinstance(runner_limits_data, dict):
        runner_limits_data = {}
    verifier_limits_data = limits_data.get("verifier", {})
    if not isinstance(verifier_limits_data, dict):
        verifier_limits_data = {}
    compactor_limits_data = limits_data.get("compactor", {})
    if not isinstance(compactor_limits_data, dict):
        compactor_limits_data = {}

    runner_compaction_policy_mode = str(
        runner_limits_data.get(
            "runner_compaction_policy_mode",
            RunnerLimitsConfig.runner_compaction_policy_mode,
        ),
    ).strip().lower()
    if runner_compaction_policy_mode not in {"legacy", "tiered", "off"}:
        runner_compaction_policy_mode = RunnerLimitsConfig.runner_compaction_policy_mode
    compaction_pressure_ratio_soft = _float_from(
        runner_limits_data,
        "compaction_pressure_ratio_soft",
        RunnerLimitsConfig.compaction_pressure_ratio_soft,
        minimum=0.4,
        maximum=2.5,
    )
    compaction_pressure_ratio_hard = _float_from(
        runner_limits_data,
        "compaction_pressure_ratio_hard",
        RunnerLimitsConfig.compaction_pressure_ratio_hard,
        minimum=0.41,
        maximum=3.0,
    )
    compaction_pressure_ratio_hard = max(
        compaction_pressure_ratio_soft + 0.01,
        compaction_pressure_ratio_hard,
    )

    runner_limits = RunnerLimitsConfig(
        max_tool_iterations=_int_from(
            runner_limits_data,
            "max_tool_iterations",
            RunnerLimitsConfig.max_tool_iterations,
            minimum=4,
            maximum=60,
        ),
        max_subtask_wall_clock_seconds=_int_from(
            runner_limits_data,
            "max_subtask_wall_clock_seconds",
            RunnerLimitsConfig.max_subtask_wall_clock_seconds,
            minimum=60,
            maximum=86_400,
        ),
        max_model_context_tokens=_int_from(
            runner_limits_data,
            "max_model_context_tokens",
            RunnerLimitsConfig.max_model_context_tokens,
            minimum=2048,
            maximum=500_000,
        ),
        max_state_summary_chars=_int_from(
            runner_limits_data,
            "max_state_summary_chars",
            RunnerLimitsConfig.max_state_summary_chars,
            minimum=120,
            maximum=20_000,
        ),
        max_verification_summary_chars=_int_from(
            runner_limits_data,
            "max_verification_summary_chars",
            RunnerLimitsConfig.max_verification_summary_chars,
            minimum=400,
            maximum=40_000,
        ),
        default_tool_result_output_chars=_int_from(
            runner_limits_data,
            "default_tool_result_output_chars",
            RunnerLimitsConfig.default_tool_result_output_chars,
            minimum=400,
            maximum=40_000,
        ),
        heavy_tool_result_output_chars=_int_from(
            runner_limits_data,
            "heavy_tool_result_output_chars",
            RunnerLimitsConfig.heavy_tool_result_output_chars,
            minimum=200,
            maximum=20_000,
        ),
        compact_tool_result_output_chars=_int_from(
            runner_limits_data,
            "compact_tool_result_output_chars",
            RunnerLimitsConfig.compact_tool_result_output_chars,
            minimum=80,
            maximum=20_000,
        ),
        compact_text_output_chars=_int_from(
            runner_limits_data,
            "compact_text_output_chars",
            RunnerLimitsConfig.compact_text_output_chars,
            minimum=80,
            maximum=20_000,
        ),
        minimal_text_output_chars=_int_from(
            runner_limits_data,
            "minimal_text_output_chars",
            RunnerLimitsConfig.minimal_text_output_chars,
            minimum=40,
            maximum=10_000,
        ),
        tool_call_argument_context_chars=_int_from(
            runner_limits_data,
            "tool_call_argument_context_chars",
            RunnerLimitsConfig.tool_call_argument_context_chars,
            minimum=80,
            maximum=20_000,
        ),
        compact_tool_call_argument_chars=_int_from(
            runner_limits_data,
            "compact_tool_call_argument_chars",
            RunnerLimitsConfig.compact_tool_call_argument_chars,
            minimum=40,
            maximum=10_000,
        ),
        runner_compaction_policy_mode=runner_compaction_policy_mode,
        enable_filetype_ingest_router=_bool_from(
            runner_limits_data,
            "enable_filetype_ingest_router",
            RunnerLimitsConfig.enable_filetype_ingest_router,
        ),
        enable_artifact_telemetry_events=_bool_from(
            runner_limits_data,
            "enable_artifact_telemetry_events",
            RunnerLimitsConfig.enable_artifact_telemetry_events,
        ),
        artifact_telemetry_max_metadata_chars=_int_from(
            runner_limits_data,
            "artifact_telemetry_max_metadata_chars",
            RunnerLimitsConfig.artifact_telemetry_max_metadata_chars,
            minimum=120,
            maximum=20_000,
        ),
        enable_model_overflow_fallback=_bool_from(
            runner_limits_data,
            "enable_model_overflow_fallback",
            RunnerLimitsConfig.enable_model_overflow_fallback,
        ),
        ingest_artifact_retention_max_age_days=_int_from(
            runner_limits_data,
            "ingest_artifact_retention_max_age_days",
            RunnerLimitsConfig.ingest_artifact_retention_max_age_days,
            minimum=0,
            maximum=3650,
        ),
        ingest_artifact_retention_max_files_per_scope=_int_from(
            runner_limits_data,
            "ingest_artifact_retention_max_files_per_scope",
            RunnerLimitsConfig.ingest_artifact_retention_max_files_per_scope,
            minimum=1,
            maximum=200_000,
        ),
        ingest_artifact_retention_max_bytes_per_scope=_int_from(
            runner_limits_data,
            "ingest_artifact_retention_max_bytes_per_scope",
            RunnerLimitsConfig.ingest_artifact_retention_max_bytes_per_scope,
            minimum=1024,
            maximum=20_000_000_000,
        ),
        preserve_recent_critical_messages=_int_from(
            runner_limits_data,
            "preserve_recent_critical_messages",
            RunnerLimitsConfig.preserve_recent_critical_messages,
            minimum=2,
            maximum=30,
        ),
        compaction_pressure_ratio_soft=compaction_pressure_ratio_soft,
        compaction_pressure_ratio_hard=compaction_pressure_ratio_hard,
        compaction_no_gain_min_delta_chars=_int_from(
            runner_limits_data,
            "compaction_no_gain_min_delta_chars",
            RunnerLimitsConfig.compaction_no_gain_min_delta_chars,
            minimum=1,
            maximum=5000,
        ),
        compaction_no_gain_attempt_limit=_int_from(
            runner_limits_data,
            "compaction_no_gain_attempt_limit",
            RunnerLimitsConfig.compaction_no_gain_attempt_limit,
            minimum=1,
            maximum=25,
        ),
        compaction_timeout_guard_seconds=_int_from(
            runner_limits_data,
            "compaction_timeout_guard_seconds",
            RunnerLimitsConfig.compaction_timeout_guard_seconds,
            minimum=0,
            maximum=3600,
        ),
        extractor_timeout_guard_seconds=_int_from(
            runner_limits_data,
            "extractor_timeout_guard_seconds",
            RunnerLimitsConfig.extractor_timeout_guard_seconds,
            minimum=0,
            maximum=3600,
        ),
        extractor_tool_args_max_chars=_int_from(
            runner_limits_data,
            "extractor_tool_args_max_chars",
            RunnerLimitsConfig.extractor_tool_args_max_chars,
            minimum=80,
            maximum=20_000,
        ),
        extractor_tool_trace_max_chars=_int_from(
            runner_limits_data,
            "extractor_tool_trace_max_chars",
            RunnerLimitsConfig.extractor_tool_trace_max_chars,
            minimum=200,
            maximum=80_000,
        ),
        extractor_prompt_max_chars=_int_from(
            runner_limits_data,
            "extractor_prompt_max_chars",
            RunnerLimitsConfig.extractor_prompt_max_chars,
            minimum=400,
            maximum=120_000,
        ),
        compaction_churn_warning_calls=_int_from(
            runner_limits_data,
            "compaction_churn_warning_calls",
            RunnerLimitsConfig.compaction_churn_warning_calls,
            minimum=1,
            maximum=500,
        ),
    )

    verifier_limits = VerifierLimitsConfig(
        max_tool_args_chars=_int_from(
            verifier_limits_data,
            "max_tool_args_chars",
            VerifierLimitsConfig.max_tool_args_chars,
            minimum=80,
            maximum=20_000,
        ),
        max_tool_status_chars=_int_from(
            verifier_limits_data,
            "max_tool_status_chars",
            VerifierLimitsConfig.max_tool_status_chars,
            minimum=80,
            maximum=20_000,
        ),
        max_tool_calls_tokens=_int_from(
            verifier_limits_data,
            "max_tool_calls_tokens",
            VerifierLimitsConfig.max_tool_calls_tokens,
            minimum=400,
            maximum=60_000,
        ),
        max_verifier_prompt_tokens=_int_from(
            verifier_limits_data,
            "max_verifier_prompt_tokens",
            VerifierLimitsConfig.max_verifier_prompt_tokens,
            minimum=800,
            maximum=120_000,
        ),
        max_result_summary_chars=_int_from(
            verifier_limits_data,
            "max_result_summary_chars",
            VerifierLimitsConfig.max_result_summary_chars,
            minimum=200,
            maximum=100_000,
        ),
        compact_result_summary_chars=_int_from(
            verifier_limits_data,
            "compact_result_summary_chars",
            VerifierLimitsConfig.compact_result_summary_chars,
            minimum=120,
            maximum=100_000,
        ),
        max_evidence_section_chars=_int_from(
            verifier_limits_data,
            "max_evidence_section_chars",
            VerifierLimitsConfig.max_evidence_section_chars,
            minimum=200,
            maximum=100_000,
        ),
        max_evidence_section_compact_chars=_int_from(
            verifier_limits_data,
            "max_evidence_section_compact_chars",
            VerifierLimitsConfig.max_evidence_section_compact_chars,
            minimum=120,
            maximum=100_000,
        ),
        max_artifact_section_chars=_int_from(
            verifier_limits_data,
            "max_artifact_section_chars",
            VerifierLimitsConfig.max_artifact_section_chars,
            minimum=200,
            maximum=100_000,
        ),
        max_artifact_section_compact_chars=_int_from(
            verifier_limits_data,
            "max_artifact_section_compact_chars",
            VerifierLimitsConfig.max_artifact_section_compact_chars,
            minimum=120,
            maximum=100_000,
        ),
        max_tool_output_excerpt_chars=_int_from(
            verifier_limits_data,
            "max_tool_output_excerpt_chars",
            VerifierLimitsConfig.max_tool_output_excerpt_chars,
            minimum=120,
            maximum=40_000,
        ),
        max_artifact_file_excerpt_chars=_int_from(
            verifier_limits_data,
            "max_artifact_file_excerpt_chars",
            VerifierLimitsConfig.max_artifact_file_excerpt_chars,
            minimum=120,
            maximum=40_000,
        ),
    )

    compactor_limits = CompactorLimitsConfig(
        max_chunk_chars=_int_from(
            compactor_limits_data,
            "max_chunk_chars",
            CompactorLimitsConfig.max_chunk_chars,
            minimum=300,
            maximum=200_000,
        ),
        max_chunks_per_round=_int_from(
            compactor_limits_data,
            "max_chunks_per_round",
            CompactorLimitsConfig.max_chunks_per_round,
            minimum=1,
            maximum=100,
        ),
        max_reduction_rounds=_int_from(
            compactor_limits_data,
            "max_reduction_rounds",
            CompactorLimitsConfig.max_reduction_rounds,
            minimum=1,
            maximum=20,
        ),
        min_compact_target_chars=_int_from(
            compactor_limits_data,
            "min_compact_target_chars",
            CompactorLimitsConfig.min_compact_target_chars,
            minimum=20,
            maximum=20_000,
        ),
        response_tokens_floor=_int_from(
            compactor_limits_data,
            "response_tokens_floor",
            CompactorLimitsConfig.response_tokens_floor,
            minimum=0,
            maximum=100_000,
        ),
        response_tokens_ratio=_float_from(
            compactor_limits_data,
            "response_tokens_ratio",
            CompactorLimitsConfig.response_tokens_ratio,
            minimum=0.0,
            maximum=8.0,
        ),
        response_tokens_buffer=_int_from(
            compactor_limits_data,
            "response_tokens_buffer",
            CompactorLimitsConfig.response_tokens_buffer,
            minimum=0,
            maximum=100_000,
        ),
        json_headroom_chars_floor=_int_from(
            compactor_limits_data,
            "json_headroom_chars_floor",
            CompactorLimitsConfig.json_headroom_chars_floor,
            minimum=0,
            maximum=20_000,
        ),
        json_headroom_chars_ratio=_float_from(
            compactor_limits_data,
            "json_headroom_chars_ratio",
            CompactorLimitsConfig.json_headroom_chars_ratio,
            minimum=0.0,
            maximum=2.0,
        ),
        json_headroom_chars_cap=_int_from(
            compactor_limits_data,
            "json_headroom_chars_cap",
            CompactorLimitsConfig.json_headroom_chars_cap,
            minimum=0,
            maximum=100_000,
        ),
        chars_per_token_estimate=_float_from(
            compactor_limits_data,
            "chars_per_token_estimate",
            CompactorLimitsConfig.chars_per_token_estimate,
            minimum=0.1,
            maximum=16.0,
        ),
        token_headroom=_int_from(
            compactor_limits_data,
            "token_headroom",
            CompactorLimitsConfig.token_headroom,
            minimum=0,
            maximum=20_000,
        ),
        target_chars_ratio=_float_from(
            compactor_limits_data,
            "target_chars_ratio",
            CompactorLimitsConfig.target_chars_ratio,
            minimum=0.01,
            maximum=1.0,
        ),
    )

    limits = LimitsConfig(
        planning_response_max_tokens=_int_from(
            limits_data,
            "planning_response_max_tokens",
            LimitsConfig.planning_response_max_tokens,
            minimum=0,
            maximum=500_000,
        ),
        adhoc_repair_source_max_chars=_int_from(
            limits_data,
            "adhoc_repair_source_max_chars",
            LimitsConfig.adhoc_repair_source_max_chars,
            minimum=0,
            maximum=500_000,
        ),
        evidence_context_text_max_chars=_int_from(
            limits_data,
            "evidence_context_text_max_chars",
            LimitsConfig.evidence_context_text_max_chars,
            minimum=200,
            maximum=100_000,
        ),
        runner=runner_limits,
        verifier=verifier_limits,
        compactor=compactor_limits,
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
        limits=limits,
        mcp=MCPConfig(servers=mcp_servers),
    )
