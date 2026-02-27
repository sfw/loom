"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from loom.config import Config, MemoryConfig, WorkspaceConfig, load_config


class TestDefaultConfig:
    """Test default configuration values."""

    def test_default_server(self):
        config = Config()
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000

    def test_default_models_empty(self):
        config = Config()
        assert config.models == {}

    def test_default_mcp_servers_empty(self):
        config = Config()
        assert config.mcp.servers == {}

    def test_default_execution(self):
        config = Config()
        assert config.execution.max_subtask_retries == 3
        assert config.execution.max_loop_iterations == 50
        assert config.execution.auto_approve_confidence_threshold == 0.8
        assert config.execution.enable_global_run_budget is False
        assert config.execution.max_task_wall_clock_seconds == 0
        assert config.execution.max_task_total_tokens == 0
        assert config.execution.max_task_model_invocations == 0
        assert config.execution.max_task_tool_calls == 0
        assert config.execution.max_task_mutating_tool_calls == 0
        assert config.execution.max_task_replans == 0
        assert config.execution.max_task_remediation_attempts == 0
        assert config.execution.executor_completion_contract_mode == "off"
        assert config.execution.planner_degraded_mode == "allow"
        assert config.execution.enable_sqlite_remediation_queue is False
        assert config.execution.enable_durable_task_runner is False
        assert config.execution.enable_mutation_idempotency is False
        assert config.execution.enable_slo_metrics is False
        assert config.execution.delegate_task_timeout_seconds == 3600
        assert config.execution.model_call_max_attempts == 5
        assert config.execution.model_call_retry_base_delay_seconds == 0.5
        assert config.execution.model_call_retry_max_delay_seconds == 8.0
        assert config.execution.model_call_retry_jitter_seconds == 0.25

    def test_default_verification(self):
        config = Config()
        assert config.verification.tier1_enabled is True
        assert config.verification.tier2_enabled is True
        assert config.verification.tier3_enabled is False
        assert config.verification.tier3_vote_count == 3
        assert config.verification.policy_engine_enabled is True
        assert config.verification.regex_default_advisory is True
        assert config.verification.phase_scope_default == "current_phase"
        assert config.verification.allow_partial_verified is True
        assert config.verification.unconfirmed_supporting_threshold == 0.30
        assert config.verification.confirm_or_prune_max_attempts == 2
        assert config.verification.confirm_or_prune_backoff_seconds == 2.0
        assert config.verification.confirm_or_prune_retry_on_transient is True
        assert config.verification.contradiction_guard_enabled is True
        assert config.verification.contradiction_guard_strict_coverage is True
        assert config.verification.contradiction_scan_max_files == 80
        assert config.verification.contradiction_scan_max_total_bytes == 2_500_000
        assert config.verification.contradiction_scan_max_file_bytes == 300_000
        assert config.verification.contradiction_scan_min_files_for_sufficiency == 2
        assert ".md" in config.verification.contradiction_scan_allowed_suffixes

    def test_default_process_flags(self):
        config = Config()
        assert config.process.require_rule_scope_metadata is False
        assert config.process.require_v2_contract is False
        assert config.process.tui_run_scoped_workspace_enabled is True
        assert config.process.llm_run_folder_naming_enabled is True

    def test_default_limits(self):
        config = Config()
        assert config.limits.planning_response_max_tokens == 16384
        assert config.limits.adhoc_repair_source_max_chars == 0
        assert config.limits.evidence_context_text_max_chars == 4000
        assert config.limits.runner.default_tool_result_output_chars == 4000
        assert config.limits.runner.runner_compaction_policy_mode == "tiered"
        assert config.limits.runner.enable_filetype_ingest_router is True
        assert config.limits.runner.enable_artifact_telemetry_events is True
        assert config.limits.runner.artifact_telemetry_max_metadata_chars == 1200
        assert config.limits.runner.enable_model_overflow_fallback is True
        assert config.limits.runner.ingest_artifact_retention_max_age_days == 14
        assert config.limits.runner.ingest_artifact_retention_max_files_per_scope == 96
        assert config.limits.runner.ingest_artifact_retention_max_bytes_per_scope == 268_435_456
        assert config.limits.runner.compaction_pressure_ratio_soft == 0.86
        assert config.limits.verifier.max_verifier_prompt_tokens == 12000
        assert config.limits.compactor.response_tokens_ratio == 0.75
        assert config.limits.compactor.json_headroom_chars_floor == 48
        assert config.limits.compactor.json_headroom_chars_ratio == 0.08
        assert config.limits.compactor.json_headroom_chars_cap == 320
        assert config.limits.compactor.chars_per_token_estimate == 3.6
        assert config.limits.compactor.token_headroom == 24
        assert config.limits.compactor.target_chars_ratio == 0.75

    def test_database_path_expands_user(self):
        config = Config()
        assert "~" not in str(config.database_path)

    def test_temp_database_path_redirects_to_scratch(self):
        config = Config(
            memory=MemoryConfig(database_path=".tmp_loom.db"),
            workspace=WorkspaceConfig(scratch_dir="~/.loom/scratch"),
        )
        assert config.database_path == Path.home() / ".loom" / "scratch" / ".tmp_loom.db"

    def test_relative_database_path_keeps_non_temp_location(self):
        config = Config(memory=MemoryConfig(database_path="data/loom.db"))
        assert config.database_path == Path("data/loom.db")

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
reasoning_effort = "none"
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
        assert config.models["test_model"].reasoning_effort == "none"
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

    def test_enable_streaming_loaded(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[execution]
enable_streaming = true
""")
        config = load_config(toml_file)
        assert config.execution.enable_streaming is True

    def test_enable_streaming_default_false(self):
        config = Config()
        assert config.execution.enable_streaming is False

    def test_execution_model_retry_policy_loaded(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[execution]
delegate_task_timeout_seconds = 7200
model_call_max_attempts = 9
model_call_retry_base_delay_seconds = 0.75
model_call_retry_max_delay_seconds = 12.0
model_call_retry_jitter_seconds = 0.10
""")
        config = load_config(toml_file)
        assert config.execution.delegate_task_timeout_seconds == 7200
        assert config.execution.model_call_max_attempts == 9
        assert config.execution.model_call_retry_base_delay_seconds == 0.75
        assert config.execution.model_call_retry_max_delay_seconds == 12.0
        assert config.execution.model_call_retry_jitter_seconds == 0.10

    def test_execution_model_retry_policy_clamps_values(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[execution]
delegate_task_timeout_seconds = 0
model_call_max_attempts = 42
model_call_retry_base_delay_seconds = -1.0
model_call_retry_max_delay_seconds = -2.0
model_call_retry_jitter_seconds = -0.5
""")
        config = load_config(toml_file)
        assert config.execution.delegate_task_timeout_seconds == 1
        assert config.execution.model_call_max_attempts == 10
        assert config.execution.model_call_retry_base_delay_seconds == 0.0
        assert config.execution.model_call_retry_max_delay_seconds == 0.0
        assert config.execution.model_call_retry_jitter_seconds == 0.0

    def test_execution_refactor_flags_loaded(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[execution]
enable_global_run_budget = true
max_task_wall_clock_seconds = 7200
max_task_total_tokens = 250000
max_task_model_invocations = 5000
max_task_tool_calls = 4000
max_task_mutating_tool_calls = 250
max_task_replans = 10
max_task_remediation_attempts = 120
executor_completion_contract_mode = "enforce"
planner_degraded_mode = "require_approval"
enable_sqlite_remediation_queue = true
enable_durable_task_runner = true
enable_mutation_idempotency = true
enable_slo_metrics = true
""")
        config = load_config(toml_file)
        assert config.execution.enable_global_run_budget is True
        assert config.execution.max_task_wall_clock_seconds == 7200
        assert config.execution.max_task_total_tokens == 250000
        assert config.execution.max_task_model_invocations == 5000
        assert config.execution.max_task_tool_calls == 4000
        assert config.execution.max_task_mutating_tool_calls == 250
        assert config.execution.max_task_replans == 10
        assert config.execution.max_task_remediation_attempts == 120
        assert config.execution.executor_completion_contract_mode == "enforce"
        assert config.execution.planner_degraded_mode == "require_approval"
        assert config.execution.enable_sqlite_remediation_queue is True
        assert config.execution.enable_durable_task_runner is True
        assert config.execution.enable_mutation_idempotency is True
        assert config.execution.enable_slo_metrics is True

    def test_verification_policy_flags_loaded(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[verification]
policy_engine_enabled = false
regex_default_advisory = false
phase_scope_default = "global"
allow_partial_verified = false
unconfirmed_supporting_threshold = 0.55
auto_confirm_prune_critical_path = false
confirm_or_prune_max_attempts = 4
confirm_or_prune_backoff_seconds = 1.25
confirm_or_prune_retry_on_transient = false
contradiction_guard_enabled = false
contradiction_guard_strict_coverage = false
contradiction_scan_max_files = 120
contradiction_scan_max_total_bytes = 3600000
contradiction_scan_max_file_bytes = 240000
contradiction_scan_allowed_suffixes = [".md", "txt", ".csv"]
contradiction_scan_min_files_for_sufficiency = 4
""")
        config = load_config(toml_file)
        assert config.verification.policy_engine_enabled is False
        assert config.verification.regex_default_advisory is False
        assert config.verification.phase_scope_default == "global"
        assert config.verification.allow_partial_verified is False
        assert config.verification.unconfirmed_supporting_threshold == 0.55
        assert config.verification.auto_confirm_prune_critical_path is False
        assert config.verification.confirm_or_prune_max_attempts == 4
        assert config.verification.confirm_or_prune_backoff_seconds == 1.25
        assert config.verification.confirm_or_prune_retry_on_transient is False
        assert config.verification.contradiction_guard_enabled is False
        assert config.verification.contradiction_guard_strict_coverage is False
        assert config.verification.contradiction_scan_max_files == 120
        assert config.verification.contradiction_scan_max_total_bytes == 3_600_000
        assert config.verification.contradiction_scan_max_file_bytes == 240_000
        assert config.verification.contradiction_scan_allowed_suffixes == [
            ".md",
            ".txt",
            ".csv",
        ]
        assert config.verification.contradiction_scan_min_files_for_sufficiency == 4

    def test_verification_contradiction_scan_values_are_clamped_and_safe(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[verification]
contradiction_guard_strict_coverage = "maybe"
contradiction_scan_max_files = -3
contradiction_scan_max_total_bytes = "bad"
contradiction_scan_max_file_bytes = 999999999
contradiction_scan_allowed_suffixes = [123, "", ".MD", "txt"]
contradiction_scan_min_files_for_sufficiency = 999
""")
        config = load_config(toml_file)
        assert config.verification.contradiction_guard_strict_coverage is True
        assert config.verification.contradiction_scan_max_files == 1
        assert config.verification.contradiction_scan_max_total_bytes == 2_500_000
        assert config.verification.contradiction_scan_max_file_bytes == 2_500_000
        assert config.verification.contradiction_scan_allowed_suffixes == [
            ".123",
            ".md",
            ".txt",
        ]
        assert config.verification.contradiction_scan_min_files_for_sufficiency == 100

    def test_process_flags_loaded(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[process]
require_rule_scope_metadata = true
require_v2_contract = true
tui_run_scoped_workspace_enabled = false
llm_run_folder_naming_enabled = false
""")
        config = load_config(toml_file)
        assert config.process.require_rule_scope_metadata is True
        assert config.process.require_v2_contract is True
        assert config.process.tui_run_scoped_workspace_enabled is False
        assert config.process.llm_run_folder_naming_enabled is False

    def test_load_mcp_servers(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[mcp.servers.notion]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-notion"]
timeout_seconds = 45
enabled = true

[mcp.servers.notion.env]
NOTION_TOKEN = "secret-token"
""")
        config = load_config(toml_file)
        assert "notion" in config.mcp.servers
        server = config.mcp.servers["notion"]
        assert server.command == "npx"
        assert server.args == ["-y", "@modelcontextprotocol/server-notion"]
        assert server.timeout_seconds == 45
        assert server.enabled is True
        assert server.env["NOTION_TOKEN"] == "secret-token"

    def test_load_limits_sections(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[limits]
planning_response_max_tokens = 24000
adhoc_repair_source_max_chars = 30000
evidence_context_text_max_chars = 7200

[limits.runner]
max_model_context_tokens = 64000
heavy_tool_result_output_chars = 3000
runner_compaction_policy_mode = "tiered"
enable_filetype_ingest_router = false
enable_artifact_telemetry_events = true
artifact_telemetry_max_metadata_chars = 1800
enable_model_overflow_fallback = false
ingest_artifact_retention_max_age_days = 30
ingest_artifact_retention_max_files_per_scope = 240
ingest_artifact_retention_max_bytes_per_scope = 104857600
preserve_recent_critical_messages = 8
compaction_pressure_ratio_soft = 0.9
compaction_pressure_ratio_hard = 1.1
compaction_no_gain_min_delta_chars = 32
compaction_no_gain_attempt_limit = 4
compaction_timeout_guard_seconds = 42
extractor_timeout_guard_seconds = 27
extractor_tool_args_max_chars = 420
extractor_tool_trace_max_chars = 5100
extractor_prompt_max_chars = 12000
compaction_churn_warning_calls = 14

[limits.verifier]
max_verifier_prompt_tokens = 20000
max_tool_output_excerpt_chars = 1800

[limits.compactor]
response_tokens_floor = 512
response_tokens_ratio = 1.0
response_tokens_buffer = 320
json_headroom_chars_floor = 64
json_headroom_chars_ratio = 0.12
json_headroom_chars_cap = 512
chars_per_token_estimate = 4.2
token_headroom = 16
target_chars_ratio = 0.6
""")
        config = load_config(toml_file)
        assert config.limits.planning_response_max_tokens == 24000
        assert config.limits.adhoc_repair_source_max_chars == 30000
        assert config.limits.evidence_context_text_max_chars == 7200
        assert config.limits.runner.max_model_context_tokens == 64000
        assert config.limits.runner.heavy_tool_result_output_chars == 3000
        assert config.limits.runner.runner_compaction_policy_mode == "tiered"
        assert config.limits.runner.enable_filetype_ingest_router is False
        assert config.limits.runner.enable_artifact_telemetry_events is True
        assert config.limits.runner.artifact_telemetry_max_metadata_chars == 1800
        assert config.limits.runner.enable_model_overflow_fallback is False
        assert config.limits.runner.ingest_artifact_retention_max_age_days == 30
        assert config.limits.runner.ingest_artifact_retention_max_files_per_scope == 240
        assert config.limits.runner.ingest_artifact_retention_max_bytes_per_scope == 104_857_600
        assert config.limits.runner.preserve_recent_critical_messages == 8
        assert config.limits.runner.compaction_pressure_ratio_soft == 0.9
        assert config.limits.runner.compaction_pressure_ratio_hard == 1.1
        assert config.limits.runner.compaction_no_gain_min_delta_chars == 32
        assert config.limits.runner.compaction_no_gain_attempt_limit == 4
        assert config.limits.runner.compaction_timeout_guard_seconds == 42
        assert config.limits.runner.extractor_timeout_guard_seconds == 27
        assert config.limits.runner.extractor_tool_args_max_chars == 420
        assert config.limits.runner.extractor_tool_trace_max_chars == 5100
        assert config.limits.runner.extractor_prompt_max_chars == 12000
        assert config.limits.runner.compaction_churn_warning_calls == 14
        assert config.limits.verifier.max_verifier_prompt_tokens == 20000
        assert config.limits.verifier.max_tool_output_excerpt_chars == 1800
        assert config.limits.compactor.response_tokens_floor == 512
        assert config.limits.compactor.response_tokens_ratio == 1.0
        assert config.limits.compactor.response_tokens_buffer == 320
        assert config.limits.compactor.json_headroom_chars_floor == 64
        assert config.limits.compactor.json_headroom_chars_ratio == 0.12
        assert config.limits.compactor.json_headroom_chars_cap == 512
        assert config.limits.compactor.chars_per_token_estimate == 4.2
        assert config.limits.compactor.token_headroom == 16
        assert config.limits.compactor.target_chars_ratio == 0.6

    def test_load_runner_compaction_policy_mode_off(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[limits.runner]
runner_compaction_policy_mode = "off"
""")
        config = load_config(toml_file)
        assert config.limits.runner.runner_compaction_policy_mode == "off"

    def test_invalid_runner_compaction_policy_mode_falls_back(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[limits.runner]
runner_compaction_policy_mode = "invalid-mode"
""")
        config = load_config(toml_file)
        assert config.limits.runner.runner_compaction_policy_mode == "tiered"

    def test_can_disable_artifact_telemetry_events(self, tmp_path: Path):
        toml_file = tmp_path / "loom.toml"
        toml_file.write_text("""\
[limits.runner]
enable_artifact_telemetry_events = false
""")
        config = load_config(toml_file)
        assert config.limits.runner.enable_artifact_telemetry_events is False
