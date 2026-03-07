"""Runner configuration hydration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loom.config import Config
from loom.engine.semantic_compactor import SemanticCompactor


@dataclass(frozen=True)
class RunnerSettings:
    max_tool_iterations: int
    max_subtask_wall_clock_seconds: int
    max_model_context_tokens: int
    max_state_summary_chars: int
    max_verification_summary_chars: int
    default_tool_result_output_chars: int
    heavy_tool_result_output_chars: int
    compact_tool_result_output_chars: int
    compact_text_output_chars: int
    minimal_text_output_chars: int
    tool_call_argument_context_chars: int
    compact_tool_call_argument_chars: int
    runner_compaction_policy_mode: str
    preserve_recent_critical_messages: int
    compaction_pressure_ratio_soft: float
    compaction_pressure_ratio_hard: float
    compaction_no_gain_min_delta_chars: int
    compaction_no_gain_attempt_limit: int
    compaction_timeout_guard_seconds: float
    extractor_timeout_guard_seconds: float
    extractor_tool_args_max_chars: int
    extractor_tool_trace_max_chars: int
    extractor_prompt_max_chars: int
    compaction_churn_warning_calls: int
    enable_filetype_ingest_router: bool
    enable_artifact_telemetry_events: bool
    artifact_telemetry_max_metadata_chars: int
    enable_model_overflow_fallback: bool
    ingest_artifact_retention_max_age_days: int
    ingest_artifact_retention_max_files_per_scope: int
    ingest_artifact_retention_max_bytes_per_scope: int
    executor_completion_contract_mode: str
    enable_mutation_idempotency: bool
    ask_user_v2_enabled: bool
    ask_user_runtime_blocking_enabled: bool
    ask_user_policy: str
    ask_user_timeout_seconds: int
    ask_user_timeout_default_response: str
    ask_user_max_pending_per_task: int
    ask_user_max_questions_per_subtask: int
    ask_user_min_seconds_between_questions: float
    evidence_context_text_max_chars: int
    compactor_kwargs: dict[str, int | float]

    @classmethod
    def from_config(cls, config: Config, *, runner_defaults: Any) -> RunnerSettings:
        runner_limits = getattr(getattr(config, "limits", None), "runner", None)

        max_tool_iterations = int(
            getattr(
                runner_limits,
                "max_tool_iterations",
                runner_defaults.MAX_TOOL_ITERATIONS,
            ),
        )
        max_subtask_wall_clock_seconds = int(
            getattr(
                runner_limits,
                "max_subtask_wall_clock_seconds",
                runner_defaults.MAX_SUBTASK_WALL_CLOCK,
            ),
        )
        max_model_context_tokens = int(
            getattr(
                runner_limits,
                "max_model_context_tokens",
                runner_defaults.MAX_MODEL_CONTEXT_TOKENS,
            ),
        )
        max_state_summary_chars = int(
            getattr(
                runner_limits,
                "max_state_summary_chars",
                runner_defaults.MAX_STATE_SUMMARY_CHARS,
            ),
        )
        max_verification_summary_chars = int(
            getattr(
                runner_limits,
                "max_verification_summary_chars",
                runner_defaults.MAX_VERIFICATION_SUMMARY_CHARS,
            ),
        )
        default_tool_result_output_chars = int(
            getattr(
                runner_limits,
                "default_tool_result_output_chars",
                runner_defaults.DEFAULT_TOOL_RESULT_OUTPUT_CHARS,
            ),
        )
        heavy_tool_result_output_chars = int(
            getattr(
                runner_limits,
                "heavy_tool_result_output_chars",
                runner_defaults.HEAVY_TOOL_RESULT_OUTPUT_CHARS,
            ),
        )
        compact_tool_result_output_chars = int(
            getattr(
                runner_limits,
                "compact_tool_result_output_chars",
                runner_defaults.COMPACT_TOOL_RESULT_OUTPUT_CHARS,
            ),
        )
        compact_text_output_chars = int(
            getattr(
                runner_limits,
                "compact_text_output_chars",
                runner_defaults.COMPACT_TEXT_OUTPUT_CHARS,
            ),
        )
        minimal_text_output_chars = int(
            getattr(
                runner_limits,
                "minimal_text_output_chars",
                runner_defaults.MINIMAL_TEXT_OUTPUT_CHARS,
            ),
        )
        tool_call_argument_context_chars = int(
            getattr(
                runner_limits,
                "tool_call_argument_context_chars",
                runner_defaults.TOOL_CALL_ARGUMENT_CONTEXT_CHARS,
            ),
        )
        compact_tool_call_argument_chars = int(
            getattr(
                runner_limits,
                "compact_tool_call_argument_chars",
                runner_defaults.COMPACT_TOOL_CALL_ARGUMENT_CHARS,
            ),
        )
        policy_mode = str(
            getattr(
                runner_limits,
                "runner_compaction_policy_mode",
                runner_defaults.RUNNER_COMPACTION_POLICY_MODE,
            ),
        ).strip().lower()
        runner_compaction_policy_mode = (
            policy_mode
            if policy_mode in {"legacy", "tiered", "off"}
            else runner_defaults.RUNNER_COMPACTION_POLICY_MODE
        )
        preserve_recent_critical_messages = max(
            2,
            int(
                getattr(
                    runner_limits,
                    "preserve_recent_critical_messages",
                    runner_defaults.PRESERVE_RECENT_CRITICAL_MESSAGES,
                ),
            ),
        )
        compaction_pressure_ratio_soft = max(
            0.4,
            float(
                getattr(
                    runner_limits,
                    "compaction_pressure_ratio_soft",
                    runner_defaults.COMPACTION_PRESSURE_RATIO_SOFT,
                ),
            ),
        )
        hard_ratio = float(
            getattr(
                runner_limits,
                "compaction_pressure_ratio_hard",
                runner_defaults.COMPACTION_PRESSURE_RATIO_HARD,
            ),
        )
        compaction_pressure_ratio_hard = max(
            compaction_pressure_ratio_soft + 0.01,
            hard_ratio,
        )
        compaction_no_gain_min_delta_chars = max(
            1,
            int(
                getattr(
                    runner_limits,
                    "compaction_no_gain_min_delta_chars",
                    runner_defaults.COMPACTION_NO_GAIN_MIN_DELTA_CHARS,
                ),
            ),
        )
        compaction_no_gain_attempt_limit = max(
            1,
            int(
                getattr(
                    runner_limits,
                    "compaction_no_gain_attempt_limit",
                    runner_defaults.COMPACTION_NO_GAIN_ATTEMPT_LIMIT,
                ),
            ),
        )
        compaction_timeout_guard_seconds = max(
            0.0,
            float(
                getattr(
                    runner_limits,
                    "compaction_timeout_guard_seconds",
                    runner_defaults.COMPACTION_TIMEOUT_GUARD_SECONDS,
                ),
            ),
        )
        extractor_timeout_guard_seconds = max(
            0.0,
            float(
                getattr(
                    runner_limits,
                    "extractor_timeout_guard_seconds",
                    runner_defaults.EXTRACTOR_TIMEOUT_GUARD_SECONDS,
                ),
            ),
        )
        extractor_tool_args_max_chars = max(
            80,
            int(
                getattr(
                    runner_limits,
                    "extractor_tool_args_max_chars",
                    runner_defaults.EXTRACTOR_TOOL_ARGS_MAX_CHARS,
                ),
            ),
        )
        extractor_tool_trace_max_chars = max(
            300,
            int(
                getattr(
                    runner_limits,
                    "extractor_tool_trace_max_chars",
                    runner_defaults.EXTRACTOR_TOOL_TRACE_MAX_CHARS,
                ),
            ),
        )
        extractor_prompt_max_chars = max(
            600,
            int(
                getattr(
                    runner_limits,
                    "extractor_prompt_max_chars",
                    runner_defaults.EXTRACTOR_PROMPT_MAX_CHARS,
                ),
            ),
        )
        compaction_churn_warning_calls = max(
            1,
            int(
                getattr(
                    runner_limits,
                    "compaction_churn_warning_calls",
                    runner_defaults.COMPACTION_CHURN_WARNING_CALLS,
                ),
            ),
        )
        enable_filetype_ingest_router = bool(
            getattr(
                runner_limits,
                "enable_filetype_ingest_router",
                runner_defaults.ENABLE_FILETYPE_INGEST_ROUTER,
            ),
        )
        enable_artifact_telemetry_events = bool(
            getattr(
                runner_limits,
                "enable_artifact_telemetry_events",
                runner_defaults.ENABLE_ARTIFACT_TELEMETRY_EVENTS,
            ),
        )
        artifact_telemetry_max_metadata_chars = max(
            120,
            int(
                getattr(
                    runner_limits,
                    "artifact_telemetry_max_metadata_chars",
                    runner_defaults.ARTIFACT_TELEMETRY_MAX_METADATA_CHARS,
                ),
            ),
        )
        enable_model_overflow_fallback = bool(
            getattr(
                runner_limits,
                "enable_model_overflow_fallback",
                runner_defaults.ENABLE_MODEL_OVERFLOW_FALLBACK,
            ),
        )
        ingest_artifact_retention_max_age_days = max(
            0,
            int(
                getattr(
                    runner_limits,
                    "ingest_artifact_retention_max_age_days",
                    14,
                ),
            ),
        )
        ingest_artifact_retention_max_files_per_scope = max(
            1,
            int(
                getattr(
                    runner_limits,
                    "ingest_artifact_retention_max_files_per_scope",
                    96,
                ),
            ),
        )
        ingest_artifact_retention_max_bytes_per_scope = max(
            1024,
            int(
                getattr(
                    runner_limits,
                    "ingest_artifact_retention_max_bytes_per_scope",
                    268_435_456,
                ),
            ),
        )

        execution_cfg = getattr(config, "execution", None)
        completion_mode = str(
            getattr(
                execution_cfg,
                "executor_completion_contract_mode",
                runner_defaults.EXECUTOR_COMPLETION_CONTRACT_MODE,
            ),
        ).strip().lower()
        if completion_mode not in {"off", "warn", "enforce"}:
            completion_mode = runner_defaults.EXECUTOR_COMPLETION_CONTRACT_MODE
        executor_completion_contract_mode = completion_mode
        enable_mutation_idempotency = bool(
            getattr(
                execution_cfg,
                "enable_mutation_idempotency",
                runner_defaults.ENABLE_MUTATION_IDEMPOTENCY,
            ),
        )
        ask_user_v2_enabled = bool(
            getattr(execution_cfg, "ask_user_v2_enabled", False),
        )
        ask_user_runtime_blocking_enabled = bool(
            getattr(execution_cfg, "ask_user_runtime_blocking_enabled", False),
        )
        ask_user_policy = str(
            getattr(execution_cfg, "ask_user_policy", "block"),
        ).strip().lower()
        if ask_user_policy not in {"block", "timeout_default", "fail_closed"}:
            ask_user_policy = "block"
        ask_user_timeout_seconds = max(
            0,
            int(getattr(execution_cfg, "ask_user_timeout_seconds", 0) or 0),
        )
        ask_user_timeout_default_response = str(
            getattr(execution_cfg, "ask_user_timeout_default_response", "") or "",
        ).strip()
        ask_user_max_pending_per_task = max(
            1,
            int(getattr(execution_cfg, "ask_user_max_pending_per_task", 3) or 3),
        )
        ask_user_max_questions_per_subtask = max(
            1,
            int(getattr(execution_cfg, "ask_user_max_questions_per_subtask", 25) or 25),
        )
        ask_user_min_seconds_between_questions = max(
            0.0,
            float(
                getattr(execution_cfg, "ask_user_min_seconds_between_questions", 10) or 0,
            ),
        )

        evidence_context_text_max_chars = int(
            getattr(
                getattr(config, "limits", None),
                "evidence_context_text_max_chars",
                4000,
            ),
        )

        compactor_limits = getattr(getattr(config, "limits", None), "compactor", None)
        compactor_kwargs: dict[str, int | float] = {
            "max_chunk_chars": int(
                getattr(
                    compactor_limits,
                    "max_chunk_chars",
                    SemanticCompactor._MAX_CHUNK_CHARS,
                ),
            ),
            "max_chunks_per_round": int(
                getattr(
                    compactor_limits,
                    "max_chunks_per_round",
                    SemanticCompactor._MAX_CHUNKS_PER_ROUND,
                ),
            ),
            "max_reduction_rounds": int(
                getattr(
                    compactor_limits,
                    "max_reduction_rounds",
                    SemanticCompactor._MAX_REDUCTION_ROUNDS,
                ),
            ),
            "min_compact_target_chars": int(
                getattr(
                    compactor_limits,
                    "min_compact_target_chars",
                    SemanticCompactor._MIN_COMPACT_TARGET_CHARS,
                ),
            ),
            "response_tokens_floor": int(
                getattr(
                    compactor_limits,
                    "response_tokens_floor",
                    SemanticCompactor._RESPONSE_TOKENS_FLOOR,
                ),
            ),
            "response_tokens_ratio": float(
                getattr(
                    compactor_limits,
                    "response_tokens_ratio",
                    SemanticCompactor._RESPONSE_TOKENS_RATIO,
                ),
            ),
            "response_tokens_buffer": int(
                getattr(
                    compactor_limits,
                    "response_tokens_buffer",
                    SemanticCompactor._RESPONSE_TOKENS_BUFFER,
                ),
            ),
            "json_headroom_chars_floor": int(
                getattr(
                    compactor_limits,
                    "json_headroom_chars_floor",
                    SemanticCompactor._JSON_HEADROOM_CHARS_FLOOR,
                ),
            ),
            "json_headroom_chars_ratio": float(
                getattr(
                    compactor_limits,
                    "json_headroom_chars_ratio",
                    SemanticCompactor._JSON_HEADROOM_CHARS_RATIO,
                ),
            ),
            "json_headroom_chars_cap": int(
                getattr(
                    compactor_limits,
                    "json_headroom_chars_cap",
                    SemanticCompactor._JSON_HEADROOM_CHARS_CAP,
                ),
            ),
            "chars_per_token_estimate": float(
                getattr(
                    compactor_limits,
                    "chars_per_token_estimate",
                    SemanticCompactor._CHARS_PER_TOKEN_ESTIMATE,
                ),
            ),
            "token_headroom": int(
                getattr(
                    compactor_limits,
                    "token_headroom",
                    SemanticCompactor._TOKEN_HEADROOM,
                ),
            ),
            "target_chars_ratio": float(
                getattr(
                    compactor_limits,
                    "target_chars_ratio",
                    SemanticCompactor._TARGET_CHARS_RATIO,
                ),
            ),
        }

        return cls(
            max_tool_iterations=max_tool_iterations,
            max_subtask_wall_clock_seconds=max_subtask_wall_clock_seconds,
            max_model_context_tokens=max_model_context_tokens,
            max_state_summary_chars=max_state_summary_chars,
            max_verification_summary_chars=max_verification_summary_chars,
            default_tool_result_output_chars=default_tool_result_output_chars,
            heavy_tool_result_output_chars=heavy_tool_result_output_chars,
            compact_tool_result_output_chars=compact_tool_result_output_chars,
            compact_text_output_chars=compact_text_output_chars,
            minimal_text_output_chars=minimal_text_output_chars,
            tool_call_argument_context_chars=tool_call_argument_context_chars,
            compact_tool_call_argument_chars=compact_tool_call_argument_chars,
            runner_compaction_policy_mode=runner_compaction_policy_mode,
            preserve_recent_critical_messages=preserve_recent_critical_messages,
            compaction_pressure_ratio_soft=compaction_pressure_ratio_soft,
            compaction_pressure_ratio_hard=compaction_pressure_ratio_hard,
            compaction_no_gain_min_delta_chars=compaction_no_gain_min_delta_chars,
            compaction_no_gain_attempt_limit=compaction_no_gain_attempt_limit,
            compaction_timeout_guard_seconds=compaction_timeout_guard_seconds,
            extractor_timeout_guard_seconds=extractor_timeout_guard_seconds,
            extractor_tool_args_max_chars=extractor_tool_args_max_chars,
            extractor_tool_trace_max_chars=extractor_tool_trace_max_chars,
            extractor_prompt_max_chars=extractor_prompt_max_chars,
            compaction_churn_warning_calls=compaction_churn_warning_calls,
            enable_filetype_ingest_router=enable_filetype_ingest_router,
            enable_artifact_telemetry_events=enable_artifact_telemetry_events,
            artifact_telemetry_max_metadata_chars=artifact_telemetry_max_metadata_chars,
            enable_model_overflow_fallback=enable_model_overflow_fallback,
            ingest_artifact_retention_max_age_days=ingest_artifact_retention_max_age_days,
            ingest_artifact_retention_max_files_per_scope=ingest_artifact_retention_max_files_per_scope,
            ingest_artifact_retention_max_bytes_per_scope=ingest_artifact_retention_max_bytes_per_scope,
            executor_completion_contract_mode=executor_completion_contract_mode,
            enable_mutation_idempotency=enable_mutation_idempotency,
            ask_user_v2_enabled=ask_user_v2_enabled,
            ask_user_runtime_blocking_enabled=ask_user_runtime_blocking_enabled,
            ask_user_policy=ask_user_policy,
            ask_user_timeout_seconds=ask_user_timeout_seconds,
            ask_user_timeout_default_response=ask_user_timeout_default_response,
            ask_user_max_pending_per_task=ask_user_max_pending_per_task,
            ask_user_max_questions_per_subtask=ask_user_max_questions_per_subtask,
            ask_user_min_seconds_between_questions=ask_user_min_seconds_between_questions,
            evidence_context_text_max_chars=evidence_context_text_max_chars,
            compactor_kwargs=compactor_kwargs,
        )
