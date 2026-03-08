"""Parity tests for runner settings hydration."""

from __future__ import annotations

from types import SimpleNamespace

from loom.engine.runner import SubtaskRunner
from loom.engine.runner.settings import RunnerSettings


def test_runner_settings_uses_runner_defaults_when_config_is_sparse() -> None:
    config = SimpleNamespace(
        limits=SimpleNamespace(),
        execution=SimpleNamespace(),
    )

    settings = RunnerSettings.from_config(config, runner_defaults=SubtaskRunner)  # type: ignore[arg-type]

    assert settings.max_tool_iterations == SubtaskRunner.MAX_TOOL_ITERATIONS
    assert settings.max_subtask_wall_clock_seconds == SubtaskRunner.MAX_SUBTASK_WALL_CLOCK
    assert settings.runner_compaction_policy_mode == SubtaskRunner.RUNNER_COMPACTION_POLICY_MODE
    assert (
        settings.executor_completion_contract_mode
        == SubtaskRunner.EXECUTOR_COMPLETION_CONTRACT_MODE
    )
    assert (
        settings.sealed_artifact_post_call_guard
        == SubtaskRunner.SEALED_ARTIFACT_POST_CALL_GUARD
    )
    assert settings.ask_user_policy == "block"
    assert settings.compactor_kwargs["max_chunk_chars"] > 0


def test_runner_settings_clamps_ratios_and_invalid_policy_values() -> None:
    config = SimpleNamespace(
        limits=SimpleNamespace(
            runner=SimpleNamespace(
                compaction_pressure_ratio_soft=0.9,
                compaction_pressure_ratio_hard=0.5,
                runner_compaction_policy_mode="unknown",
            ),
        ),
        execution=SimpleNamespace(
            executor_completion_contract_mode="invalid",
            sealed_artifact_post_call_guard="invalid",
            ask_user_policy="invalid",
        ),
    )

    settings = RunnerSettings.from_config(config, runner_defaults=SubtaskRunner)  # type: ignore[arg-type]

    assert settings.compaction_pressure_ratio_soft == 0.9
    assert settings.compaction_pressure_ratio_hard == 0.91
    assert settings.runner_compaction_policy_mode == SubtaskRunner.RUNNER_COMPACTION_POLICY_MODE
    assert (
        settings.executor_completion_contract_mode
        == SubtaskRunner.EXECUTOR_COMPLETION_CONTRACT_MODE
    )
    assert (
        settings.sealed_artifact_post_call_guard
        == SubtaskRunner.SEALED_ARTIFACT_POST_CALL_GUARD
    )
    assert settings.ask_user_policy == "block"
