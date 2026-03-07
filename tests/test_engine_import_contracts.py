"""Import/re-export parity checks for core engine public contracts."""

from __future__ import annotations

import importlib


def test_core_engine_public_import_contracts() -> None:
    from loom.engine.orchestrator import Orchestrator, create_task
    from loom.engine.runner import SubtaskResult, SubtaskRunner, ToolCallRecord
    from loom.engine.semantic_compactor import SemanticCompactor
    from loom.engine.verification import VerificationGates, VerificationResult

    assert callable(create_task)
    assert Orchestrator.__name__ == "Orchestrator"
    assert SubtaskRunner.__name__ == "SubtaskRunner"
    assert SubtaskResult.__name__ == "SubtaskResult"
    assert ToolCallRecord.__name__ == "ToolCallRecord"
    assert VerificationGates.__name__ == "VerificationGates"
    assert VerificationResult.__name__ == "VerificationResult"
    assert SemanticCompactor.__name__ == "SemanticCompactor"


def test_orchestrator_package_import_smoke() -> None:
    module = importlib.import_module("loom.engine.orchestrator")

    assert module.__name__ == "loom.engine.orchestrator"
    assert callable(module.create_task)
    assert callable(module.Orchestrator)


def test_orchestrator_budget_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.orchestrator.budget")

    assert module.__name__ == "loom.engine.orchestrator.budget"
    assert callable(module._RunBudget)


def test_orchestrator_seam_submodule_import_smoke() -> None:
    module_names = [
        "loom.engine.orchestrator.core",
        "loom.engine.orchestrator.planning",
        "loom.engine.orchestrator.dispatch",
        "loom.engine.orchestrator.remediation",
        "loom.engine.orchestrator.runtime",
        "loom.engine.orchestrator.output",
        "loom.engine.orchestrator.validity",
        "loom.engine.orchestrator.evidence",
        "loom.engine.orchestrator.telemetry",
        "loom.engine.orchestrator.task_factory",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_orchestrator_core_planning_dispatch_remediation_exports() -> None:
    core = importlib.import_module("loom.engine.orchestrator.core")
    planning = importlib.import_module("loom.engine.orchestrator.planning")
    dispatch = importlib.import_module("loom.engine.orchestrator.dispatch")
    remediation = importlib.import_module("loom.engine.orchestrator.remediation")
    runtime = importlib.import_module("loom.engine.orchestrator.runtime")

    assert callable(core.create_task)
    assert callable(planning.phase_mode)
    assert callable(planning.topology_retry_attempts)
    assert callable(dispatch.iteration_retry_mode)
    assert callable(dispatch.observe_iteration_runtime_usage)
    assert callable(remediation.remediation_queue_limits)
    assert callable(remediation.bounded_remediation_backoff_seconds)
    assert callable(runtime.task_run_id)
    assert callable(runtime.initialize_task_run_id)
    assert callable(runtime.emit_event)


def test_orchestrator_task_factory_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.orchestrator.task_factory")

    assert module.__name__ == "loom.engine.orchestrator.task_factory"
    assert callable(module.create_task)


def test_semantic_compactor_parse_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.semantic_compactor.parse")

    assert module.__name__ == "loom.engine.semantic_compactor.parse"
    assert callable(module.extract_compacted_text)
    assert callable(module.extract_partial_compressed_text)


def test_semantic_compactor_config_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.semantic_compactor.config")

    assert module.__name__ == "loom.engine.semantic_compactor.config"
    assert callable(module.compactor_hard_limit_chars)
    assert callable(module.compactor_response_max_tokens)


def test_semantic_compactor_cache_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.semantic_compactor.cache")

    assert module.__name__ == "loom.engine.semantic_compactor.cache"
    assert callable(module.lookup_cache_or_inflight)
    assert callable(module.store_cache_and_release_inflight)


def test_semantic_compactor_core_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.semantic_compactor.core")

    assert module.__name__ == "loom.engine.semantic_compactor.core"
    assert callable(module.SemanticCompactor)


def test_semantic_compactor_model_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.semantic_compactor.model")

    assert module.__name__ == "loom.engine.semantic_compactor.model"
    assert callable(module.select_model)
    assert callable(module.invoke_compactor_model)


def test_semantic_compactor_pipeline_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.semantic_compactor.pipeline")

    assert module.__name__ == "loom.engine.semantic_compactor.pipeline"
    assert callable(module.compact_with_model)
    assert callable(module.chunked_compaction)


def test_runner_types_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.runner.types")

    assert module.__name__ == "loom.engine.runner.types"
    assert callable(module.ToolCallRecord)
    assert callable(module.SubtaskResult)


def test_runner_settings_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.runner.settings")

    assert module.__name__ == "loom.engine.runner.settings"
    assert callable(module.RunnerSettings)


def test_runner_session_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.runner.session")

    assert module.__name__ == "loom.engine.runner.session"
    assert callable(module.RunnerSession)
    assert callable(module.new_runner_session)


def test_runner_core_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.runner.core")

    assert module.__name__ == "loom.engine.runner.core"
    assert callable(module.SubtaskRunner)


def test_runner_policy_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.runner.policy")

    assert module.__name__ == "loom.engine.runner.policy"
    assert callable(module.validate_deliverable_write_policy)
    assert callable(module.validate_sealed_artifact_mutation_policy)


def test_runner_compaction_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.runner.compaction")

    assert module.__name__ == "loom.engine.runner.compaction"
    assert callable(module.is_model_request_overflow_error)
    assert callable(module.rewrite_tool_payload_for_overflow)


def test_runner_telemetry_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.runner.telemetry")

    assert module.__name__ == "loom.engine.runner.telemetry"
    assert callable(module.emit_artifact_ingest_telemetry)
    assert callable(module.emit_compaction_policy_decision_from_diagnostics)


def test_runner_memory_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.runner.memory")

    assert module.__name__ == "loom.engine.runner.memory"
    assert callable(module.extract_memory)
    assert callable(module.parse_memory_entries)


def test_runner_execution_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.runner.execution")

    assert module.__name__ == "loom.engine.runner.execution"
    assert callable(module.run_subtask)
    assert module._COMPACTOR_EVENT_CONTEXT is not None


def test_engine_path_policy_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.path_policy")

    assert module.__name__ == "loom.engine.path_policy"
    assert callable(module.normalize_path_for_policy)
    assert callable(module.normalize_deliverable_paths)
    assert callable(module.looks_like_deliverable_variant)


def test_verification_types_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.types")

    assert module.__name__ == "loom.engine.verification.types"
    assert callable(module.Check)
    assert callable(module.VerificationResult)


def test_verification_tier1_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.tier1")

    assert module.__name__ == "loom.engine.verification.tier1"
    assert callable(module.DeterministicVerifier)


def test_verification_tier2_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.tier2")

    assert module.__name__ == "loom.engine.verification.tier2"
    assert callable(module.LLMVerifier)


def test_verification_gates_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.gates")

    assert module.__name__ == "loom.engine.verification.gates"
    assert callable(module.VerificationGates)
    assert callable(module.VotingVerifier)


def test_verification_parsing_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.parsing")

    assert module.__name__ == "loom.engine.verification.parsing"
    assert callable(module.parse_verifier_response)
    assert callable(module.coerce_assessment_from_text)


def test_verification_prompting_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.prompting")

    assert module.__name__ == "loom.engine.verification.prompting"
    assert callable(module.build_verifier_prompt)
    assert callable(module.build_repair_prompt)


def test_verification_placeholder_guard_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.placeholder_guard")

    assert module.__name__ == "loom.engine.verification.placeholder_guard"
    assert callable(module.is_placeholder_claim_failure)
    assert callable(module.scan_placeholder_markers)


def test_verification_claims_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.claims")

    assert module.__name__ == "loom.engine.verification.claims"
    assert callable(module.extract_claim_lifecycle)
    assert callable(module.attach_claim_lifecycle)


def test_verification_events_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.events")

    assert module.__name__ == "loom.engine.verification.events"
    assert callable(module.emit_verification_outcome)
    assert callable(module.emit_instrumentation_events)


def test_verification_policy_submodule_import_smoke() -> None:
    module = importlib.import_module("loom.engine.verification.policy")

    assert module.__name__ == "loom.engine.verification.policy"
    assert callable(module.classify_shadow_diff)
    assert callable(module.aggregate_non_failing)
    assert callable(module.legacy_result_from_tiers)
