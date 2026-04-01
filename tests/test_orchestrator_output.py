"""Focused tests for extracted orchestrator output helpers."""

from __future__ import annotations

from types import SimpleNamespace

from loom.engine.orchestrator import output as orchestrator_output
from loom.recovery.retry import RetryStrategy


def test_output_helpers_use_defaults_without_process() -> None:
    assert orchestrator_output.output_intermediate_root(None) == ".loom/phase-artifacts"
    assert orchestrator_output.output_enforce_single_writer(None) is True
    assert orchestrator_output.output_conflict_policy(None) == "defer_fifo"
    assert orchestrator_output.output_publish_mode(None) == "transactional"
    assert (
        orchestrator_output.phase_finalizer_input_policy(None, "phase-a")
        == "require_all_workers"
    )


def test_output_helpers_read_coordination_config() -> None:
    process = SimpleNamespace(
        output_coordination=SimpleNamespace(
            intermediate_root=".loom/custom",
            enforce_single_writer=False,
            conflict_policy="fail_fast",
            publish_mode="best_effort",
        ),
        phase_finalizer_input_policy=lambda phase_id: (
            "allow_partial" if phase_id == "phase-a" else "require_all_workers"
        ),
    )

    assert orchestrator_output.output_intermediate_root(process) == ".loom/custom"
    assert orchestrator_output.output_enforce_single_writer(process) is False
    assert orchestrator_output.output_conflict_policy(process) == "fail_fast"
    assert orchestrator_output.output_publish_mode(process) == "best_effort"
    assert orchestrator_output.phase_finalizer_input_policy(process, "phase-a") == "allow_partial"


def test_output_helpers_reject_invalid_values() -> None:
    process = SimpleNamespace(
        output_coordination=SimpleNamespace(
            conflict_policy="unknown",
            publish_mode="weird",
        ),
        phase_finalizer_input_policy=lambda _phase_id: "invalid",
    )

    assert orchestrator_output.output_conflict_policy(process) == "defer_fifo"
    assert orchestrator_output.output_publish_mode(process) == "transactional"
    assert (
        orchestrator_output.phase_finalizer_input_policy(process, "phase-a")
        == "require_all_workers"
    )


def test_prioritize_runnable_for_output_conflicts_prefers_starved_subtasks() -> None:
    runnable = [
        SimpleNamespace(id="s1", phase_id="p"),
        SimpleNamespace(id="s2", phase_id="p"),
        SimpleNamespace(id="s3", phase_id="p"),
    ]
    tracker = {
        "s2": {"streak": 1, "first_seq": 4},
        "s3": {"streak": 3, "first_seq": 2},
    }

    prioritized = orchestrator_output.prioritize_runnable_for_output_conflicts(
        runnable=runnable,
        conflict_tracker=tracker,
        starvation_threshold=3,
    )

    assert [item.id for item in prioritized] == ["s3", "s2", "s1"]


def test_select_conflict_safe_batch_defers_overlapping_deliverables() -> None:
    subtask_a = SimpleNamespace(id="s1", phase_id="phase-a")
    subtask_b = SimpleNamespace(id="s2", phase_id="phase-a")
    runnable = [subtask_a, subtask_b]
    tracker: dict[str, dict[str, object]] = {}
    paths = {
        "s1": ["deliverable.md"],
        "s2": ["deliverable.md"],
    }

    selected, deferred, next_seq = orchestrator_output.select_conflict_safe_batch(
        runnable=runnable,
        max_parallel=2,
        conflict_tracker=tracker,
        sequence_counter=1,
        canonical_paths_for_subtask=lambda subtask: list(paths[subtask.id]),
        starvation_threshold=2,
        active_pending_ids={"s1", "s2"},
    )

    assert [item.id for item in selected] == ["s1"]
    assert deferred and deferred[0]["subtask_id"] == "s2"
    assert deferred[0]["conflicting_with"] == ["s1"]
    assert deferred[0]["deferral_streak"] == 1
    assert next_seq == 2


def test_normalize_deliverable_paths_for_conflict_normalizes_workspace_relative(tmp_path) -> None:
    normalized = orchestrator_output.normalize_deliverable_paths_for_conflict(
        ["reports/summary.md", str(tmp_path / "reports" / "summary.md")],
        workspace=tmp_path,
    )

    assert normalized == ["reports/summary.md"]


def test_canonical_deliverable_paths_for_subtask_uses_output_policy(tmp_path) -> None:
    task = SimpleNamespace(workspace=str(tmp_path))
    subtask = SimpleNamespace(id="s1", phase_id="phase-a")

    paths = orchestrator_output.canonical_deliverable_paths_for_subtask(
        task=task,
        subtask=subtask,
        output_write_policy_for_subtask=lambda _subtask: {
            "expected_deliverables": ["report.md", "report.md"],
        },
    )

    assert paths == ["report.md"]


def test_manifest_only_input_violations_flags_unallowed_intermediate_reads(tmp_path) -> None:
    manifest_path = tmp_path / ".loom/phase-artifacts/run-1/phase-a/manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")

    orchestrator = SimpleNamespace(
        _intermediate_phase_prefix=lambda **_kwargs: (
            ".loom/phase-artifacts/run-1/phase-a"
        ),
        _phase_artifact_manifest_path=lambda **_kwargs: manifest_path,
        _intermediate_read_paths_from_tool_calls=lambda **_kwargs: [
            ".loom/phase-artifacts/run-1/phase-a/worker-a/report.md",
            ".loom/phase-artifacts/run-1/phase-a/worker-b/secret.md",
            "notes.md",
        ],
    )
    task = SimpleNamespace(workspace=str(tmp_path))
    subtask = SimpleNamespace(phase_id="phase-a")

    violations = orchestrator_output.manifest_only_input_violations(
        orchestrator,
        task=task,
        subtask=subtask,
        tool_calls=[],
        allowed_manifest_paths=[
            ".loom/phase-artifacts/run-1/phase-a/worker-a/report.md",
        ],
        allowed_extra_prefixes=[],
    )

    assert violations == [".loom/phase-artifacts/run-1/phase-a/worker-b/secret.md"]


def test_commit_finalizer_stage_publish_commits_staged_files(tmp_path) -> None:
    stage_rel = "staging/report.md"
    canonical_rel = "report.md"
    stage_path = tmp_path / stage_rel
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text("published", encoding="utf-8")
    task = SimpleNamespace(id="task-1", workspace=str(tmp_path), metadata={})
    subtask = SimpleNamespace(id="finalizer")

    class _Stub:
        def __init__(self) -> None:
            self.sealed: list[str] = []
            self.restored = False

        def _artifact_seals_snapshot(self, _task) -> dict[str, dict[str, object]]:
            return {}

        def _seal_paths_after_commit(self, *, task, subtask_id: str, paths: list[str]) -> None:
            del task, subtask_id
            self.sealed = list(paths)

        def _restore_artifact_seals_snapshot(self, *, task, snapshot) -> None:
            del task, snapshot
            self.restored = True

    stub = _Stub()
    success, message = orchestrator_output.commit_finalizer_stage_publish(
        stub,
        task=task,
        subtask=subtask,
        stage_plan={
            "enabled": True,
            "stage_to_canonical": {stage_rel: canonical_rel},
        },
    )

    assert success is True
    assert message == ""
    assert (tmp_path / canonical_rel).read_text(encoding="utf-8") == "published"
    assert not stage_path.exists()
    assert stub.sealed == [canonical_rel]
    assert stub.restored is False


def test_augment_retry_context_for_outputs_describes_direct_one_shot_writes() -> None:
    orchestrator = SimpleNamespace(
        _files_from_attempts=lambda _attempts: [],
        _output_intermediate_root=lambda: ".loom/phase-artifacts",
        _OUTPUT_ROLE_PHASE_FINALIZER="phase_finalizer",
    )
    subtask = SimpleNamespace(
        id="draft-report",
        phase_id="research",
        output_role="worker",
        output_strategy="direct",
    )

    context = orchestrator_output._augment_retry_context_for_outputs(
        orchestrator,
        subtask=subtask,
        attempts=[],
        strategy=RetryStrategy.GENERIC,
        expected_deliverables=["report.md"],
        forbidden_deliverables=[],
        base_context="Base",
    )

    assert "CANONICAL DELIVERABLE FILES FOR THIS SUBTASK" in context
    assert "Each listed deliverable may be written at most once" in context
