from __future__ import annotations

import csv

from loom.engine.correction import (
    Blocker,
    BlockerClass,
    CorrectionDecision,
    CorrectionExecutor,
    CorrectionHandler,
    CorrectionState,
    ProgressVector,
    Repairability,
    RepairAction,
)


def _decision(*, target: str, actual: int, expected: int) -> CorrectionDecision:
    return CorrectionDecision(
        cycle_id="corr-test",
        blockers=(
            Blocker(
                code="csv_schema_mismatch",
                message="row width mismatch",
                blocking=True,
                repairability=Repairability.AUTOMATIC,
                blocker_class=BlockerClass.ARTIFACT_SCHEMA,
                targets=(target,),
            ),
        ),
        repairability=Repairability.AUTOMATIC,
        handler=CorrectionHandler.SCHEMA_REPAIR,
        state=CorrectionState.PLANNED,
        actions=(
            RepairAction(
                action_type="repair_structured_output_schema",
                handler=CorrectionHandler.SCHEMA_REPAIR,
                arguments={
                    "targets": [target],
                    "diagnostics": [{
                        "target": target,
                        "row_number": 2,
                        "actual_columns": actual,
                        "expected_columns": expected,
                    }],
                },
            ),
        ),
        progress=ProgressVector(1, 1, 1, 0, 0, 1, 0.0),
        progress_made=True,
        no_progress_count=0,
        stop_for_no_progress=False,
    )


def test_executor_repairs_only_unambiguous_trailing_empty_overflow(tmp_path):
    target = tmp_path / "register.csv"
    target.write_text("id,risk,owner\n1,delay,ops,,\n", encoding="utf-8")

    result = CorrectionExecutor().execute(
        workspace=tmp_path,
        decision=_decision(target="register.csv", actual=5, expected=3),
    )

    assert result.applied is True
    assert result.changed_targets == ("register.csv",)
    with target.open(newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [
            ["id", "risk", "owner"],
            ["1", "delay", "ops"],
        ]


def test_executor_refuses_ambiguous_nonempty_overflow(tmp_path):
    target = tmp_path / "register.csv"
    original = "id,risk,owner\n1,delay,ops,unexpected\n"
    target.write_text(original, encoding="utf-8")

    result = CorrectionExecutor().execute(
        workspace=tmp_path,
        decision=_decision(target="register.csv", actual=4, expected=3),
    )

    assert result.applied is False
    assert result.reason == "ambiguous_nonempty_overflow"
    assert target.read_text(encoding="utf-8") == original


def test_executor_refuses_workspace_escape(tmp_path):
    result = CorrectionExecutor().execute(
        workspace=tmp_path,
        decision=_decision(target="../outside.csv", actual=4, expected=3),
    )

    assert result.applied is False
    assert result.reason == "target_not_safe"
