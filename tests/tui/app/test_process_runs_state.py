from __future__ import annotations

from types import SimpleNamespace

from loom.tui.app.constants import (
    _PROCESS_RUN_LAUNCH_STAGE_INDEX,
    _PROCESS_RUN_LAUNCH_STAGE_LABEL,
    _PROCESS_RUN_LAUNCH_STAGES,
)
from loom.tui.app.process_runs import state


def test_elapsed_and_pause_bookkeeping() -> None:
    run = SimpleNamespace(
        status="running",
        started_at=10.0,
        ended_at=70.0,
        paused_started_at=0.0,
        paused_accumulated_seconds=5.0,
    )
    assert state.elapsed_seconds_for_run(run) == 55.0

    state.set_process_run_status(run, "paused", now=80.0)
    assert run.status == "paused"
    assert run.paused_started_at == 80.0

    state.set_process_run_status(run, "running", now=90.0)
    assert run.status == "running"
    assert run.paused_started_at == 0.0
    assert run.paused_accumulated_seconds == 15.0


def test_stage_rows_and_summary() -> None:
    run = SimpleNamespace(
        launch_stage="queueing_delegate",
        status="cancel_requested",
        launch_error="",
    )
    rows = state.process_run_stage_rows(
        run,
        stages=_PROCESS_RUN_LAUNCH_STAGES,
        stage_index=_PROCESS_RUN_LAUNCH_STAGE_INDEX,
        one_line=lambda text, _max_len: str(text),
    )
    assert any(row["id"] == "stage:cancel-requested" for row in rows)

    summary = state.process_run_stage_summary_row(
        run,
        stage_labels=_PROCESS_RUN_LAUNCH_STAGE_LABEL,
        launch_stage_label=lambda stage: state.process_run_launch_stage_label(
            stage,
            stage_labels=_PROCESS_RUN_LAUNCH_STAGE_LABEL,
        ),
    )
    assert summary is not None
    assert "Launch stage:" in summary["content"]


def test_status_normalization() -> None:
    assert state.normalize_process_run_status("planning") == "running"
    assert state.normalize_process_run_status("completed") == "completed"
    assert state.is_process_run_busy_status("running") is True
    assert state.format_elapsed(3661) == "1:01:01"
