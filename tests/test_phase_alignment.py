"""Tests for process phase alignment helpers."""

from loom.processes.phase_alignment import infer_phase_id_for_subtask, match_phase_id_for_subtask


def test_match_phase_id_exact_id() -> None:
    phase_id, score = match_phase_id_for_subtask(
        subtask_id="market-sizing",
        text="Estimate TAM and SAM",
        phase_ids=["market-sizing", "risk-map"],
        phase_descriptions={
            "market-sizing": "Estimate TAM and SAM with assumptions.",
            "risk-map": "List risks and mitigations.",
        },
    )
    assert phase_id == "market-sizing"
    assert score >= 1.0


def test_infer_phase_id_from_description_and_deliverables() -> None:
    phase_id = infer_phase_id_for_subtask(
        subtask_id="build-slogan-longlist",
        text=(
            "Generate a high-volume longlist of slogan options and write "
            "slogan-longlist.csv before filtering."
        ),
        phase_ids=["slogan-divergence", "final-selection"],
        phase_descriptions={
            "slogan-divergence": (
                "Generate a high-volume longlist of slogan options before filtering."
            ),
            "final-selection": "Select finalists and finalize launch copy.",
        },
        phase_deliverables={
            "slogan-divergence": ["slogan-longlist.csv"],
            "final-selection": ["final-slogans.md"],
        },
    )
    assert phase_id == "slogan-divergence"


def test_infer_phase_id_returns_empty_on_ambiguous_text() -> None:
    phase_id = infer_phase_id_for_subtask(
        subtask_id="planner-generated-id",
        text="Do analysis",
        phase_ids=["phase-a", "phase-b"],
        phase_descriptions={
            "phase-a": "Collect data and analyze results.",
            "phase-b": "Collect data and analyze results.",
        },
        phase_deliverables={
            "phase-a": ["file-a.md"],
            "phase-b": ["file-b.md"],
        },
    )
    assert phase_id == ""
