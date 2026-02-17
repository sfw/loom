"""Opt-in live canary tests for built-in process definitions."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from loom.config import Config, load_config
from loom.processes.schema import ProcessLoader, ProcessTestCase
from loom.processes.testing import ProcessCaseResult, run_process_case_live

BUILTIN_PROCESSES = (
    "investment-analysis",
    "marketing-strategy",
    "research-report",
    "competitive-intel",
    "consulting-engagement",
)


def _live_enabled() -> bool:
    value = os.getenv("LOOM_PROCESS_LIVE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _load_live_config() -> Config:
    configured = os.getenv("LOOM_PROCESS_LIVE_CONFIG", "").strip()
    if configured:
        return load_config(Path(configured))
    return load_config(None)


def _live_timeout_seconds() -> int:
    raw = os.getenv("LOOM_PROCESS_LIVE_TIMEOUT", "").strip()
    try:
        timeout = int(raw) if raw else 900
    except ValueError:
        timeout = 900
    return max(120, timeout)


def _emit_live_result(process_name: str, result: ProcessCaseResult) -> None:
    artifact_dir = os.getenv("LOOM_PROCESS_LIVE_ARTIFACT_DIR", "").strip()
    if not artifact_dir:
        return

    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "process": process_name,
        "case_id": result.case_id,
        "mode": result.mode,
        "passed": result.passed,
        "task_status": result.task_status,
        "duration_seconds": result.duration_seconds,
        "message": result.message,
        "details": result.details,
        "event_log_path": result.event_log_path,
    }
    with open(path / f"{process_name}.json", "w") as f:
        json.dump(payload, f, indent=2)
    if result.event_log_path:
        src = Path(result.event_log_path)
        if src.exists():
            shutil.copy2(src, path / f"{process_name}-events.jsonl")


@pytest.mark.asyncio
@pytest.mark.process_live
@pytest.mark.network
@pytest.mark.integration
@pytest.mark.parametrize("process_name", BUILTIN_PROCESSES)
async def test_builtin_processes_live_canary(process_name: str, tmp_path):
    if not _live_enabled():
        pytest.skip("Live process canary disabled (set LOOM_PROCESS_LIVE=1).")

    config = _load_live_config()
    if not config.models:
        pytest.skip("Live process canary requires configured models.")

    process_def = ProcessLoader(workspace=tmp_path).load(process_name)
    live_case = ProcessTestCase(
        id="live-canary",
        mode="live",
        goal=f"Run live canary for process {process_name} and produce required deliverables.",
        timeout_seconds=_live_timeout_seconds(),
        requires_network=True,
    )

    result = await run_process_case_live(
        process_def,
        live_case,
        config=config,
        workspace=tmp_path / process_name,
    )
    _emit_live_result(process_name, result)

    assert result.passed, "\n".join([result.message, *result.details])
