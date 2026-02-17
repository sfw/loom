"""Deterministic contract tests for built-in process definitions."""

from __future__ import annotations

import pytest

from loom.config import Config
from loom.processes.schema import ProcessLoader
from loom.processes.testing import run_process_tests

BUILTIN_PROCESSES = (
    "investment-analysis",
    "marketing-strategy",
    "research-report",
    "competitive-intel",
    "consulting-engagement",
)


@pytest.mark.asyncio
@pytest.mark.process_contract
@pytest.mark.parametrize("process_name", BUILTIN_PROCESSES)
async def test_builtin_process_contracts_deterministic(process_name: str, tmp_path):
    """Each built-in process must pass deterministic contract execution."""
    loader = ProcessLoader(workspace=tmp_path)
    process_def = loader.load(process_name)

    results = await run_process_tests(
        process_def,
        config=Config(),
        workspace=tmp_path / process_name,
        include_live=False,
    )

    assert results, f"No deterministic cases selected for {process_name}"
    failed = [r for r in results if not r.passed]
    failure_lines: list[str] = []
    for result in failed:
        failure_lines.append(
            f"{process_name}:{result.case_id} -> {result.message}"
        )
        failure_lines.extend(
            f"  - {detail}" for detail in result.details
        )
    assert not failed, "\n".join(failure_lines)
