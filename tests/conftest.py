"""Shared test fixtures for Loom."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.config import Config, ExecutionConfig, MemoryConfig, ServerConfig


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test workspace."""
    return tmp_path


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Provide a test configuration with temp paths."""
    db_path = tmp_path / "test_loom.db"
    return Config(
        server=ServerConfig(host="127.0.0.1", port=9999),
        models={},
        memory=MemoryConfig(database_path=str(db_path)),
        execution=ExecutionConfig(max_subtask_retries=2, max_loop_iterations=10),
    )
