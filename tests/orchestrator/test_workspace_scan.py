"""Orchestrator workspace document scan tests."""

from __future__ import annotations

import pytest

from loom.engine.orchestrator import Orchestrator
from tests.orchestrator.conftest import (
    _make_config,
    _make_event_bus,
    _make_mock_memory,
    _make_mock_prompts,
    _make_mock_router,
    _make_mock_tools,
    _make_state_manager,
)


class TestWorkspaceDocumentScan:
    """Tests for the expanded workspace analysis that includes non-code files."""

    def _make_orchestrator(self, tmp_path):
        return Orchestrator(
            model_router=_make_mock_router(),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

    def test_scan_finds_documents_by_category(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / "README.md").write_text("# Hello")
        (workspace / "spec.pdf").write_bytes(b"%PDF-fake")
        (workspace / "data.csv").write_text("a,b\n1,2")
        (workspace / "logo.png").write_bytes(b"\x89PNG")
        (workspace / "slides.pptx").write_bytes(b"PK-fake")

        orch = self._make_orchestrator(tmp_path)
        result = orch._scan_workspace_documents(workspace)

        assert "Documents and non-code files:" in result
        assert "README.md" in result
        assert "spec.pdf" in result
        assert "data.csv" in result
        assert "logo.png" in result
        assert "slides.pptx" in result

    def test_scan_skips_hidden_and_noise_dirs(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / ".git").mkdir()
        (workspace / ".git" / "notes.md").write_text("internal")
        (workspace / "node_modules").mkdir()
        (workspace / "node_modules" / "pkg.json").write_text("{}")
        # This one should be found
        (workspace / "docs").mkdir()
        (workspace / "docs" / "guide.md").write_text("# Guide")

        orch = self._make_orchestrator(tmp_path)
        result = orch._scan_workspace_documents(workspace)

        assert "guide.md" in result
        assert "notes.md" not in result
        assert "pkg.json" not in result

    def test_scan_returns_empty_for_code_only_workspace(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / "main.py").write_text("print('hello')")
        (workspace / "utils.go").write_text("package main")

        orch = self._make_orchestrator(tmp_path)
        result = orch._scan_workspace_documents(workspace)

        assert result == ""

    def test_scan_respects_max_per_category(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        for i in range(20):
            (workspace / f"doc_{i:02d}.md").write_text(f"# Doc {i}")

        orch = self._make_orchestrator(tmp_path)
        result = orch._scan_workspace_documents(workspace, max_per_category=5)

        # Should only list 5
        md_lines = [line for line in result.splitlines() if "doc_" in line]
        assert len(md_lines) == 5

    @pytest.mark.asyncio
    async def test_analyze_workspace_includes_documents(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / "app.py").write_text("class App:\n    pass\n")
        (workspace / "README.md").write_text("# My App")
        (workspace / "data.csv").write_text("x,y\n1,2")

        orch = self._make_orchestrator(tmp_path)
        result = await orch._analyze_workspace(workspace)

        # Code analysis should find the Python file
        assert "App" in result
        # Document scan should find the non-code files
        assert "README.md" in result
        assert "data.csv" in result
