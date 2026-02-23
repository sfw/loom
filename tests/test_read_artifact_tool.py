"""Tests for the read_artifact tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.ingest.artifacts import persist_fetch_artifact
from loom.ingest.router import ContentKind
from loom.tools.read_artifact import ReadArtifactTool
from loom.tools.registry import ToolContext


class TestReadArtifactTool:
    @pytest.mark.asyncio
    async def test_reads_persisted_artifact_by_ref(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        record = persist_fetch_artifact(
            content_bytes=b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n",
            source_url="https://example.com/report.pdf",
            media_type="application/pdf",
            content_kind=ContentKind.PDF,
            workspace=workspace,
            scratch_dir=scratch,
            subtask_id="research-pass",
        )

        tool = ReadArtifactTool()
        ctx = ToolContext(
            workspace=workspace,
            read_roots=[],
            scratch_dir=scratch,
            changelog=None,
            subtask_id="research-pass",
            auth_context=None,
        )
        result = await tool.execute({"artifact_ref": record.artifact_ref}, ctx)

        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data.get("artifact_ref") == record.artifact_ref
        assert result.data.get("artifact_path") == str(record.path)
        assert "artifact" in result.output.lower()

    @pytest.mark.asyncio
    async def test_fails_when_artifact_missing(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        tool = ReadArtifactTool()
        ctx = ToolContext(workspace=workspace)

        result = await tool.execute({"artifact_ref": "af_does_not_exist"}, ctx)

        assert result.success is False
        assert "Artifact not found" in (result.error or "")
