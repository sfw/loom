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
        assert result.data.get("url") == "https://example.com/report.pdf"
        assert result.data.get("source_url") == "https://example.com/report.pdf"
        assert isinstance(result.data.get("extracted_chars"), int)
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

    @pytest.mark.asyncio
    async def test_query_reads_relevant_snippets_from_text_artifact(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        content = (
            "# Platform Reference\n\n"
            "Background information about general account settings.\n\n"
            "The auth token refresh endpoint rotates bearer tokens every 15 minutes "
            "and supports manual revocation for compromised sessions.\n\n"
            "Additional product notes about billing and user seats.\n"
        )
        record = persist_fetch_artifact(
            content_bytes=content.encode("utf-8"),
            source_url="https://example.com/reference",
            media_type="text/markdown",
            content_kind=ContentKind.TEXT,
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
        result = await tool.execute(
            {
                "artifact_ref": record.artifact_ref,
                "query": "auth token refresh",
                "max_snippets": 2,
            },
            ctx,
        )

        assert result.success is True
        assert "Relevant snippets for query: auth token refresh" in result.output
        assert "rotates bearer tokens every 15 minutes" in result.output
        assert isinstance(result.data, dict)
        assert result.data.get("retrieval_strategy") == "query_snippets"
        assert result.data.get("query") == "auth token refresh"
