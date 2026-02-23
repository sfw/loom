"""Tests for research-oriented built-in tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.research.models import AcademicResult, ArchiveResult
from loom.tools.academic_search import AcademicSearchTool
from loom.tools.archive_access import ArchiveAccessTool
from loom.tools.citation_manager import CitationManagerTool
from loom.tools.fact_checker import FactCheckerTool
from loom.tools.inflation_calculator import InflationCalculatorTool
from loom.tools.peer_review_simulator import PeerReviewSimulatorTool
from loom.tools.registry import ToolContext
from loom.tools.timeline_visualizer import TimelineVisualizerTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace)


class TestAcademicSearchTool:
    async def test_requires_query(self, ctx: ToolContext):
        tool = AcademicSearchTool()
        result = await tool.execute({}, ctx)
        assert not result.success

    async def test_merges_provider_results(self, monkeypatch, ctx: ToolContext):
        tool = AcademicSearchTool()

        async def fake_query(provider, *, query, max_results, client):
            del query, max_results, client
            if provider == "crossref":
                return [
                    AcademicResult(
                        title="Roman Roads and Trade",
                        authors=["A. Historian"],
                        year=2020,
                        venue="Journal of History",
                        url="https://example.org/roads",
                        doi="10.1/roads",
                        source_db="crossref",
                        source_type="journal",
                        citation_count=12,
                        confidence=0.9,
                    )
                ]
            return [
                AcademicResult(
                    title="Roman Roads and Trade",
                    authors=["A. Historian"],
                    year=2020,
                    venue="arXiv",
                    url="https://example.org/roads",
                    source_db="arxiv",
                    source_type="preprint",
                    confidence=0.8,
                )
            ]

        monkeypatch.setattr("loom.tools.academic_search._query_provider", fake_query)
        result = await tool.execute(
            {
                "query": "roman trade",
                "providers": ["crossref", "arxiv"],
                "max_results": 10,
            },
            ctx,
        )
        assert result.success
        assert result.data["count"] == 1
        assert result.data["results"][0]["title"] == "Roman Roads and Trade"


class TestArchiveAccessTool:
    async def test_requires_query(self, ctx: ToolContext):
        tool = ArchiveAccessTool()
        result = await tool.execute({}, ctx)
        assert not result.success

    async def test_collects_archive_results(self, monkeypatch, ctx: ToolContext):
        tool = ArchiveAccessTool()

        async def fake_query(source, *, query, max_results, client):
            del query, max_results, client
            return [
                ArchiveResult(
                    title=f"{source} result",
                    creator="Archivist",
                    date="1932-01-01",
                    repository=source,
                    record_url=f"https://example.org/{source}",
                    access_url=f"https://example.org/{source}",
                    rights="public domain",
                    snippet="snapshot",
                    media_type="text",
                )
            ]

        monkeypatch.setattr("loom.tools.archive_access._query_archive", fake_query)
        result = await tool.execute(
            {
                "query": "dust bowl",
                "archive_sources": ["internet_archive", "wikimedia"],
                "media_types": ["text"],
            },
            ctx,
        )
        assert result.success
        assert result.data["count"] == 2


class TestCitationManagerTool:
    async def test_add_validate_and_format(self, ctx: ToolContext, workspace: Path):
        tool = CitationManagerTool()

        add = await tool.execute(
            {
                "operation": "add",
                "path": "references.json",
                "citation": {
                    "title": "A Primary Source",
                    "authors": ["Jane Doe"],
                    "year": 1945,
                    "url": "https://example.org/primary",
                    "doi": "10.10/test",
                },
            },
            ctx,
        )
        assert add.success

        validate = await tool.execute(
            {"operation": "validate", "path": "references.json"},
            ctx,
        )
        assert validate.success
        assert validate.data["issue_count"] == 0

        formatted = await tool.execute(
            {
                "operation": "format",
                "path": "references.json",
                "style": "apa",
                "output_path": "references-apa.md",
            },
            ctx,
        )
        assert formatted.success
        assert (workspace / "references-apa.md").exists()

    async def test_map_claims(self, ctx: ToolContext, workspace: Path):
        tool = CitationManagerTool()
        await tool.execute(
            {
                "operation": "add",
                "path": "references.json",
                "citation": {
                    "title": "Industrial Output in 1939",
                    "authors": ["Ada Smith"],
                    "year": 1939,
                    "url": "https://example.org/output",
                },
            },
            ctx,
        )

        result = await tool.execute(
            {
                "operation": "map_claims",
                "path": "references.json",
                "claims": ["Industrial output rose in 1939"],
                "output_path": "claim-map.csv",
            },
            ctx,
        )
        assert result.success
        assert (workspace / "claim-map.csv").exists()
        assert result.data["rows"] >= 1


class TestFactCheckerTool:
    async def test_fact_checker_with_local_sources(self, ctx: ToolContext, workspace: Path):
        source = workspace / "source.txt"
        source.write_text(
            "The city population in 1900 was 12000 according to the census.",
            encoding="utf-8",
        )

        tool = FactCheckerTool()
        result = await tool.execute(
            {
                "claims": ["The city population in 1900 was 12000"],
                "sources": ["source.txt"],
                "strictness": "standard",
            },
            ctx,
        )
        assert result.success
        assert result.data["counts"]["supported"] >= 1
        assert (workspace / "fact-check-report.md").exists()
        assert (workspace / "fact-check-report.csv").exists()


class TestPeerReviewSimulatorTool:
    async def test_peer_review_outputs(self, ctx: ToolContext, workspace: Path):
        tool = PeerReviewSimulatorTool()
        content = """
# Draft

This report argues that transit demand increased in 1924.

## Method
We used public records and archival notes.

## Limitations
Data quality varies by district.

Sources: https://example.org/notes
"""
        result = await tool.execute(
            {
                "content": content,
                "review_type": "methodology",
                "num_reviewers": 2,
            },
            ctx,
        )
        assert result.success
        assert result.data["weighted_score"] > 0
        assert (workspace / "peer-review.md").exists()
        assert (workspace / "peer-review.json").exists()


class TestInflationCalculatorTool:
    async def test_inflation_conversion(self, ctx: ToolContext):
        tool = InflationCalculatorTool()
        result = await tool.execute(
            {
                "amount": 100,
                "from_year": 2000,
                "to_year": 2020,
                "index": "cpi_u",
            },
            ctx,
        )
        assert result.success
        assert result.data["adjusted_amount"] > 100
        assert result.data["series_version"]


class TestTimelineVisualizerTool:
    async def test_builds_timeline_artifacts(self, ctx: ToolContext, workspace: Path):
        tool = TimelineVisualizerTool()
        events = [
            {
                "date": "1930",
                "title": "Policy enacted",
                "entity": "Gov",
                "source": "https://example.org/policy",
            },
            {
                "date": "1932-05",
                "title": "Policy enacted",
                "entity": "Gov",
                "source": "https://example.org/update",
            },
            {
                "date": "1935-01-03",
                "title": "Outcomes published",
                "entity": "Gov",
            },
        ]
        result = await tool.execute(
            {
                "events": events,
                "group_by": "entity",
                "output_formats": ["markdown", "mermaid", "json", "csv"],
                "output_prefix": "policy-timeline",
            },
            ctx,
        )
        assert result.success
        assert result.data["event_count"] == 3
        assert result.data["conflict_count"] >= 1
        assert (workspace / "policy-timeline.md").exists()
        assert (workspace / "policy-timeline.mermaid").exists()
        assert (workspace / "policy-timeline.json").exists()
        assert (workspace / "policy-timeline.csv").exists()
