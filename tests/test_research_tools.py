"""Tests for research-oriented built-in tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.research.models import AcademicResult, ArchiveResult
from loom.tools.academic_search import AcademicSearchTool
from loom.tools.archive_access import ArchiveAccessTool
from loom.tools.citation_manager import CitationManagerTool
from loom.tools.correspondence_analysis import CorrespondenceAnalysisTool
from loom.tools.economic_data_api import EconomicDataApiTool
from loom.tools.fact_checker import FactCheckerTool
from loom.tools.historical_currency_normalizer import HistoricalCurrencyNormalizerTool
from loom.tools.humanize_writing import HumanizeWritingTool
from loom.tools.inflation_calculator import InflationCalculatorTool
from loom.tools.peer_review_simulator import PeerReviewSimulatorTool
from loom.tools.primary_source_ocr import OcrPage, PrimarySourceOcrTool
from loom.tools.registry import ToolContext
from loom.tools.social_network_mapper import SocialNetworkMapperTool
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


class TestEconomicDataApiTool:
    async def test_search_with_mock_provider(self, monkeypatch, ctx: ToolContext):
        tool = EconomicDataApiTool()

        async def fake_search(*, provider, query, max_results, client):
            del query, max_results, client
            return [
                {
                    "provider": provider,
                    "series_id": "FP.CPI.TOTL",
                    "title": "Consumer price index",
                    "dataset": "WDI",
                    "source_url": "https://example.org/worldbank",
                }
            ]

        monkeypatch.setattr("loom.tools.economic_data_api.economic_search", fake_search)
        result = await tool.execute(
            {"operation": "search", "query": "inflation", "provider": "world_bank"},
            ctx,
        )
        assert result.success
        assert result.data["count"] == 1
        assert result.data["results"][0]["series_id"] == "FP.CPI.TOTL"

    async def test_get_observations_with_fallback(self, monkeypatch, ctx: ToolContext):
        tool = EconomicDataApiTool()

        async def fake_obs(*, provider, series_id, **kwargs):
            del kwargs
            if provider == "world_bank":
                raise RuntimeError("provider down")
            return {
                "provider": provider,
                "series_id": series_id,
                "title": "CPI",
                "coverage": {"count": 2, "start": "2020", "end": "2021"},
                "observations": [
                    {"period": "2020", "value": 100.0},
                    {"period": "2021", "value": 110.0},
                ],
            }

        monkeypatch.setattr(
            "loom.tools.economic_data_api.economic_get_observations",
            fake_obs,
        )
        result = await tool.execute(
            {
                "operation": "get_observations",
                "providers": ["world_bank", "dbnomics"],
                "series_id": "WB/WDI/FP.CPI.TOTL",
            },
            ctx,
        )
        assert result.success
        assert result.data["provider"] == "dbnomics"
        assert result.data["coverage"]["count"] == 2


class TestHistoricalCurrencyNormalizerTool:
    async def test_fx_only(self, monkeypatch, ctx: ToolContext):
        tool = HistoricalCurrencyNormalizerTool()

        async def fake_convert(**kwargs):
            return {
                "converted_amount": 120.0,
                "source": "ecb_reference_rates",
                "warnings": [],
                "from_date_effective": "2020-01-01",
                "to_date_effective": "2021-01-01",
            }

        monkeypatch.setattr(
            "loom.tools.historical_currency_normalizer.convert_via_ecb_reference_rates",
            fake_convert,
        )
        result = await tool.execute(
            {
                "amount": 100,
                "from_currency": "USD",
                "to_currency": "EUR",
                "from_date": "2020-01-01",
                "to_date": "2021-01-01",
                "mode": "fx_only",
            },
            ctx,
        )
        assert result.success
        assert result.data["normalized_amount"] == 120.0
        assert result.data["mode"] == "fx_only"

    async def test_fx_and_us_inflation(self, monkeypatch, ctx: ToolContext):
        tool = HistoricalCurrencyNormalizerTool()

        async def fake_convert(*, amount, from_currency, to_currency, **kwargs):
            del kwargs
            if from_currency == "USD" and to_currency == "USD":
                return {
                    "converted_amount": amount,
                    "source": "ecb_reference_rates",
                    "warnings": [],
                }
            if from_currency == "USD" and to_currency == "EUR":
                return {
                    "converted_amount": amount * 0.9,
                    "source": "ecb_reference_rates",
                    "warnings": [],
                }
            return {
                "converted_amount": amount,
                "source": "ecb_reference_rates",
                "warnings": [],
            }

        class _Inflated:
            adjusted_amount = 150.0
            multiplier = 1.5
            percent_change = 50.0

        def fake_inflation(**kwargs):
            del kwargs
            return _Inflated()

        monkeypatch.setattr(
            "loom.tools.historical_currency_normalizer.convert_via_ecb_reference_rates",
            fake_convert,
        )
        monkeypatch.setattr(
            "loom.tools.historical_currency_normalizer.calculate_inflation",
            fake_inflation,
        )
        result = await tool.execute(
            {
                "amount": 100,
                "from_currency": "USD",
                "to_currency": "EUR",
                "from_date": "2010-01-01",
                "to_date": "2020-01-01",
                "mode": "fx_and_us_inflation",
            },
            ctx,
        )
        assert result.success
        assert result.data["inflation"]["inflation_multiplier"] == 1.5


class TestPrimarySourceOcrTool:
    async def test_image_ocr_success(self, monkeypatch, ctx: ToolContext, workspace: Path):
        tool = PrimarySourceOcrTool()
        img = workspace / "scan.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        monkeypatch.setattr(
            "loom.tools.primary_source_ocr._which",
            lambda name: "/usr/bin/" + name,
        )

        async def fake_ocr_image(path, *, language):
            del path, language
            return OcrPage(page=1, text="Extracted text", engine="tesseract")

        monkeypatch.setattr("loom.tools.primary_source_ocr._ocr_image", fake_ocr_image)
        result = await tool.execute({"path": "scan.png", "cleanup": "none"}, ctx)
        assert result.success
        assert result.data["page_count"] == 1
        assert "Extracted text" in result.data["text"]

    async def test_image_ocr_requires_tesseract(
        self,
        monkeypatch,
        ctx: ToolContext,
        workspace: Path,
    ):
        tool = PrimarySourceOcrTool()
        img = workspace / "scan2.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        monkeypatch.setattr("loom.tools.primary_source_ocr._which", lambda _name: None)
        result = await tool.execute({"path": "scan2.png"}, ctx)
        assert not result.success
        assert "Tesseract" in result.error


class TestCorrespondenceAnalysisTool:
    async def test_runs_on_inline_table(self, ctx: ToolContext, workspace: Path):
        tool = CorrespondenceAnalysisTool()
        result = await tool.execute(
            {
                "table": [[10, 20], [30, 40]],
                "row_labels": ["A", "B"],
                "column_labels": ["X", "Y"],
                "output_formats": ["json"],
                "output_prefix": "ca-test",
            },
            ctx,
        )
        assert result.success
        assert result.data["dimensions_retained"] >= 1
        assert (workspace / "ca-test.json").exists()

    async def test_records_mode(self, ctx: ToolContext):
        tool = CorrespondenceAnalysisTool()
        result = await tool.execute(
            {
                "records": [
                    {"segment": "A", "outcome": "Y", "count": 2},
                    {"segment": "A", "outcome": "N", "count": 1},
                    {"segment": "B", "outcome": "Y", "count": 4},
                    {"segment": "B", "outcome": "N", "count": 3},
                ],
                "row_field": "segment",
                "column_field": "outcome",
                "value_field": "count",
            },
            ctx,
        )
        assert result.success
        assert len(result.data["rows"]) == 2


class TestSocialNetworkMapperTool:
    async def test_structured_edges(self, ctx: ToolContext, workspace: Path):
        tool = SocialNetworkMapperTool()
        result = await tool.execute(
            {
                "nodes": ["Alice", "Bob", "Carol"],
                "edges": [
                    {"source": "Alice", "target": "Bob"},
                    {"source": "Bob", "target": "Carol"},
                ],
                "output_formats": ["json", "csv"],
                "output_prefix": "network-test",
            },
            ctx,
        )
        assert result.success
        assert result.data["node_count"] == 3
        assert (workspace / "network-test.json").exists()
        assert (workspace / "network-test-nodes.csv").exists()
        assert (workspace / "network-test-edges.csv").exists()

    async def test_text_extraction(self, ctx: ToolContext):
        tool = SocialNetworkMapperTool()
        result = await tool.execute(
            {
                "text": "Alice met Bob. Bob worked with Carol.",
                "extract_relations_from_text": True,
            },
            ctx,
        )
        assert result.success
        assert result.data["edge_count"] >= 2


class TestHumanizeWritingTool:
    _ROBOTIC_TEXT = (
        "Our platform delivers value for your business. "
        "Our platform delivers value for your business. "
        "It is very important to note that our platform delivers value for your business. "
        "In conclusion, our platform is very good."
    )

    _NATURAL_TEXT = (
        "Last quarter, the support team resolved 418 tickets in under six hours on average. "
        "Most requests were onboarding-related, so we replaced the setup checklist with annotated "
        "screenshots and a short walkthrough video. "
        "Two weeks later, repeat tickets dropped by 31 percent."
    )

    async def test_analyze_returns_score_and_issues(self, ctx: ToolContext):
        tool = HumanizeWritingTool()
        result = await tool.execute(
            {
                "operation": "analyze",
                "content": self._ROBOTIC_TEXT,
                "mode": "report",
            },
            ctx,
        )
        assert result.success
        assert result.data["operation"] == "analyze"
        assert result.data["report"]["humanization_score"] < 85
        assert len(result.data["report"]["issues"]) >= 1
        assert len(result.data["recommended_edits"]) >= 1

    async def test_plan_rewrite_applies_constraints(self, ctx: ToolContext):
        tool = HumanizeWritingTool()
        result = await tool.execute(
            {
                "operation": "plan_rewrite",
                "content": self._ROBOTIC_TEXT,
                "constraints": {
                    "preserve_terms": ["Loom", "support team"],
                    "banned_phrases": ["very important"],
                },
                "max_recommendations": 6,
            },
            ctx,
        )
        assert result.success
        edits = result.data["recommended_edits"]
        assert len(edits) >= 1
        assert any("Preserve these exact terms" in action for action in edits)
        assert "preserve_terms" in result.data["constraints_applied"]

    async def test_compare_detects_improvement(self, ctx: ToolContext):
        tool = HumanizeWritingTool()
        result = await tool.execute(
            {
                "operation": "compare",
                "content": self._NATURAL_TEXT,
                "baseline_content": self._ROBOTIC_TEXT,
                "mode": "report",
            },
            ctx,
        )
        assert result.success
        assert result.data["improved"] is True
        assert result.data["score_delta"] > 0
        assert "clarity" in result.data["sub_score_delta"]

    async def test_evaluate_writes_artifacts_from_path(
        self,
        ctx: ToolContext,
        workspace: Path,
    ):
        tool = HumanizeWritingTool()
        draft_path = workspace / "draft.md"
        draft_path.write_text(self._NATURAL_TEXT, encoding="utf-8")

        result = await tool.execute(
            {
                "operation": "evaluate",
                "path": "draft.md",
                "target_score": 40,
                "output_path": "reports/humanized-writing.md",
                "output_json_path": "reports/humanized-writing.json",
            },
            ctx,
        )
        assert result.success
        assert result.data["report"]["passes_target"] is True
        assert (workspace / "reports" / "humanized-writing.md").exists()
        assert (workspace / "reports" / "humanized-writing.json").exists()

    async def test_compare_requires_baseline(self, ctx: ToolContext):
        tool = HumanizeWritingTool()
        result = await tool.execute(
            {
                "operation": "compare",
                "content": "Short draft",
            },
            ctx,
        )
        assert not result.success
        assert "baseline" in result.error.lower()
