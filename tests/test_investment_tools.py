"""Tests for the investment/economic tool suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.factor_exposure_engine import FactorExposureEngineTool
from loom.tools.filing_event_parser import FilingEventParserTool
from loom.tools.macro_regime_engine import MacroRegimeEngineTool
from loom.tools.market_data_api import MarketDataApiTool
from loom.tools.opportunity_ranker import OpportunityRankerTool
from loom.tools.portfolio_evaluator import PortfolioEvaluatorTool
from loom.tools.portfolio_optimizer import PortfolioOptimizerTool
from loom.tools.portfolio_recommender import PortfolioRecommenderTool
from loom.tools.registry import ToolContext
from loom.tools.sec_fundamentals_api import SecFundamentalsApiTool
from loom.tools.sentiment_feeds_api import SentimentFeedsApiTool
from loom.tools.symbol_universe_api import SymbolUniverseApiTool
from loom.tools.valuation_engine import ValuationEngineTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace)


class TestMarketDataApiTool:
    async def test_get_returns(self, monkeypatch, ctx: ToolContext):
        tool = MarketDataApiTool()

        async def fake_prices(*, symbol, start, end, client):
            del start, end, client
            base = 100.0 if symbol.upper() == "AAPL" else 50.0
            return {
                "provider": "stooq",
                "symbol": symbol,
                "as_of": "2025-01-03",
                "source_url": "https://example.test/stooq",
                "rows": [
                    {"date": "2025-01-01", "close": base},
                    {"date": "2025-01-02", "close": base * 1.01},
                    {"date": "2025-01-03", "close": base * 1.02},
                ],
            }

        monkeypatch.setattr("loom.tools.market_data_api.fetch_stooq_daily_prices", fake_prices)
        result = await tool.execute(
            {
                "operation": "get_returns",
                "symbols": ["AAPL", "MSFT"],
            },
            ctx,
        )
        assert result.success
        assert result.data["operation"] == "get_returns"
        assert len(result.data["series"]) == 2


class TestSymbolUniverseApiTool:
    async def test_map_ticker_cik(self, monkeypatch, ctx: ToolContext):
        tool = SymbolUniverseApiTool()

        async def fake_map(*, client):
            del client
            return {
                "AAPL": {
                    "ticker": "AAPL",
                    "cik": "0000320193",
                    "name": "Apple Inc.",
                    "source": "sec",
                },
                "MSFT": {
                    "ticker": "MSFT",
                    "cik": "0000789019",
                    "name": "Microsoft",
                    "source": "sec",
                },
            }

        monkeypatch.setattr("loom.tools.symbol_universe_api.fetch_sec_ticker_map", fake_map)
        result = await tool.execute(
            {
                "operation": "map_ticker_cik",
                "tickers": ["AAPL", "MSFT", "XXXX"],
            },
            ctx,
        )
        assert result.success
        assert result.data["count"] == 2
        assert "XXXX" in result.data["missing"]


class TestSecFundamentalsApiTool:
    async def test_get_ttm_metrics(self, monkeypatch, ctx: ToolContext):
        tool = SecFundamentalsApiTool()

        async def fake_resolve(*, ticker, client):
            del ticker, client
            return {"ticker": "AAPL", "cik": "0000320193"}

        async def fake_facts(*, cik, client):
            del cik, client
            return {
                "entityName": "Apple Inc.",
                "facts": {
                    "us-gaap": {
                        "Revenues": {
                            "units": {
                                "USD": [
                                    {"val": 100.0, "end": "2024-03-31", "form": "10-Q"},
                                    {"val": 110.0, "end": "2024-06-30", "form": "10-Q"},
                                    {"val": 120.0, "end": "2024-09-30", "form": "10-Q"},
                                    {"val": 130.0, "end": "2024-12-31", "form": "10-Q"},
                                ]
                            }
                        },
                        "NetIncomeLoss": {
                            "units": {
                                "USD": [
                                    {"val": 20.0, "end": "2024-03-31", "form": "10-Q"},
                                    {"val": 22.0, "end": "2024-06-30", "form": "10-Q"},
                                    {"val": 25.0, "end": "2024-09-30", "form": "10-Q"},
                                    {"val": 26.0, "end": "2024-12-31", "form": "10-Q"},
                                ]
                            }
                        },
                        "NetCashProvidedByUsedInOperatingActivities": {
                            "units": {
                                "USD": [
                                    {"val": 30.0, "end": "2024-03-31", "form": "10-Q"},
                                    {"val": 30.0, "end": "2024-06-30", "form": "10-Q"},
                                    {"val": 30.0, "end": "2024-09-30", "form": "10-Q"},
                                    {"val": 30.0, "end": "2024-12-31", "form": "10-Q"},
                                ]
                            }
                        },
                        "PaymentsToAcquirePropertyPlantAndEquipment": {
                            "units": {
                                "USD": [
                                    {"val": -5.0, "end": "2024-03-31", "form": "10-Q"},
                                    {"val": -5.0, "end": "2024-06-30", "form": "10-Q"},
                                    {"val": -5.0, "end": "2024-09-30", "form": "10-Q"},
                                    {"val": -5.0, "end": "2024-12-31", "form": "10-Q"},
                                ]
                            }
                        },
                        "Assets": {
                            "units": {"USD": [{"val": 1000.0, "end": "2024-12-31", "form": "10-K"}]}
                        },
                        "Liabilities": {
                            "units": {"USD": [{"val": 700.0, "end": "2024-12-31", "form": "10-K"}]}
                        },
                        "StockholdersEquity": {
                            "units": {"USD": [{"val": 300.0, "end": "2024-12-31", "form": "10-K"}]}
                        },
                    }
                },
            }

        monkeypatch.setattr("loom.tools.sec_fundamentals_api.resolve_ticker_to_cik", fake_resolve)
        monkeypatch.setattr("loom.tools.sec_fundamentals_api.fetch_company_facts", fake_facts)
        result = await tool.execute({"operation": "get_ttm_metrics", "ticker": "AAPL"}, ctx)
        assert result.success
        assert result.data["metrics"]["revenue_ttm"] == pytest.approx(460.0)
        assert result.data["metrics"]["free_cash_flow_ttm"] == pytest.approx(100.0)


class TestFilingEventParserTool:
    async def test_extract_guidance_changes(self, ctx: ToolContext, workspace: Path):
        path = workspace / "filing.txt"
        path.write_text(
            "The company raised guidance for fiscal year 2026 by 5%.",
            encoding="utf-8",
        )
        tool = FilingEventParserTool()
        result = await tool.execute(
            {"operation": "extract_guidance_changes", "path": "filing.txt"},
            ctx,
        )
        assert result.success
        assert result.data["event_count"] >= 1


class TestSentimentFeedsApiTool:
    async def test_score_sentiment(self, ctx: ToolContext):
        tool = SentimentFeedsApiTool()
        result = await tool.execute(
            {
                "operation": "score_sentiment",
                "signals": [
                    {
                        "name": "put_call",
                        "value": 0.8,
                        "neutral": 1.0,
                        "scale": 0.3,
                        "direction": "higher_is_bearish",
                    },
                    {
                        "name": "short_flow",
                        "value": 0.35,
                        "neutral": 0.45,
                        "scale": 0.2,
                        "direction": "higher_is_bearish",
                    },
                ],
            },
            ctx,
        )
        assert result.success
        assert result.data["signal_count"] == 2


class TestMacroRegimeEngineTool:
    async def test_score_headwinds_tailwinds(self, ctx: ToolContext):
        tool = MacroRegimeEngineTool()
        result = await tool.execute(
            {
                "operation": "score_headwinds_tailwinds",
                "indicators": {
                    "inflation_yoy": 2.4,
                    "gdp_growth": 2.8,
                    "policy_rate": 4.0,
                    "yield_curve_spread": 0.4,
                    "unemployment_rate": 4.0,
                    "credit_spread": 1.1,
                },
                "exposures": {"technology": 0.4, "financials": 0.2, "defensives": 0.1},
            },
            ctx,
        )
        assert result.success
        assert "tailwind_score" in result.data


class TestFactorExposureEngineTool:
    async def test_estimate_betas(self, ctx: ToolContext):
        tool = FactorExposureEngineTool()
        result = await tool.execute(
            {
                "operation": "estimate_betas",
                "asset_returns": {"AAPL": [0.01, -0.005, 0.012, 0.004, -0.002]},
                "factor_returns": {"MKT": [0.008, -0.004, 0.009, 0.003, -0.001]},
            },
            ctx,
        )
        assert result.success
        assert "AAPL" in result.data["betas"]


class TestValuationEngineTool:
    async def test_intrinsic_value_range(self, ctx: ToolContext):
        tool = ValuationEngineTool()
        result = await tool.execute(
            {
                "operation": "intrinsic_value_range",
                "free_cash_flow": 10000000000,
                "shares_outstanding": 1000000000,
                "discount_rate": 0.1,
                "terminal_growth": 0.02,
            },
            ctx,
        )
        assert result.success
        assert result.data["per_share_value"]["base"] > 0


class TestOpportunityRankerTool:
    async def test_rank_candidates(self, ctx: ToolContext):
        tool = OpportunityRankerTool()
        result = await tool.execute(
            {
                "operation": "rank_candidates",
                "candidates": [
                    {"symbol": "AAA", "expected_return": 0.15, "risk": 0.2, "confidence": 0.7},
                    {"symbol": "BBB", "expected_return": 0.12, "risk": 0.4, "confidence": 0.9},
                ],
            },
            ctx,
        )
        assert result.success
        assert result.data["count"] == 2
        assert result.data["ranked"][0]["symbol"] == "AAA"


class TestPortfolioOptimizerTool:
    async def test_optimize_mvo(self, ctx: ToolContext):
        tool = PortfolioOptimizerTool()
        result = await tool.execute(
            {
                "operation": "optimize_mvo",
                "expected_returns": {"AAA": 0.12, "BBB": 0.08},
                "asset_returns": {
                    "AAA": [0.01, 0.02, -0.01, 0.01, 0.0, 0.02],
                    "BBB": [0.005, 0.004, -0.002, 0.006, 0.001, 0.003],
                },
                "constraints": {"grid_step": 0.25, "max_weight": 0.8},
            },
            ctx,
        )
        assert result.success
        assert "weights" in result.data


class TestPortfolioEvaluatorTool:
    async def test_benchmark_attribution(self, ctx: ToolContext):
        tool = PortfolioEvaluatorTool()
        result = await tool.execute(
            {
                "operation": "benchmark_attribution",
                "portfolio_returns": [0.01, -0.005, 0.007, 0.002, -0.001, 0.004],
                "benchmark_returns": [0.009, -0.006, 0.006, 0.001, -0.002, 0.003],
            },
            ctx,
        )
        assert result.success
        assert "information_ratio" in result.data


class TestPortfolioRecommenderTool:
    async def test_recommend_and_rebalance(self, ctx: ToolContext):
        tool = PortfolioRecommenderTool()
        rec = await tool.execute(
            {
                "operation": "recommend_portfolio",
                "risk_profile": "balanced",
                "max_positions": 2,
                "candidates": [
                    {"symbol": "AAA", "score": 1.2, "conviction": 0.8, "risk": 0.2},
                    {"symbol": "BBB", "score": 1.0, "conviction": 0.9, "risk": 0.3},
                    {"symbol": "CCC", "score": 0.7, "conviction": 0.7, "risk": 0.4},
                ],
            },
            ctx,
        )
        assert rec.success
        assert rec.data["position_count"] == 2

        reb = await tool.execute(
            {
                "operation": "propose_rebalance",
                "target_weights": rec.data["target_weights"],
                "current_weights": {"AAA": 0.1, "BBB": 0.1, "CCC": 0.8},
            },
            ctx,
        )
        assert reb.success
        assert "trades" in reb.data
