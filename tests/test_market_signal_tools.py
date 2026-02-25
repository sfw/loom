"""Tests for market-signal tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.earnings_surprise_predictor import EarningsSurprisePredictorTool
from loom.tools.insider_trading_tracker import InsiderTradingTrackerTool
from loom.tools.options_flow_analyzer import OptionsFlowAnalyzerTool
from loom.tools.registry import ToolContext
from loom.tools.short_interest_analyzer import ShortInterestAnalyzerTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace)


class TestOptionsFlowAnalyzerTool:
    async def test_score_flow_from_source_path(self, ctx: ToolContext, workspace: Path):
        csv_path = workspace / "options.csv"
        csv_path.write_text(
            "date,symbol,put_volume,call_volume\n"
            "2026-01-01,AAPL,100,200\n"
            "2026-01-02,AAPL,150,200\n"
            "2026-01-03,AAPL,300,150\n",
            encoding="utf-8",
        )

        tool = OptionsFlowAnalyzerTool()
        result = await tool.execute(
            {
                "operation": "score_flow",
                "symbol": "AAPL",
                "source_path": "options.csv",
            },
            ctx,
        )
        assert result.success
        assert result.data["operation"] == "score_flow"
        assert "score" in result.data

    async def test_get_put_call_history_uses_cboe_fetch(self, monkeypatch, ctx: ToolContext):
        tool = OptionsFlowAnalyzerTool()

        async def fake_history(*, start_date, end_date, max_rows, source_url, client):
            del start_date, end_date, max_rows, source_url, client
            return {
                "rows": [
                    {
                        "date": "2026-01-01",
                        "put_call_ratio": 0.9,
                        "put_volume": 90,
                        "call_volume": 100,
                    },
                    {
                        "date": "2026-01-02",
                        "put_call_ratio": 1.1,
                        "put_volume": 110,
                        "call_volume": 100,
                    },
                ],
                "source_url": "https://example.test/totalpc.csv",
            }

        monkeypatch.setattr(
            "loom.tools.options_flow_analyzer.fetch_cboe_put_call_history",
            fake_history,
        )
        result = await tool.execute({"operation": "get_put_call_history"}, ctx)
        assert result.success
        assert result.data["count"] == 2


class TestInsiderTradingTrackerTool:
    async def test_summarize_insider_activity(self, monkeypatch, ctx: ToolContext):
        tool = InsiderTradingTrackerTool()

        async def fake_resolve(*, ticker, client):
            del ticker, client
            return {"ticker": "AAPL", "cik": "0000320193", "name": "Apple Inc."}

        async def fake_submissions(*, cik, client):
            del cik, client
            return {
                "cik": "0000320193",
                "filings": {
                    "recent": {
                        "accessionNumber": ["0000000000-26-000001"],
                        "filingDate": ["2026-01-20"],
                        "reportDate": ["2026-01-20"],
                        "form": ["4"],
                        "primaryDocument": ["xslF345X03/form4.xml"],
                    }
                },
            }

        async def fake_filing_tx(*, filing_url, client):
            del filing_url, client
            return {
                "transactions": [
                    {
                        "transaction_date": "2026-01-20",
                        "transaction_code": "P",
                        "acquired_disposed": "A",
                        "shares": 1000,
                        "price": 200,
                        "transaction_value": 200000,
                        "owner_name": "Doe John",
                        "owner_role_weight": 1.2,
                    }
                ],
                "source_url": "https://example.test/form4.xml",
            }

        monkeypatch.setattr(
            "loom.tools.insider_trading_tracker.resolve_ticker_to_cik",
            fake_resolve,
        )
        monkeypatch.setattr(
            "loom.tools.insider_trading_tracker.fetch_sec_submissions",
            fake_submissions,
        )
        monkeypatch.setattr(
            "loom.tools.insider_trading_tracker.fetch_sec_filing_transactions",
            fake_filing_tx,
        )

        result = await tool.execute(
            {"operation": "summarize_insider_activity", "ticker": "AAPL"},
            ctx,
        )
        assert result.success
        assert result.data["summary"]["buy_count"] == 1
        assert result.data["summary"]["net_value"] > 0


class TestShortInterestAnalyzerTool:
    async def test_compute_short_pressure_from_inline_inputs(self, ctx: ToolContext):
        tool = ShortInterestAnalyzerTool()
        result = await tool.execute(
            {
                "operation": "compute_short_pressure",
                "short_interest_shares": 1_500_000,
                "float_shares": 10_000_000,
                "average_daily_volume": 200_000,
                "short_volume": 120_000,
                "total_volume": 250_000,
            },
            ctx,
        )
        assert result.success
        assert result.data["metrics"]["short_pressure_score"] > 0

    async def test_detect_squeeze_setup(self, ctx: ToolContext):
        tool = ShortInterestAnalyzerTool()
        result = await tool.execute(
            {
                "operation": "detect_squeeze_setup",
                "short_pressure_score": 82,
                "price_momentum_20d": 0.12,
            },
            ctx,
        )
        assert result.success
        assert result.data["setup"]["squeeze_setup"] is True


class TestEarningsSurprisePredictorTool:
    async def test_predict_surprise_model_only(self, ctx: ToolContext):
        tool = EarningsSurprisePredictorTool()
        result = await tool.execute(
            {
                "operation": "predict_surprise",
                "latest_eps": 1.1,
                "prior_eps": [0.9, 1.0, 1.1],
                "price_returns": [0.01, -0.005, 0.012, 0.003, 0.004],
                "sentiment_score": 0.3,
                "options_flow_score": 0.2,
                "short_pressure_score": 60,
                "insider_score": 0.25,
            },
            ctx,
        )
        assert result.success
        assert result.data["operation"] == "predict_surprise"
        assert "predicted_eps_range" in result.data

    async def test_compare_to_consensus(self, ctx: ToolContext):
        tool = EarningsSurprisePredictorTool()
        result = await tool.execute(
            {
                "operation": "compare_to_consensus",
                "latest_eps": 1.2,
                "prior_eps": [1.0, 1.1, 1.2],
                "consensus_eps": 1.15,
                "actual_eps": 1.28,
            },
            ctx,
        )
        assert result.success
        assert result.data["comparison"]["consensus_eps"] == pytest.approx(1.15)
        assert "winner" in result.data["comparison"]

    async def test_backtest_model(self, ctx: ToolContext):
        tool = EarningsSurprisePredictorTool()
        history = [
            {
                "latest_eps": 1.0,
                "model_inputs": {
                    "eps_trend": 0.05,
                    "margin": 0.2,
                    "momentum_20d": 0.05,
                    "volatility_20d": 0.03,
                    "sentiment_score": 0.1,
                    "options_flow_score": 0.2,
                    "short_squeeze_potential": 0.1,
                    "insider_score": 0.05,
                    "guidance_delta": 0.1,
                    "revenue_scale": 1.0,
                },
                "actual_surprise_pct": 0.04,
            },
            {
                "latest_eps": 1.0,
                "model_inputs": {
                    "eps_trend": -0.04,
                    "margin": 0.1,
                    "momentum_20d": -0.02,
                    "volatility_20d": 0.05,
                    "sentiment_score": -0.1,
                    "options_flow_score": -0.15,
                    "short_squeeze_potential": -0.1,
                    "insider_score": -0.05,
                    "guidance_delta": -0.08,
                    "revenue_scale": 1.0,
                },
                "actual_surprise_pct": -0.03,
            },
        ]
        result = await tool.execute(
            {
                "operation": "backtest_model",
                "history": history,
            },
            ctx,
        )
        assert result.success
        assert result.data["backtest"]["sample_count"] == 2
