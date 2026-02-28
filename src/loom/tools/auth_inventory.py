"""First-party tool auth classification inventory.

This inventory is intentionally explicit: every built-in non-MCP tool must be
classified so auth behavior reviews stay current as tools are added/changed.
"""

from __future__ import annotations

from typing import Literal

ToolAuthClassification = Literal["no_auth", "optional_auth", "required_auth"]


FIRST_PARTY_TOOL_AUTH_CLASSIFICATION: dict[str, ToolAuthClassification] = {
    "academic_search": "no_auth",
    "analyze_code": "no_auth",
    "archive_access": "no_auth",
    "ask_user": "no_auth",
    "calculator": "no_auth",
    "citation_manager": "no_auth",
    "conversation_recall": "no_auth",
    "correspondence_analysis": "no_auth",
    "delegate_task": "no_auth",
    "delete_file": "no_auth",
    "document_write": "no_auth",
    "earnings_surprise_predictor": "no_auth",
    "economic_data_api": "no_auth",
    "edit_file": "no_auth",
    "fact_checker": "no_auth",
    "factor_exposure_engine": "no_auth",
    "filing_event_parser": "no_auth",
    "git_command": "no_auth",
    "glob_find": "no_auth",
    "historical_currency_normalizer": "no_auth",
    "humanize_writing": "no_auth",
    "inflation_calculator": "no_auth",
    "insider_trading_tracker": "no_auth",
    "list_directory": "no_auth",
    "list_tools": "no_auth",
    "macro_regime_engine": "no_auth",
    "market_data_api": "no_auth",
    "move_file": "no_auth",
    "opportunity_ranker": "no_auth",
    "options_flow_analyzer": "no_auth",
    "peer_review_simulator": "no_auth",
    "portfolio_evaluator": "no_auth",
    "portfolio_optimizer": "no_auth",
    "portfolio_recommender": "no_auth",
    "primary_source_ocr": "no_auth",
    "read_artifact": "no_auth",
    "read_file": "no_auth",
    "ripgrep_search": "no_auth",
    "run_tool": "no_auth",
    "search_files": "no_auth",
    "sec_fundamentals_api": "no_auth",
    "sentiment_feeds_api": "no_auth",
    "shell_execute": "no_auth",
    "short_interest_analyzer": "no_auth",
    "social_network_mapper": "no_auth",
    "spreadsheet": "no_auth",
    "symbol_universe_api": "no_auth",
    "task_tracker": "no_auth",
    "timeline_visualizer": "no_auth",
    "valuation_engine": "no_auth",
    "web_fetch": "no_auth",
    "web_fetch_html": "no_auth",
    "web_search": "no_auth",
    "write_file": "no_auth",
}


def tool_auth_classification(tool_name: str) -> ToolAuthClassification | None:
    """Return first-party auth classification for one tool name."""
    key = str(tool_name or "").strip()
    if not key:
        return None
    return FIRST_PARTY_TOOL_AUTH_CLASSIFICATION.get(key)
