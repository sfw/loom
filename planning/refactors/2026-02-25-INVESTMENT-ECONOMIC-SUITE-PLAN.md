# Investment + Economic Tool Suite Plan (2026-02-25)

## Objective
Add a full built-in, no-new-credentials investment/economic tool suite that supports:
1. Equity/ETF market data research.
2. Macro headwind/tailwind evaluation.
3. Security-level opportunity ranking and thesis support.
4. Portfolio construction, evaluation, and recommendation.
5. Investor sentiment signal aggregation.

## Hard Constraints
1. No required paid API services.
2. No required custom API keys/credentials.
3. Use keyless public data endpoints where live data is needed.
4. Keep deterministic fallback paths for offline or partial-data runs.
5. Every decision-grade output must include confidence, provenance, and caveats.

## Existing Baseline
1. `economic_data_api` already supports keyless macro datasets (World Bank/OECD/Eurostat/DBnomics/BLS/FRED).
2. `historical_currency_normalizer` and `inflation_calculator` already cover FX + inflation normalization.
3. Tool auto-discovery and registry contracts are stable.
4. Existing research-tool test patterns can be extended for this suite.

## Tool Suite (v1)

### Data / Ingestion
1. `market_data_api`
   - Ops: `get_prices`, `get_returns`, `get_actions`
   - Keyless source chain: Stooq CSV endpoint (primary) with deterministic validation.
2. `symbol_universe_api`
   - Ops: `resolve_symbol`, `map_ticker_cik`, `list_symbols`
   - Sources: SEC ticker mapping + optional local CSV input.
3. `sec_fundamentals_api`
   - Ops: `get_statement`, `get_ttm_metrics`, `get_quality_flags`
   - Sources: SEC `company_tickers.json` + `companyfacts` API.
4. `filing_event_parser`
   - Ops: `extract_guidance_changes`, `extract_buyback_dividend_changes`, `extract_insider_activity`
   - Sources: filing text inputs (raw text/Markdown paths) with deterministic pattern extraction.
5. `sentiment_feeds_api`
   - Ops: `score_sentiment`, `get_put_call`, `get_short_flow`, `get_cot_positioning`
   - Inputs: inline signal bundles, CSV files, optional keyless endpoint fetches.

### Analytics / Decisioning
1. `macro_regime_engine`
   - Ops: `classify_regime`, `score_headwinds_tailwinds`
2. `factor_exposure_engine`
   - Ops: `estimate_betas`, `factor_contribution`, `factor_correlation`
3. `valuation_engine`
   - Ops: `intrinsic_value_range`, `implied_growth`, `scenario_valuation`
4. `opportunity_ranker`
   - Ops: `rank_candidates`, `explain_rank`, `thesis_breakers`
5. `portfolio_optimizer`
   - Ops: `optimize_mvo`, `optimize_risk_parity`, `optimize_cvar`
6. `portfolio_evaluator`
   - Ops: `performance_stats`, `risk_stats`, `drawdown`, `benchmark_attribution`
7. `portfolio_recommender`
   - Ops: `recommend_portfolio`, `propose_rebalance`, `monitor_alerts`

## Cross-Cutting Contracts
All tools should return:
1. `as_of` timestamp.
2. `keyless` boolean.
3. `sources` list (URL/path provenance).
4. `confidence` score `0..1`.
5. `warnings` list.

Where numeric outputs are generated, include:
1. `assumptions` object.
2. `quality_flags` list.
3. Explicit units and period alignment metadata.

## Shared Implementation Layer
Add shared utility modules:
1. `src/loom/research/finance.py`
   - Return/volatility/correlation/covariance helpers.
   - Drawdown/performance metrics.
   - Lightweight optimization helpers (grid/heuristic based, no heavy deps).
2. `src/loom/research/providers/markets.py`
   - Keyless market-data fetch + parse helpers.
3. `src/loom/research/providers/sec_finance.py`
   - SEC ticker map and companyfacts normalization.

## Phased Work Plan

### Phase 1: Foundation
1. Add shared finance math utilities.
2. Add market and SEC provider adapters.
3. Add unit tests for parsing/metrics helpers.

### Phase 2: Data Tools
1. Implement `market_data_api`.
2. Implement `symbol_universe_api`.
3. Implement `sec_fundamentals_api`.
4. Implement `filing_event_parser`.
5. Implement `sentiment_feeds_api`.

### Phase 3: Analytics Tools
1. Implement regime/factor/valuation/opportunity tools.
2. Ensure each tool supports deterministic, inline-input operation.

### Phase 4: Portfolio Tools
1. Implement optimizer/evaluator/recommender.
2. Add constraints handling (position caps, sector caps, turnover cap).
3. Add recommendation rationale + thesis-break conditions.

### Phase 5: Hardening
1. Add integration tests for all new tools.
2. Update README tool inventory and usage snippets.
3. Validate discovery/registry compatibility.

## Test Strategy
1. New file: `tests/test_investment_tools.py`.
2. Use mocked transports for networked providers.
3. Cover:
   - Parameter validation.
   - Core numeric correctness on fixed fixtures.
   - Missing/partial-data behavior.
   - Recommendation output contracts.

## Acceptance Criteria
1. All new tools are auto-discoverable and schema-valid.
2. Each tool runs without API keys/custom credentials.
3. End-to-end flow possible:
   - data -> macro/factor/valuation -> opportunity ranking -> optimized portfolio -> recommendation.
4. Tests pass for new and touched modules.
5. Outputs include provenance, caveats, and confidence metadata.

## Risks and Mitigations
1. Public endpoint schema drift.
   - Mitigation: strict parsing + fallbacks + warning-rich payloads.
2. Data gaps across symbols/time ranges.
   - Mitigation: quality flags + confidence penalties + graceful partial outputs.
3. Numeric stability without heavy math libraries.
   - Mitigation: conservative heuristics and explicit assumptions.

## Out of Scope (v1)
1. Options Greeks and derivatives pricing engines.
2. Intraday HFT-grade microstructure analytics.
3. Authenticated premium datasets.
4. Execution venue routing/trade automation.
