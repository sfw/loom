# Keyless Research Tools Implementation Plan (2026-02-24)

## Objective
Implement five new built-in tools that can run without new API keys or paid services:
1. `primary_source_ocr`
2. `historical_currency_normalizer`
3. `economic_data_api`
4. `correspondence_analysis`
5. `social_network_mapper`

The implementation must work with Loom's existing tool framework and degrade gracefully when optional local dependencies are missing.

## Hard Constraints
1. No required paid services.
2. No required API keys.
3. Existing configured LLM models are allowed.
4. Default behavior must remain safe, deterministic where possible, and testable offline with mocked network calls.
5. Missing optional binaries/libraries must return actionable `ToolResult.fail(...)` or a reduced-capability success response, never crashes.

## Current Baseline in Repo
1. Tool auto-discovery via `Tool.__init_subclass__` and `discover_tools()` is already in place.
2. Existing research tooling patterns already exist (`academic_search`, `archive_access`, `inflation_calculator`, `fact_checker`, `timeline_visualizer`).
3. `src/loom/research/models.py` already contains normalized dataclasses and can be extended.
4. `tests/test_research_tools.py` already covers research-tool behavior patterns and can be expanded.

## v1 Scope and Boundaries

### 1) `primary_source_ocr`
Goal: extract machine-readable text from scanned PDFs and images for historical primary sources.

No-key implementation path:
1. Prefer local OCR engines in priority order:
2. `tesseract` binary (CLI) if available.
3. Optional `ocrmypdf` pass for searchable PDF generation when input is PDF and user requests `make_searchable_pdf=true`.
4. Optional model-assisted cleanup using configured LLM only after deterministic OCR output is captured.

v1 boundaries:
1. Do not require cloud OCR APIs.
2. Support image files and PDFs.
3. Provide page-level text, confidence metadata when available, and extraction warnings.
4. Include provenance fields (`engine`, `engine_version`, `language`, `preprocess_steps`).

### 2) `historical_currency_normalizer`
Goal: normalize historical monetary values across currency and time (FX + inflation context).

No-key implementation path:
1. FX sources: ECB reference-rate feeds and Fed H10 downloadable series (public, keyless).
2. Inflation sources: existing bundled US CPI series plus keyless public CPI adapters (World Bank, BLS unregistered mode) for non-US expansion.
3. Local cache of fetched series to reduce repeated network dependency.

v1 boundaries:
1. v1 guarantees USD + major fiat currencies with explicit coverage windows.
2. When target date/index unavailable, return partial conversion with explicit uncertainty flags.
3. Keep formulas deterministic and surface data provenance.

### 3) `economic_data_api`
Goal: unified keyless access to public macroeconomic indicators.

No-key implementation path:
1. Provider adapters for keyless endpoints:
2. World Bank Indicators API.
3. OECD SDMX API.
4. Eurostat API.
5. DBnomics API.
6. BLS public API in unregistered mode (daily limits documented).

v1 boundaries:
1. Operations: `search`, `get_series`, `get_observations`.
2. Normalize output schema across providers (provider metadata retained).
3. Add bounded pagination and rate-limit-aware retry/backoff.
4. Do not add providers that require paid credentials in v1.

### 4) `correspondence_analysis`
Goal: run correspondence analysis (CA) on contingency tables and return interpretable factors.

No-key implementation path:
1. Local numeric compute only (no external API).
2. Preferred implementation via optional scientific stack (`numpy` plus lightweight CA routines).
3. Optional integration with `prince` when installed; fallback to internal CA computation path.

v1 boundaries:
1. Accept table input from inline matrix, CSV path, or JSON records.
2. Return row/column principal coordinates, explained inertia, and contribution tables.
3. Emit Markdown/CSV/JSON artifacts when requested.

### 5) `social_network_mapper`
Goal: build and analyze relationship graphs from structured edges and optionally from text.

No-key implementation path:
1. Core graph engine using local library (`networkx`) or internal adjacency fallback.
2. Optional NLP extraction mode from text using local NLP pipeline (spaCy if installed).
3. Deterministic mode accepts explicit nodes/edges without NLP dependency.

v1 boundaries:
1. Compute centrality and connected components in deterministic mode.
2. Optional community detection in advanced mode.
3. Export graph artifacts (`.json`, `.csv`, optional `.graphml`) for downstream visualization.

## Cross-Cutting Architecture

### New Tool Modules
1. `src/loom/tools/primary_source_ocr.py`
2. `src/loom/tools/historical_currency_normalizer.py`
3. `src/loom/tools/economic_data_api.py`
4. `src/loom/tools/correspondence_analysis.py`
5. `src/loom/tools/social_network_mapper.py`

### Research/Data Support Modules
1. Add provider adapters under `src/loom/research/providers/` (currently empty).
2. Extend `src/loom/research/models.py` with normalized records for:
3. OCR segments and page metadata.
4. Economic series metadata and observations.
5. Currency normalization provenance payloads.
6. Graph analysis summaries.

### Optional Dependency Strategy
1. Keep base install lean; introduce optional extras groups in `pyproject.toml` (for example `research-ocr`, `research-stats`, `research-graph`).
2. Runtime checks decide capability availability.
3. Each tool returns a clear capability report when optional components are absent.

### Security and Safety
1. Reuse bounded HTTP patterns from existing networked tools (`httpx`, timeouts, max results, retries).
2. Maintain SSRF-safe behavior for remote fetches by constraining providers to known domains/endpoints.
3. Preserve workspace path safety via existing `Tool` path-resolution methods.

## Workstreams

### W1: Contracts and Schemas
1. Define JSON schemas for all five tools (inputs and required fields).
2. Define normalized `data` payload shape for each tool.
3. Document failure semantics (`invalid_input`, `capability_missing`, `provider_unavailable`, `partial_result`).

Exit criteria:
1. Tool contracts are stable and validated by unit tests.

### W2: Shared Provider Layer
1. Implement provider adapter interfaces in `src/loom/research/providers/`.
2. Add deterministic normalization for economic and FX/CPI datasets.
3. Add simple local caching for fetched public datasets with timestamped provenance.

Exit criteria:
1. Provider responses map to one schema.
2. Cache behavior is deterministic and test-covered.

### W3: Build `economic_data_api` First
1. Implement provider selection and operation routing.
2. Add keyless provider set (World Bank/OECD/Eurostat/DBnomics/BLS public mode).
3. Add merge/sort/filter behavior and provenance metadata.

Exit criteria:
1. Keyless data retrieval works through a single tool contract.
2. Provider failures are isolated and surfaced without total tool failure.

### W4: Build `historical_currency_normalizer`
1. Integrate with shared FX/CPI adapters and local cache.
2. Implement deterministic conversion paths:
3. FX-only conversion.
4. Inflation-only normalization.
5. FX + inflation normalization.
6. Provide uncertainty and coverage notes for missing dates/indices.

Exit criteria:
1. Tool returns reproducible numeric outputs with explicit provenance.
2. Unsupported combinations fail with actionable guidance.

### W5: Build `primary_source_ocr`
1. Implement file-type routing (image vs PDF).
2. Add OCR engine abstraction with local engine detection.
3. Implement page segmentation, text aggregation, and optional searchable-PDF output.
4. Add post-OCR cleanup options (`none`, `light`, `llm_cleanup`) with default deterministic mode.

Exit criteria:
1. OCR works without API keys.
2. Missing OCR engine returns clear remediation instructions.

### W6: Build `correspondence_analysis`
1. Implement input parsers and contingency-table validation.
2. Implement CA compute path and artifact generation.
3. Add interpretation helpers (top-contributing rows/columns by dimension).

Exit criteria:
1. CA metrics are reproducible for fixed inputs.
2. Output artifacts are generated in at least JSON and Markdown formats.

### W7: Build `social_network_mapper`
1. Implement deterministic graph mode (explicit nodes/edges).
2. Add optional text-to-edge extraction mode.
3. Implement centrality/component metrics and exports.

Exit criteria:
1. Deterministic mode has no NLP dependency.
2. Optional NLP mode is feature-gated and documented.

### W8: Docs, Process Coverage, and UX
1. Update README built-in tools section and docs pages to include new tools and no-key constraints.
2. Add process guidance snippets for when each tool should be used.
3. Add examples showing keyless workflows.

Exit criteria:
1. Users can discover all five tools and their capability limits from docs alone.

## Test Plan

### Unit Tests
1. Add/extend tests for parameter validation and schema conformance.
2. Mock provider adapters for deterministic `economic_data_api` and `historical_currency_normalizer` tests.
3. Verify graceful handling when optional binaries/libraries are absent.

### Integration Tests
1. Extend `tests/test_research_tools.py` with positive-path coverage for all five tools.
2. Add fixture-based sample inputs:
3. Scanned image/PDF OCR fixtures.
4. Economic time series fixture payloads.
5. Contingency table fixtures.
6. Graph edge-list fixtures.

### Regression Tests
1. Ensure no API key environment variables are required for happy-path tests.
2. Ensure provider outage in one source does not block the entire response when alternatives succeed.

## Rollout Plan
1. Phase 1: `economic_data_api` + shared provider layer.
2. Phase 2: `historical_currency_normalizer` (depends on provider layer).
3. Phase 3: `primary_source_ocr` (local capability with optional extras).
4. Phase 4: `correspondence_analysis` + `social_network_mapper`.
5. Phase 5: docs + process guidance + final hardening.

## Acceptance Criteria
1. All five tools appear in auto-discovered tool registry with stable schemas.
2. Core functionality for each tool works without new API keys or paid services.
3. Optional capability gaps are surfaced clearly, not silently ignored.
4. Research-tool test suite covers success, partial-result, and failure paths.
5. Documentation explicitly lists provider/data-source provenance and operational limits.

## Risks and Mitigations
1. Risk: public API schema drift.
Mitigation: adapter isolation, strict response parsing, fallback providers.

2. Risk: local OCR/statistics dependencies vary by machine.
Mitigation: capability detection + optional extras + clear error messaging.

3. Risk: historical FX/CPI gaps for specific dates/currencies.
Mitigation: explicit coverage metadata, partial-result mode, deterministic fallback logic.

4. Risk: NLP extraction quality noise in social graph mode.
Mitigation: deterministic edge-input mode as primary path, optional NLP mode clearly marked.

## Out of Scope for v1
1. Paid dataset connectors.
2. Authenticated/private economics APIs.
3. Fully automated citation-grade OCR correction for all scripts/languages.
4. Interactive graph UI rendering inside Loom (artifact export only in v1).
