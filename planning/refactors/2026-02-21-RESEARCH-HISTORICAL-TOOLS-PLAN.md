# 2026-02-21 Research/Historical Built-In Tools Plan

## Objective
Add first-class built-in tools for historical and research-heavy `/run` workflows:
1. `academic_search` (historical/archival discovery)
2. `archive_access` (public archive retrieval and metadata access)
3. `citation_manager` (bibliography + citation consistency)
4. `peer_review_simulator` (structured critique/rubric review)
5. `inflation_calculator` (inflation-adjusted conversions and CPI lookups)
6. `timeline_visualizer` (chronological evidence rendering)
7. `fact_checker` (claim verification against cited evidence)
8. `image_analysis` (visual analysis, including ruins/maps/diagrams)

Also add PDF chart/graph support as a first-class capability, not a best-effort text fallback.

## Why This Is Needed Now
Current tooling is strong for generic web search and text extraction, but weak for:
1. Scholarly source discovery with publication metadata and stable IDs (DOI/arXiv/etc.).
2. End-to-end citation hygiene (claim -> source traceability, dedupe, bibliography formatting).
3. Structured fact-checking with explicit verdicts and confidence.
4. Extracting meaning from visual evidence in PDFs (charts, graphs, maps, figures).

Existing `read_file` already supports PDF text extraction and image content blocks, but it does not provide:
1. Chart/table extraction into structured data.
2. Claim-level verification workflow.
3. Citation lifecycle management.

## Feasibility Assessment for Latest `/run` Recommendations (2026-02-21)
Requested list:
1. `academic_search`
2. `archive_access`
3. `citation_manager`
4. `peer_review_simulator`
5. `inflation_calculator`
6. `timeline_visualizer`
7. `fact_checker`

Assessment:
1. `academic_search`: feasible now.
- Build path: API-backed metadata search + normalized results.

2. `archive_access`: feasible with scoped v1 (public/open archives only).
- Build path: connectors for open datasets and archived web snapshots.
- Constraint: no guaranteed access to paywalled/private archives without user-provided credentials/connectors.

3. `citation_manager`: feasible now.
- Build path: deterministic parsing/dedupe/format/validate pipeline.

4. `peer_review_simulator`: feasible now.
- Build path: rubric-driven LLM critique returning structured findings and confidence.

5. `inflation_calculator`: feasible now.
- Build path: deterministic CPI-based calculations with local cache + optional refresh.

6. `timeline_visualizer`: feasible now.
- Build path: generate timeline artifacts as Markdown, Mermaid, and CSV/JSON.
- Constraint: image-rendered timeline charts are optional and can be deferred.

7. `fact_checker`: feasible now.
- Build path: claim extraction, evidence linking, verdict classification.

## Proposed Tool Set

### Required Built-Ins (User Requested)
1. `academic_search`
2. `archive_access`
3. `citation_manager`
4. `peer_review_simulator`
5. `inflation_calculator`
6. `timeline_visualizer`
7. `fact_checker`
8. `image_analysis`

### Additional Built-Ins Recommended
1. `pdf_figure_extract`
- Purpose: render selected PDF pages/figures and expose chart/graph/table regions.
- Why: essential bridge from PDF documents to visual analysis.

2. `table_extract`
- Purpose: extract tabular data from PDF pages/images into CSV/JSON.
- Why: research and historical reports often include key evidence in tables.

3. `source_snapshot`
- Purpose: persist source metadata + fetch timestamp + content hash for reproducibility.
- Why: avoids citation drift and link rot for archival claims.

4. `claim_extractor`
- Purpose: extract candidate factual claims from prose into structured records.
- Why: improves `fact_checker` and `citation_manager` workflows with deterministic preprocessing.

## Tool Contracts (v1)

### 1) `academic_search`
Inputs:
1. `query` (required)
2. `year_from`, `year_to` (optional)
3. `source_types` (`journal`, `conference`, `preprint`, `archive`, `book`, `thesis`)
4. `max_results`
5. `sort_by` (`relevance`, `year`, `citations`)

Outputs:
1. Human-readable ranked results.
2. Structured `data.results[]` with: `title`, `authors`, `year`, `venue`, `url`, `doi`, `abstract`, `source_db`, `confidence`.

Implementation notes:
1. Provider adapter layer with graceful fallback order (for example: OpenAlex/Crossref/Semantic Scholar/arXiv).
2. Normalize to one internal schema before returning.

### 2) `citation_manager`
Inputs:
1. `operation` (`add`, `dedupe`, `format`, `validate`, `map_claims`)
2. `citation` (for `add`)
3. `path` (default `references.bib` or `references.json`)
4. `style` (`apa`, `mla`, `chicago`, `ieee`, `bibtex`)
5. `claims_path` (optional; for claim-citation mapping)

Outputs:
1. Updated bibliography artifact(s).
2. Validation report: missing fields, duplicate references, unresolved citations.
3. Optional `claim_citation_map.csv` artifact.

Implementation notes:
1. Deterministic parser/formatter first; style rendering can be extended incrementally.
2. Keep model-independent for repeatability.

### 3) `fact_checker`
Inputs:
1. `claims` (list) or `claims_path`
2. `sources` (URLs/files) or `source_index_path`
3. `strictness` (`lenient`, `standard`, `strict`)
4. `require_primary_sources` (bool)

Outputs:
1. Per-claim verdict: `supported`, `partially_supported`, `contradicted`, `unverifiable`.
2. Confidence score, supporting citations, contradiction notes.
3. Structured report file (for example `fact-check-report.md` + `fact-check-report.csv`).

Implementation notes:
1. Rule-based evidence linking first (exact + fuzzy match), then model arbitration for ambiguous matches.
2. Never return binary pass/fail only; always include rationale and source links.

### 4) `image_analysis`
Inputs:
1. `path` (image file or page render output)
2. `analysis_type` (`general`, `architecture`, `map`, `artifact`, `chart`, `table`, `ocr`)
3. `regions` (optional bounding boxes)
4. `question` (optional targeted prompt)

Outputs:
1. Structured observations with confidence and uncertainty flags.
2. Optional extracted text (OCR), entities, and detected visual features.
3. Artifacts for downstream verification (`image-observations.json`).

Implementation notes:
1. Reuse existing multimodal content flow (`ImageBlock`) and keep a deterministic fallback mode.
2. For chart mode, forward to chart extraction pipeline when available.

### 5) `pdf_figure_extract` (recommended)
Inputs:
1. `path`
2. `page_start`, `page_end`
3. `extract` (`figures`, `charts`, `tables`, `all`)

Outputs:
1. Figure inventory with page index + bounding regions.
2. Rendered image assets for each figure/table region.
3. Manifest JSON for handoff to `image_analysis` / `table_extract`.

### 6) `table_extract` (recommended)
Inputs:
1. `path` (PDF/image)
2. `page` and optional `region`
3. `output_format` (`csv`, `json`, `markdown`)

Outputs:
1. Extracted table artifact(s).
2. Column quality report (missing cells, confidence, parse warnings).

### 7) `archive_access`
Inputs:
1. `query` (required)
2. `archive_sources` (for example `library_of_congress`, `internet_archive`, `national_archives`, `openverse`, `wikimedia`)
3. `date_from`, `date_to` (optional)
4. `media_types` (`text`, `image`, `audio`, `video`, `mixed`)
5. `max_results`

Outputs:
1. Ranked archival hits with source provenance.
2. Structured `data.results[]` with: `title`, `creator`, `date`, `repository`, `record_url`, `access_url`, `rights`, `snippet`.
3. Optional snapshot artifact manifest for downstream verification.

Implementation notes:
1. Scope v1 to publicly accessible archives with stable HTTP APIs/pages.
2. Keep a connector interface so private/premium archives can be added later via auth profiles/MCP.

### 8) `peer_review_simulator`
Inputs:
1. `path` or `content`
2. `review_type` (`historical_accuracy`, `methodology`, `argument_quality`, `citation_quality`, `general`)
3. `rubric` (optional custom criteria)
4. `strictness` (`light`, `standard`, `strict`)
5. `num_reviewers` (1-3 synthetic reviewer perspectives)

Outputs:
1. Structured review report: strengths, weaknesses, major issues, minor issues, revision actions.
2. Scored rubric table with confidence.
3. Optional artifacts (`peer-review.md`, `peer-review.json`).

Implementation notes:
1. Run with role-routed review model and deterministic schema validation.
2. Include explicit uncertainty and disagreement fields when reviewers diverge.

### 9) `inflation_calculator`
Inputs:
1. `amount` (required)
2. `from_year` (required)
3. `to_year` (required)
4. `index` (`cpi_u`, `cpi_w`, optional future indices)
5. `region` (default `US`)

Outputs:
1. Inflation-adjusted amount with index values used.
2. Structured `data` including multiplier, percent change, and data provenance.
3. Optional table artifact for scenario comparisons.

Implementation notes:
1. Deterministic calculations over cached CPI series (versioned data file).
2. Optional online refresh job; offline behavior remains fully functional.

### 10) `timeline_visualizer`
Inputs:
1. `events` (inline) or `events_path` (CSV/JSON)
2. `granularity` (`day`, `month`, `year`, `auto`)
3. `group_by` (`entity`, `region`, `topic`, `none`)
4. `output_formats` (`markdown`, `mermaid`, `csv`, `json`)

Outputs:
1. Chronological timeline artifact(s) in requested format(s).
2. Conflict diagnostics for contradictory dates or missing time precision.
3. Optional event index with source links.

Implementation notes:
1. Start with text-native outputs (Markdown + Mermaid) for zero extra runtime deps.
2. If graphical rendering is needed later, add optional image renderer behind feature flag.

## Architecture Changes

### A1) Tool Discovery and Registration
No registry redesign required: new classes under `/Users/sfw/Development/loom/src/loom/tools/` are auto-discovered.

Primary files:
1. `/Users/sfw/Development/loom/src/loom/tools/__init__.py`
2. `/Users/sfw/Development/loom/src/loom/tools/registry.py`
3. New modules under `/Users/sfw/Development/loom/src/loom/tools/`

### A2) Shared Research Data Models
Add normalized dataclasses (or typed dicts) for:
1. `AcademicResult`
2. `CitationRecord`
3. `FactCheckVerdict`
4. `FigureRegion` / `TableRegion`

Proposed location:
1. `/Users/sfw/Development/loom/src/loom/research/models.py`

### A3) Provider Adapters
Add provider abstraction with retry/rate-limit policy consistent with current web tools.

Proposed location:
1. `/Users/sfw/Development/loom/src/loom/research/providers/`

### A4) Multimodal Bridge for PDFs
Reuse existing `DocumentBlock`/`ImageBlock` pipeline and add PDF visual extraction path so chart pages can be analyzed as images when text extraction is insufficient.

Primary files:
1. `/Users/sfw/Development/loom/src/loom/content.py`
2. `/Users/sfw/Development/loom/src/loom/tools/file_ops.py`
3. `/Users/sfw/Development/loom/src/loom/engine/runner.py`

### A5) Process Integration
Update built-in process definitions so research processes can require these tools explicitly.

Primary files:
1. `/Users/sfw/Development/loom/src/loom/processes/builtin/research-report.yaml`
2. `/Users/sfw/Development/loom/src/loom/processes/builtin/market-research.yaml`
3. `/Users/sfw/Development/loom/src/loom/processes/schema.py`

### A6) Research Data Packs and Cache
Add local data/cache support for deterministic computations and reproducible archive references.

Primary files:
1. `/Users/sfw/Development/loom/src/loom/research/data/cpi_us.csv`
2. `/Users/sfw/Development/loom/src/loom/research/cache.py`
3. `/Users/sfw/Development/loom/src/loom/research/snapshots.py`

## Workstreams and PR Sequence

### W1: Tool Scaffolding + Contracts
1. Add tool modules, schemas, argument validation, and deterministic error handling.
2. Add shared research models and normalized result schema.
3. Ensure outputs are both human-readable and machine-parseable (`data` payloads).

### W2: `academic_search` + `citation_manager`
1. Implement search providers with normalization and provenance fields.
2. Implement citation add/dedupe/format/validate operations.
3. Generate reusable bibliography artifacts.

### W3: `fact_checker` + `peer_review_simulator`
1. Implement claim ingestion and evidence linking.
2. Add structured verdict schema and confidence scoring.
3. Emit claim-to-source mapping artifacts.
4. Implement rubric-based peer review simulation and revision guidance outputs.

### W4: Archive + Time + Inflation Utilities
1. Implement `archive_access` connectors for public archive sources.
2. Implement `timeline_visualizer` with Markdown/Mermaid/CSV outputs.
3. Implement `inflation_calculator` with deterministic CPI dataset and provenance fields.

### W5: PDF Visual Pipeline (`pdf_figure_extract`, `table_extract`, `image_analysis` chart mode)
1. Add PDF page/region extraction for figures/tables.
2. Add chart/table extraction path and quality diagnostics.
3. Integrate image/chart analysis outputs into downstream fact-checking.

### W6: Process + Prompt Integration
1. Update process YAML tool requirements and guidance.
2. Add verification rules for citation completeness and claim traceability.
3. Add verification rules for timeline consistency and inflation-source provenance.
4. Update docs and examples for historical/archival runs.

### W7: Rollout Controls + Telemetry
1. Add feature flags for each new tool family.
2. Log tool usage, failure modes, extraction confidence, and fallback frequency.
3. Keep fail-safe behavior: if advanced extraction fails, preserve baseline text workflow.

## Test Plan

### Unit Tests
1. Add tool-level tests to `/Users/sfw/Development/loom/tests/test_new_tools.py` and `/Users/sfw/Development/loom/tests/test_tools.py`.
2. Add search-provider normalization tests and error/retry tests.
3. Add citation dedupe/format/validation matrix tests.
4. Add fact-check verdict consistency tests with deterministic fixtures.
5. Add PDF figure/table extraction fixture tests.
6. Add inflation conversion golden tests against fixture CPI values.
7. Add timeline normalization tests (ordering, grouping, ambiguity handling).
8. Add peer-review schema stability tests (required fields, score ranges).

### Integration Tests
1. Process-level tests in `/Users/sfw/Development/loom/tests/test_processes.py` verifying required tools and deliverables.
2. End-to-end research run that produces:
   - source log
   - bibliography
   - fact-check report
   - peer-review report
   - timeline artifacts
   - inflation-adjusted summary table
   - chart/table extraction artifacts

### Regression Tests
1. Verify existing `read_file` behavior is unchanged for plain text-only workflows.
2. Verify tool auto-discovery still loads all built-ins.
3. Verify no process fails when new tools are absent but not required.
4. Verify `archive_access` fails gracefully for unavailable/private sources with actionable errors.

## Acceptance Criteria
1. `/run` can execute a research process using `academic_search`, `archive_access` (public scope), `citation_manager`, and `fact_checker` without custom tooling.
2. Citation artifacts are deduped and format-valid, with claim-to-source traceability.
3. Fact-check output includes per-claim verdict + confidence + linked evidence.
4. `peer_review_simulator` produces structured critique and revision actions against a declared rubric.
5. `inflation_calculator` returns deterministic inflation-adjusted outputs with data provenance.
6. `timeline_visualizer` generates timeline artifacts and flags contradictory chronology.
7. PDF charts/graphs/tables can be extracted into structured artifacts with fallback diagnostics.
8. `image_analysis` can analyze extracted figures (including architecture/ruin imagery) and emit structured observations.
9. Research built-in process definitions document and enforce these capabilities via `tools.required` and verification rules.

## Risks and Mitigations
1. API/source instability and rate limits.
- Mitigation: provider fallback order, retries, and cached snapshots.

2. Noisy chart/table extraction on low-quality PDFs.
- Mitigation: quality scores, confidence thresholds, and explicit fallback to manual review.

3. Citation format edge cases.
- Mitigation: strict validation and deterministic canonicalization before formatting.

4. Tool bloat in non-research workflows.
- Mitigation: process-level tool requirement scoping and default guidance to call these only in research/historical runs.

5. Archive source accessibility variability.
- Mitigation: explicit public-source scope in v1, connector-level diagnostics, and optional authenticated connectors later.

6. Inflation dataset staleness.
- Mitigation: versioned local CPI data with update metadata and optional refresh path.

## Recommended Implementation Priority
1. `academic_search`
2. `citation_manager`
3. `fact_checker`
4. `peer_review_simulator`
5. `inflation_calculator`
6. `timeline_visualizer`
7. `archive_access` (public-source v1)
8. `pdf_figure_extract` + `table_extract`
9. `image_analysis` advanced modes
