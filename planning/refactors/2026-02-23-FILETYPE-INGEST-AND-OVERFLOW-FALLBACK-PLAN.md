# Filetype-Aware Ingest + Overflow Fallback Plan (2026-02-23)

## Objective
Prevent context blowups caused by binary/large document fetches by moving type handling to ingest time (not compaction time), while preserving output quality and keeping retrieval flexible for many task types.

## Why This Is Needed
Recent failures are dominated by oversized tool payload accumulation, not normal reasoning growth:

- Run log: `/Users/sfw/.loom/logs/20260223-141254-cowork-1c7df8e9.events.jsonl`
- `web_fetch` on large PDFs occurs immediately before failure (`seq 2119`, `seq 2123`)
- Request envelopes then exceed provider hard limits:
  - total message size exceeded 4MB hard cap
  - token request exceeded model context limit
- Retries repeated with near-identical oversized payloads and no successful recovery

Compaction can reduce text, but cannot reliably recover from bad binary/text ingestion decisions after the fact.

## Design Principles
1. Ingest first, compact second.
2. Never decode unknown binary as prompt text.
3. Keep source fidelity via artifact references; do not discard source material.
4. Keep context lightweight by default; retrieve depth on demand.
5. Use deterministic routing; no model-in-the-loop for filetype classification.
6. Add a narrow overflow fallback path for provider-limit errors.
7. Keep v1 simple: minimal new primitives, clear telemetry, safe rollout.

## Scope
### In scope
- Filetype-aware routing for fetched content.
- Handler pipeline for text-like vs binary/document media.
- Artifact reference model for remote fetched binaries.
- Immediate value extraction for documents (PDF first, extensible to others).
- Runtime overflow fallback for model input hard-limit errors.
- Telemetry and tests.

### Out of scope
- Full document indexing/search system.
- Semantic ranking or model-based ingest triage.
- Large architectural changes to planner/orchestrator behavior.

## Target End State
1. `web_fetch` classifies content by MIME + magic bytes + URL extension hints.
2. Text-like payloads return bounded text as today.
3. Binary/document payloads are persisted as artifacts and represented in context by:
   - concise summary text,
   - structured metadata,
   - optional extracted text excerpt,
   - stable artifact reference.
4. Runner no longer retries identical oversized requests when provider returns hard size/token limit errors.
5. Runner performs one deterministic overflow fallback rewrite and retries once.
6. Final deliverables remain high quality because source artifacts are still available for targeted reads.

## Architecture
### A. Ingest Router (new)
Add `src/loom/ingest/router.py` with:

- `detect_content_kind(headers, first_bytes, url) -> ContentKind`
- `ContentKind`: `text`, `html`, `pdf`, `office_doc`, `image`, `archive`, `unknown_binary`
- Deterministic precedence:
  1. Trusted MIME type from response header (if specific)
  2. Magic-byte sniff
  3. URL extension hint
  4. Fallback to `unknown_binary`

### B. Artifact Store (new, lightweight)
Add `src/loom/ingest/artifacts.py`:

- Persist fetched binary blobs under run-scoped scratch path.
- Return `artifact_ref` (UUID/string), `path`, `media_type`, `size_bytes`, `source_url`, `created_at`.
- Maintain small manifest JSON per run for lookup.

No global indexing in v1.

### C. Filetype Handlers (new)
Add `src/loom/ingest/handlers.py` with a minimal handler contract:

- `can_handle(kind, media_type) -> bool`
- `extract_summary(path, max_chars) -> ExtractedPayload`

Initial handlers:
- `PdfHandler`:
  - extract text (bounded pages/characters),
  - produce summary + page range metadata.
- `TextHandler`:
  - decode text safely with existing HTML stripping behavior when needed.
- `BinaryFallbackHandler`:
  - no text decode; emit metadata-only summary.

Future handlers (docx/pptx/images) slot into same contract.

### D. Tool Integration (update existing `web_fetch`)
Update `src/loom/tools/web.py` flow:

1. Download bounded bytes as today.
2. Route content via ingest router.
3. For text/html: existing behavior (decode + optional strip).
4. For binary/document:
   - save artifact,
   - run handler extraction if supported,
   - return compact tool output summary (not raw bytes decode),
   - include rich `data` fields:
     - `artifact_ref`
     - `content_kind`
     - `media_type`
     - `size_bytes`
     - `source_url`
     - extraction stats (`extracted_chars`, `page_range`, `truncated`)

Optionally add `content_blocks` for documents to align with existing multimodal flow.

### E. Overflow Fallback (runner)
Update runner model invocation retry path to detect provider hard-limit errors:

- error signals include:
  - `"total message size ... exceeds limit"`
  - `"exceeded model token limit"`

On first such failure in an invocation:
1. Rewrite prior tool messages with `content_kind in {pdf, office_doc, unknown_binary}` to artifact stubs:
   - keep short summary + key metadata + artifact_ref
   - drop large embedded output text from those tool messages
2. Preserve:
   - system prompt
   - current user turn
   - latest assistant/tool exchange
3. Retry once with rewritten context.

If still failing, fail fast as non-retryable for that invocation to avoid thrash.

## Data Contract (v1)
### Tool result `data` additions
For binary/document fetch results:

- `artifact_ref: str`
- `content_kind: str`
- `media_type: str`
- `size_bytes: int`
- `source_url: str`
- `extracted_chars: int`
- `extraction_truncated: bool`
- `handler: str`

### Optional summary template in tool `output`
Human/model-readable short text:

`[Fetched PDF artifact: <name>, <size>, pages <n>. Extracted <k> chars (truncated=<bool>). Use artifact_ref=<id> for targeted follow-up.]`

## Rollout Strategy
Add feature flags in config (off by default initially):

- `runner.enable_filetype_ingest_router`
- `runner.enable_model_overflow_fallback`

Rollout phases:
1. Shadow telemetry mode (router active, behavior parity where possible).
2. Enable binary/document handler path for `web_fetch`.
3. Enable overflow fallback.
4. Promote to default after validation on heavy-document runs.

## Workstreams
### Workstream 1: Ingest Router + Artifact Model
Deliverables:
- router + content kind enums
- artifact storage + manifest
- MIME + magic-byte detection tests

Acceptance:
- Binary payloads are never decoded as plain text by default path.
- Router classification is deterministic and test-covered.

### Workstream 2: `web_fetch` Filetype-Aware Execution
Deliverables:
- integrate router in `web_fetch`
- binary/document branch using artifact store
- bounded extraction for PDF handler

Acceptance:
- Fetching a PDF returns metadata/summary + artifact ref, not raw binary-decoded output.
- Existing HTML/text fetch behavior remains intact.

### Workstream 3: Extensible Handler Interface
Deliverables:
- handler contract + registry
- PDF + binary fallback handlers in v1

Acceptance:
- New handler types can be added without touching `web_fetch` core logic.

### Workstream 4: Runner Overflow Fallback
Deliverables:
- hard-limit error classification
- one-pass context rewrite using artifact stubs
- one retry max for fallback path

Acceptance:
- No repeated identical oversized request retries after hard-limit errors.
- Retry path emits clear event diagnostics.

### Workstream 5: Telemetry + Diagnostics
Deliverables:
- events for ingest decisions and fallback outcomes
- counters for bytes in/out and extracted chars

Acceptance:
- Logs show why content was treated as binary/text and whether fallback fired.
- Failures can be diagnosed without guesswork.

### Workstream 6: Tests + Rollout Hardening
Deliverables:
- unit + integration tests
- config parsing/flag tests
- regression tests for existing tool/message flows

Acceptance:
- Existing orchestration/verification tests remain green.
- New failure mode is covered with deterministic reproduction fixtures.

## Test Plan
### Unit tests
- Ingest router classification:
  - PDF MIME + magic
  - mislabeled MIME with PDF magic
  - plain text and HTML paths
  - unknown binary path
- Artifact store write/read manifest lifecycle.
- PDF handler extraction truncation boundaries.

### Integration tests
- `web_fetch` PDF URL fixture returns artifact summary (not giant text blob).
- Runner receives hard-limit provider error and executes one overflow fallback rewrite + single retry.
- Overflow fallback preserves latest critical turns and tool-call continuity.

### Regression tests
- No regression in `web_fetch_html`.
- No regression in normal text `web_fetch`.
- No regression in verification/extractor behavior when no binary artifacts are present.

## Risks and Mitigations
1. Risk: false binary detection for uncommon text payloads.
   - Mitigation: MIME + sniff precedence with explicit allowlist for text media types.
2. Risk: extraction libraries missing in runtime.
   - Mitigation: graceful fallback to metadata-only summary; keep artifact_ref available.
3. Risk: fallback rewrite removes useful context.
   - Mitigation: preserve latest exchange + system/user critical turns; one-pass conservative rewrite.
4. Risk: artifact growth on disk.
   - Mitigation: run-scoped storage + retention cleanup policy in follow-up.

## PR Sequence (suggested)
1. PR-1: Router + artifact primitives + tests.
2. PR-2: `web_fetch` integration + PDF handler + telemetry.
3. PR-3: Runner overflow fallback + retry semantics + tests.
4. PR-4: Config flags, rollout defaults, docs, and final regression pass.

## Implementation Checklist (File-Level)
### 1) Ingest primitives
- [ ] Add `src/loom/ingest/router.py`
  - [ ] `ContentKind` enum/string constants.
  - [ ] MIME + magic-byte + extension detection helpers.
  - [ ] `detect_content_kind(...)`.
- [ ] Add `src/loom/ingest/artifacts.py`
  - [ ] Run-scope artifact directory resolution.
  - [ ] Persist blob + metadata manifest write.
  - [ ] Artifact reference generation and lookup helpers.
- [ ] Add `src/loom/ingest/handlers.py`
  - [ ] Handler contract.
  - [ ] `PdfHandler` bounded extraction.
  - [ ] `BinaryFallbackHandler` metadata summary.

### 2) Web tool integration
- [ ] Update `src/loom/tools/web.py`
  - [ ] Route fetched content through ingest router.
  - [ ] Preserve current text/html behavior.
  - [ ] Add binary/document artifact branch with summary output.
  - [ ] Populate `ToolResult.data` with `artifact_ref` and ingest metadata.
  - [ ] Add optional `content_blocks` for PDF when extraction exists.

### 3) Runner overflow fallback
- [ ] Update `src/loom/engine/runner.py`
  - [ ] Add overflow error classifier (`size/token limit` patterns).
  - [ ] Add message rewrite pass for artifact-heavy tool messages.
  - [ ] Integrate one-shot fallback into model invocation retry path.
  - [ ] Emit model event diagnostics for fallback decisions/outcomes.
  - [ ] Ensure no repeated identical retries after deterministic overflow errors.

### 4) Config and rollout flags
- [ ] Update `src/loom/config.py`
  - [ ] Extend `RunnerLimitsConfig` with:
    - [ ] `enable_filetype_ingest_router`
    - [ ] `enable_model_overflow_fallback`
  - [ ] Parse + validate both flags in loader.
- [ ] Update `src/loom/engine/runner.py` init to read new flags.

### 5) Tests
- [ ] Update `tests/test_web_tool.py`
  - [ ] Add content-kind detection tests (PDF magic/mime, unknown binary).
  - [ ] Add binary summary formatting tests.
- [ ] Update `tests/test_orchestrator.py`
  - [ ] Add runner unit tests for overflow classification.
  - [ ] Add fallback rewrite tests preserving latest critical turns.
  - [ ] Add non-retry behavior for non-recoverable overflow errors after fallback.
- [ ] Add new focused tests if needed (e.g. `tests/test_ingest_router.py`).

### 6) Validation
- [ ] `uv run pytest -q`
- [ ] `uv run pytest /Users/sfw/Development/loom/tests/test_orchestrator.py -q`
- [ ] `uv run pytest /Users/sfw/Development/loom/tests/test_web_tool.py -q`
- [ ] `uv run pytest /Users/sfw/Development/loom/tests/test_semantic_compactor.py -q`
- [ ] `uv run pytest /Users/sfw/Development/loom/tests/test_verification.py -q`
- [ ] `uv run ruff check /Users/sfw/Development/loom/src /Users/sfw/Development/loom/tests`

## Definition of Done
1. Large fetched PDFs/binaries no longer inflate prompt context with decoded garbage text.
2. At least one known overflow reproduction run succeeds or degrades cleanly without retry thrash.
3. Overflow hard-limit retries are bounded and non-redundant.
4. Logs clearly expose ingest classification and fallback activity.
5. System remains general-purpose across heterogeneous task types.
