# TUI Model Introspection and Catalog Hardening Plan (2026-03-01)

## Objective
Deliver production-ready model introspection UX in the TUI by:
1. Upgrading `/model` from a minimal alias print to a full, safe active-model report.
2. Adding `/models` to show the same safe detail for all configured models.
3. Preserving current runtime behavior (no model-switch semantics change in this effort).

## Problem Statement
Current behavior is too thin for operations/debugging:
1. `/model` only reports a simple active alias and omits practical diagnostics (provider, endpoint, protocol, limits, capabilities, roles).
2. There is no single command to inspect the full configured model catalog.
3. No explicit contract defines which model fields are safe to display vs always-redacted.

## Scope
In scope:
1. TUI slash command UX: `/model` enrichment and new `/models`.
2. Shared rendering/helpers for model introspection output.
3. Slash help/hint/completion updates and tests.
4. Secret-safety contract and regression tests.

Out of scope:
1. Runtime model switching (`/model <alias>` or equivalent).
2. Changes to model routing logic or selection policy.
3. API/CLI behavior changes outside TUI unless explicitly needed for consistency.

## Command Semantics Contract
1. `/model` (no args): show detailed active-model report.
2. `/model <anything>`: return usage error and explicit note that runtime switching is not supported yet.
3. `/models` (no args): show full configured model catalog with active marker.
4. `/models <anything>`: return usage error.
5. `/models` with zero configured entries: show `No configured models` plus active runtime block when `self._model` exists.

## Current Baseline (Repo-Accurate)
1. `/model` is handled in `src/loom/tui/app.py` and currently emits `Model: {name}` only.
2. Palette action `model_info` calls `_show_model_info()` and uses the same minimal output.
3. Model configuration is available via `Config.models: dict[str, ModelConfig]`.
4. `ModelConfig` includes provider, base URL, model ID, max tokens, temperature, roles, reasoning effort, tier, capabilities, and `api_key`.
5. Active model instance is `self._model` (a `ModelProvider`), selected at setup/startup via `ModelRouter`.

## Production Requirements
1. Security and privacy:
2. Never display API keys, tokens, auth headers, or userinfo credentials in URLs.
3. Never rely on object `repr` for display payloads.
4. Strict allowlist for output fields.
5. Correctness:
6. `/model` must reflect active runtime provider and identify the matching configured alias when possible.
7. `/models` must include all configured aliases in deterministic order.
8. Output must remain valid when no model is configured or config is partial.
9. Active alias matching must avoid false positives when multiple config entries share provider/model/base URL.
9. UX and operability:
10. Output format should be easy to scan in TUI chat (stable line labels, deterministic ordering).
11. Active model must be clearly marked in `/models`.
12. Slash hint/help/completion must include `/models`.
13. `/models` rendering should remain responsive with large catalogs (target: 200 configured models without perceptible UI stall).
13. Testability:
14. Add explicit tests for redaction, output shape, no-config fallback, and command discoverability.
15. Keep behavior deterministic to avoid flaky snapshot-like tests.
16. Delivery confidence:
17. Include phased implementation gates and a rollback path.

## Non-Goals
1. Implementing `/model <alias>` switching now.
2. Performing network calls for model health from these commands.
3. Reformatting unrelated slash command output surfaces.

## Design Decisions
1. Source of truth:
2. Use `self._config.models` for catalog data.
3. Use `self._model` for active runtime identity.
4. Active alias resolution:
5. Resolve active alias by exact provider-name match first (`provider.name` to config alias), then by provider/model/base URL heuristic.
6. If heuristic returns multiple matches, do not guess; mark active alias as `ambiguous` and list candidate aliases.
7. If unresolved, show active runtime block as `alias: (runtime-only)` and still render configured catalog separately.
7. Protocol derivation:
8. `anthropic` -> `anthropic-messages`.
9. `openai`/`openai_compatible` -> `openai-chat-completions`.
10. `ollama` -> `ollama-chat`.
11. Endpoint display:
12. Use configured `base_url` when present.
13. For Anthropic, default to `https://api.anthropic.com` when unset.
14. Strip credentials/query/fragment from displayed URLs.
15. If URL parsing fails, show `endpoint: (invalid-configured-url)` and never echo raw malformed input.
15. Redaction strategy:
16. Never show `api_key`.
17. Never print raw config dicts.
18. Display only curated fields via explicit formatter helpers.
19. Add a denylist scan in tests for key/token/header markers to catch accidental leakage.

## Output Contract (Proposed)
For each model block (`/model` and `/models`):
1. `alias`
2. `active` marker (`yes` for current, omitted or `no` for others)
3. `provider` (`anthropic`, `openai_compatible`, `ollama`, etc.)
4. `protocol` (derived from provider mapping)
5. `endpoint` (sanitized base URL or default)
6. `model_id` (configured model string)
7. `roles` (comma-separated, deterministic)
8. `tier` (explicit or inferred label)
9. `temperature`
10. `max_tokens`
11. `reasoning_effort` (if set)
12. `capabilities` (vision/native_pdf/thinking/citations/audio_input/audio_output)

Notes:
1. The active `/model` view may add a small runtime section when values differ from config (for example inferred tier or runtime fallback values).
2. Empty/missing values should render as `-` rather than omitting lines unpredictably.
3. When active alias is ambiguous, report `active_alias: ambiguous` and `candidates: <aliases>` to prevent incorrect attribution.
4. `/models` ordering: active alias first when uniquely resolved, then remaining aliases in case-insensitive lexical order.
5. Tier resolution: prefer explicit config tier when `tier > 0`; otherwise show runtime/inferred tier and mark as inferred.

## Workstreams and File Touchpoints

### Workstream A: Introspection Payload + Redaction Helpers
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`

Deliverables:
1. Add helper methods for:
2. Active alias resolution.
3. Provider-to-protocol mapping.
4. Endpoint sanitization and safe display.
5. One-model and all-model rendering with deterministic field order.
6. Add explicit no-config fallback renderer.
7. Add usage renderers for `/model` and `/models` argument validation.

Acceptance:
1. Helpers never access/display secret-bearing fields except through redaction-safe path.
2. Deterministic formatting across runs.
3. Ambiguous active-model resolution never assigns an incorrect single alias.

### Workstream B: Slash Command and Palette Wiring
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`
2. `/Users/sfw/Development/loom/src/loom/tui/commands.py`

Deliverables:
1. Add `/models` to `_SLASH_COMMANDS` and priority order.
2. Route `/model` to rich active-model renderer.
3. Add `/models` handler for full catalog output.
4. Add command palette entry for all models (`Show models catalog`) and map to the same renderer.
5. Add explicit usage output for invalid args to `/model` and `/models`.

Acceptance:
1. `/`, `/help`, and tab completion expose `/models`.
2. `/model` and `/models` share formatter logic (no duplicate divergent code paths).
3. Invalid arg forms for `/model` and `/models` return deterministic usage text.

### Workstream C: Test Hardening
Files:
1. `/Users/sfw/Development/loom/tests/test_tui.py`
2. `/Users/sfw/Development/loom/tests/test_tui_commands.py` (if palette entry added)

Deliverables:
1. `/model` detailed output test includes provider/protocol/endpoint/model_id fields.
2. `/models` lists all aliases with active marker and deterministic order.
3. Redaction test guarantees secrets never appear in output.
4. Slash completion/hint/help tests updated for `/models`.
5. Fallback tests for no configured model / no active model.
6. Arg-validation tests for `/model foo` and `/models foo`.
7. Ambiguous alias-resolution tests ensure no false active-alias assignment.

Acceptance:
1. New tests fail if any secret key value leaks.
2. Existing slash behavior tests stay green except intentional expectation updates.
3. Added tests prove deterministic behavior for ambiguity and invalid-arg paths.

### Workstream D: Docs and Operator Notes
Files:
1. `/Users/sfw/Development/loom/CHANGELOG.md`
2. `/Users/sfw/Development/loom/README.md` (if slash command docs mention `/model`)

Deliverables:
1. Changelog entry for richer model introspection and new `/models`.
2. Short usage examples for operators.

Acceptance:
1. `CHANGELOG.md` documents `/model` enrichment and new `/models`.
2. If README includes slash command docs, update those examples to match output and redaction guarantees.

## Test Strategy

### Unit/Behavioral Tests
1. `/model` when active alias is resolved from config.
2. `/model` when active runtime model has no matching alias.
3. `/models` with mixed providers and missing optional fields.
4. Endpoint sanitizer strips userinfo and querystring safely.
5. Provider-to-protocol mapping defaults cleanly for unknown providers.
6. Heuristic alias match returns ambiguous state when multiple candidates are valid.
7. `/model <arg>` and `/models <arg>` return usage errors.
8. Invalid URL input does not leak raw value in output.
9. Large-catalog formatting test (for example 200 aliases) remains deterministic and complete.

### UX Regression Tests
1. Slash hint root catalog includes `/models`.
2. Slash completion for `/m` includes `/mcp`, `/model`, `/models` in deterministic order.
3. Help output includes `/models` description.

### Security Regression Tests
1. Output does not contain raw `api_key` values.
2. Output does not contain `Authorization` or `x-api-key` strings.
3. Output does not expose URL embedded credentials.
4. Output does not contain obvious token markers (for example `sk-`, `Bearer `, `token=`).

## Risks and Mitigations
1. Risk: Secret leakage via incidental stringification of config/provider objects.
2. Mitigation: strict field allowlist + targeted leak tests.
3. Risk: Active alias mismatch when runtime provider name differs from config alias.
4. Mitigation: explicit multi-step alias resolution and clear `(runtime-only)` fallback label.
5. Risk: Future provider additions bypass protocol mapping.
6. Mitigation: centralized mapping helper with explicit default and test coverage.
7. Risk: Output churn breaks tests and user scripts.
8. Mitigation: stable label ordering and documented output contract.
9. Risk: Alias heuristic misidentifies active model in duplicate configurations.
10. Mitigation: ambiguity-safe resolution that never emits a single alias unless confidence is exact.

## Implementation Phases and Gates
1. Phase 1 (Safety primitives):
2. Implement redaction, sanitization, protocol mapping, and ambiguity-safe alias resolution helpers.
3. Gate: helper-level tests green.
4. Phase 2 (Command surfaces):
5. Wire `/model`, `/models`, slash metadata, palette action, and usage errors.
6. Gate: slash behavior tests green.
7. Phase 3 (Regression + docs):
8. Add leak tests, ambiguity tests, and docs/changelog updates.
9. Gate: targeted + full TUI test suites green before merge.

## Rollout Plan
1. Implement helpers + command wiring in one PR with tests.
2. Validate manually in TUI with:
3. single configured model,
4. multi-model mixed-provider config,
5. no-model setup path.
6. Run targeted suites:
7. `uv run pytest tests/test_tui.py -k "slash or model"`
8. `uv run pytest tests/test_tui_commands.py` (if palette touched)
9. Merge after green CI and changelog update.

## Rollback Plan
1. Revert to prior `/model` and remove `/models` registration in one patch if any production regression is found.
2. Keep helper methods isolated so rollback can occur without touching unrelated slash-command logic.
3. If only formatting causes instability, keep `/models` command but switch to minimal safe payload while preserving command contract.

## Acceptance Criteria (Release Gate)
1. `/model` prints a detailed, redaction-safe active-model report.
2. `/models` prints detailed, redaction-safe blocks for all configured models.
3. Help/hints/completion discover `/models`.
4. No secrets appear in rendered output under test.
5. No regressions in existing slash command handling paths.

## Follow-On (Separate Proposal)
1. Add `/model <alias>` runtime switching with explicit scope semantics:
2. per-chat-session vs app-global behavior,
3. persistence rules on restart/setup,
4. compatibility with role-routed helper models.
