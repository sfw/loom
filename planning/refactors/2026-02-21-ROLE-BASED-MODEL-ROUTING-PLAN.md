# 2026-02-21 Role-Based Model Routing Plan

## Objective
Ensure every implicit LLM invocation in production code selects a model through `loom.toml` role assignment (`planner`, `executor`, `extractor`, `verifier`), except explicit user-directed cowork overrides (`--model`).

## Audit Scope
- Reviewed model selection and invocation paths in:
  - `/Users/sfw/Development/loom/src/loom/models/router.py`
  - `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
  - `/Users/sfw/Development/loom/src/loom/engine/runner.py`
  - `/Users/sfw/Development/loom/src/loom/engine/verification.py`
  - `/Users/sfw/Development/loom/src/loom/engine/semantic_compactor.py`
  - `/Users/sfw/Development/loom/src/loom/__main__.py`
  - `/Users/sfw/Development/loom/src/loom/tui/app.py`
  - `/Users/sfw/Development/loom/src/loom/cowork/session.py`
  - `/Users/sfw/Development/loom/src/loom/learning/reflection.py`

## Findings
### Compliant (role-routed)
- Router is role-first (`role -> candidates -> tier preference`):
  - `/Users/sfw/Development/loom/src/loom/models/router.py:57`
- Planner/replanner use `role="planner"`:
  - `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py:1437`
  - `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py:1720`
- Subtask execution uses `role="executor"`:
  - `/Users/sfw/Development/loom/src/loom/engine/runner.py:280`
- Memory extraction uses `role="extractor"`:
  - `/Users/sfw/Development/loom/src/loom/engine/runner.py:636`
- Verification uses `role="verifier"`:
  - `/Users/sfw/Development/loom/src/loom/engine/verification.py:1762`
  - `/Users/sfw/Development/loom/src/loom/engine/verification.py:1709`
- Default cowork model selection uses `role="executor"`:
  - `/Users/sfw/Development/loom/src/loom/__main__.py:286`
  - `/Users/sfw/Development/loom/src/loom/tui/app.py:712`

### Intentional override (accepted)
- Explicit cowork model override bypasses role enforcement by design.
- `--model` path returns provider by name directly with no role check:
  - `/Users/sfw/Development/loom/src/loom/__main__.py:279`
  - `/Users/sfw/Development/loom/src/loom/__main__.py:280`
  - `/Users/sfw/Development/loom/src/loom/__main__.py:281`
- Accepted behavior: user explicitly requested this model.

### Non-compliant (role bypass or cross-role fallback)
1. TUI ad hoc process synthesis uses active cowork model directly.
- Direct calls to `self._model.complete(...)`:
  - `/Users/sfw/Development/loom/src/loom/tui/app.py:2073`
  - `/Users/sfw/Development/loom/src/loom/tui/app.py:2109`
- Effect: synthesis/repair tasks are not routed by role (should be planner-oriented).

2. TUI run-folder naming uses active cowork model directly.
- Direct call to `self._model.complete(...)`:
  - `/Users/sfw/Development/loom/src/loom/tui/app.py:3293`
- Effect: utility extraction task is not routed by role.

3. Semantic compactor has cross-role fallback.
- Requested role falls back to `verifier`, then `executor`:
  - `/Users/sfw/Development/loom/src/loom/engine/semantic_compactor.py:99`
- Effect: compaction may silently run on a model not assigned to the requested role.

4. Reflection engine accepts an injected model, not a role router.
- Direct calls to `_model.complete(...)`:
  - `/Users/sfw/Development/loom/src/loom/learning/reflection.py:388`
  - `/Users/sfw/Development/loom/src/loom/learning/reflection.py:454`
- Note: currently appears not wired in TUI startup, but path is role-agnostic if enabled.

## Refactor Plan
1. Route TUI helper LLM tasks through `ModelRouter` with explicit roles.
- Add a reusable role selection helper on `LoomApp` (or store a router instance).
- Use `planner` role for ad hoc process synthesis + repair.
- Use `extractor` role (or `planner` if product decision prefers) for run-folder naming.
- Preserve fallback behavior when role is unavailable (return deterministic fallback output, no crash).

2. Make semantic compactor role fallback policy explicit.
- Replace implicit cross-role fallback with one of:
  - strict role mode: if requested role missing, skip LLM compaction and use whitespace compaction;
  - or config-gated fallback mode with explicit opt-in.
- Add logs/metadata indicating whether compaction used requested role or fallback.

3. Align reflection path to role routing (if/when enabled).
- Inject a role-selected model (recommended `extractor`) instead of arbitrary model instance.
- If no model for that role, keep existing non-fatal behavior and skip LLM-assisted reflection.

4. Add regression tests for role enforcement.
- `tui/app`: ad hoc synthesis and run-folder naming call role-selected model (mock router select calls).
- `semantic_compactor`: strict mode does not cross role boundaries.
- `learning/reflection`: optional test for role-based model injection if wired.

5. Update docs.
- Document which role is used by cowork helper tasks.
- Document that `--model` in cowork is an explicit override and may bypass role routing.
- Document strict-vs-fallback compactor behavior.

## Acceptance Criteria
- No implicit production LLM invocation path chooses a model by name alone without role validation.
- Explicit `--model` cowork override remains supported and documented as an intentional bypass.
- Cowork helper LLM calls are role-routed or explicitly degraded to non-LLM fallback.
- Cross-role model substitution only happens when explicitly configured.
- Automated tests cover all previously non-compliant paths listed above.
