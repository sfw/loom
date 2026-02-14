# Loom Code Review Audit

Comprehensive code review conducted across 5 rounds examining the entire codebase.

## Review Scope

### Round 1-2 (Previous Session)
- 52 findings across all modules, all fixed
- Focus: initial security, data integrity, edge cases

### Round 3: Tools, State, Events, Learning
- **Scope:** All 15 tool files, state management, events, learning
- **Key finding:** Ripgrep subprocess leak on timeout
- **Result:** 1 HIGH fix (subprocess kill on timeout)

### Round 4: API, MCP, TUI, Model Providers
- **Scope:** API routes, MCP server, TUI app, all model providers, config
- **Key findings:** SSRF via redirects, webhook SSRF, MCP crash on invalid JSON, SSE handler leaks, background task error loss
- **Result:** 3 CRITICAL + 5 HIGH fixes

### Round 5: Cross-Cutting Data Flow
- **Scope:** Module boundaries, interface mismatches, error propagation
- **Key findings:** SubtaskResult wrong field names (crash), VerificationResult missing required fields (crash), wrong import path
- **Result:** 2 CRITICAL + 3 HIGH fixes

## Total Fixes Across All Rounds

| Severity | Round 1-2 | Round 3-5 | Total |
|----------|-----------|-----------|-------|
| CRITICAL | 5         | 4         | 9     |
| HIGH     | 12        | 7         | 19    |
| MEDIUM   | 20        | 4         | 24    |
| LOW      | 15        | 0         | 15    |
| **Total** | **52**   | **15**    | **67** |

## Files Modified

### Round 1-2
25 files (see commit `0bbfb49`)

### Round 3 (commit `d3cf583`)
16 files across all subsystems

### Round 4-5 (current commit)
- `src/loom/engine/orchestrator.py` - SubtaskResult fields, VerificationResult init
- `src/loom/engine/runner.py` - Memory extraction logging
- `src/loom/tools/web.py` - SSRF redirect protection, Content-Length check
- `src/loom/tools/ripgrep.py` - Subprocess kill on timeout
- `src/loom/events/webhook.py` - Callback URL SSRF validation
- `src/loom/events/bus.py` - unsubscribe/unsubscribe_all methods
- `src/loom/api/routes.py` - Background task error handling, SSE handler cleanup
- `src/loom/integrations/mcp_server.py` - JSON parse safety, error handling

## Remaining Known Issues (Won't Fix / Deferred)

1. **Shared mutable tool bindings** - Old async generators can reference stale bindings after session switch. Requires per-session tool instances.
2. **Session locking** - No application-level lock prevents concurrent resume of same session. SQLite WAL helps but doesn't fully prevent it.
3. **Unused Config properties** - `Config.database_path`, `Config.scratch_path`, `Config.log_path` defined but never called. Harmless dead code.
4. **CORS wildcards** - `http://localhost:*` may not work as expected with all CORS middleware. Works for development but should be reviewed for production.
5. **Ollama synthetic tool IDs** - Uses `call_{i}` instead of real API-provided IDs. Works because the ID is only used for matching tool results back to calls within a single turn.

## Test Coverage

All 685 tests pass across all rounds. Tests updated:
- `test_auto_approve_on_timeout` → `test_deny_on_timeout`
- `test_no_callback_permissive` → `test_no_callback_denies_by_default`
- `test_passes_with_low_confidence_on_no_verifier` → `test_fails_safe_on_no_verifier`
- `test_summary_truncation_in_yaml` → updated for 200-char limit
