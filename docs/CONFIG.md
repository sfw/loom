# Loom Configuration Reference

This document lists supported `loom.toml` keys, legacy `[mcp]` keys, defaults,
and normalization behavior for the current runtime.

## Config Resolution Order

Loom loads core config in this order:

1. `--config /explicit/path/loom.toml` (CLI flag, when provided)
2. `./loom.toml` (current working directory)
3. `~/.loom/loom.toml` (user config)
4. Built-in defaults

MCP server config layers are resolved in this order:

1. `--mcp-config <path>`
2. `./.loom/mcp.toml`
3. `~/.loom/mcp.toml`
4. Legacy `[mcp]` section inside `loom.toml`

## `loom.toml` Reference

### `[server]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `host` | `string` | `"127.0.0.1"` | API bind host for `loom serve`. |
| `port` | `int` | `9000` | API bind port for `loom serve`. |

### `[models.<name>]`

`<name>` is any model alias (for example `primary`, `utility`, `planner`).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | `string` | required | One of `ollama`, `openai_compatible`, `anthropic`. |
| `base_url` | `string` | `""` | Provider API endpoint base URL. |
| `model` | `string` | `""` | Model identifier to call. |
| `max_tokens` | `int` | `8192` | Max completion tokens for that model profile. |
| `temperature` | `float` | `0.1` | Sampling temperature. |
| `roles` | `list[string]` | `["executor"]` | Assigned roles: `planner`, `executor`, `extractor`, `verifier`, `compactor`. |
| `api_key` | `string` | `""` | API key (if provider requires auth). |
| `reasoning_effort` | `string` | `""` | Optional provider-specific reasoning control hint. |
| `tier` | `int` | `0` | Optional explicit quality/cost tier (`0` = auto-detect). |
| `capabilities.vision` | `bool` | auto-detected | Override model vision support. |
| `capabilities.native_pdf` | `bool` | auto-detected | Override native PDF support. |
| `capabilities.thinking` | `bool` | auto-detected | Override long-reasoning support hint. |
| `capabilities.citations` | `bool` | auto-detected | Override citation support hint. |
| `capabilities.audio_input` | `bool` | auto-detected | Override audio input support hint. |
| `capabilities.audio_output` | `bool` | auto-detected | Override audio output support hint. |

Recommended two-model split:
- `primary` roles: `["planner", "verifier"]`
- `utility` roles: `["executor", "extractor", "compactor"]`
- Ensure at least one configured model has the `executor` role.

### `[workspace]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `default_path` | `string` | `"~/projects"` | Default workspace root when none is supplied. |
| `scratch_dir` | `string` | `"~/.loom/scratch"` | Scratch/temp storage path. |

### `[execution]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_subtask_retries` | `int` | `3` | Retry budget per subtask in orchestrated runs. |
| `max_loop_iterations` | `int` | `50` | Loop/dispatch budget for orchestration progress. |
| `max_parallel_subtasks` | `int` | `3` | Max concurrently runnable subtasks. |
| `auto_approve_confidence_threshold` | `float` | `0.8` | Auto-approval threshold in confidence-gated flows. |
| `enable_streaming` | `bool` | `false` | Enables streaming behavior where supported. |
| `enable_global_run_budget` | `bool` | `false` | Enforces task-level global resource budgets when limits are configured. |
| `max_task_wall_clock_seconds` | `int` | `0` | Task-level wall-clock cap (`0` disables). |
| `max_task_total_tokens` | `int` | `0` | Task-level total model token cap (`0` disables). |
| `max_task_model_invocations` | `int` | `0` | Task-level model invocation cap (`0` disables). |
| `max_task_tool_calls` | `int` | `0` | Task-level total tool call cap (`0` disables). |
| `max_task_mutating_tool_calls` | `int` | `0` | Task-level mutating tool call cap (`0` disables). |
| `max_task_replans` | `int` | `0` | Task-level replan cap (`0` disables). |
| `max_task_remediation_attempts` | `int` | `0` | Task-level remediation-attempt cap (`0` disables). |
| `executor_completion_contract_mode` | `string` | `"off"` | Executor completion protocol (`off`, `warn`, `enforce`). |
| `planner_degraded_mode` | `string` | `"allow"` | Planner fallback policy (`allow`, `require_approval`, `deny`). |
| `enable_sqlite_remediation_queue` | `bool` | `false` | Dual-write remediation queue and retry lineage to SQLite tables. |
| `enable_durable_task_runner` | `bool` | `false` | Enables durable queued/running task run leasing and recovery. |
| `enable_mutation_idempotency` | `bool` | `false` | Enables mutating-tool idempotency ledger dedupe. |
| `enable_slo_metrics` | `bool` | `false` | Enables `/slo` snapshot endpoint. |
| `delegate_task_timeout_seconds` | `int` | `3600` | Timeout for delegated orchestration calls (`/run`, `delegate_task`). |
| `model_call_max_attempts` | `int` | `5` | Max retry attempts for model invocation retry policy. |
| `model_call_retry_base_delay_seconds` | `float` | `0.5` | Base exponential backoff delay. |
| `model_call_retry_max_delay_seconds` | `float` | `8.0` | Max delay cap for retry backoff. |
| `model_call_retry_jitter_seconds` | `float` | `0.25` | Added random jitter on retries. |

### `[verification]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `tier1_enabled` | `bool` | `true` | Enables deterministic verification checks. |
| `tier2_enabled` | `bool` | `true` | Enables independent LLM verification. |
| `tier3_enabled` | `bool` | `false` | Enables multi-vote verification tier. |
| `tier3_vote_count` | `int` | `3` | Number of votes when tier 3 is enabled. |
| `policy_engine_enabled` | `bool` | `true` | Enables process policy evaluation path. |
| `regex_default_advisory` | `bool` | `true` | Treat regex rules as advisory unless hard-enforced. |
| `strict_output_protocol` | `bool` | `true` | Enforce verifier output protocol/shape handling. |
| `shadow_compare_enabled` | `bool` | `false` | Emit shadow comparison telemetry for policy diffs. |
| `phase_scope_default` | `string` | `"current_phase"` | Rule scope fallback (`current_phase` or `global`). |
| `allow_partial_verified` | `bool` | `true` | Allow partial-verified outcomes where supported. |
| `unconfirmed_supporting_threshold` | `float` | `0.30` | Threshold for unconfirmed supporting evidence handling. |
| `auto_confirm_prune_critical_path` | `bool` | `true` | Auto-run remediation attempts for critical-path failures. |
| `confirm_or_prune_max_attempts` | `int` | `2` | Max remediation attempts in confirm/prune path. |
| `confirm_or_prune_backoff_seconds` | `float` | `2.0` | Backoff between remediation attempts. |
| `confirm_or_prune_retry_on_transient` | `bool` | `true` | Retry remediation when transient failures are detected. |
| `contradiction_guard_enabled` | `bool` | `true` | Enables contradiction scanning against workspace artifacts. |
| `contradiction_guard_strict_coverage` | `bool` | `true` | Requires enough evidence coverage before contradiction pass. |
| `contradiction_scan_max_files` | `int` | `80` | Max files scanned during contradiction checks. |
| `contradiction_scan_max_total_bytes` | `int` | `2500000` | Total byte cap across scanned files. |
| `contradiction_scan_max_file_bytes` | `int` | `300000` | Per-file byte cap during contradiction scanning. |
| `contradiction_scan_allowed_suffixes` | `list[string]` | text/code suffix allowlist | File suffix allowlist for contradiction scanning. |
| `contradiction_scan_min_files_for_sufficiency` | `int` | `2` | Minimum scanned files required for sufficiency. |
| `remediation_queue_max_attempts` | `int` | `3` | Max queued remediation attempts after confirm/prune. |
| `remediation_queue_backoff_seconds` | `float` | `2.0` | Base backoff for queued remediation retries. |
| `remediation_queue_max_backoff_seconds` | `float` | `30.0` | Max backoff cap for queued remediation retries. |

Default `contradiction_scan_allowed_suffixes`:

`.md`, `.txt`, `.rst`, `.csv`, `.tsv`, `.json`, `.yaml`, `.yml`,
`.toml`, `.ini`, `.cfg`, `.conf`, `.xml`, `.html`, `.htm`, `.py`,
`.js`, `.ts`, `.tsx`, `.jsx`, `.sql`, `.sh`

### `[limits]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `planning_response_max_tokens` | `int` | `16384` | Planner synthesis response budget. |
| `adhoc_repair_source_max_chars` | `int` | `0` | Source truncation limit for ad hoc JSON repair (`0` means disabled). |
| `evidence_context_text_max_chars` | `int` | `4000` | Evidence context cap fed into planning/verification prompts. |

### `[limits.runner]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_tool_iterations` | `int` | `20` | Max tool/model loop iterations per subtask execution pass. |
| `max_subtask_wall_clock_seconds` | `int` | `1200` | Per-subtask wall-clock timeout budget. |
| `max_model_context_tokens` | `int` | `24000` | Runner model-context budget hint. |
| `max_state_summary_chars` | `int` | `480` | Target size for compacted state summaries. |
| `max_verification_summary_chars` | `int` | `8000` | Target size for verification summary payloads. |
| `default_tool_result_output_chars` | `int` | `4000` | Default tool output compaction target. |
| `heavy_tool_result_output_chars` | `int` | `2000` | Output target for heavy tools. |
| `compact_tool_result_output_chars` | `int` | `500` | Output target for aggressively compacted tool excerpts. |
| `compact_text_output_chars` | `int` | `900` | Generic compacted text target. |
| `minimal_text_output_chars` | `int` | `260` | Tiny fallback compacted text target. |
| `tool_call_argument_context_chars` | `int` | `500` | Argument context extraction target. |
| `compact_tool_call_argument_chars` | `int` | `220` | Aggressive tool-argument compaction target. |
| `runner_compaction_policy_mode` | `string` | `"tiered"` | Runner compaction policy (`legacy`, `tiered`, `off`). |
| `enable_filetype_ingest_router` | `bool` | `true` | Routes fetched binary/doc payloads into artifact-backed summaries. |
| `enable_artifact_telemetry_events` | `bool` | `true` | Emits artifact ingest/read/retention and compaction/overflow transparency events to run logs (set `false` to disable). |
| `artifact_telemetry_max_metadata_chars` | `int` | `1200` | Max serialized chars allowed for `handler_metadata` telemetry payload fields. |
| `enable_model_overflow_fallback` | `bool` | `true` | Enables one-shot overflow fallback rewrite when model request size is exceeded. |
| `ingest_artifact_retention_max_age_days` | `int` | `14` | Max artifact age (days) before retention cleanup removes old fetched artifacts. |
| `ingest_artifact_retention_max_files_per_scope` | `int` | `96` | Max retained fetched artifact files per scope/subtask directory. |
| `ingest_artifact_retention_max_bytes_per_scope` | `int` | `268435456` | Max retained bytes per fetched artifact scope directory (256 MiB). |
| `preserve_recent_critical_messages` | `int` | `6` | Count of most-recent critical messages protected from compaction pruning. |
| `compaction_pressure_ratio_soft` | `float` | `0.86` | Soft pressure ratio threshold for compaction policy. |
| `compaction_pressure_ratio_hard` | `float` | `1.02` | Hard pressure ratio threshold for compaction policy. |
| `compaction_no_gain_min_delta_chars` | `int` | `24` | Minimum char delta to treat compaction as meaningful gain. |
| `compaction_no_gain_attempt_limit` | `int` | `2` | Max consecutive no-gain compaction attempts before skipping. |
| `compaction_timeout_guard_seconds` | `int` | `30` | Timeout guard for compaction operations. |
| `extractor_timeout_guard_seconds` | `int` | `20` | Timeout guard for asynchronous memory extraction. |
| `extractor_tool_args_max_chars` | `int` | `260` | Tool argument excerpt cap for extraction prompts. |
| `extractor_tool_trace_max_chars` | `int` | `3600` | Tool trace excerpt cap for extraction prompts. |
| `extractor_prompt_max_chars` | `int` | `9000` | Max prompt size budget for extraction calls. |
| `compaction_churn_warning_calls` | `int` | `10` | Threshold for compaction churn warning telemetry. |

### `[limits.verifier]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_tool_args_chars` | `int` | `360` | Max tool-argument excerpt size in verifier prompts. |
| `max_tool_status_chars` | `int` | `320` | Max status excerpt size in verifier prompts. |
| `max_tool_calls_tokens` | `int` | `4000` | Budget for serialized tool-call context in verification. |
| `max_verifier_prompt_tokens` | `int` | `12000` | Verifier prompt assembly budget. |
| `max_result_summary_chars` | `int` | `7000` | Max result summary size before extra compaction. |
| `compact_result_summary_chars` | `int` | `2600` | Aggressive result-summary target. |
| `max_evidence_section_chars` | `int` | `4200` | Evidence section limit for verification output. |
| `max_evidence_section_compact_chars` | `int` | `2200` | Compacted evidence section target. |
| `max_artifact_section_chars` | `int` | `4200` | Artifact section limit for verification output. |
| `max_artifact_section_compact_chars` | `int` | `2200` | Compacted artifact section target. |
| `max_tool_output_excerpt_chars` | `int` | `1100` | Tool output excerpt cap in verifier payloads. |
| `max_artifact_file_excerpt_chars` | `int` | `800` | Artifact file excerpt cap in verifier payloads. |

### `[limits.compactor]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_chunk_chars` | `int` | `9000` | Chunk size before hierarchical map/reduce compaction. |
| `max_chunks_per_round` | `int` | `12` | Max chunks compacted per reduction round. |
| `max_reduction_rounds` | `int` | `4` | Max full compaction reduction rounds per payload. |
| `min_compact_target_chars` | `int` | `140` | Floor target when reducing per-attempt character budget. |
| `response_tokens_floor` | `int` | `256` | Minimum `max_tokens` sent to compactor model calls. |
| `response_tokens_ratio` | `float` | `0.75` | Token-budget ratio derived from hard character limits. |
| `response_tokens_buffer` | `int` | `256` | Fixed token headroom added to compactor budget. |
| `json_headroom_chars_floor` | `int` | `48` | Minimum extra characters reserved for JSON envelope overhead. |
| `json_headroom_chars_ratio` | `float` | `0.08` | Ratio-based JSON envelope headroom. |
| `json_headroom_chars_cap` | `int` | `320` | Upper cap for JSON envelope headroom. |
| `chars_per_token_estimate` | `float` | `3.6` | Character/token estimator used for budget calculations. |
| `token_headroom` | `int` | `24` | Extra tokens added after char/token conversion. |
| `target_chars_ratio` | `float` | `0.75` | Attempt target ratio under hard limit. |

Compactor validation warnings:
- If final retry still exceeds target chars, Loom keeps the compacted output and emits warning telemetry fields (`compactor_warning`, `compactor_warning_reason`, `compactor_warning_delta_chars`, `compactor_warning_target_chars`, `compactor_warning_received_chars`) instead of truncating.

### Run Telemetry Event Contracts

When `limits.runner.enable_artifact_telemetry_events = true`, Loom emits these
run-log events (`.events.jsonl`) from orchestration boundaries:

- `artifact_ingest_classified`
- `artifact_ingest_completed`
- `artifact_retention_pruned` (only when `files_deleted > 0`)
- `artifact_read_completed`
- `compaction_policy_decision`
- `overflow_fallback_applied` (only when fallback rewrite executes)
- `telemetry_run_summary` (once per task finalization)
- `task_budget_exhausted`
- `task_plan_degraded`
- `tool_call_deduplicated`
- `task_run_acquired`
- `task_run_heartbeat`
- `task_run_recovered`

Artifact event required fields:
- `subtask_id`, `tool`, `url` (sanitized), `content_kind`, `content_type`, `status` (`ok|error`)

Artifact event optional fields (when available):
- `artifact_ref`
- `artifact_workspace_relpath` (preferred) or `artifact_path` (fallback)
- `size_bytes`
- `declared_size_bytes`
- `handler`
- `extracted_chars`
- `extraction_truncated`
- `handler_metadata` (bounded by `artifact_telemetry_max_metadata_chars`)

Retention event fields:
- `scopes_scanned`, `files_deleted`, `bytes_deleted`

Compaction decision event fields:
- `subtask_id`, `pressure_ratio`, `policy_mode`
- `decision` (`skip|compact_tool|compact_history|fallback_rewrite`)
- `reason` (deterministic short code)

Overflow fallback fields:
- `rewritten_messages`, `chars_reduced`, `preserved_recent_messages`

Run summary fields:
- `run_id`
- `model_invocations`
- `tool_calls`
- `mutating_tool_calls`
- `artifact_ingests`
- `artifact_reads`
- `artifact_retention_deletes`
- `compaction_policy_decisions`
- `overflow_fallback_count`
- `compactor_warning_count`
- `budget_snapshot`

Safety notes:
- Telemetry events do not include raw extracted document text or binary payload snippets.
- URL telemetry removes query strings and fragments.
- Prefer `artifact_workspace_relpath` over absolute paths.
- Oversize `handler_metadata` is reduced to deterministic summary metadata instead of raw payload slicing.

### `[memory]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `database_path` | `string` | `"~/.loom/loom.db"` | SQLite database location. |

### `[logging]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `level` | `string` | `"INFO"` | Log level (`DEBUG`, `INFO`, etc.). |
| `event_log_path` | `string` | `"~/.loom/logs"` | Directory for event log files. |

### `[process]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `default` | `string` | `""` | Default process name/path to apply when omitted. |
| `search_paths` | `list[string]` | `[]` | Additional process discovery paths. |
| `require_rule_scope_metadata` | `bool` | `false` | Enforce stricter rule scope metadata validation. |
| `require_v2_contract` | `bool` | `false` | Require `schema_version: 2` process contracts. |
| `tui_run_scoped_workspace_enabled` | `bool` | `true` | Create per-run subfolders for TUI `/run` executions. |
| `llm_run_folder_naming_enabled` | `bool` | `true` | Allow model-generated run folder names in TUI. Low-quality/echoed names are rejected and fallback naming is used. |

### `[tui]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `chat_resume_page_size` | `int` | `250` | Number of transcript rows loaded during initial session chat hydration. |
| `chat_resume_max_rendered_rows` | `int` | `1200` | Maximum chat transcript rows kept mounted before oldest rows are trimmed. |
| `chat_resume_use_event_journal` | `bool` | `true` | Prefer replaying persisted chat event journal rows when available. |
| `chat_resume_enable_legacy_fallback` | `bool` | `true` | Fallback to synthesizing chat transcript rows from `conversation_turns` when needed. |
| `realtime_refresh_enabled` | `bool` | `true` | Enables live UI refresh for workspace tree, files panel, and progress surfaces. |
| `workspace_watch_backend` | `string` | `"poll"` | Workspace watch backend (`poll`, `native`). `native` currently falls back to polling. |
| `workspace_poll_interval_ms` | `int` | `1000` | Poll interval for workspace change detection when realtime refresh is enabled. |
| `workspace_refresh_debounce_ms` | `int` | `250` | Debounce window for coalescing repeated workspace refresh requests. |
| `workspace_refresh_max_wait_ms` | `int` | `1500` | Maximum delay before forcing a queued workspace refresh. |
| `workspace_scan_max_entries` | `int` | `20000` | Max filesystem entries sampled per poll signature snapshot. |
| `chat_stream_flush_interval_ms` | `int` | `120` | Sparse flush cadence for buffered streaming chat chunks. |
| `files_panel_max_rows` | `int` | `2000` | Maximum retained rows in the Files panel before oldest rows are dropped. |
| `delegate_progress_max_lines` | `int` | `150` | Maximum retained lines per collapsed delegate-progress section. |
| `run_launch_heartbeat_interval_ms` | `int` | `6000` | Heartbeat cadence for `/run` launch/running stage lines when progress is quiet. |
| `run_launch_timeout_seconds` | `int` | `300` | Timeout for `/run` preflight launch stages before delegate execution starts. |
| `run_preflight_async_enabled` | `bool` | `true` | Runs `/run` preflight in a background worker; set `false` for rollback to inline preflight behavior. |

### Legacy `[mcp]` in `loom.toml` (supported)

Preferred MCP config lives in `mcp.toml`, but Loom still accepts server
definitions inside `loom.toml`:

`[mcp.servers.<alias>]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `command` | `string` | `""` | Command used to launch MCP server process. |
| `args` | `list[string]` | `[]` | Command arguments. |
| `env` | `table` | `{}` | Environment variables for MCP server process. |
| `cwd` | `string` | `""` | Working directory for MCP server process. |
| `timeout_seconds` | `int` | `30` | MCP call timeout budget. |
| `enabled` | `bool` | `true` | Enables/disables that MCP server. |

## Environment Overrides

| Env var | Overrides | Notes |
| --- | --- | --- |
| `LOOM_DELEGATE_TIMEOUT_SECONDS` | `execution.delegate_task_timeout_seconds` | Used by delegated orchestration (`/run`, `delegate_task`). |

## Normalization Rules

- `execution.delegate_task_timeout_seconds` is clamped to at least `1`.
- `execution.model_call_max_attempts` is clamped to `1..10`.
- `execution.model_call_retry_base_delay_seconds` and jitter are clamped to `>= 0`.
- `execution.model_call_retry_max_delay_seconds` is clamped to `>= base_delay`.
- `verification.unconfirmed_supporting_threshold` is clamped to `0..1`.
- `verification.confirm_or_prune_max_attempts` is clamped to at least `1`.
- `verification.confirm_or_prune_backoff_seconds` is clamped to `>= 0`.
- `verification.remediation_queue_max_attempts` is clamped to at least `1`.
- `verification.remediation_queue_backoff_seconds` is clamped to `>= 0`.
- `verification.remediation_queue_max_backoff_seconds` is clamped to `>= remediation_queue_backoff_seconds`.
- `verification.contradiction_scan_max_files` is clamped to `1..1000`.
- `verification.contradiction_scan_max_total_bytes` is clamped to `1024..50000000`.
- `verification.contradiction_scan_max_file_bytes` is clamped to `1..10000000` and cannot exceed `contradiction_scan_max_total_bytes`.
- `verification.contradiction_scan_min_files_for_sufficiency` is clamped to `1..100`.
- `verification.contradiction_scan_allowed_suffixes` is normalized to unique lowercase suffixes prefixed with `.`.
- `limits.planning_response_max_tokens` is clamped to `0..500000`.
- `limits.adhoc_repair_source_max_chars` is clamped to `0..500000`.
- `limits.evidence_context_text_max_chars` is clamped to `200..100000`.
- `limits.runner.max_tool_iterations` is clamped to `4..60`.
- `limits.runner.max_subtask_wall_clock_seconds` is clamped to `60..86400`.
- `limits.runner.max_model_context_tokens` is clamped to `2048..500000`.
- `limits.runner.max_state_summary_chars` is clamped to `120..20000`.
- `limits.runner.max_verification_summary_chars` is clamped to `400..40000`.
- `limits.runner.default_tool_result_output_chars` is clamped to `400..40000`.
- `limits.runner.heavy_tool_result_output_chars` is clamped to `200..20000`.
- `limits.runner.compact_tool_result_output_chars` is clamped to `80..20000`.
- `limits.runner.compact_text_output_chars` is clamped to `80..20000`.
- `limits.runner.minimal_text_output_chars` is clamped to `40..10000`.
- `limits.runner.tool_call_argument_context_chars` is clamped to `80..20000`.
- `limits.runner.compact_tool_call_argument_chars` is clamped to `40..10000`.
- Invalid `limits.runner.runner_compaction_policy_mode` falls back to `"tiered"`.
- `limits.runner.artifact_telemetry_max_metadata_chars` is clamped to `120..20000`.
- `limits.runner.ingest_artifact_retention_max_age_days` is clamped to `0..3650`.
- `limits.runner.ingest_artifact_retention_max_files_per_scope` is clamped to `1..200000`.
- `limits.runner.ingest_artifact_retention_max_bytes_per_scope` is clamped to `1024..20000000000`.
- `limits.runner.preserve_recent_critical_messages` is clamped to `2..30`.
- `limits.runner.compaction_pressure_ratio_soft` is clamped to `0.4..2.5`.
- `limits.runner.compaction_pressure_ratio_hard` is clamped to `0.41..3.0` and kept above `soft + 0.01`.
- `limits.runner.compaction_no_gain_min_delta_chars` is clamped to `1..5000`.
- `limits.runner.compaction_no_gain_attempt_limit` is clamped to `1..25`.
- `limits.runner.compaction_timeout_guard_seconds` is clamped to `0..3600`.
- `limits.runner.extractor_timeout_guard_seconds` is clamped to `0..3600`.
- `limits.runner.extractor_tool_args_max_chars` is clamped to `80..20000`.
- `limits.runner.extractor_tool_trace_max_chars` is clamped to `200..80000`.
- `limits.runner.extractor_prompt_max_chars` is clamped to `400..120000`.
- `limits.runner.compaction_churn_warning_calls` is clamped to `1..500`.
- `limits.verifier.max_tool_args_chars` is clamped to `80..20000`.
- `limits.verifier.max_tool_status_chars` is clamped to `80..20000`.
- `limits.verifier.max_tool_calls_tokens` is clamped to `400..60000`.
- `limits.verifier.max_verifier_prompt_tokens` is clamped to `800..120000`.
- `limits.verifier.max_result_summary_chars` is clamped to `200..100000`.
- `limits.verifier.compact_result_summary_chars` is clamped to `120..100000`.
- `limits.verifier.max_evidence_section_chars` is clamped to `200..100000`.
- `limits.verifier.max_evidence_section_compact_chars` is clamped to `120..100000`.
- `limits.verifier.max_artifact_section_chars` is clamped to `200..100000`.
- `limits.verifier.max_artifact_section_compact_chars` is clamped to `120..100000`.
- `limits.verifier.max_tool_output_excerpt_chars` is clamped to `120..40000`.
- `limits.verifier.max_artifact_file_excerpt_chars` is clamped to `120..40000`.
- `limits.compactor.max_chunk_chars` is clamped to `300..200000`.
- `limits.compactor.max_chunks_per_round` is clamped to `1..100`.
- `limits.compactor.max_reduction_rounds` is clamped to `1..20`.
- `limits.compactor.min_compact_target_chars` is clamped to `20..20000`.
- `limits.compactor.response_tokens_floor` is clamped to `0..100000`.
- `limits.compactor.response_tokens_ratio` is clamped to `0.0..8.0`.
- `limits.compactor.response_tokens_buffer` is clamped to `0..100000`.
- `limits.compactor.json_headroom_chars_floor` is clamped to `0..20000`.
- `limits.compactor.json_headroom_chars_ratio` is clamped to `0.0..2.0`.
- `limits.compactor.json_headroom_chars_cap` is clamped to `0..100000`.
- `limits.compactor.chars_per_token_estimate` is clamped to `0.1..16.0`.
- `limits.compactor.token_headroom` is clamped to `0..20000`.
- `limits.compactor.target_chars_ratio` is clamped to `0.01..1.0`.
- `tui.chat_resume_page_size` is clamped to `20..500`.
- `tui.chat_resume_max_rendered_rows` is clamped to `100..10000`.
- Invalid `tui.workspace_watch_backend` falls back to `"poll"`.
- `tui.workspace_poll_interval_ms` is clamped to `200..10000`.
- `tui.workspace_refresh_debounce_ms` is clamped to `50..5000`.
- `tui.workspace_refresh_max_wait_ms` is clamped to `200..30000`.
- `tui.workspace_scan_max_entries` is clamped to `500..200000`.
- `tui.chat_stream_flush_interval_ms` is clamped to `40..2000`.
- `tui.files_panel_max_rows` is clamped to `100..20000`.
- `tui.delegate_progress_max_lines` is clamped to `20..5000`.
- `tui.run_launch_heartbeat_interval_ms` is clamped to `500..30000`.
- `tui.run_launch_timeout_seconds` is clamped to `5..600`.
- MCP `timeout_seconds` falls back to `30` when invalid/non-positive.
