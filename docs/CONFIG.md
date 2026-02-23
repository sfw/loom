# Loom Configuration Reference

This document lists all supported `loom.toml` and MCP config keys, their defaults,
and what each option controls.

## Config Resolution Order

Loom loads configuration in this order:

1. `--config /explicit/path/loom.toml` (CLI flag, when provided)
2. `./loom.toml` (current working directory)
3. `~/.loom/loom.toml` (user config)
4. Built-in defaults

MCP server configuration can be supplied in:

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

`<name>` is any model alias you choose (for example `primary`, `utility`, `planner`).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `provider` | `string` | required | One of `ollama`, `openai_compatible`, `anthropic`. |
| `base_url` | `string` | `""` | Provider API endpoint base URL. |
| `model` | `string` | `""` | Model identifier to call. |
| `max_tokens` | `int` | `4096` | Max completion tokens for that model profile. |
| `temperature` | `float` | `0.1` | Sampling temperature. |
| `roles` | `list[string]` | `["executor"]` | Assigned roles: `planner`, `executor`, `extractor`, `verifier`. |
| `api_key` | `string` | `""` | API key (if provider requires auth). |
| `tier` | `int` | `0` | Optional explicit quality/cost tier (`0` = auto-detect). |
| `capabilities.vision` | `bool` | auto-detected | Override model vision support. |
| `capabilities.native_pdf` | `bool` | auto-detected | Override native PDF support. |
| `capabilities.thinking` | `bool` | auto-detected | Override long-reasoning support hint. |
| `capabilities.citations` | `bool` | auto-detected | Override citation support hint. |
| `capabilities.audio_input` | `bool` | auto-detected | Override audio input support hint. |
| `capabilities.audio_output` | `bool` | auto-detected | Override audio output support hint. |

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

### `[limits]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `planning_response_max_tokens` | `int` | `16384` | Planner synthesis response budget. |
| `adhoc_repair_source_max_chars` | `int` | `0` | Source truncation limit for ad hoc JSON repair (`0` means disabled). |
| `evidence_context_text_max_chars` | `int` | `8192` | Evidence context cap fed into planning/verification prompts. |

### `[limits.runner]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_tool_iterations` | `int` | `20` | Max tool/model loop iterations per subtask execution pass. |
| `max_subtask_wall_clock_seconds` | `int` | `1200` | Per-subtask wall-clock timeout budget. |
| `max_model_context_tokens` | `int` | `24000` | Runner model-context budget hint. |
| `max_state_summary_chars` | `int` | `640` | Target size for compacted state summaries. |
| `max_verification_summary_chars` | `int` | `8000` | Target size for verification summary payloads. |
| `default_tool_result_output_chars` | `int` | `2800` | Default tool output compaction target. |
| `heavy_tool_result_output_chars` | `int` | `3600` | Output target for heavy tools. |
| `compact_tool_result_output_chars` | `int` | `900` | Output target for aggressively compacted tool excerpts. |
| `compact_text_output_chars` | `int` | `1400` | Generic compacted text target. |
| `minimal_text_output_chars` | `int` | `260` | Tiny fallback compacted text target. |
| `tool_call_argument_context_chars` | `int` | `700` | Argument context extraction target. |
| `compact_tool_call_argument_chars` | `int` | `1600` | Aggressive tool-argument compaction target. |

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
| `max_chunk_chars` | `int` | `8000` | Chunk size before hierarchical map/reduce compaction. |
| `max_chunks_per_round` | `int` | `10` | Max chunks compacted per reduction round. |
| `max_reduction_rounds` | `int` | `2` | Max full compaction reduction rounds per payload. |
| `min_compact_target_chars` | `int` | `220` | Floor target when reducing per-attempt character budget. |
| `response_tokens_floor` | `int` | `256` | Minimum `max_tokens` sent to compactor model calls. |
| `response_tokens_ratio` | `float` | `0.55` | Token-budget ratio derived from hard character limits. |
| `response_tokens_buffer` | `int` | `256` | Fixed token headroom added to compactor budget. |
| `json_headroom_chars_floor` | `int` | `128` | Minimum extra characters reserved for JSON envelope overhead. |
| `json_headroom_chars_ratio` | `float` | `0.30` | Ratio-based JSON envelope headroom. |
| `json_headroom_chars_cap` | `int` | `1024` | Upper cap for JSON envelope headroom. |
| `chars_per_token_estimate` | `float` | `2.8` | Character/token estimator used for budget calculations. |
| `token_headroom` | `int` | `128` | Extra tokens added after char/token conversion. |
| `target_chars_ratio` | `float` | `0.82` | Attempt target ratio under hard limit. |

Compactor validation warnings:
- If final retry still exceeds target chars, Loom keeps the compacted output and emits warning telemetry fields (`compactor_warning`, `compactor_warning_reason`, `compactor_warning_delta_chars`) instead of truncating.

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

### Legacy `[mcp]` in `loom.toml` (supported)

The preferred MCP config lives in `mcp.toml`, but Loom still accepts MCP server
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
- MCP `timeout_seconds` falls back to `30` when invalid/non-positive.
