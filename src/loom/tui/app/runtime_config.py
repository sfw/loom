"""Runtime and config-derived accessor helpers."""

from __future__ import annotations

from pathlib import Path

from loom.models.retry import ModelRetryPolicy

from .constants import (
    _DEFAULT_CHAT_RESUME_MAX_RENDERED_ROWS,
    _DEFAULT_CHAT_RESUME_PAGE_SIZE,
    _DEFAULT_TUI_CHAT_STREAM_FLUSH_INTERVAL_MS,
    _DEFAULT_TUI_DELEGATE_PROGRESS_MAX_LINES,
    _DEFAULT_TUI_FILES_PANEL_MAX_ROWS,
    _DEFAULT_TUI_REALTIME_REFRESH_ENABLED,
    _DEFAULT_TUI_RUN_CANCEL_WAIT_TIMEOUT_SECONDS,
    _DEFAULT_TUI_RUN_CLOSE_MODAL_TIMEOUT_SECONDS,
    _DEFAULT_TUI_RUN_LAUNCH_HEARTBEAT_INTERVAL_MS,
    _DEFAULT_TUI_RUN_LAUNCH_TIMEOUT_SECONDS,
    _DEFAULT_TUI_RUN_PREFLIGHT_ASYNC_ENABLED,
    _DEFAULT_TUI_RUN_PROGRESS_REFRESH_INTERVAL_MS,
    _DEFAULT_TUI_WORKSPACE_POLL_INTERVAL_MS,
    _DEFAULT_TUI_WORKSPACE_REFRESH_DEBOUNCE_MS,
    _DEFAULT_TUI_WORKSPACE_REFRESH_MAX_WAIT_MS,
    _DEFAULT_TUI_WORKSPACE_SCAN_MAX_ENTRIES,
    _DEFAULT_TUI_WORKSPACE_WATCH_BACKEND,
    _MUTATING_TOOL_FALLBACK,
)


def model_retry_policy(self) -> ModelRetryPolicy:
    if self._config is None:
        return ModelRetryPolicy()
    return ModelRetryPolicy.from_execution_config(self._config.execution)


def cowork_tool_exposure_mode(self) -> str:
    if self._config is None:
        return "hybrid"
    mode = str(
        getattr(self._config.execution, "cowork_tool_exposure_mode", "hybrid")
        or "hybrid",
    ).strip().lower()
    if mode in {"full", "adaptive", "hybrid"}:
        return mode
    return "hybrid"


def cowork_memory_index_enabled(self) -> bool:
    if self._config is None:
        return True
    execution = getattr(self._config, "execution", None)
    if execution is None:
        return True
    return bool(
        getattr(execution, "cowork_memory_index_enabled", True),
    )


def cowork_memory_index_v2_actions_enabled(self) -> bool:
    if self._config is None:
        return True
    execution = getattr(self._config, "execution", None)
    if execution is None:
        return True
    return bool(
        getattr(
            execution,
            "cowork_memory_index_v2_actions_enabled",
            True,
        ),
    )


def cowork_memory_index_force_fts(self) -> bool:
    if self._config is None:
        return False
    execution = getattr(self._config, "execution", None)
    if execution is None:
        return False
    return bool(
        getattr(execution, "cowork_memory_index_force_fts", False),
    )


def cowork_indexer_model_role_strict(self) -> bool:
    if self._config is None:
        return False
    execution = getattr(self._config, "execution", None)
    if execution is None:
        return False
    return bool(
        getattr(execution, "cowork_indexer_model_role_strict", False),
    )


def cowork_memory_index_llm_extraction_enabled(self) -> bool:
    if self._config is None:
        return True
    execution = getattr(self._config, "execution", None)
    if execution is None:
        return True
    return bool(
        getattr(
            execution,
            "cowork_memory_index_llm_extraction_enabled",
            True,
        ),
    )


def cowork_memory_index_queue_max_batches(self) -> int:
    if self._config is None:
        return 32
    execution = getattr(self._config, "execution", None)
    if execution is None:
        return 32
    return max(
        1,
        int(
            getattr(
                execution,
                "cowork_memory_index_queue_max_batches",
                32,
            ),
        ),
    )


def cowork_memory_index_section_limit(self) -> int:
    if self._config is None:
        return 4
    execution = getattr(self._config, "execution", None)
    if execution is None:
        return 4
    return max(
        1,
        int(
            getattr(
                execution,
                "cowork_memory_index_section_limit",
                4,
            ),
        ),
    )


def cowork_recall_index_max_chars(self) -> int:
    if self._config is None:
        return 1200
    execution = getattr(self._config, "execution", None)
    if execution is None:
        return 1200
    return max(
        600,
        int(
            getattr(
                execution,
                "cowork_recall_index_max_chars",
                1200,
            ),
        ),
    )


def cowork_max_context_tokens(self) -> int:
    runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
    if runner_limits is None:
        return 24_000
    return max(
        4096,
        int(getattr(runner_limits, "max_model_context_tokens", 24_000)),
    )


def cowork_scratch_dir(self) -> Path | None:
    if self._config is None:
        return None
    try:
        return self._config.scratch_path
    except Exception:
        return None


def cowork_enable_filetype_ingest_router(self) -> bool:
    runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
    if runner_limits is None:
        return True
    return bool(getattr(runner_limits, "enable_filetype_ingest_router", True))


def cowork_ingest_artifact_retention_max_age_days(self) -> int:
    runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
    if runner_limits is None:
        return 14
    return max(0, int(getattr(runner_limits, "ingest_artifact_retention_max_age_days", 14)))


def cowork_ingest_artifact_retention_max_files_per_scope(self) -> int:
    runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
    if runner_limits is None:
        return 96
    return max(
        1,
        int(getattr(runner_limits, "ingest_artifact_retention_max_files_per_scope", 96)),
    )


def cowork_ingest_artifact_retention_max_bytes_per_scope(self) -> int:
    runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
    if runner_limits is None:
        return 268_435_456
    return max(
        1024,
        int(getattr(
            runner_limits,
            "ingest_artifact_retention_max_bytes_per_scope",
            268_435_456,
        )),
    )


def chat_resume_page_size(self) -> int:
    tui_cfg = getattr(self._config, "tui", None)
    if tui_cfg is None:
        return _DEFAULT_CHAT_RESUME_PAGE_SIZE
    try:
        value = int(getattr(tui_cfg, "chat_resume_page_size", _DEFAULT_CHAT_RESUME_PAGE_SIZE))
    except Exception:
        return _DEFAULT_CHAT_RESUME_PAGE_SIZE
    return max(20, min(value, 500))


def chat_resume_max_rendered_rows(self) -> int:
    tui_cfg = getattr(self._config, "tui", None)
    if tui_cfg is None:
        return _DEFAULT_CHAT_RESUME_MAX_RENDERED_ROWS
    try:
        value = int(getattr(
            tui_cfg,
            "chat_resume_max_rendered_rows",
            _DEFAULT_CHAT_RESUME_MAX_RENDERED_ROWS,
        ))
    except Exception:
        return _DEFAULT_CHAT_RESUME_MAX_RENDERED_ROWS
    return max(100, min(value, 10_000))


def chat_resume_use_event_journal(self) -> bool:
    tui_cfg = getattr(self._config, "tui", None)
    if tui_cfg is None:
        return True
    try:
        return bool(getattr(tui_cfg, "chat_resume_use_event_journal", True))
    except Exception:
        return True


def chat_resume_enable_legacy_fallback(self) -> bool:
    tui_cfg = getattr(self._config, "tui", None)
    if tui_cfg is None:
        return True
    try:
        return bool(getattr(tui_cfg, "chat_resume_enable_legacy_fallback", True))
    except Exception:
        return True


def tui_progress_auto_follow(self) -> bool:
    return True


def tui_realtime_refresh_enabled(self) -> bool:
    tui_cfg = getattr(self._config, "tui", None)
    if tui_cfg is None:
        return _DEFAULT_TUI_REALTIME_REFRESH_ENABLED
    try:
        return bool(
            getattr(
                tui_cfg,
                "realtime_refresh_enabled",
                _DEFAULT_TUI_REALTIME_REFRESH_ENABLED,
            )
        )
    except Exception:
        return _DEFAULT_TUI_REALTIME_REFRESH_ENABLED


def tui_workspace_watch_backend(self) -> str:
    tui_cfg = getattr(self._config, "tui", None)
    if tui_cfg is None:
        return _DEFAULT_TUI_WORKSPACE_WATCH_BACKEND
    value = str(
        getattr(
            tui_cfg,
            "workspace_watch_backend",
            _DEFAULT_TUI_WORKSPACE_WATCH_BACKEND,
        )
        or _DEFAULT_TUI_WORKSPACE_WATCH_BACKEND,
    ).strip().lower()
    return value if value in {"poll", "native"} else _DEFAULT_TUI_WORKSPACE_WATCH_BACKEND


def tui_workspace_poll_interval_seconds(self) -> float:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_WORKSPACE_POLL_INTERVAL_MS
    if tui_cfg is None:
        return default / 1000.0
    try:
        value_ms = int(getattr(tui_cfg, "workspace_poll_interval_ms", default))
    except Exception:
        value_ms = default
    return max(0.2, min(value_ms, 10_000) / 1000.0)


def tui_workspace_refresh_debounce_seconds(self) -> float:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_WORKSPACE_REFRESH_DEBOUNCE_MS
    if tui_cfg is None:
        return default / 1000.0
    try:
        value_ms = int(getattr(tui_cfg, "workspace_refresh_debounce_ms", default))
    except Exception:
        value_ms = default
    return max(0.05, min(value_ms, 5000) / 1000.0)


def tui_workspace_refresh_max_wait_seconds(self) -> float:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_WORKSPACE_REFRESH_MAX_WAIT_MS
    if tui_cfg is None:
        return default / 1000.0
    try:
        value_ms = int(getattr(tui_cfg, "workspace_refresh_max_wait_ms", default))
    except Exception:
        value_ms = default
    return max(0.2, min(value_ms, 30_000) / 1000.0)


def tui_workspace_scan_max_entries(self) -> int:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_WORKSPACE_SCAN_MAX_ENTRIES
    if tui_cfg is None:
        return default
    try:
        value = int(getattr(tui_cfg, "workspace_scan_max_entries", default))
    except Exception:
        value = default
    return max(500, min(value, 200_000))


def tui_chat_stream_flush_interval_ms(self) -> int:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_CHAT_STREAM_FLUSH_INTERVAL_MS
    if tui_cfg is None:
        return default
    try:
        value = int(getattr(tui_cfg, "chat_stream_flush_interval_ms", default))
    except Exception:
        value = default
    return max(40, min(value, 2000))


def tui_files_panel_max_rows(self) -> int:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_FILES_PANEL_MAX_ROWS
    if tui_cfg is None:
        return default
    try:
        value = int(getattr(tui_cfg, "files_panel_max_rows", default))
    except Exception:
        value = default
    return max(100, min(value, 20_000))


def tui_delegate_progress_max_lines(self) -> int:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_DELEGATE_PROGRESS_MAX_LINES
    if tui_cfg is None:
        return default
    try:
        value = int(getattr(tui_cfg, "delegate_progress_max_lines", default))
    except Exception:
        value = default
    return max(20, min(value, 5000))


def tui_run_launch_heartbeat_interval_seconds(self) -> float:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_RUN_LAUNCH_HEARTBEAT_INTERVAL_MS
    if tui_cfg is None:
        return default / 1000.0
    try:
        value_ms = int(getattr(tui_cfg, "run_launch_heartbeat_interval_ms", default))
    except Exception:
        value_ms = default
    return max(0.5, min(value_ms, 30_000) / 1000.0)


def tui_run_launch_timeout_seconds(self) -> float:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_RUN_LAUNCH_TIMEOUT_SECONDS
    if tui_cfg is None:
        return float(default)
    try:
        value = float(getattr(tui_cfg, "run_launch_timeout_seconds", default))
    except Exception:
        value = float(default)
    return max(5.0, min(value, 600.0))


def tui_run_close_modal_timeout_seconds(self) -> float:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_RUN_CLOSE_MODAL_TIMEOUT_SECONDS
    if tui_cfg is None:
        return float(default)
    try:
        value = float(getattr(tui_cfg, "run_close_modal_timeout_seconds", default))
    except Exception:
        value = float(default)
    return max(5.0, min(value, 300.0))


def tui_run_cancel_wait_timeout_seconds(self) -> float:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_RUN_CANCEL_WAIT_TIMEOUT_SECONDS
    if tui_cfg is None:
        return float(default)
    try:
        value = float(getattr(tui_cfg, "run_cancel_wait_timeout_seconds", default))
    except Exception:
        value = float(default)
    return max(1.0, min(value, 120.0))


def tui_run_progress_refresh_interval_seconds(self) -> float:
    tui_cfg = getattr(self._config, "tui", None)
    default = _DEFAULT_TUI_RUN_PROGRESS_REFRESH_INTERVAL_MS
    if tui_cfg is None:
        return default / 1000.0
    try:
        value_ms = int(getattr(tui_cfg, "run_progress_refresh_interval_ms", default))
    except Exception:
        value_ms = default
    return max(0.05, min(value_ms, 2000) / 1000.0)


def tui_run_preflight_async_enabled(self) -> bool:
    tui_cfg = getattr(self._config, "tui", None)
    if tui_cfg is None:
        return _DEFAULT_TUI_RUN_PREFLIGHT_ASYNC_ENABLED
    try:
        return bool(
            getattr(
                tui_cfg,
                "run_preflight_async_enabled",
                _DEFAULT_TUI_RUN_PREFLIGHT_ASYNC_ENABLED,
            )
        )
    except Exception:
        return _DEFAULT_TUI_RUN_PREFLIGHT_ASYNC_ENABLED


def is_mutating_tool(self, tool_name: str) -> bool:
    name = str(tool_name or "").strip()
    if not name:
        return False
    tool = self._tools.get(name)
    if tool is not None:
        try:
            return bool(getattr(tool, "is_mutating", False))
        except Exception:
            return name in _MUTATING_TOOL_FALLBACK
    return name in _MUTATING_TOOL_FALLBACK
