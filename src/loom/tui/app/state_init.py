"""LoomApp constructor state initialization."""

from __future__ import annotations

from datetime import UTC, datetime

from loom.config_runtime.store import ConfigRuntimeStore
from loom.cowork.approval import ApprovalDecision


def initialize_app_state(
    self,
    *,
    model,
    tools,
    workspace,
    config,
    db,
    store,
    resume_session,
    explicit_mcp_path,
    legacy_config_path,
    explicit_auth_path,
) -> None:
    self._model = model
    self._tools = tools
    self._workspace = workspace
    self._config = config
    self._config_source_path = legacy_config_path
    self._config_runtime_store = ConfigRuntimeStore(
        config,
        source_path=legacy_config_path,
    )
    self._db = db
    self._store = store
    self._resume_session = resume_session
    self._explicit_mcp_path = explicit_mcp_path
    self._legacy_config_path = legacy_config_path
    self._explicit_auth_path = explicit_auth_path
    self._process_defn = None
    self._session = None
    self._chat_busy = False
    self._chat_turn_worker = None
    self._chat_stop_requested = False
    self._chat_stop_inflight = False
    self._chat_stop_requested_at = 0.0
    self._chat_stop_last_path = ""
    self._chat_stop_last_error = ""
    self._chat_redirect_inflight = False
    self._cowork_inflight_tool_counts = {}
    self._pending_inject_directives = []
    self._last_rendered_steer_queue_signature = ()
    self._active_redirect_directive = None
    self._last_applied_directive_id = ""
    self._steer_last_error = ""
    self._total_tokens = 0
    self._telemetry_runtime_override_mode = None
    self._telemetry_mode_updated_at = datetime.now(UTC).isoformat()

    self._approval_event = None
    self._approval_result = ApprovalDecision.DENY

    self._recall_tool = None
    self._delegate_tool = None
    self._confirm_exit_waiter = None
    self._slash_cycle_seed = ""
    self._slash_cycle_candidates = []
    self._applying_slash_tab_completion = False
    self._skip_slash_cycle_reset_once = False
    self._input_history = []
    self._input_history_nav_index = None
    self._input_history_nav_draft = ""
    self._applying_input_history_navigation = False
    self._skip_input_history_reset_once = False
    self._process_runs = {}
    self._process_elapsed_timer = None
    self._process_command_map = {}
    self._blocked_process_commands = []
    self._cached_process_catalog = []
    self._process_command_index_last_refresh_at = 0.0
    self._process_command_index_refresh_inflight = False
    self._adhoc_process_cache = {}
    self._adhoc_package_doc_cache = None
    self._sidebar_cowork_tasks = []
    self._process_close_hint_shown = False
    self._close_process_tab_inflight = set()
    self._process_run_cancel_handlers = {}
    self._process_run_pause_handlers = {}
    self._process_run_play_handlers = {}
    self._process_run_inject_handlers = {}
    self._process_run_answer_handlers = {}
    self._process_run_question_locks = {}
    self._process_run_seen_questions = {}
    self._process_run_pending_inject = {}
    self._process_run_user_input_pause_depths = {}
    self._process_run_user_input_pause_started_at = {}
    self._process_run_user_input_paused_accumulated_seconds = {}
    self._auto_resume_workspace_on_init = True
    self._run_auth_profile_overrides = {}
    self._chat_replay_events = []
    self._chat_history_source = ""
    self._chat_history_oldest_seq = None
    self._chat_history_oldest_turn = None
    self._chat_trimmed_total = 0
    self._active_delegate_streams = {}
    self._workspace_refresh_pending_reasons = set()
    self._workspace_refresh_first_request_at = 0.0
    self._workspace_refresh_timer_pending = False
    self._workspace_refresh_timer = None
    self._workspace_poll_timer = None
    self._workspace_poll_inflight = False
    self._workspace_signature = None
    self._workspace_scan_overflow_notified = False
    self._files_panel_recent_ops = {}
    self._files_panel_dedupe_window_seconds = 1.5
    self._startup_landing_active = False
    self._last_landing_slash_hint_text = ""
