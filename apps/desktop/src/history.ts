import type {
  ConversationApproval,
  ConversationMessage,
  ConversationPrompt,
  ConversationStreamEvent,
  NotificationEvent,
  RunTimelineEvent,
} from "./api";

export function firstMeaningfulString(value: unknown): string {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const text = firstMeaningfulString(item);
      if (text) {
        return text;
      }
    }
    return "";
  }
  if (!value || typeof value !== "object") {
    return "";
  }
  const record = value as Record<string, unknown>;
  for (const key of ["command", "path", "question", "text", "output", "error", "label", "message"]) {
    const text = firstMeaningfulString(record[key]);
    if (text) {
      return text;
    }
  }
  for (const entry of Object.values(record)) {
    const text = firstMeaningfulString(entry);
    if (text) {
      return text;
    }
  }
  return "";
}

function summarizeTimelineEvent(event: RunTimelineEvent): string {
  const message = event.data.message;
  if (typeof message === "string" && message.trim()) {
    return message;
  }
  const status = event.data.status;
  if (typeof status === "string" && status.trim()) {
    return status;
  }
  const goal = event.data.goal;
  if (typeof goal === "string" && goal.trim()) {
    return goal;
  }
  return event.event_type.replace(/_/g, " ");
}

export function startCaseEventName(value: string): string {
  return value
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase() + part.slice(1))
    .join(" ");
}

function runEventPreview(data: Record<string, unknown>): string {
  for (const key of [
    "message",
    "summary",
    "goal",
    "instruction",
    "status",
    "tool_name",
    "command",
    "path",
    "rule_name",
    "error",
  ]) {
    const text = firstMeaningfulString(data[key]);
    if (text) {
      return text;
    }
  }
  return "";
}

// ---------------------------------------------------------------------------
// Noise filter — hide events that add no value for the end user
// ---------------------------------------------------------------------------

const NOISE_EVENT_TYPES = new Set([
  "telemetry_diagnostic",
  "telemetry_run_summary",
  "telemetry_mode_changed",
  "telemetry_settings_warning",
  "webhook_delivery_attempted",
  "webhook_delivery_succeeded",
  "webhook_delivery_failed",
  "webhook_delivery_dropped",
  "db_migration_start",
  "db_migration_applied",
  "db_migration_verify_failed",
  "db_migration_failed",
  "db_schema_ready",
  "task_run_heartbeat",
  "compaction_policy_decision",
  "overflow_fallback_applied",
  "token_streamed",
  "tool_call_completed",
  "model_invocation",
]);

export function isRunTimelineNoise(event: RunTimelineEvent): boolean {
  if (event.event_type === "model_invocation" && event.data.retry_scheduled === true) {
    return false;
  }
  if (event.event_type === "claim_verification_summary") {
    const d = event.data;
    const extracted = typeof d.extracted === "number" ? Number(d.extracted) : 0;
    const supported = typeof d.supported === "number" ? Number(d.supported) : 0;
    const partiallySupported =
      typeof d.partially_supported === "number" ? Number(d.partially_supported) : 0;
    const contradicted = typeof d.contradicted === "number" ? Number(d.contradicted) : 0;
    const insufficient =
      typeof d.insufficient_evidence === "number" ? Number(d.insufficient_evidence) : 0;
    const pruned = typeof d.pruned === "number" ? Number(d.pruned) : 0;
    if (
      extracted <= 0 &&
      supported <= 0 &&
      partiallySupported <= 0 &&
      contradicted <= 0 &&
      insufficient <= 0 &&
      pruned <= 0
    ) {
      return true;
    }
  }
  return NOISE_EVENT_TYPES.has(event.event_type);
}

// ---------------------------------------------------------------------------
// Human-readable title, detail, and pills
// ---------------------------------------------------------------------------

function _str(val: unknown): string {
  return typeof val === "string" ? val.trim() : "";
}

function _fileList(data: Record<string, unknown>): string[] {
  const raw = data.files_changed ?? data.files_changed_paths;
  if (Array.isArray(raw)) {
    return raw.filter((f): f is string => typeof f === "string" && f.trim() !== "");
  }
  return [];
}

function _toolName(data: Record<string, unknown>): string {
  return _str(data.tool) || _str(data.tool_name) || "";
}

function _subtaskLabel(data: Record<string, unknown>): string {
  return _str(data.subtask_id) || _str(data.phase_id) || "";
}

function _retryCountdownLabel(
  data: Record<string, unknown>,
  nowMs: number,
): string {
  const resumeAtMs =
    typeof data.retry_resume_at_ms === "number" ? Number(data.retry_resume_at_ms) : NaN;
  if (Number.isFinite(resumeAtMs) && resumeAtMs > 0) {
    const remainingSeconds = Math.max(0, Math.ceil((resumeAtMs - nowMs) / 1000));
    return remainingSeconds > 0 ? `retry in ${remainingSeconds}s` : "retrying now";
  }
  const delaySeconds =
    typeof data.retry_delay_seconds === "number" ? Number(data.retry_delay_seconds) : NaN;
  if (Number.isFinite(delaySeconds) && delaySeconds > 0) {
    return delaySeconds >= 10
      ? `retry in ${Math.round(delaySeconds)}s`
      : `retry in ${delaySeconds.toFixed(1)}s`;
  }
  return "retry scheduled";
}

export function runTimelineTitle(event: RunTimelineEvent): string {
  const d = event.data;
  const sub = _subtaskLabel(d);

  switch (event.event_type) {
    // --- Tool calls (most actionable) ---
    case "tool_call_started": {
      const tool = _toolName(d) || "tool";
      const args = (typeof d.args === "object" && d.args !== null ? d.args : {}) as Record<string, unknown>;
      const path = _str(d.path) || _str(args.path);
      if (path) return `${tool} → ${path}`;
      return sub ? `${tool} for ${sub}` : tool;
    }
    case "tool_call_completed": {
      const tool = _toolName(d) || "tool";
      const files = _fileList(d);
      if (d.success === false) return sub ? `${tool} failed for ${sub}` : `${tool} failed`;
      if (files.length === 1) return `${tool} → ${files[0]}`;
      if (files.length > 1) return `${tool} → ${files.length} files`;
      return sub ? `${tool} for ${sub}` : tool;
    }

    // --- Subtask lifecycle ---
    case "subtask_started":
      return sub ? `Started ${sub}` : "Subtask started";
    case "subtask_completed":
      return sub ? `Completed ${sub}` : "Subtask completed";
    case "subtask_failed":
      return sub ? `Failed ${sub}` : "Subtask failed";
    case "subtask_blocked":
      return sub ? `Blocked ${sub}` : "Subtask blocked";
    case "subtask_retrying":
      return sub ? `Retrying ${sub}` : "Subtask retrying";

    // --- Task lifecycle ---
    case "task_completed": return "Task completed";
    case "task_failed": return "Task failed";
    case "task_cancelled": return "Task cancelled";
    case "task_cancel_requested": return "Cancel requested";
    case "task_paused": return "Task paused";
    case "task_resumed": return "Task resumed";
    case "task_planning": return "Planning";
    case "task_plan_ready": return "Plan ready";
    case "task_executing": return "Executing";
    case "task_replanning": return "Replanning";
    case "task_stalled": return "Task stalled";
    case "task_budget_exhausted": return "Budget exhausted";

    // --- Verification ---
    case "verification_started":
      return sub ? `Verifying ${sub}` : "Verification started";
    case "verification_passed":
      return sub ? `Verified ${sub} ✓` : "Verification passed";
    case "verification_failed":
      return sub ? `Verification failed ${sub}` : "Verification failed";
    case "verification_rule_applied":
      return `Rule applied: ${_str(d.rule_id) || "rule"}`;
    case "verification_rule_skipped":
      return `Rule skipped: ${_str(d.rule_id) || "rule"}`;
    case "verification_outcome":
      return d.passed ? "Verification passed" : "Verification failed";

    // --- Model invocation ---
    case "model_invocation":
      if (d.retry_scheduled === true) {
        return sub ? `Retry scheduled for ${sub}` : "Model retry scheduled";
      }
      return sub ? `Model call for ${sub}` : "Model call";

    // --- Approval / user interaction ---
    case "approval_requested":
      return `Approval needed: ${_toolName(d) || "action"}`;
    case "approval_received":
      return `Approval: ${_str(d.decision) || "received"}`;
    case "ask_user_requested":
      return "Waiting for input";
    case "ask_user_answered":
      return "Input received";
    case "steer_instruction":
      return "Instruction injected";

    // --- Artifact events ---
    case "artifact_ingest_classified": {
      const path = _str(d.path) || _str(d.artifact_path);
      return path ? `Classified ${path}` : "Artifact classified";
    }
    case "artifact_ingest_completed": {
      const path = _str(d.path) || _str(d.artifact_path);
      return path ? `Ingested ${path}` : "Artifact ingested";
    }
    case "artifact_retention_pruned": {
      const path = _str(d.path) || _str(d.artifact_path);
      return path ? `Pruned ${path}` : "Artifact pruned";
    }
    case "artifact_read_completed": {
      const path = _str(d.path) || _str(d.artifact_path);
      return path ? `Read ${path}` : "Artifact read";
    }
    case "artifact_seal_validation":
      return "Artifact seal validation";
    case "artifact_confinement_violation": {
      const path = _str(d.path) || _str(d.artifact_path);
      return path ? `Confinement violation: ${path}` : "Confinement violation";
    }

    // --- Run validity ---
    case "run_validity_scorecard":
      return "Run validity scorecard";

    default:
      return startCaseEventName(event.event_type);
  }
}

export function runTimelineDetail(event: RunTimelineEvent, nowMs: number = Date.now()): string {
  const d = event.data;
  const args = (typeof d.args === "object" && d.args !== null ? d.args : {}) as Record<string, unknown>;

  switch (event.event_type) {
    // --- Tool calls: show file paths, args, or error ---
    case "tool_call_completed": {
      const files = _fileList(d);
      if (files.length > 0) return files.join(", ");
      const error = _str(d.error);
      if (error) return error;
      const path = _str(args.path);
      if (path) return path;
      const command = _str(args.command);
      if (command) return command;
      return _str(d.summary) || _str(d.message) || "";
    }
    case "tool_call_started": {
      const path = _str(d.path) || _str(args.path);
      if (path) return path;
      const command = _str(args.command);
      if (command) return command;
      const query = _str(args.query) || _str(args.search_query) || _str(args.url);
      if (query) return query;
      // Show first meaningful arg value
      for (const v of Object.values(args)) {
        const s = _str(v);
        if (s && s.length > 2) return s.length > 200 ? s.slice(0, 200) + "…" : s;
      }
      return "";
    }

    // --- Subtask lifecycle: show summary, goal, duration ---
    case "subtask_started":
      return _str(d.goal) || _str(d.description) || _str(d.message) || "";
    case "subtask_completed": {
      const parts: string[] = [];
      const summary = _str(d.summary) || _str(d.message);
      if (summary) parts.push(summary);
      const dur = typeof d.duration === "number" ? `${d.duration.toFixed(1)}s` : "";
      const outcome = _str(d.verification_outcome);
      if (dur || outcome) parts.push([dur, outcome].filter(Boolean).join(" · "));
      return parts.join(" — ") || "";
    }
    case "subtask_failed":
      return _str(d.error) || _str(d.summary) || _str(d.message) || _str(d.reason) || "";
    case "subtask_blocked":
      return _str(d.reason) || _str(d.message) || "";
    case "subtask_retrying":
      return _str(d.reason) || _str(d.error) || `Attempt ${d.attempt || "?"}`;

    // --- Verification: show rule details, outcome, confidence ---
    case "verification_started":
      return _str(d.description) || (typeof d.target_tier === "number" ? `Tier ${d.target_tier} verification` : "");
    case "verification_passed":
    case "verification_failed": {
      const parts: string[] = [];
      const outcome = _str(d.outcome) || _str(d.reason) || _str(d.reason_code);
      if (outcome) parts.push(outcome);
      if (typeof d.confidence === "number") parts.push(`confidence: ${Math.round(Number(d.confidence) * 100)}%`);
      return parts.join(" · ") || "";
    }
    case "verification_outcome": {
      const passed = d.passed ? "Passed" : "Failed";
      const reason = _str(d.reason_code) || _str(d.reason);
      const conf = typeof d.confidence === "number" ? `${Math.round(Number(d.confidence) * 100)}%` : "";
      return [passed, reason, conf].filter(Boolean).join(" · ");
    }
    case "verification_rule_applied": {
      const rule = _str(d.rule_id);
      const ruleType = _str(d.rule_type);
      const severity = _str(d.severity);
      const scope = _str(d.scope);
      return [ruleType, severity, scope].filter(Boolean).join(" · ") || rule;
    }
    case "verification_rule_skipped":
      return _str(d.reason) || _str(d.rule_type) || "";
    case "verification_contradiction_detected":
      return _str(d.description) || _str(d.message) || _str(d.detail) || "";

    // --- Claim verification ---
    case "claim_verification_summary": {
      const parts: string[] = [];
      if (typeof d.extracted === "number") parts.push(`${d.extracted} claims`);
      if (typeof d.supported === "number") parts.push(`${d.supported} supported`);
      if (typeof d.contradicted === "number" && Number(d.contradicted) > 0) parts.push(`${d.contradicted} contradicted`);
      if (typeof d.insufficient_evidence === "number" && Number(d.insufficient_evidence) > 0) parts.push(`${d.insufficient_evidence} insufficient`);
      if (typeof d.pruned === "number" && Number(d.pruned) > 0) parts.push(`${d.pruned} pruned`);
      return parts.join(" · ") || "";
    }

    // --- Model invocation ---
    case "model_invocation": {
      if (d.retry_scheduled === true) {
        const status = typeof d.http_status === "number" ? `HTTP ${d.http_status}` : "";
        const errorCode = _str(d.model_error_code);
        const model = _str(d.model);
        const countdown = _retryCountdownLabel(d, nowMs);
        return [model, countdown, status, errorCode].filter(Boolean).join(" · ");
      }
      const model = _str(d.model);
      const phase = _str(d.phase);
      const tokens = typeof d.request_est_tokens === "number" ? `~${d.request_est_tokens} tokens` : "";
      const tools = typeof d.assistant_tool_calls === "number" ? `${d.assistant_tool_calls} tool calls` : "";
      return [model, phase, tokens, tools].filter(Boolean).join(" · ");
    }

    // --- Task lifecycle ---
    case "task_completed":
    case "task_failed":
    case "task_cancelled":
    case "task_created":
      return _str(d.goal) || _str(d.summary) || _str(d.message) || "";
    case "task_restarted":
      return _str(d.message) || "Run restarted in place";
    case "task_plan_ready":
      return _str(d.message) || _str(d.summary) || "";
    case "task_stalled":
    case "task_budget_exhausted":
      return _str(d.reason) || _str(d.message) || "";
    case "task_replanning":
      return _str(d.reason) || _str(d.trigger) || "";

    // --- Approval / user interaction ---
    case "approval_requested":
      return _str(d.message) || `${_str(d.tool_name)}: approval needed`;
    case "approval_received":
      return d.approved === false ? "Rejected by operator" : _str(d.decision) || "Approved by operator";
    case "approval_rejected":
      return _str(d.reason) || "Rejected by operator";
    case "approval_timed_out":
      return typeof d.timeout_seconds === "number"
        ? `Approval timed out after ${d.timeout_seconds}s`
        : "Approval timed out";
    case "ask_user_requested":
      return _str(d.question) || _str(d.message) || "";
    case "ask_user_answered":
      return _str(d.answer) || _str(d.response) || "";
    case "steer_instruction":
      return _str(d.instruction) || "";

    // --- Artifact events ---
    case "artifact_ingest_classified": {
      const category = _str(d.category) || _str(d.classification);
      const desc = _str(d.description) || _str(d.summary);
      return [category, desc].filter(Boolean).join(" — ") || "";
    }
    case "artifact_ingest_completed": {
      const size = typeof d.size_bytes === "number" ? `${(Number(d.size_bytes) / 1024).toFixed(1)} KB` : "";
      const category = _str(d.category);
      return [category, size].filter(Boolean).join(" · ") || "";
    }
    case "artifact_retention_pruned":
    case "artifact_read_completed":
    case "artifact_confinement_violation":
      return _str(d.reason) || _str(d.message) || _str(d.description) || "";
    case "artifact_seal_validation":
      return _str(d.outcome) || _str(d.message) || "";

    // --- Run validity ---
    case "run_validity_scorecard": {
      const score = typeof d.score === "number" ? `Score: ${d.score}` : "";
      const verdict = _str(d.verdict);
      return [score, verdict].filter(Boolean).join(" · ") || "";
    }

    default: {
      // Try meaningful fields in priority order
      for (const key of ["summary", "message", "description", "reason", "goal", "error", "outcome", "detail"]) {
        const val = _str(d[key]);
        if (val) return val.length > 300 ? val.slice(0, 300) + "…" : val;
      }
      return "";
    }
  }
}

/** Extract tool call args as a formatted string for display. */
export function runTimelineToolArgs(event: RunTimelineEvent): string {
  if (event.event_type !== "tool_call_started" && event.event_type !== "tool_call_completed") {
    return "";
  }
  const args = event.data.args;
  if (!args || typeof args !== "object") return "";
  try {
    return JSON.stringify(args, null, 2);
  } catch {
    return "";
  }
}

export function runTimelineToolName(event: RunTimelineEvent): string {
  return _toolName(event.data);
}

export function runTimelinePills(event: RunTimelineEvent): string[] {
  const pills: string[] = [];
  const d = event.data;
  const sub = _subtaskLabel(d);
  const tool = _toolName(d);

  // Subtask label (most useful context)
  if (sub && !event.event_type.startsWith("subtask_")) {
    pills.push(sub);
  }

  if (tool && (event.event_type === "tool_call_started" || event.event_type === "tool_call_completed")) {
    pills.push(tool);
  }

  const sequence = typeof d.sequence === "number" ? d.sequence : undefined;
  if (sequence != null) {
    pills.push(`seq ${sequence}`);
  }

  // Duration
  const elapsed = typeof d.elapsed_ms === "number" ? d.elapsed_ms : (typeof d.duration === "number" ? d.duration * 1000 : 0);
  if (elapsed > 0) {
    pills.push(elapsed >= 1000 ? `${(elapsed / 1000).toFixed(1)} s` : `${Math.round(elapsed)} ms`);
  }

  // File count for multi-file tool calls
  const files = _fileList(d);
  if (files.length > 1) {
    pills.push(`${files.length} files`);
  }

  // Verification tier/confidence
  if (typeof d.tier === "number") {
    pills.push(`tier ${d.tier}`);
  }
  if (typeof d.confidence === "number") {
    pills.push(`${Math.round(Number(d.confidence) * 100)}%`);
  }

  if (event.event_type === "tool_call_completed") {
    pills.push(Boolean(d.success) ? "success" : "failed");
  }

  return pills.slice(0, 4);
}

export function summarizeMessage(message: ConversationMessage): string {
  if (typeof message.content === "string" && message.content.trim()) {
    return message.content.trim();
  }
  if (typeof message.tool_name === "string" && message.tool_name.trim()) {
    return `Tool call: ${message.tool_name}`;
  }
  return "No content";
}

export interface ConversationTurnSeparatorStats {
  tool_count: number;
  tokens: number;
  model: string;
  tokens_per_second: number;
  latency_ms: number;
  total_time_ms: number;
  context_tokens: number;
  context_messages: number;
  omitted_messages: number;
  recall_index_used: boolean;
}

function coerceConversationTurnInt(value: unknown, defaultValue = 0): number {
  const numeric = typeof value === "number"
    ? value
    : typeof value === "string" && value.trim()
      ? Number(value)
      : Number.NaN;
  if (!Number.isFinite(numeric)) {
    return defaultValue;
  }
  return Math.max(0, Math.trunc(numeric));
}

function coerceConversationTurnFloat(value: unknown, defaultValue = 0): number {
  const numeric = typeof value === "number"
    ? value
    : typeof value === "string" && value.trim()
      ? Number(value)
      : Number.NaN;
  if (!Number.isFinite(numeric)) {
    return defaultValue;
  }
  return Math.max(0, numeric);
}

function coerceConversationTurnBool(value: unknown, defaultValue = false): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return defaultValue;
    }
    return value !== 0;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes", "y", "on"].includes(normalized)) {
      return true;
    }
    if (["0", "false", "no", "n", "off"].includes(normalized)) {
      return false;
    }
  }
  return defaultValue;
}

export function normalizeConversationTurnSeparatorPayload(
  payload: Record<string, unknown>,
): ConversationTurnSeparatorStats {
  return {
    tool_count: coerceConversationTurnInt(payload.tool_count, 0),
    tokens: coerceConversationTurnInt(payload.tokens, 0),
    model: typeof payload.model === "string" ? payload.model.trim() : "",
    tokens_per_second: coerceConversationTurnFloat(payload.tokens_per_second, 0),
    latency_ms: coerceConversationTurnInt(payload.latency_ms, 0),
    total_time_ms: coerceConversationTurnInt(payload.total_time_ms, 0),
    context_tokens: coerceConversationTurnInt(payload.context_tokens, 0),
    context_messages: coerceConversationTurnInt(payload.context_messages, 0),
    omitted_messages: coerceConversationTurnInt(payload.omitted_messages, 0),
    recall_index_used: coerceConversationTurnBool(payload.recall_index_used, false),
  };
}

export function formatConversationTurnSeparatorDuration(durationMs: number): string {
  if (durationMs >= 1000) {
    return `${(durationMs / 1000).toFixed(1)}s`;
  }
  return `${durationMs}ms`;
}

export function conversationTurnSeparatorParts(
  stats: ConversationTurnSeparatorStats,
): string[] {
  const parts: string[] = [];
  if (stats.tool_count > 0) {
    const suffix = stats.tool_count === 1 ? "" : "s";
    parts.push(`${stats.tool_count.toLocaleString()} tool${suffix}`);
  }
  parts.push(`${stats.tokens.toLocaleString()} tokens`);
  if (stats.tokens_per_second > 0) {
    parts.push(`${stats.tokens_per_second.toFixed(1)} tok/s`);
  }
  if (stats.latency_ms > 0) {
    parts.push(`${formatConversationTurnSeparatorDuration(stats.latency_ms)} latency`);
  }
  if (stats.total_time_ms > 0) {
    parts.push(`${formatConversationTurnSeparatorDuration(stats.total_time_ms)} total`);
  }
  if (stats.context_tokens > 0) {
    parts.push(`ctx ${stats.context_tokens.toLocaleString()} tok`);
  }
  if (stats.context_messages > 0) {
    parts.push(`${stats.context_messages.toLocaleString()} ctx msg`);
  }
  if (stats.omitted_messages > 0) {
    parts.push(`${stats.omitted_messages.toLocaleString()} archived`);
  }
  if (stats.recall_index_used) {
    parts.push("recall-index");
  }
  if (stats.model) {
    parts.push(stats.model);
  }
  return parts;
}

export function summarizeConversationTurnSeparator(
  payload: Record<string, unknown>,
): string {
  return conversationTurnSeparatorParts(
    normalizeConversationTurnSeparatorPayload(payload),
  ).join(" · ");
}

function summarizeConversationEvent(event: ConversationStreamEvent): string {
  const payload = event.payload;
  if (event.event_type === "user_message") {
    return `You: ${String(payload.text || "").trim() || "Sent a message"}`;
  }
  if (event.event_type === "assistant_text") {
    return String(payload.text || "").trim() || "Assistant replied";
  }
  if (event.event_type === "tool_call_started") {
    return `Started ${String(payload.tool_name || "tool")}`;
  }
  if (event.event_type === "tool_call_completed") {
    const label = String(payload.tool_name || "tool");
    if (label === "ask_user") {
      const questionPayload = payload.question_payload;
      if (
        questionPayload
        && typeof questionPayload === "object"
        && typeof (questionPayload as { question?: unknown }).question === "string"
      ) {
        return `Asked: ${String((questionPayload as { question: string }).question).trim()}`;
      }
    }
    return Boolean(payload.success) ? `${label} completed` : `${label} failed`;
  }
  if (event.event_type === "approval_requested") {
    return `Approval needed for ${String(payload.tool_name || "tool")}`;
  }
  if (event.event_type === "approval_resolved") {
    return `${String(payload.tool_name || "tool")} ${String(payload.decision || "resolved")}`;
  }
  if (event.event_type === "steering_instruction") {
    return String(payload.instruction || "").trim() || "Steering instruction queued";
  }
  if (event.event_type === "turn_separator") {
    return summarizeConversationTurnSeparator(payload);
  }
  if (event.event_type === "turn_interrupted") {
    return String(payload.message || "Conversation turn interrupted");
  }
  if (event.event_type === "content_indicator") {
    const blocks = Array.isArray(payload.content_blocks) ? payload.content_blocks.length : 0;
    return `Attached ${blocks} content block${blocks === 1 ? "" : "s"}`;
  }
  return event.event_type.replace(/_/g, " ");
}

export function conversationApprovalPreview(approval: ConversationApproval | null): string {
  if (!approval) {
    return "";
  }
  const riskInfo = approval.risk_info;
  const impactPreview =
    riskInfo && typeof riskInfo.impact_preview === "string"
      ? riskInfo.impact_preview.trim()
      : "";
  if (impactPreview) {
    return impactPreview;
  }
  for (const value of Object.values(approval.args || {})) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function conversationEventPreviewFromPayload(payload: Record<string, unknown>): string {
  const questionPayload = normalizeConversationPrompt(payload.question_payload);
  if (questionPayload) {
    return questionPayload.question;
  }
  const directQuestion = typeof payload.question === "string" ? payload.question.trim() : "";
  if (directQuestion) {
    return directQuestion;
  }
  const fromArgs = firstMeaningfulString(payload.args);
  if (fromArgs) {
    return fromArgs;
  }
  const fromOutput = firstMeaningfulString(payload.output);
  if (fromOutput) {
    return fromOutput;
  }
  const fromError = firstMeaningfulString(payload.error);
  if (fromError) {
    return fromError;
  }
  return "";
}

export function conversationEventTitle(event: ConversationStreamEvent): string {
  const payload = event.payload;
  if (event.event_type === "user_message") {
    return "You";
  }
  if (event.event_type === "assistant_text") {
    return "Assistant";
  }
  if (event.event_type === "tool_call_started") {
    return `Started ${String(payload.tool_name || "tool")}`;
  }
  if (event.event_type === "tool_call_completed") {
    const toolName = String(payload.tool_name || "tool");
    if (toolName === "ask_user") {
      return "Asked for guidance";
    }
    return Boolean(payload.success) ? `${toolName} completed` : `${toolName} failed`;
  }
  if (event.event_type === "approval_requested") {
    return `Approval needed for ${String(payload.tool_name || "tool")}`;
  }
  if (event.event_type === "approval_resolved") {
    const decision = String(payload.decision || "resolved").replace(/_/g, " ");
    return `${String(payload.tool_name || "tool")} ${decision}`;
  }
  if (event.event_type === "steering_instruction") {
    return "Instruction queued";
  }
  if (event.event_type === "turn_separator") {
    return "Turn completed";
  }
  if (event.event_type === "turn_interrupted") {
    return "Conversation interrupted";
  }
  if (event.event_type === "content_indicator") {
    return "Attached content";
  }
  return event.event_type.replace(/_/g, " ");
}

export function conversationEventDetail(event: ConversationStreamEvent): string {
  const payload = event.payload;
  if (event.event_type === "user_message" || event.event_type === "assistant_text") {
    return String(payload.text || "").trim() || summarizeConversationEvent(event);
  }
  if (event.event_type === "tool_call_started" || event.event_type === "tool_call_completed") {
    const preview = conversationEventPreviewFromPayload(payload);
    return preview || summarizeConversationEvent(event);
  }
  if (event.event_type === "approval_requested") {
    const preview = conversationApprovalPreview(payload as unknown as ConversationApproval);
    return preview || summarizeConversationEvent(event);
  }
  if (event.event_type === "approval_resolved") {
    const preview = conversationApprovalPreview(payload as unknown as ConversationApproval);
    return preview || summarizeConversationEvent(event);
  }
  if (event.event_type === "steering_instruction") {
    return String(payload.instruction || "").trim() || summarizeConversationEvent(event);
  }
  if (event.event_type === "turn_separator") {
    return summarizeConversationEvent(event);
  }
  if (event.event_type === "turn_interrupted") {
    return String(payload.message || "").trim() || summarizeConversationEvent(event);
  }
  if (event.event_type === "content_indicator") {
    return summarizeConversationEvent(event);
  }
  return summarizeConversationEvent(event);
}

export function conversationEventPills(event: ConversationStreamEvent): string[] {
  const payload = event.payload;
  const pills: string[] = [];
  if (typeof payload.tool_name === "string" && payload.tool_name.trim()) {
    pills.push(payload.tool_name.trim());
  }
  if (event.event_type === "tool_call_completed") {
    pills.push(Boolean(payload.success) ? "success" : "failed");
    const questionPayload = normalizeConversationPrompt(payload.question_payload);
    if (questionPayload?.options.length) {
      pills.push(`${questionPayload.options.length} option${questionPayload.options.length === 1 ? "" : "s"}`);
    }
  }
  if (event.event_type === "approval_resolved") {
    const decision = String(payload.decision || "").trim();
    if (decision) {
      pills.push(decision.replace(/_/g, " "));
    }
  }
  if (event.event_type === "approval_requested") {
    const riskInfo =
      payload.risk_info && typeof payload.risk_info === "object"
        ? (payload.risk_info as Record<string, unknown>)
        : null;
    const riskLevel = typeof riskInfo?.risk_level === "string" ? riskInfo.risk_level.trim() : "";
    if (riskLevel) {
      pills.push(riskLevel);
    }
  }
  if (event.event_type === "steering_instruction") {
    const count = Number(payload.pending_inject_count || 0);
    if (count > 0) {
      pills.push(`${count} queued`);
    }
  }
  if (event.event_type === "turn_separator") {
    const stats = normalizeConversationTurnSeparatorPayload(payload);
    pills.push(`${stats.tokens.toLocaleString()} tokens`);
    if (stats.tool_count > 0) {
      const suffix = stats.tool_count === 1 ? "" : "s";
      pills.push(`${stats.tool_count.toLocaleString()} tool${suffix}`);
    }
  }
  if (event.event_type === "content_indicator") {
    const blocks = Array.isArray(payload.content_blocks) ? payload.content_blocks.length : 0;
    pills.push(`${blocks} block${blocks === 1 ? "" : "s"}`);
  }
  return pills;
}

export function normalizeConversationPrompt(payload: unknown): ConversationPrompt | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  const record = payload as Record<string, unknown>;
  const question = String(record.question || "").trim();
  if (!question) {
    return null;
  }
  return {
    question,
    question_type: String(record.question_type || "free_text"),
    options: Array.isArray(record.options)
      ? record.options
          .filter((option): option is Record<string, unknown> => Boolean(option) && typeof option === "object")
          .map((option) => ({
            id: String(option.id || option.label || ""),
            label: String(option.label || option.id || ""),
            description: String(option.description || ""),
          }))
          .filter((option) => option.label.trim())
      : [],
    allow_custom_response: Boolean(record.allow_custom_response),
    min_selections: Number(record.min_selections || 0),
    max_selections: Number(record.max_selections || 0),
    context_note: String(record.context_note || ""),
    urgency: String(record.urgency || "normal"),
    default_option_id: String(record.default_option_id || ""),
    tool_call_id: String(record.tool_call_id || ""),
  };
}

export function notificationSummary(event: NotificationEvent): string {
  return event.summary || startCaseEventName(event.event_type);
}

export function matchesWorkspaceSearch(query: string, ...values: Array<unknown>): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    return true;
  }
  return values.some((value) => firstMeaningfulString(value).toLowerCase().includes(normalized));
}
