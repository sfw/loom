const RUN_STATUS_ALIASES: Record<string, string> = {
  canceled: "cancelled",
  cancel_requested: "cancelled",
  complete: "completed",
  done: "completed",
  error: "failed",
  failure: "failed",
  in_progress: "running",
  succeeded: "completed",
  success: "completed",
};

const ACTIVE_RUN_STATUSES = new Set(["executing", "planning", "running"]);
const TERMINAL_RUN_STATUSES = new Set(["completed", "failed", "cancelled"]);

export function normalizeRunStatus(status: string | null | undefined): string {
  const normalized = String(status || "").trim().toLowerCase();
  if (!normalized) {
    return "";
  }
  return RUN_STATUS_ALIASES[normalized] || normalized;
}

export function displayRunStatus(status: string | null | undefined): string {
  const normalized = normalizeRunStatus(status);
  if (normalized) {
    return normalized;
  }
  return String(status || "").trim();
}

export function isRunActiveStatus(status: string | null | undefined): boolean {
  return ACTIVE_RUN_STATUSES.has(normalizeRunStatus(status));
}

export function isRunTerminalStatus(status: string | null | undefined): boolean {
  return TERMINAL_RUN_STATUSES.has(normalizeRunStatus(status));
}

export function canPauseRunStatus(status: string | null | undefined): boolean {
  return ACTIVE_RUN_STATUSES.has(normalizeRunStatus(status));
}

export function canResumeRunStatus(status: string | null | undefined): boolean {
  return normalizeRunStatus(status) === "paused";
}

export function canMessageRunStatus(status: string | null | undefined): boolean {
  return ACTIVE_RUN_STATUSES.has(normalizeRunStatus(status));
}
