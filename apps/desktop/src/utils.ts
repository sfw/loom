import type { ReactNode } from "react";
import { createElement } from "react";
import type { WorkspaceFileEntry, WorkspaceSummary, ApprovalFeedItem } from "./api";

export function formatDate(value: string): string {
  if (!value) return "Not yet";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

export function formatValue(value: unknown): string {
  if (typeof value === "string") return value || "empty";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value == null) return "unset";
  try { return JSON.stringify(value, null, 2); } catch { return "complex value"; }
}

export function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function highlightText(text: string, query: string): ReactNode {
  const source = String(text || "");
  const needle = query.trim();
  if (!source || !needle) return source;
  const lowerSource = source.toLowerCase();
  const lowerNeedle = needle.toLowerCase();
  const parts: ReactNode[] = [];
  let cursor = 0;
  let matchIndex = lowerSource.indexOf(lowerNeedle);
  while (matchIndex !== -1) {
    if (matchIndex > cursor) parts.push(source.slice(cursor, matchIndex));
    const matchEnd = matchIndex + needle.length;
    parts.push(
      createElement("mark", {
        className: "match-highlight",
        key: `${matchIndex}-${matchEnd}-${parts.length}`,
      }, source.slice(matchIndex, matchEnd))
    );
    cursor = matchEnd;
    matchIndex = lowerSource.indexOf(lowerNeedle, cursor);
  }
  if (cursor < source.length) parts.push(source.slice(cursor));
  return parts.length > 0 ? parts : source;
}

export function defaultWorkspaceName(path: string): string {
  const normalized = path.trim().replace(/[\\/]+$/, "");
  if (!normalized) return "";
  const pieces = normalized.split(/[\\/]/).filter(Boolean);
  return pieces[pieces.length - 1] || normalized;
}

export function workspaceTagsFromMetadata(metadata: Record<string, unknown> | null | undefined): string[] {
  const raw = metadata?.tags;
  if (!Array.isArray(raw)) return [];
  return raw.map((tag) => String(tag || "").trim()).filter(Boolean);
}

export function workspaceNoteFromMetadata(metadata: Record<string, unknown> | null | undefined): string {
  const raw = metadata?.note;
  return typeof raw === "string" ? raw.trim() : "";
}

export function parseWorkspaceTags(raw: string): string[] {
  const seen = new Set<string>();
  const tags: string[] = [];
  for (const part of raw.split(",")) {
    const cleaned = part.trim();
    if (!cleaned) continue;
    const normalized = cleaned.toLowerCase();
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    tags.push(cleaned);
  }
  return tags;
}

export function parentDirectory(path: string): string {
  const normalized = path.replace(/[\\/]+/g, "/").replace(/\/+$/, "");
  const index = normalized.lastIndexOf("/");
  return index >= 0 ? normalized.slice(0, index) : "";
}

export function fileDepth(path: string): number {
  const normalized = path.replace(/[\\/]+/g, "/").replace(/^\/+|\/+$/g, "");
  if (!normalized) return 0;
  return normalized.split("/").length - 1;
}

export function fileSortKey(entry: WorkspaceFileEntry): string {
  return `${entry.is_dir ? "0" : "1"}:${entry.name.toLowerCase()}`;
}

export function ancestorDirectories(path: string): string[] {
  const normalized = path.replace(/[\\/]+/g, "/").replace(/^\/+|\/+$/g, "");
  if (!normalized || !normalized.includes("/")) return [];
  const parts = normalized.split("/");
  const items: string[] = [];
  for (let index = 1; index < parts.length; index += 1) {
    items.push(parts.slice(0, index).join("/"));
  }
  return items;
}

export function isTypingIntoField(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tagName = target.tagName.toUpperCase();
  if (tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT") return true;
  return Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

export function adjacentItemId<T extends { id: string }>(items: T[], currentId: string, delta: number): string {
  if (items.length === 0) return "";
  const currentIndex = items.findIndex((item) => item.id === currentId);
  const startIndex = currentIndex >= 0 ? currentIndex : 0;
  const nextIndex = Math.max(0, Math.min(items.length - 1, startIndex + delta));
  return items[nextIndex]?.id || items[0].id;
}

export function joinWorkspacePath(parentPath: string, folderName: string): string {
  const cleanParent = parentPath.trim().replace(/[\\/]+$/, "");
  const cleanFolder = folderName.trim().replace(/^[\\/]+/, "");
  if (!cleanParent) return cleanFolder;
  if (!cleanFolder) return cleanParent;
  return `${cleanParent}/${cleanFolder}`;
}

export function mergeWorkspaceSummary(current: WorkspaceSummary[], summary: WorkspaceSummary): WorkspaceSummary[] {
  const next = current.map((workspace) => workspace.id === summary.id ? summary : workspace);
  if (next.some((workspace) => workspace.id === summary.id)) return next;
  return [...next, summary];
}

export function approvalQuestionOptions(item: ApprovalFeedItem): Array<{ id: string; label: string }> {
  const raw = item.request_payload.options;
  if (!Array.isArray(raw)) return [];
  return raw
    .map((option) => {
      if (!option || typeof option !== "object") return null;
      return {
        id: String((option as { id?: unknown }).id || "").trim(),
        label: String((option as { label?: unknown }).label || "").trim(),
      };
    })
    .filter((option): option is { id: string; label: string } => Boolean(option?.id && option.label));
}

export function approvalQuestionType(item: ApprovalFeedItem): string {
  return typeof item.request_payload.question_type === "string"
    ? item.request_payload.question_type.trim() : "";
}

export function approvalQuestionContext(item: ApprovalFeedItem): string {
  return typeof item.request_payload.context_note === "string"
    ? item.request_payload.context_note.trim() : "";
}

export function scrollToSection(ref: { current: HTMLElement | null }) {
  ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export function isTransientRequestError(error: unknown): boolean {
  const message = String(error instanceof Error ? error.message : error || "").trim().toLowerCase();
  if (!message) return false;
  return [
    "load failed",
    "failed to fetch",
    "networkerror",
    "network request failed",
    "the network connection was lost",
    "request timed out",
    "signal timed out",
    "timed out",
    "timeout",
    "aborterror",
    "aborted",
  ].some((needle) => message.includes(needle));
}

export type ViewTab = "overview" | "threads" | "runs" | "files" | "integrations" | "settings";
