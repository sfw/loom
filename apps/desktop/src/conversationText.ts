const TOOL_CALL_CONTEXT_PLACEHOLDERS = new Set([
  "tool call context omitted.",
  "tool call required to continue.",
]);

export function stripConversationToolCallPlaceholders(text: string): string {
  const normalized = String(text || "").replace(/\r\n?/g, "\n");
  if (normalized.length === 0) {
    return "";
  }
  return normalized
    .split("\n")
    .filter((line) => !TOOL_CALL_CONTEXT_PLACEHOLDERS.has(line.trim().toLowerCase()))
    .join("\n");
}
