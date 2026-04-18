import type { WorkspaceSearchItem, WorkspaceSearchResponse } from "./api";

export type CommandOption = {
  id: string;
  label: string;
  command: string;
  description: string;
  keywords: string[];
};

export type PaletteEntry = {
  id: string;
  title: string;
  description: string;
  keyword: string;
  kind: "command" | "result";
  badge?: string;
  command?: CommandOption;
  result?: WorkspaceSearchItem;
};

export const RECENT_PALETTE_STORAGE_KEY = "loom.desktop.palette.recents.v1";

  const RESULT_GROUPS: Array<{
    key: keyof Pick<
      WorkspaceSearchResponse,
    "workspaces" | "conversations" | "runs" | "approvals" | "artifacts" | "files" | "processes" | "accounts" | "mcp_servers" | "tools"
  >;
  label: string;
}> = [
  { key: "workspaces", label: "Workspaces" },
  { key: "conversations", label: "Threads" },
  { key: "runs", label: "Runs" },
  { key: "approvals", label: "Approvals" },
  { key: "artifacts", label: "Artifacts" },
  { key: "files", label: "Files" },
  { key: "processes", label: "Processes" },
  { key: "accounts", label: "Accounts" },
  { key: "mcp_servers", label: "MCP Servers" },
  { key: "tools", label: "Tools" },
];

export const BASE_COMMAND_OPTIONS: CommandOption[] = [
  {
    id: "open-integrations",
    label: "Open integrations",
    command: "integrations",
    description: "Open MCP servers, accounts, and trust state for this workspace.",
    keywords: ["integrations", "mcp", "auth", "accounts", "servers"],
  },
  {
    id: "add-local-server",
    label: "Add local server",
    command: "add local server",
    description: "Jump to integrations and start adding a local MCP server.",
    keywords: ["integrations", "mcp", "server", "local", "stdio", "add"],
  },
  {
    id: "add-remote-server",
    label: "Add remote server",
    command: "add remote server",
    description: "Jump to integrations and start adding a remote MCP server.",
    keywords: ["integrations", "mcp", "server", "remote", "oauth", "add"],
  },
  {
    id: "connect-account",
    label: "Connect account",
    command: "connect account",
    description: "Open integrations to connect or switch an account for a server.",
    keywords: ["integrations", "auth", "account", "oauth", "connect"],
  },
  {
    id: "broken-integrations",
    label: "Show broken integrations",
    command: "broken integrations",
    description: "Open integrations and review items that need attention.",
    keywords: ["integrations", "broken", "issues", "repair", "attention"],
  },
  {
    id: "new-conversation",
    label: "New thread",
    command: "new thread",
    description: "Focus the thread composer for the selected workspace.",
    keywords: ["conversation", "thread", "chat", "new"],
  },
  {
    id: "new-run",
    label: "New run",
    command: "new run",
    description: "Focus the run launcher for the selected workspace.",
    keywords: ["run", "task", "launch", "new"],
  },
  {
    id: "starter-thread",
    label: "Starter thread",
    command: "starter thread",
    description: "Load a useful first conversation prompt.",
    keywords: ["starter", "thread", "conversation", "prompt"],
  },
  {
    id: "starter-run",
    label: "Starter run",
    command: "starter run",
    description: "Load a useful first run goal.",
    keywords: ["starter", "run", "goal"],
  },
  {
    id: "latest-conversation",
    label: "Latest thread",
    command: "latest thread",
    description: "Open the most recent thread in this workspace.",
    keywords: ["latest", "recent", "conversation", "thread"],
  },
  {
    id: "latest-run",
    label: "Latest run",
    command: "latest run",
    description: "Open the most recent run in this workspace.",
    keywords: ["latest", "recent", "run", "task"],
  },
  {
    id: "clear-search",
    label: "Clear workspace search",
    command: "clear search",
    description: "Reset the top-bar workspace search query.",
    keywords: ["clear", "search", "reset"],
  },
];

const PINNED_COMMAND_IDS = new Set([
  "open-integrations",
  "new-conversation",
  "new-run",
  "latest-conversation",
  "latest-run",
]);

function paletteResultEntryId(item: WorkspaceSearchItem): string {
  return [
    "result",
    item.kind,
    item.workspace_id || "global",
    item.item_id || item.title,
  ].join("-");
}

export function isPaletteEntry(value: unknown): value is PaletteEntry {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as Record<string, unknown>;
  const kind = String(record.kind || "");
  return Boolean(
    (kind === "command" || kind === "result")
    && typeof record.id === "string"
    && typeof record.title === "string"
    && typeof record.description === "string"
    && typeof record.keyword === "string",
  );
}

function matchesCommandOption(option: CommandOption, query: string): boolean {
  const haystack = [
    option.label,
    option.command,
    option.description,
    option.keywords.join(" "),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

export function buildCommandOptions(commandDraft: string): CommandOption[] {
  const trimmedCommandDraft = commandDraft.trim();
  const normalizedCommandDraft = trimmedCommandDraft.toLowerCase();

  return BASE_COMMAND_OPTIONS.filter((option) => {
    if (!normalizedCommandDraft) {
      return true;
    }
    return matchesCommandOption(option, normalizedCommandDraft);
  });
}

export function buildPinnedPaletteEntries(): PaletteEntry[] {
  return BASE_COMMAND_OPTIONS
    .filter((option) => PINNED_COMMAND_IDS.has(option.id))
    .map((option) => ({
      id: `command-${option.id}`,
      title: option.label,
      description: option.description,
      keyword: option.command,
      kind: "command" as const,
      badge: "Pinned",
      command: option,
    }));
}

export function buildCommandPaletteEntries(options: CommandOption[]): PaletteEntry[] {
  return options.slice(0, 4).map((option) => ({
    id: `command-${option.id}`,
    title: option.label,
    description: option.description,
    keyword: option.command,
    kind: "command" as const,
    command: option,
  }));
}

export function buildResultPaletteEntries(
  results: WorkspaceSearchResponse | null,
): PaletteEntry[] {
  if (!results) {
    return [];
  }
  return RESULT_GROUPS.flatMap(({ key }) => results[key] || [])
    .map((item) => {
      const details = item.kind === "workspace"
        ? [item.workspace_path || item.subtitle, item.snippet]
        : [
            item.workspace_display_name,
            item.subtitle,
            item.snippet,
          ];
      return {
        id: paletteResultEntryId(item),
        title: item.title,
        description: details.filter(Boolean).join(" · "),
        keyword: item.badges[0] || item.kind.replace(/_/g, " "),
        kind: "result" as const,
        badge: item.kind === "workspace" ? "Workspace" : undefined,
        result: item,
      };
    });
}

export function rememberPaletteEntry(
  current: PaletteEntry[],
  entry: PaletteEntry,
): PaletteEntry[] {
  const next = [
    { ...entry, badge: "Recent" },
    ...current.filter((item) => item.id !== entry.id),
  ];
  return next.slice(0, 6);
}

export function buildPaletteEntries(
  commandDraft: string,
  recentEntries: PaletteEntry[],
  commandEntries: PaletteEntry[],
  resultEntries: PaletteEntry[],
  pinnedEntries: PaletteEntry[],
): PaletteEntry[] {
  return commandDraft.trim()
    ? [...commandEntries, ...resultEntries]
    : [...recentEntries, ...pinnedEntries].filter(
        (entry, index, rows) => rows.findIndex((candidate) => candidate.id === entry.id) === index,
      );
}

export function buildPaletteSections(
  commandDraft: string,
  recentEntries: PaletteEntry[],
  commandEntries: PaletteEntry[],
  resultEntries: PaletteEntry[],
  pinnedEntries: PaletteEntry[],
  results?: WorkspaceSearchResponse | null,
): Array<{ label: string; entries: PaletteEntry[] }> {
  if (!commandDraft.trim()) {
    return [
      { label: "Recent", entries: recentEntries.slice(0, 4) },
      {
        label: "Pinned",
        entries: pinnedEntries
          .filter((entry) => !recentEntries.some((recent) => recent.id === entry.id))
          .slice(0, 4),
      },
    ];
  }

  const sections: Array<{ label: string; entries: PaletteEntry[] }> = [];
  if (commandEntries.length > 0) {
    sections.push({ label: "Actions", entries: commandEntries.slice(0, 3) });
  }
  if (!results) {
    return sections;
  }
  const resultEntryMap = new Map(resultEntries.map((entry) => [entry.id, entry]));
  for (const { key, label } of RESULT_GROUPS) {
    const rows = (results[key] || []).map((item) =>
      resultEntryMap.get(paletteResultEntryId(item)),
    ).filter((entry): entry is PaletteEntry => Boolean(entry));
    if (rows.length > 0) {
      sections.push({ label, entries: rows.slice(0, 6) });
    }
  }
  return sections;
}
