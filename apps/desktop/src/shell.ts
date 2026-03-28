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

export const BASE_COMMAND_OPTIONS: CommandOption[] = [
  {
    id: "new-conversation",
    label: "New conversation",
    command: "new conversation",
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
    label: "Latest conversation",
    command: "latest conversation",
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
  "new-conversation",
  "new-run",
  "latest-conversation",
  "latest-run",
]);

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
  const commandSearchTerm = normalizedCommandDraft.startsWith("search ")
    ? trimmedCommandDraft.slice(7).trim()
    : trimmedCommandDraft;

  return [
    ...BASE_COMMAND_OPTIONS.filter((option) => {
      if (!normalizedCommandDraft) {
        return true;
      }
      return matchesCommandOption(option, normalizedCommandDraft);
    }),
    ...(commandSearchTerm
      ? [{
          id: `search-${commandSearchTerm.toLowerCase()}`,
          label: `Search workspace for "${commandSearchTerm}"`,
          command: `search ${commandSearchTerm}`,
          description: "Run the workspace-wide search from the command palette.",
          keywords: ["search", "workspace", commandSearchTerm.toLowerCase()],
        } satisfies CommandOption]
      : []),
  ].filter((option, index, rows) =>
    rows.findIndex((candidate) => candidate.command === option.command) === index,
  );
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
  return options.slice(0, 6).map((option) => ({
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
  return [
    ...results.conversations,
    ...results.runs,
    ...results.approvals,
    ...results.artifacts,
    ...results.files,
  ]
    .slice(0, 6)
    .map((item) => ({
      id: `result-${item.kind}-${item.item_id || item.title}`,
      title: item.title,
      description: item.subtitle || item.snippet || item.kind,
      keyword: item.kind.replace(/_/g, " "),
      kind: "result" as const,
      badge: "Result",
      result: item,
    }));
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
): Array<{ label: string; entries: PaletteEntry[] }> {
  return commandDraft.trim()
    ? [
        { label: "Actions", entries: commandEntries.slice(0, 4) },
        { label: "Results", entries: resultEntries.slice(0, 4) },
      ]
    : [
        { label: "Recent", entries: recentEntries.slice(0, 4) },
        {
          label: "Pinned",
          entries: pinnedEntries
            .filter((entry) => !recentEntries.some((recent) => recent.id === entry.id))
            .slice(0, 4),
        },
      ];
}
