import { describe, expect, it } from "vitest";

import type { WorkspaceSearchResponse } from "./api";
import {
  buildCommandOptions,
  buildCommandPaletteEntries,
  buildPaletteEntries,
  buildPaletteSections,
  buildPinnedPaletteEntries,
  buildResultPaletteEntries,
  rememberPaletteEntry,
  type PaletteEntry,
} from "./shell";

const workspaceSearchResponse: WorkspaceSearchResponse = {
  workspace: {
    id: "ws-1",
    canonical_path: "/tmp/workspace",
    display_name: "Workspace One",
    workspace_type: "local",
    is_archived: false,
    sort_order: 0,
    last_opened_at: "2026-03-24T10:00:00",
    created_at: "2026-03-24T10:00:00",
    updated_at: "2026-03-24T10:00:00",
    metadata: {},
    exists_on_disk: true,
    conversation_count: 1,
    run_count: 1,
    active_run_count: 0,
    last_activity_at: "2026-03-24T10:00:00",
  },
  query: "auth",
  total_results: 2,
  conversations: [
    {
      kind: "conversation",
      item_id: "conv-1",
      title: "Auth thread",
      subtitle: "gpt-test",
      snippet: "Authentication findings",
      badges: ["conversation"],
      conversation_id: "conv-1",
      run_id: "",
      approval_item_id: "",
      path: "",
      metadata: {},
    },
  ],
  runs: [
    {
      kind: "run",
      item_id: "run-1",
      title: "Auth run",
      subtitle: "running",
      snippet: "Investigating auth failures",
      badges: ["run"],
      conversation_id: "",
      run_id: "run-1",
      approval_item_id: "",
      path: "",
      metadata: {},
    },
  ],
  approvals: [],
  artifacts: [],
  files: [],
  processes: [],
  mcp_servers: [],
  tools: [],
};

function sampleRecentEntry(): PaletteEntry {
  return {
    id: "command-new-run",
    title: "New run",
    description: "Focus the run launcher for the selected workspace.",
    keyword: "new run",
    kind: "command",
    badge: "Recent",
    command: buildCommandOptions("new run").find((option) => option.id === "new-run"),
  };
}

describe("shell command palette helpers", () => {
  it("builds base commands plus a dynamic search action", () => {
    const baseOptions = buildCommandOptions("");
    const options = buildCommandOptions("auth");

    expect(baseOptions.some((option) => option.id === "new-run")).toBe(true);
    expect(options.some((option) => option.id === "new-run")).toBe(false);
    expect(options).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          command: "search auth",
          label: 'Search workspace for "auth"',
        }),
      ]),
    );
  });

  it("builds pinned command entries", () => {
    const entries = buildPinnedPaletteEntries();

    expect(entries.length).toBeGreaterThan(0);
    expect(entries.every((entry) => entry.badge === "Pinned")).toBe(true);
    expect(entries.map((entry) => entry.id)).toContain("command-new-run");
  });

  it("remembers recent entries without duplicates and keeps newest first", () => {
    const starter = sampleRecentEntry();
    const latestRun: PaletteEntry = {
      id: "command-latest-run",
      title: "Latest run",
      description: "Open the most recent run in this workspace.",
      keyword: "latest run",
      kind: "command",
      command: buildCommandOptions("latest run").find((option) => option.id === "latest-run"),
    };

    const next = rememberPaletteEntry([starter], latestRun);
    const deduped = rememberPaletteEntry(next, starter);

    expect(next[0]?.id).toBe("command-latest-run");
    expect(deduped[0]?.id).toBe("command-new-run");
    expect(deduped.filter((entry) => entry.id === "command-new-run")).toHaveLength(1);
    expect(deduped[0]?.badge).toBe("Recent");
  });

  it("turns workspace search results into palette result entries", () => {
    const entries = buildResultPaletteEntries(workspaceSearchResponse);

    expect(entries).toHaveLength(2);
    expect(entries[0]).toEqual(
      expect.objectContaining({
        kind: "result",
        badge: "Result",
      }),
    );
    expect(entries.map((entry) => entry.title)).toEqual(["Auth thread", "Auth run"]);
  });

  it("shows recent and pinned sections when the command draft is empty", () => {
    const recentEntries = [sampleRecentEntry()];
    const pinnedEntries = buildPinnedPaletteEntries();
    const commandEntries = buildCommandPaletteEntries(buildCommandOptions(""));
    const resultEntries = buildResultPaletteEntries(null);

    const entries = buildPaletteEntries(
      "",
      recentEntries,
      commandEntries,
      resultEntries,
      pinnedEntries,
    );
    const sections = buildPaletteSections(
      "",
      recentEntries,
      commandEntries,
      resultEntries,
      pinnedEntries,
    );

    expect(entries[0]?.id).toBe("command-new-run");
    expect(sections.map((section) => section.label)).toEqual(["Recent", "Pinned"]);
    expect(sections[1]?.entries.some((entry) => entry.id === "command-new-run")).toBe(false);
  });

  it("shows action and result sections when the command draft is populated", () => {
    const commandEntries = buildCommandPaletteEntries(buildCommandOptions("auth"));
    const resultEntries = buildResultPaletteEntries(workspaceSearchResponse);
    const pinnedEntries = buildPinnedPaletteEntries();

    const entries = buildPaletteEntries(
      "auth",
      [],
      commandEntries,
      resultEntries,
      pinnedEntries,
    );
    const sections = buildPaletteSections(
      "auth",
      [],
      commandEntries,
      resultEntries,
      pinnedEntries,
    );

    expect(entries.some((entry) => entry.kind === "command")).toBe(true);
    expect(entries.some((entry) => entry.kind === "result")).toBe(true);
    expect(sections.map((section) => section.label)).toEqual(["Actions", "Results"]);
  });
});
