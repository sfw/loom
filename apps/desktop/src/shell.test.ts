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
  workspace: null,
  query: "auth",
  total_results: 4,
  workspaces: [
    {
      kind: "workspace",
      item_id: "ws-1",
      title: "Workspace One",
      subtitle: "/tmp/workspace",
      snippet: "Primary workspace",
      badges: ["1 thread", "1 run"],
      workspace_id: "ws-1",
      workspace_display_name: "Workspace One",
      workspace_path: "/tmp/workspace",
      conversation_id: "",
      run_id: "",
      approval_item_id: "",
      path: "",
      metadata: {},
    },
  ],
  conversations: [
    {
      kind: "conversation",
      item_id: "conv-1",
      title: "Auth thread",
      subtitle: "gpt-test",
      snippet: "Authentication findings",
      badges: ["conversation"],
      workspace_id: "ws-1",
      workspace_display_name: "Workspace One",
      workspace_path: "/tmp/workspace",
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
      workspace_id: "ws-2",
      workspace_display_name: "Workspace Two",
      workspace_path: "/tmp/workspace-two",
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
  accounts: [
    {
      kind: "account",
      item_id: "auth_search_profile",
      title: "Auth Search Account",
      subtitle: "notion · oauth2_pkce",
      snippet: "Linked to notion · Connected",
      badges: ["ready", "linked:notion"],
      workspace_id: "ws-1",
      workspace_display_name: "Workspace One",
      workspace_path: "/tmp/workspace",
      conversation_id: "",
      run_id: "",
      approval_item_id: "",
      path: "",
      metadata: {},
    },
  ],
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
    const integrationOptions = buildCommandOptions("integr");

    expect(baseOptions.some((option) => option.id === "new-run")).toBe(true);
    expect(baseOptions.some((option) => option.id === "open-integrations")).toBe(true);
    expect(options.some((option) => option.id === "new-run")).toBe(false);
    expect(options.map((option) => option.id)).toEqual([
      "open-integrations",
      "add-remote-server",
      "connect-account",
    ]);
    expect(integrationOptions.map((option) => option.id)).toContain("open-integrations");
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

    expect(entries).toHaveLength(4);
    expect(entries[0]).toEqual(
      expect.objectContaining({
        kind: "result",
        badge: "Workspace",
      }),
    );
    expect(entries.map((entry) => entry.title)).toEqual([
      "Workspace One",
      "Auth thread",
      "Auth run",
      "Auth Search Account",
    ]);
    expect(entries[1]?.description).toContain("Workspace One");
    expect(entries[2]?.description).toContain("Workspace Two");
    expect(entries[3]?.description).toContain("Workspace One");
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

  it("shows grouped result sections when the command draft is populated", () => {
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
      workspaceSearchResponse,
    );

    expect(entries.some((entry) => entry.kind === "result")).toBe(true);
    expect(sections.map((section) => section.label)).toEqual(["Actions", "Workspaces", "Threads", "Runs", "Accounts"]);
  });
});
