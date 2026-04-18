import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import CommandPalette, { type CommandPaletteSection } from "./CommandPalette";
import type { PaletteEntry } from "./shell";

function makeEntry(overrides: Partial<PaletteEntry> = {}): PaletteEntry {
  return {
    id: "command-new-run",
    title: "New run",
    description: "Focus the run launcher for the selected workspace.",
    keyword: "new run",
    kind: "command",
    badge: "Pinned",
    ...overrides,
  };
}

function renderHighlight(text: string): string {
  return text;
}

describe("CommandPalette", () => {
  it("does not render when closed with an empty draft", () => {
    const { container } = render(
      <CommandPalette
        activeIndex={0}
        commandDraft=""
        onSelect={() => {}}
        open={false}
        paletteEntries={[]}
        paletteSections={[]}
        renderHighlight={renderHighlight}
        searching={false}
      />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("renders empty-state guidance for an open palette with no results", () => {
    render(
      <CommandPalette
        activeIndex={0}
        commandDraft=""
        onSelect={() => {}}
        open
        paletteEntries={[]}
        paletteSections={[]}
        renderHighlight={renderHighlight}
        searching={false}
      />,
    );

    expect(
      screen.getByText("Type to search workspaces, threads, runs, and files."),
    ).toBeInTheDocument();
  });

  it("renders sections, active state, and selection callbacks", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const actionEntry = makeEntry();
    const resultEntry = makeEntry({
      id: "result-artifact-auth-report",
      title: "auth-report.md",
      description: "Authentication findings",
      keyword: "artifact",
      kind: "result",
      badge: "Result",
    });
    const paletteEntries = [actionEntry, resultEntry];
    const paletteSections: CommandPaletteSection[] = [
      { label: "Actions", entries: [actionEntry] },
      { label: "Runs", entries: [resultEntry] },
    ];

    render(
      <CommandPalette
        activeIndex={1}
        commandDraft="auth"
        onSelect={onSelect}
        open
        paletteEntries={paletteEntries}
        paletteSections={paletteSections}
        renderHighlight={renderHighlight}
        searching={true}
      />,
    );

    expect(screen.getByText("Searching...")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();
    expect(screen.getByText("Runs")).toBeInTheDocument();

    // The second button (index 1) should have the active highlight class
    const buttons = screen.getAllByRole("button");
    expect(buttons.length).toBe(2);

    await user.click(screen.getByText("auth-report.md"));
    expect(onSelect).toHaveBeenCalledWith(resultEntry);
  });
});
