import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import type { PaletteEntry } from "./shell";

export type CommandPaletteSection = {
  label: string;
  entries: PaletteEntry[];
};

type CommandPaletteProps = {
  open: boolean;
  commandDraft: string;
  searching: boolean;
  activeIndex: number;
  paletteEntries: PaletteEntry[];
  paletteSections: CommandPaletteSection[];
  onSelect: (entry: PaletteEntry) => void;
  renderHighlight: (text: string, query: string) => ReactNode;
};

export default function CommandPalette({
  open,
  commandDraft,
  searching,
  activeIndex,
  paletteEntries,
  paletteSections,
  onSelect,
  renderHighlight,
}: CommandPaletteProps) {
  if (!open && !commandDraft.trim()) {
    return null;
  }

  return (
    <div className="flex-1 overflow-y-auto py-2">
      {searching && (
        <p className="px-4 py-2 text-xs text-zinc-500">Searching...</p>
      )}

      {paletteEntries.length === 0 ? (
        <p className="px-4 py-6 text-center text-sm text-zinc-600">
          {commandDraft.trim()
            ? "No matching commands or results."
            : "Type to search or use a command."}
        </p>
      ) : (
        paletteSections.map((section) =>
          section.entries.length > 0 ? (
            <div key={section.label} className="mb-1">
              <p className="px-4 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-zinc-600">
                {section.label}
              </p>
              {section.entries.map((entry) => {
                const index = paletteEntries.findIndex((item) => item.id === entry.id);
                const isActive = activeIndex === index;
                return (
                  <button
                    key={entry.id}
                    type="button"
                    className={cn(
                      "flex w-full items-center justify-between gap-3 px-4 py-2 text-left transition-colors",
                      isActive
                        ? "bg-[#8a9a7b]/10 text-zinc-100"
                        : "text-zinc-300 hover:bg-zinc-800/50",
                    )}
                    onClick={() => onSelect(entry)}
                    onMouseDown={(e) => e.preventDefault()}
                  >
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium truncate">
                        {renderHighlight(entry.title, commandDraft)}
                      </p>
                      <p className="text-xs text-zinc-500 truncate mt-0.5">
                        {renderHighlight(entry.description, commandDraft)}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {entry.badge && (
                        <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-500">
                          {entry.badge}
                        </span>
                      )}
                      <span className="font-mono text-[11px] text-zinc-600">
                        {renderHighlight(entry.keyword, commandDraft)}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          ) : null
        )
      )}
    </div>
  );
}
