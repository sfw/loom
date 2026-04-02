import {
  startTransition,
  type FormEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  fetchGlobalSearch,
  type WorkspaceOverview,
  type WorkspaceSearchResponse,
} from "../api";
import {
  buildCommandOptions,
  buildCommandPaletteEntries,
  buildPaletteEntries,
  buildPaletteSections,
  buildPinnedPaletteEntries,
  buildResultPaletteEntries,
  isPaletteEntry,
  rememberPaletteEntry as nextRecentPaletteEntries,
  RECENT_PALETTE_STORAGE_KEY,
  type CommandOption,
  type PaletteEntry,
} from "../shell";
import type { ViewTab } from "../utils";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface CommandPaletteState {
  commandDraft: string;
  commandPaletteOpen: boolean;
  activeCommandIndex: number;
  commandSearchResults: WorkspaceSearchResponse | null;
  searchingCommandPalette: boolean;
  recentPaletteEntries: PaletteEntry[];

  // Computed
  trimmedCommandDraft: string;
  normalizedCommandDraft: string;
  commandSearchTerm: string;
  commandOptions: CommandOption[];
  pinnedPaletteEntries: PaletteEntry[];
  commandPaletteEntries: PaletteEntry[];
  resultPaletteEntries: PaletteEntry[];
  paletteEntries: PaletteEntry[];
  paletteSections: Array<{ label: string; entries: PaletteEntry[] }>;

  // Refs
  commandInputRef: React.RefObject<HTMLInputElement | null>;
}

export interface CommandPaletteActions {
  setCommandDraft: React.Dispatch<React.SetStateAction<string>>;
  setCommandPaletteOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setActiveCommandIndex: React.Dispatch<React.SetStateAction<number>>;
  focusCommandBar: () => void;
  handleCommandAction: (rawCommand: string) => void;
  executeCommandOption: (option: CommandOption) => void;
  executePaletteEntry: (entry: PaletteEntry) => void;
  handleCommandInputKeyDown: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  handleCommandSubmit: (event: FormEvent<HTMLFormElement>) => void;
  rememberPaletteEntry: (entry: PaletteEntry) => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useCommandPalette(deps: {
  selectedWorkspaceId: string;
  overview: WorkspaceOverview | null;
  setSelectedConversationId: React.Dispatch<React.SetStateAction<string>>;
  setSelectedRunId: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceSearchQuery: React.Dispatch<React.SetStateAction<string>>;
  setActiveTab: React.Dispatch<React.SetStateAction<ViewTab>>;
  setError: React.Dispatch<React.SetStateAction<string>>;
  setNotice: React.Dispatch<React.SetStateAction<string>>;
  focusSearch: () => void;
  focusConversationComposer: () => void;
  focusRunComposer: () => void;
  handlePrefillStarterConversation: () => void;
  handlePrefillStarterRun: () => void;
  handleSearchResultSelection: (result: {
    workspace_id?: string;
    conversation_id?: string;
    run_id?: string;
    approval_item_id?: string;
    item_id?: string;
    title?: string;
    path?: string;
    kind?: string;
  }) => void;
}): CommandPaletteState & CommandPaletteActions {
  const {
    selectedWorkspaceId,
    overview,
    setSelectedConversationId,
    setSelectedRunId,
    setWorkspaceSearchQuery,
    setActiveTab,
    setError,
    setNotice,
    focusSearch,
    focusConversationComposer,
    focusRunComposer,
    handlePrefillStarterConversation,
    handlePrefillStarterRun,
    handleSearchResultSelection,
  } = deps;

  // State
  const [commandDraft, setCommandDraft] = useState("");
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const [activeCommandIndex, setActiveCommandIndex] = useState(0);
  const [commandSearchResults, setCommandSearchResults] =
    useState<WorkspaceSearchResponse | null>(null);
  const [searchingCommandPalette, setSearchingCommandPalette] = useState(false);
  const [recentPaletteEntries, setRecentPaletteEntries] = useState<PaletteEntry[]>([]);

  // Refs
  const commandInputRef = useRef<HTMLInputElement | null>(null);
  const commandBlurTimerRef = useRef<number | null>(null);
  const commandSearchTimerRef = useRef<number | null>(null);

  // ---------------------------------------------------------------------------
  // Computed values
  // ---------------------------------------------------------------------------

  const trimmedCommandDraft = useMemo(() => commandDraft.trim(), [commandDraft]);
  const normalizedCommandDraft = useMemo(
    () => trimmedCommandDraft.toLowerCase(),
    [trimmedCommandDraft],
  );
  const commandSearchTerm = useMemo(() => (
    normalizedCommandDraft.startsWith("search ")
      ? trimmedCommandDraft.slice(7).trim()
      : trimmedCommandDraft
  ), [normalizedCommandDraft, trimmedCommandDraft]);
  const commandOptions = useMemo(
    () => buildCommandOptions(commandDraft),
    [commandDraft],
  );
  const pinnedPaletteEntries = useMemo(() => buildPinnedPaletteEntries(), []);
  const commandPaletteEntriesComputed = useMemo(
    () => buildCommandPaletteEntries(commandOptions),
    [commandOptions],
  );
  const resultPaletteEntries = useMemo(
    () => buildResultPaletteEntries(commandSearchResults),
    [commandSearchResults],
  );
  const paletteEntries = useMemo(() => buildPaletteEntries(
    commandDraft,
    recentPaletteEntries,
    commandPaletteEntriesComputed,
    resultPaletteEntries,
    pinnedPaletteEntries,
  ), [
    commandDraft,
    commandPaletteEntriesComputed,
    pinnedPaletteEntries,
    recentPaletteEntries,
    resultPaletteEntries,
  ]);
  const paletteSections = useMemo(() => buildPaletteSections(
    commandDraft,
    recentPaletteEntries,
    commandPaletteEntriesComputed,
    resultPaletteEntries,
    pinnedPaletteEntries,
    commandSearchResults,
  ), [
    commandDraft,
    commandPaletteEntriesComputed,
    commandSearchResults,
    pinnedPaletteEntries,
    recentPaletteEntries,
    resultPaletteEntries,
  ]);

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (commandBlurTimerRef.current !== null) {
        window.clearTimeout(commandBlurTimerRef.current);
      }
      if (commandSearchTimerRef.current !== null) {
        window.clearTimeout(commandSearchTimerRef.current);
      }
    };
  }, []);

  // Load recent palette entries from localStorage
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(RECENT_PALETTE_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return;
      }
      setRecentPaletteEntries(parsed.filter(isPaletteEntry).slice(0, 6));
    } catch {
      return;
    }
  }, []);

  // Persist recent palette entries to localStorage
  useEffect(() => {
    try {
      window.localStorage.setItem(
        RECENT_PALETTE_STORAGE_KEY,
        JSON.stringify(recentPaletteEntries.slice(0, 6)),
      );
    } catch {
      return;
    }
  }, [recentPaletteEntries]);

  // Keep palette index in bounds
  useEffect(() => {
    if (paletteEntries.length === 0) {
      setActiveCommandIndex(0);
      return;
    }
    setActiveCommandIndex((current) =>
      Math.max(0, Math.min(paletteEntries.length - 1, current)),
    );
  }, [paletteEntries.length]);

  // Debounced command palette search
  useEffect(() => {
    const query = commandSearchTerm.trim();
    if (commandSearchTimerRef.current !== null) {
      window.clearTimeout(commandSearchTimerRef.current);
      commandSearchTimerRef.current = null;
    }
    if (!commandPaletteOpen || !query) {
      setSearchingCommandPalette(false);
      setCommandSearchResults(null);
      return;
    }
    let cancelled = false;
    setSearchingCommandPalette(true);
    commandSearchTimerRef.current = window.setTimeout(() => {
      commandSearchTimerRef.current = null;
      void (async () => {
        try {
          const results = await fetchGlobalSearch(query, 5);
          if (!cancelled) {
            setCommandSearchResults(results);
          }
        } catch {
          if (!cancelled) {
            setCommandSearchResults(null);
          }
        } finally {
          if (!cancelled) {
            setSearchingCommandPalette(false);
          }
        }
      })();
    }, 160);
    return () => {
      cancelled = true;
      if (commandSearchTimerRef.current !== null) {
        window.clearTimeout(commandSearchTimerRef.current);
        commandSearchTimerRef.current = null;
      }
    };
  }, [commandPaletteOpen, commandSearchTerm]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  function focusCommandBar() {
    if (commandBlurTimerRef.current !== null) {
      window.clearTimeout(commandBlurTimerRef.current);
      commandBlurTimerRef.current = null;
    }
    setCommandPaletteOpen(true);
    commandInputRef.current?.focus();
    commandInputRef.current?.select();
  }

  function handleCommandAction(rawCommand: string) {
    const normalized = rawCommand.trim().toLowerCase();
    if (!normalized) {
      return;
    }
    if (normalized === "new conversation" || normalized === "new thread" || normalized === "thread") {
      startTransition(() => {
        setSelectedRunId("");
        setSelectedConversationId("");
        setActiveTab("threads");
      });
      focusConversationComposer();
      return;
    }
    if (normalized === "new run" || normalized === "run") {
      startTransition(() => {
        setSelectedConversationId("");
        setSelectedRunId("");
        setActiveTab("runs");
      });
      focusRunComposer();
      return;
    }
    if (normalized === "starter thread" || normalized === "starter conversation") {
      startTransition(() => {
        setSelectedRunId("");
        setSelectedConversationId("");
        setActiveTab("threads");
      });
      handlePrefillStarterConversation();
      focusConversationComposer();
      return;
    }
    if (normalized === "starter run") {
      startTransition(() => {
        setSelectedConversationId("");
        setSelectedRunId("");
        setActiveTab("runs");
      });
      handlePrefillStarterRun();
      focusRunComposer();
      return;
    }
    if (normalized === "latest conversation" || normalized === "latest thread") {
      const latestConversationId = [...(overview?.recent_conversations || [])]
        .sort((a, b) => String(b.last_active_at || "").localeCompare(String(a.last_active_at || "")))[0]?.id || "";
      if (!latestConversationId) {
        setError("No thread is available in this workspace.");
        return;
      }
      startTransition(() => {
        setSelectedRunId("");
        setSelectedConversationId(latestConversationId);
        setActiveTab("threads");
      });
      return;
    }
    if (normalized === "latest run") {
      const latestRunId = [...(overview?.recent_runs || [])]
        .sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")))[0]?.id || "";
      if (!latestRunId) {
        setError("No run is available in this workspace.");
        return;
      }
      startTransition(() => {
        setSelectedConversationId("");
        setSelectedRunId(latestRunId);
        setActiveTab("runs");
      });
      return;
    }
    if (normalized === "clear search") {
      setWorkspaceSearchQuery("");
      return;
    }
    if (normalized.startsWith("search ")) {
      setWorkspaceSearchQuery(rawCommand.trim().slice(7));
      focusSearch();
      return;
    }
    setError(
      "Unknown command. Try new thread, new run, starter thread, starter run, latest thread, latest run, or search <term>.",
    );
  }

  function rememberPaletteEntry(entry: PaletteEntry) {
    setRecentPaletteEntries((current) => nextRecentPaletteEntries(current, entry));
  }

  function executeCommandOption(option: CommandOption) {
    rememberPaletteEntry({
      id: `command-${option.id}`,
      title: option.label,
      description: option.description,
      keyword: option.command,
      kind: "command",
      command: option,
    });
    setCommandDraft(option.command);
    handleCommandAction(option.command);
    setCommandPaletteOpen(false);
    setActiveCommandIndex(0);
    setCommandDraft("");
  }

  function executePaletteEntry(entry: PaletteEntry) {
    if (entry.kind === "command" && entry.command) {
      executeCommandOption(entry.command);
      return;
    }
    if (entry.kind === "result" && entry.result) {
      rememberPaletteEntry(entry);
      handleSearchResultSelection(entry.result);
      setCommandPaletteOpen(false);
      setActiveCommandIndex(0);
      setCommandDraft("");
    }
  }

  function handleCommandInputKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (!commandPaletteOpen || paletteEntries.length === 0) {
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveCommandIndex((current) => (current + 1) % paletteEntries.length);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveCommandIndex((current) => (
        (current - 1 + paletteEntries.length) % paletteEntries.length
      ));
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      setCommandPaletteOpen(false);
      return;
    }
    if (event.key === "Enter" && paletteEntries.length > 0) {
      event.preventDefault();
      executePaletteEntry(paletteEntries[activeCommandIndex] || paletteEntries[0]);
    }
  }

  function handleCommandSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setNotice("");
    handleCommandAction(commandDraft);
    setCommandPaletteOpen(false);
    setActiveCommandIndex(0);
    setCommandDraft("");
  }

  return {
    // State
    commandDraft,
    commandPaletteOpen,
    activeCommandIndex,
    commandSearchResults,
    searchingCommandPalette,
    recentPaletteEntries,

    // Computed
    trimmedCommandDraft,
    normalizedCommandDraft,
    commandSearchTerm,
    commandOptions,
    pinnedPaletteEntries,
    commandPaletteEntries: commandPaletteEntriesComputed,
    resultPaletteEntries,
    paletteEntries,
    paletteSections,

    // Refs
    commandInputRef,

    // Actions
    setCommandDraft,
    setCommandPaletteOpen,
    setActiveCommandIndex,
    focusCommandBar,
    handleCommandAction,
    executeCommandOption,
    executePaletteEntry,
    handleCommandInputKeyDown,
    handleCommandSubmit,
    rememberPaletteEntry,
  };
}
