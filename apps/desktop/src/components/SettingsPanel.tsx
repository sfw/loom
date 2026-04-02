import { useCallback, useEffect, useState } from "react";
import {
  shallowEqual,
  useAppSelector,
} from "@/context/AppContext";
import { cn } from "@/lib/utils";
import {
  Sun,
  Moon,
  Monitor,
  Database,
  FileCode,
  Server,
  Cpu,
  Settings,
  Shield,
  Layers,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Theme types & helpers
// ---------------------------------------------------------------------------

type ThemePreference = "dark" | "light" | "auto";

const THEME_STORAGE_KEY = "loom-theme-preference";

function getStoredTheme(): ThemePreference {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "dark" || stored === "light" || stored === "auto") return stored;
  } catch {
    // localStorage may not be available
  }
  return "dark";
}

function applyTheme(preference: ThemePreference) {
  const root = document.documentElement;
  let effectiveTheme: "dark" | "light";

  if (preference === "auto") {
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    effectiveTheme = prefersDark ? "dark" : "light";
  } else {
    effectiveTheme = preference;
  }

  if (effectiveTheme === "dark") {
    root.classList.add("dark");
    root.classList.remove("light");
  } else {
    root.classList.remove("dark");
    root.classList.add("light");
  }
}

// ---------------------------------------------------------------------------
// SettingsPanel
// ---------------------------------------------------------------------------

export default function SettingsPanel() {
  const { runtime, models, settings } = useAppSelector((state) => ({
    runtime: state.runtime,
    models: state.models,
    settings: state.settings,
  }), shallowEqual);

  // ---- Theme state ----
  const [theme, setThemeState] = useState<ThemePreference>(getStoredTheme);

  const setTheme = useCallback((next: ThemePreference) => {
    setThemeState(next);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // ignore
    }
    applyTheme(next);
  }, []);

  // Apply theme on mount
  useEffect(() => {
    applyTheme(theme);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Listen to system preference changes when in auto mode
  useEffect(() => {
    if (theme !== "auto") return;
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => applyTheme("auto");
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, [theme]);

  // ---- Settings entries ----
  const allSettings = [
    ...(settings?.basic || []),
    ...(settings?.advanced || []),
  ];

  // ---- Theme option cards ----
  const themeOptions: Array<{
    value: ThemePreference;
    label: string;
    description: string;
    icon: typeof Moon;
  }> = [
    {
      value: "dark",
      label: "Dark",
      description: "Dark background with light text",
      icon: Moon,
    },
    {
      value: "light",
      label: "Light",
      description: "Light mode (coming soon)",
      icon: Sun,
    },
    {
      value: "auto",
      label: "Auto",
      description: "Follows your system preference",
      icon: Monitor,
    },
  ];

  return (
    <div className="flex h-full flex-col overflow-y-auto">
      {/* Page header */}
      <div className="border-b border-zinc-800/60 bg-[#0c0c0e] px-6 py-5">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#6b7a5e]/15">
            <Settings size={18} className="text-[#a3b396]" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-zinc-100">Settings</h1>
            <p className="text-xs text-zinc-500">
              Appearance, runtime info, and configuration
            </p>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-8">
        {/* ================================================================
            APPEARANCE
        ================================================================ */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Sun size={15} className="text-[#a3b396]" />
            <h2 className="text-sm font-semibold text-zinc-200">Appearance</h2>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {themeOptions.map(({ value, label, description, icon: Icon }) => {
              const isActive = theme === value;
              return (
                <button
                  key={value}
                  type="button"
                  onClick={() => setTheme(value)}
                  className={cn(
                    "group relative flex flex-col items-center gap-2.5 rounded-xl border px-4 py-5 text-center transition-all",
                    isActive
                      ? "border-[#8a9a7b] bg-[#8a9a7b]/10 ring-1 ring-[#8a9a7b]/30"
                      : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700 hover:bg-zinc-800/40",
                  )}
                >
                  {/* Active indicator dot */}
                  {isActive && (
                    <span className="absolute top-2.5 right-2.5 h-2 w-2 rounded-full bg-[#8a9a7b]" />
                  )}

                  <div
                    className={cn(
                      "flex h-10 w-10 items-center justify-center rounded-lg transition-colors",
                      isActive
                        ? "bg-[#6b7a5e]/20"
                        : "bg-zinc-800/60 group-hover:bg-zinc-700/40",
                    )}
                  >
                    <Icon
                      size={20}
                      className={cn(
                        "transition-colors",
                        isActive
                          ? "text-[#a3b396]"
                          : "text-zinc-500 group-hover:text-zinc-300",
                      )}
                    />
                  </div>

                  <div>
                    <p
                      className={cn(
                        "text-sm font-semibold",
                        isActive ? "text-[#a3b396]" : "text-zinc-300",
                      )}
                    >
                      {label}
                    </p>
                    <p className="mt-0.5 text-[11px] text-zinc-500">
                      {description}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        {/* ================================================================
            RUNTIME INFO
        ================================================================ */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Server size={15} className="text-[#a3b396]" />
            <h2 className="text-sm font-semibold text-zinc-200">
              Runtime Information
            </h2>
          </div>

          {runtime ? (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 overflow-hidden">
              {/* Runtime details grid */}
              <div className="grid grid-cols-2 gap-px bg-zinc-800/40">
                <InfoCell
                  icon={Server}
                  label="Version"
                  value={runtime.version || "unknown"}
                />
                <InfoCell
                  icon={Shield}
                  label="Role"
                  value={runtime.runtime_role || "unknown"}
                />
                <InfoCell
                  icon={Database}
                  label="Database Path"
                  value={runtime.database_path || "not set"}
                  mono
                />
                <InfoCell
                  icon={FileCode}
                  label="Config Path"
                  value={runtime.config_path || "not set"}
                  mono
                />
                <InfoCell
                  icon={Layers}
                  label="Host"
                  value={`${runtime.host || "localhost"}:${runtime.port || "?"}`}
                  mono
                />
                <InfoCell
                  icon={Cpu}
                  label="Started At"
                  value={
                    runtime.started_at
                      ? new Date(runtime.started_at).toLocaleString()
                      : "unknown"
                  }
                />
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 px-5 py-8 text-center">
              <p className="text-sm text-zinc-500">
                Runtime information not available
              </p>
            </div>
          )}

          {/* Connected Models */}
          <div className="mt-5">
            <div className="flex items-center gap-2 mb-3">
              <Cpu size={14} className="text-[#bec8b4]" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">
                Connected Models
              </h3>
              <span className="ml-auto rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-zinc-500">
                {models.length}
              </span>
            </div>

            {models.length > 0 ? (
              <div className="space-y-2">
                {models.map((model) => (
                  <div
                    key={model.name}
                    className="flex items-start gap-3 rounded-lg border border-zinc-800 bg-zinc-900/30 px-4 py-3"
                  >
                    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#6b7a5e]/10">
                      <Cpu size={14} className="text-[#a3b396]" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <p className="truncate text-sm font-semibold text-zinc-200">
                          {model.name}
                        </p>
                        <span className="shrink-0 rounded bg-[#8a9a7b]/15 px-1.5 py-px text-[10px] font-semibold text-[#a3b396]">
                          Tier {model.tier}
                        </span>
                      </div>
                      <p className="mt-0.5 truncate text-xs font-mono text-zinc-500">
                        {model.model_id || model.model}
                      </p>
                      {model.roles.length > 0 && (
                        <div className="mt-1.5 flex flex-wrap gap-1">
                          {model.roles.map((role) => (
                            <span
                              key={role}
                              className="rounded-md bg-zinc-800 px-1.5 py-0.5 text-[10px] font-medium text-zinc-400"
                            >
                              {role}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 px-4 py-6 text-center">
                <p className="text-sm text-zinc-500">No models connected</p>
              </div>
            )}
          </div>
        </section>

        {/* ================================================================
            SETTINGS SNAPSHOT
        ================================================================ */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <FileCode size={15} className="text-[#a3b396]" />
            <h2 className="text-sm font-semibold text-zinc-200">
              Configuration Snapshot
            </h2>
            <span className="ml-auto text-[10px] text-zinc-600">
              Read-only
            </span>
          </div>

          {allSettings.length > 0 ? (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 overflow-hidden">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-zinc-800/60 bg-zinc-900/50">
                    <th className="px-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                      Path
                    </th>
                    <th className="px-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                      Description
                    </th>
                    <th className="px-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                      Effective Value
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/40">
                  {allSettings.map((entry) => (
                    <tr
                      key={entry.path}
                      className="group transition-colors hover:bg-zinc-800/20"
                    >
                      <td className="px-4 py-2.5">
                        <code className="text-xs font-mono text-[#a3b396]">
                          {entry.path}
                        </code>
                      </td>
                      <td className="px-4 py-2.5 text-xs text-zinc-400 max-w-[280px]">
                        <span className="line-clamp-2">
                          {entry.description || "No description"}
                        </span>
                      </td>
                      <td className="px-4 py-2.5">
                        <code className="text-xs font-mono text-zinc-300">
                          {entry.effective_display || "unset"}
                        </code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 px-5 py-8 text-center">
              <p className="text-sm text-zinc-500">
                No configuration entries loaded
              </p>
            </div>
          )}

          {settings?.updated_at && (
            <p className="mt-2 text-[10px] text-zinc-600">
              Last updated: {new Date(settings.updated_at).toLocaleString()}
            </p>
          )}
        </section>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// InfoCell — small helper for the runtime info grid
// ---------------------------------------------------------------------------

function InfoCell({
  icon: Icon,
  label,
  value,
  mono = false,
}: {
  icon: typeof Server;
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-start gap-2.5 bg-zinc-900/30 px-4 py-3">
      <Icon size={14} className="mt-0.5 shrink-0 text-zinc-600" />
      <div className="min-w-0">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
          {label}
        </p>
        <p
          className={cn(
            "mt-0.5 truncate text-xs",
            mono ? "font-mono text-zinc-300" : "text-zinc-300",
          )}
          title={value}
        >
          {value}
        </p>
      </div>
    </div>
  );
}
