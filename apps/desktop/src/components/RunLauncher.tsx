import { useState, type FormEvent } from "react";
import {
  Play,
  Zap,
  FileText,
  ChevronDown,
  ChevronUp,
  Shield,
} from "lucide-react";
import {
  shallowEqual,
  useAppActions,
  useAppSelector,
} from "@/context/AppContext";
import { cn } from "@/lib/utils";
import type { ProcessInfo } from "@/api";

type LaunchMode = "adhoc" | "process";

export default function RunLauncher() {
  const {
    runGoal,
    runProcess,
    runApprovalMode,
    launchingRun,
    selectedWorkspaceId,
    inventory,
  } = useAppSelector((state) => ({
    runGoal: state.runGoal,
    runProcess: state.runProcess,
    runApprovalMode: state.runApprovalMode,
    launchingRun: state.launchingRun,
    selectedWorkspaceId: state.selectedWorkspaceId,
    inventory: state.inventory,
  }), shallowEqual);
  const {
    setRunGoal,
    setRunProcess,
    setRunApprovalMode,
    handleLaunchRun,
  } = useAppActions();

  const [launchMode, setLaunchMode] = useState<LaunchMode>("adhoc");
  const [showAdvanced, setShowAdvanced] = useState(false);

  const processes = inventory?.processes || [];
  const hasProcesses = processes.length > 0;
  const disabled = !selectedWorkspaceId || launchingRun;

  function selectProcess(p: ProcessInfo) {
    setRunProcess(p.name);
    setRunGoal((current) => current || p.description || "");
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    // For ad-hoc mode, clear the process synchronously in the handler
    // rather than relying on React batching setRunProcess("") before
    // handleLaunchRun reads the state.
    if (launchMode === "adhoc" && runProcess) {
      setRunProcess("");
      // Defer the launch to the next tick so the state update takes effect
      e.preventDefault();
      window.setTimeout(() => {
        const syntheticEvent = new Event("submit", { cancelable: true }) as unknown as FormEvent<HTMLFormElement>;
        syntheticEvent.preventDefault = () => {};
        handleLaunchRun(syntheticEvent);
      }, 0);
      return;
    }
    handleLaunchRun(e);
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Mode toggle */}
      <div className="flex items-center gap-1 px-3 pt-3 pb-2">
        <button
          type="button"
          onClick={() => { setLaunchMode("adhoc"); setRunProcess(""); }}
          className={cn(
            "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            launchMode === "adhoc"
              ? "bg-[#6b7a5e]/20 text-[#bec8b4] border border-[#8a9a7b]/30"
              : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 border border-transparent",
          )}
        >
          <Zap className="h-3 w-3" />
          Ad-hoc
        </button>
        <button
          type="button"
          onClick={() => setLaunchMode("process")}
          className={cn(
            "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            launchMode === "process"
              ? "bg-[#6b7a5e]/20 text-[#bec8b4] border border-[#8a9a7b]/30"
              : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 border border-transparent",
          )}
        >
          <FileText className="h-3 w-3" />
          Process
          {hasProcesses && (
            <span className="ml-0.5 text-[10px] text-zinc-500">{processes.length}</span>
          )}
        </button>
      </div>

      {/* Process catalog (when process mode) */}
      {launchMode === "process" && (
        <div className="flex flex-1 flex-col overflow-hidden min-h-0">
          <div className="flex-1 overflow-y-auto px-3 pb-2">
            {processes.length === 0 ? (
              <div className="rounded-lg border border-dashed border-zinc-700 px-3 py-4 text-center">
                <FileText className="mx-auto h-5 w-5 text-zinc-600 mb-1.5" />
                <p className="text-xs text-zinc-500">No process definitions found</p>
                <p className="text-[10px] text-zinc-600 mt-1">
                  Add process YAML files to your workspace or configure search paths
                </p>
              </div>
            ) : (
              <div className="space-y-1">
                {processes.map((p) => {
                  const isSelected = runProcess === p.name;
                  return (
                    <button
                      key={p.name}
                      type="button"
                      onClick={() => selectProcess(p)}
                      disabled={disabled}
                      className={cn(
                        "w-full rounded-lg px-3 py-2 text-left transition-colors",
                        isSelected
                          ? "bg-[#6b7a5e]/15 border border-[#8a9a7b]/30"
                          : "border border-transparent hover:bg-zinc-800/60",
                        "disabled:opacity-40 disabled:cursor-not-allowed",
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className={cn(
                          "text-xs font-semibold truncate",
                          isSelected ? "text-[#bec8b4]" : "text-zinc-200",
                        )}>
                          {p.name}
                        </span>
                        {p.version && (
                          <span className="shrink-0 text-[10px] text-zinc-600">
                            v{p.version}
                          </span>
                        )}
                      </div>
                      {p.description && (
                        <p className="mt-0.5 text-[10.5px] text-zinc-500 line-clamp-2">
                          {p.description}
                        </p>
                      )}
                      {p.author && (
                        <p className="mt-0.5 text-[10px] text-zinc-600">
                          by {p.author}
                        </p>
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Quick launch button — always visible when process selected */}
          {runProcess && (
            <div className="border-t border-zinc-800 px-3 py-2.5 shrink-0">
              <button
                type="button"
                onClick={() => {
                  const syntheticEvent = new Event("submit", { cancelable: true }) as unknown as FormEvent<HTMLFormElement>;
                  syntheticEvent.preventDefault = () => {};
                  handleLaunchRun(syntheticEvent);
                }}
                disabled={disabled}
                className={cn(
                  "flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition-colors",
                  "bg-[#6b7a5e] text-white hover:bg-[#8a9a7b]",
                  "disabled:opacity-40 disabled:cursor-not-allowed",
                )}
              >
                <Play className="h-4 w-4" />
                {launchingRun ? "Launching..." : `Launch ${runProcess}`}
              </button>
            </div>
          )}
        </div>
      )}

      {/* Ad-hoc description */}
      {launchMode === "adhoc" && (
        <div className="px-3 pb-2">
          <div className="flex items-start gap-2 rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2">
            <Zap className="h-3.5 w-3.5 text-[#a3b396] mt-0.5 shrink-0" />
            <p className="text-[11px] text-zinc-500 leading-relaxed">
              Describe your goal in natural language. The system will dynamically
              synthesize an execution plan and choose the right tools.
            </p>
          </div>
        </div>
      )}

      {/* Launch form */}
      <div className="border-t border-zinc-800 px-3 py-3">
        <form onSubmit={handleSubmit} className="space-y-2">
          {/* Goal */}
          <div>
            <label className="block text-[10.5px] font-medium text-zinc-500 mb-1">
              {launchMode === "adhoc" ? "Goal" : "Goal override"}
            </label>
            <textarea
              value={runGoal}
              onChange={(e) => setRunGoal(e.target.value)}
              placeholder={
                launchMode === "adhoc"
                  ? "Describe what the run should accomplish..."
                  : runProcess
                    ? `Override the default goal for ${runProcess}, or leave blank to use process defaults`
                    : "Select a process above, then optionally customize the goal"
              }
              rows={launchMode === "adhoc" ? 3 : 2}
              disabled={disabled}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs text-zinc-200 placeholder:text-zinc-600 resize-none focus:outline-none focus:ring-1 focus:ring-[#8a9a7b]/50 focus:border-[#8a9a7b]/50 disabled:opacity-40"
            />
          </div>

          {/* Selected process display (process mode) */}
          {launchMode === "process" && runProcess && (
            <div className="flex items-center gap-2 rounded-md bg-[#6b7a5e]/10 border border-[#8a9a7b]/20 px-2.5 py-1.5">
              <FileText className="h-3 w-3 text-[#a3b396]" />
              <span className="text-xs text-[#bec8b4] font-medium">{runProcess}</span>
              <button
                type="button"
                onClick={() => { setRunProcess(""); setRunGoal(""); }}
                className="ml-auto text-[10px] text-zinc-500 hover:text-zinc-300"
              >
                clear
              </button>
            </div>
          )}

          {/* Advanced options toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1 text-[10.5px] text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            {showAdvanced ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            Advanced options
          </button>

          {showAdvanced && (
            <div className="space-y-2 pl-1">
              {/* Approval mode */}
              <div>
                <label className="flex items-center gap-1 text-[10.5px] font-medium text-zinc-500 mb-1">
                  <Shield className="h-3 w-3" />
                  Approval mode
                </label>
                <div className="flex gap-1">
                  {([
                    { value: "auto", label: "Auto", desc: "Gate destructive ops" },
                    { value: "manual", label: "Manual", desc: "Gate every step" },
                    { value: "disabled", label: "Disabled", desc: "No gating" },
                  ] as const).map((mode) => (
                    <button
                      key={mode.value}
                      type="button"
                      onClick={() => setRunApprovalMode(mode.value)}
                      disabled={disabled}
                      title={mode.desc}
                      className={cn(
                        "flex-1 rounded-md px-2 py-1.5 text-[10.5px] font-medium transition-colors border",
                        runApprovalMode === mode.value
                          ? "bg-[#6b7a5e]/15 text-[#bec8b4] border-[#8a9a7b]/30"
                          : "text-zinc-400 border-zinc-700 hover:bg-zinc-800",
                        "disabled:opacity-40 disabled:cursor-not-allowed",
                      )}
                    >
                      {mode.label}
                    </button>
                  ))}
                </div>
                <p className="mt-1 text-[10px] text-zinc-600">
                  {runApprovalMode === "auto" && "Auto-proceed at high confidence, gate destructive and uncertain ops"}
                  {runApprovalMode === "manual" && "Pause before every subtask for human approval"}
                  {runApprovalMode === "disabled" && "No gating except always-gated destructive patterns (rm -rf, drop table, etc.)"}
                </p>
              </div>

            </div>
          )}

          {/* Launch button */}
          <button
            type="submit"
            disabled={disabled || (launchMode === "adhoc" && !runGoal.trim()) || (launchMode === "process" && !runProcess)}
            className={cn(
              "flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition-colors",
              (launchMode === "process" && runProcess)
                ? "bg-[#6b7a5e] text-white hover:bg-[#8a9a7b] shadow-sm"
                : "bg-[#6b7a5e] text-white hover:bg-[#8a9a7b]",
              "disabled:opacity-40 disabled:cursor-not-allowed",
            )}
          >
            <Play className="h-4 w-4" />
            {launchingRun
              ? "Launching..."
              : launchMode === "process" && runProcess
                ? `Launch ${runProcess}`
                : "Launch run"}
          </button>
        </form>
      </div>
    </div>
  );
}
