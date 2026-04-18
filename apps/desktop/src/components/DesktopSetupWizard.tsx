import { useState } from "react";
import { CheckCircle2, ChevronLeft, Loader2, Sparkles } from "lucide-react";

import type { SetupCompleteRequest, SetupProviderInfo, SetupStatus } from "../api";

type WizardStep = "provider" | "connection" | "roles" | "utility_prompt" | "review";
type DraftKey = "primary" | "utility";

interface ModelDraftState {
  name: DraftKey;
  provider: string;
  base_url: string;
  api_key: string;
  model: string;
  roles: string[];
  max_tokens: number;
  temperature: number;
}

interface DesktopSetupWizardProps {
  status: SetupStatus;
  onDiscoverModels: (provider: string, baseUrl: string, apiKey?: string) => Promise<string[]>;
  onCompleteSetup: (payload: SetupCompleteRequest) => Promise<void>;
}

function createDraft(name: DraftKey, provider: SetupProviderInfo, roles: string[] = []): ModelDraftState {
  return {
    name,
    provider: provider.provider_key,
    base_url: provider.default_base_url,
    api_key: "",
    model: "",
    roles,
    max_tokens: name === "primary" ? 8192 : 2048,
    temperature: name === "primary" ? 0.1 : 0,
  };
}

function titleForTarget(target: DraftKey): string {
  return target === "primary" ? "Primary model" : "Utility model";
}

function stepTitle(step: WizardStep, target: DraftKey): string {
  if (step === "provider") return `Choose a provider for your ${target === "primary" ? "first" : "utility"} model`;
  if (step === "connection") return `Connect your ${titleForTarget(target).toLowerCase()}`;
  if (step === "roles") return "Decide what the primary model should handle";
  if (step === "utility_prompt") return "Cover the remaining roles";
  return "Review your Loom setup";
}

export default function DesktopSetupWizard({
  status,
  onDiscoverModels,
  onCompleteSetup,
}: DesktopSetupWizardProps) {
  const providerMap = new Map(
    status.providers.map((provider) => [provider.provider_key, provider]),
  );
  const rolePresets = status.role_presets;
  const allRoles = rolePresets.all ?? ["planner", "executor", "extractor", "verifier", "compactor"];
  const [step, setStep] = useState<WizardStep>("provider");
  const [activeTarget, setActiveTarget] = useState<DraftKey>("primary");
  const [primaryDraft, setPrimaryDraft] = useState<ModelDraftState>(
    createDraft("primary", status.providers[0]),
  );
  const [utilityDraft, setUtilityDraft] = useState<ModelDraftState | null>(null);
  const [discoveredModels, setDiscoveredModels] = useState<Record<DraftKey, string[]>>({
    primary: [],
    utility: [],
  });
  const [discovering, setDiscovering] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const currentDraft = activeTarget === "primary" ? primaryDraft : (
    utilityDraft ?? createDraft("utility", status.providers[0], [])
  );
  const currentProvider = providerMap.get(currentDraft.provider) ?? status.providers[0];
  const currentDiscovered = discoveredModels[activeTarget] ?? [];
  const missingRoles = allRoles.filter((role) => !primaryDraft.roles.includes(role));

  const setDraft = (updater: (draft: ModelDraftState) => ModelDraftState) => {
    if (activeTarget === "primary") {
      setPrimaryDraft((current) => updater(current));
      return;
    }
    setUtilityDraft((current) => updater(current ?? createDraft("utility", currentProvider, missingRoles)));
  };

  const canContinueFromConnection = currentDraft.base_url.trim().length > 0 && currentDraft.model.trim().length > 0;
  const canContinueFromRoles = primaryDraft.roles.length > 0;

  async function handleDiscoverModels() {
    setError("");
    setDiscovering(true);
    try {
      const models = await onDiscoverModels(
        currentDraft.provider,
        currentDraft.base_url,
        currentDraft.api_key,
      );
      setDiscoveredModels((current) => ({
        ...current,
        [activeTarget]: models,
      }));
      if (models.length > 0 && !currentDraft.model.trim()) {
        setDraft((draft) => ({ ...draft, model: models[0] }));
      }
    } catch (discoverError) {
      setError(discoverError instanceof Error ? discoverError.message : "Could not discover models.");
    } finally {
      setDiscovering(false);
    }
  }

  function handleContinue() {
    setError("");
    if (step === "provider") {
      setStep("connection");
      return;
    }
    if (step === "connection") {
      if (!canContinueFromConnection) {
        setError("Enter a server URL and model name to continue.");
        return;
      }
      if (!Number.isFinite(currentDraft.temperature)) {
        setError("Enter a valid temperature before continuing.");
        return;
      }
      if (!Number.isInteger(currentDraft.max_tokens) || currentDraft.max_tokens <= 0) {
        setError("Max tokens must be a whole number greater than 0.");
        return;
      }
      if (activeTarget === "utility") {
        setStep("review");
        return;
      }
      setStep("roles");
      return;
    }
    if (step === "roles") {
      if (!canContinueFromRoles) {
        setError("Choose how the primary model should be used.");
        return;
      }
      if (missingRoles.length === 0) {
        setUtilityDraft(null);
        setStep("review");
        return;
      }
      setStep("utility_prompt");
    }
  }

  function handleBack() {
    setError("");
    if (step === "connection") {
      setStep("provider");
      return;
    }
    if (step === "roles") {
      setStep("connection");
      return;
    }
    if (step === "utility_prompt") {
      setStep("roles");
      return;
    }
    if (step === "review") {
      if (utilityDraft) {
        setActiveTarget("utility");
        setStep("connection");
        return;
      }
      setActiveTarget("primary");
      setStep(missingRoles.length > 0 ? "utility_prompt" : "roles");
    }
  }

  async function handleSave() {
    setError("");
    setSaving(true);
    try {
      await onCompleteSetup({
        models: [
          primaryDraft,
          ...(utilityDraft ? [utilityDraft] : []),
        ],
      });
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not save your Loom setup.");
    } finally {
      setSaving(false);
    }
  }

  function chooseProvider(provider: SetupProviderInfo) {
    setDiscoveredModels((current) => ({
      ...current,
      [activeTarget]: [],
    }));
    setDraft((draft) => ({
      ...draft,
      provider: provider.provider_key,
      base_url: provider.default_base_url,
      api_key: "",
      model: "",
    }));
  }

  function renderProviderStep() {
    return (
      <div className="grid gap-3">
        {status.providers.map((provider) => {
          const selected = currentDraft.provider === provider.provider_key;
          return (
            <button
              key={provider.provider_key}
              type="button"
              onClick={() => chooseProvider(provider)}
              className={`rounded-2xl border px-4 py-4 text-left transition-colors ${
                selected
                  ? "border-[#8a9a7b] bg-[#182015]"
                  : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-900"
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-semibold text-zinc-100">{provider.display_name}</p>
                  <p className="mt-1 text-xs text-zinc-500">{provider.default_base_url}</p>
                </div>
                {selected && <CheckCircle2 className="h-5 w-5 shrink-0 text-[#a3b396]" />}
              </div>
              <p className="mt-3 text-xs text-zinc-400">
                {provider.needs_api_key ? "Requires an API key." : "Works with local or hosted endpoints."}
              </p>
            </button>
          );
        })}
      </div>
    );
  }

  function renderConnectionStep() {
    return (
      <div className="space-y-4">
        <div className="grid gap-2">
          <label className="text-xs font-medium uppercase tracking-[0.2em] text-zinc-500" htmlFor="setup-base-url">
            Base URL
          </label>
          <input
            id="setup-base-url"
            value={currentDraft.base_url}
            onChange={(event) => setDraft((draft) => ({ ...draft, base_url: event.target.value }))}
            className="rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2.5 text-sm text-zinc-100 outline-none transition-colors focus:border-[#8a9a7b]"
            placeholder={currentProvider.default_base_url}
          />
        </div>

        <div className="grid gap-2">
          <label className="text-xs font-medium uppercase tracking-[0.2em] text-zinc-500" htmlFor="setup-api-key">
            API Key {currentProvider.needs_api_key ? "" : "(optional)"}
          </label>
          <input
            id="setup-api-key"
            type="password"
            value={currentDraft.api_key}
            onChange={(event) => setDraft((draft) => ({ ...draft, api_key: event.target.value }))}
            className="rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2.5 text-sm text-zinc-100 outline-none transition-colors focus:border-[#8a9a7b]"
            placeholder={currentProvider.needs_api_key ? "Required for this provider" : "Leave blank if not needed"}
          />
        </div>

        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-zinc-100">Find models automatically</p>
              <p className="text-xs text-zinc-500">
                We’ll query this endpoint and let you pick from what Loom can actually use.
              </p>
            </div>
            <button
              type="button"
              onClick={() => void handleDiscoverModels()}
              disabled={discovering || !currentDraft.base_url.trim()}
              className="inline-flex items-center gap-2 rounded-lg bg-[#6b7a5e] px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-[#8a9a7b] disabled:cursor-not-allowed disabled:opacity-60"
            >
              {discovering ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
              Discover models
            </button>
          </div>
          {currentDiscovered.length > 0 ? (
            <div className="mt-4 flex flex-wrap gap-2">
              {currentDiscovered.map((modelName) => (
                <button
                  key={modelName}
                  type="button"
                  onClick={() => setDraft((draft) => ({ ...draft, model: modelName }))}
                  className={`rounded-full border px-3 py-1.5 text-xs transition-colors ${
                    currentDraft.model === modelName
                      ? "border-[#8a9a7b] bg-[#182015] text-[#d7e0ce]"
                      : "border-zinc-700 bg-zinc-950 text-zinc-300 hover:border-zinc-600"
                  }`}
                >
                  {modelName}
                </button>
              ))}
            </div>
          ) : (
            <p className="mt-4 text-xs text-zinc-500">
              If discovery does not return anything, you can still type a model name manually below.
            </p>
          )}
        </div>

        <div className="grid gap-2">
          <label className="text-xs font-medium uppercase tracking-[0.2em] text-zinc-500" htmlFor="setup-model-name">
            Model Name
          </label>
          <input
            id="setup-model-name"
            value={currentDraft.model}
            onChange={(event) => setDraft((draft) => ({ ...draft, model: event.target.value }))}
            className="rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2.5 text-sm text-zinc-100 outline-none transition-colors focus:border-[#8a9a7b]"
            placeholder="claude-sonnet-4-5, gpt-4.1, qwen3:14b..."
          />
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <div className="grid gap-2">
            <label className="text-xs font-medium uppercase tracking-[0.2em] text-zinc-500" htmlFor="setup-temperature">
              Temperature
            </label>
            <input
              id="setup-temperature"
              type="number"
              step="0.1"
              value={currentDraft.temperature}
              onChange={(event) => {
                const next = Number(event.target.value);
                setDraft((draft) => ({ ...draft, temperature: Number.isFinite(next) ? next : draft.temperature }));
              }}
              className="rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2.5 text-sm text-zinc-100 outline-none transition-colors focus:border-[#8a9a7b]"
            />
            <p className="text-xs text-zinc-500">
              Lower is steadier. Raise this when a provider or model expects a hotter default.
            </p>
          </div>
          <div className="grid gap-2">
            <label className="text-xs font-medium uppercase tracking-[0.2em] text-zinc-500" htmlFor="setup-max-tokens">
              Max Tokens
            </label>
            <input
              id="setup-max-tokens"
              type="number"
              min="1"
              step="1"
              value={currentDraft.max_tokens}
              onChange={(event) => {
                const next = Number(event.target.value);
                setDraft((draft) => ({ ...draft, max_tokens: Number.isInteger(next) ? next : draft.max_tokens }));
              }}
              className="rounded-xl border border-zinc-800 bg-zinc-950/80 px-3 py-2.5 text-sm text-zinc-100 outline-none transition-colors focus:border-[#8a9a7b]"
            />
            <p className="text-xs text-zinc-500">
              Controls how large each response is allowed to be for this model.
            </p>
          </div>
        </div>
      </div>
    );
  }

  function renderRolesStep() {
    const options: Array<{ key: "all" | "primary" | "utility"; label: string; description: string }> = [
      {
        key: "all",
        label: "All roles",
        description: "Use one strong model for planning, execution, extraction, verification, and compaction.",
      },
      {
        key: "primary",
        label: "Primary only",
        description: "Use this model for planning and execution, then add a cheaper utility model if you want.",
      },
      {
        key: "utility",
        label: "Utility only",
        description: "Use this model for extraction, verification, and compaction.",
      },
    ];

    return (
      <div className="grid gap-3">
        {options.map((option) => {
          const roles = rolePresets[option.key] ?? [];
          const selected = roles.length === primaryDraft.roles.length
            && roles.every((role) => primaryDraft.roles.includes(role));
          return (
            <button
              key={option.key}
              type="button"
              onClick={() => setPrimaryDraft((draft) => ({ ...draft, roles }))}
              className={`rounded-2xl border px-4 py-4 text-left transition-colors ${
                selected
                  ? "border-[#8a9a7b] bg-[#182015]"
                  : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-900"
              }`}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm font-semibold text-zinc-100">{option.label}</p>
                  <p className="mt-1 text-xs text-zinc-400">{option.description}</p>
                </div>
                {selected && <CheckCircle2 className="h-5 w-5 shrink-0 text-[#a3b396]" />}
              </div>
              <p className="mt-3 text-xs text-zinc-500">{roles.join(", ")}</p>
            </button>
          );
        })}
      </div>
    );
  }

  function renderUtilityPromptStep() {
    return (
      <div className="space-y-4 rounded-2xl border border-zinc-800 bg-zinc-900/50 p-5">
        <div>
          <p className="text-sm font-semibold text-zinc-100">The primary model does not cover everything yet.</p>
          <p className="mt-1 text-sm text-zinc-400">
            Remaining roles: <span className="text-zinc-200">{missingRoles.join(", ")}</span>
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => {
              const nextProvider = utilityDraft
                ? providerMap.get(utilityDraft.provider) ?? status.providers[0]
                : status.providers[0];
              setUtilityDraft((current) => ({
                ...(current ?? createDraft("utility", nextProvider, missingRoles)),
                name: "utility",
                roles: missingRoles,
              }));
              setActiveTarget("utility");
              setStep("provider");
            }}
            className="rounded-lg bg-[#6b7a5e] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#8a9a7b]"
          >
            Add utility model
          </button>
          <button
            type="button"
            onClick={() => {
              setUtilityDraft(null);
              setStep("review");
            }}
            className="rounded-lg border border-zinc-700 px-4 py-2 text-sm font-medium text-zinc-300 transition-colors hover:border-zinc-600 hover:text-zinc-100"
          >
            Use the primary model for everything
          </button>
        </div>
      </div>
    );
  }

  function renderReviewCard(draft: ModelDraftState) {
    const provider = providerMap.get(draft.provider);
    return (
      <div key={draft.name} className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold text-zinc-100">{titleForTarget(draft.name)}</p>
            <p className="text-xs text-zinc-500">{provider?.display_name ?? draft.provider}</p>
          </div>
          <span className="rounded-full bg-[#182015] px-2.5 py-1 text-[11px] font-medium text-[#c7d3bb]">
            {draft.roles.join(", ")}
          </span>
        </div>
        <div className="mt-4 space-y-2 text-sm text-zinc-300">
          <p><span className="text-zinc-500">Endpoint:</span> {draft.base_url}</p>
          <p><span className="text-zinc-500">Model:</span> {draft.model}</p>
          <p><span className="text-zinc-500">Temperature:</span> {draft.temperature}</p>
          <p><span className="text-zinc-500">Max tokens:</span> {draft.max_tokens}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#09090b] px-6 py-10 text-zinc-100">
      <div className="w-full max-w-4xl overflow-hidden rounded-[28px] border border-zinc-800 bg-[#111114] shadow-2xl">
        <div className="grid gap-0 lg:grid-cols-[280px_1fr]">
          <aside className="border-b border-zinc-800 bg-[radial-gradient(circle_at_top,_rgba(163,179,150,0.18),_transparent_45%),linear-gradient(180deg,_rgba(24,32,21,0.92),_rgba(17,17,20,0.98))] p-6 lg:border-b-0 lg:border-r">
            <p className="text-xs font-semibold uppercase tracking-[0.28em] text-[#a3b396]">First-run setup</p>
            <h1 className="mt-3 text-2xl font-semibold text-zinc-50">Connect Loom to your models</h1>
            <p className="mt-3 text-sm leading-6 text-zinc-300">
              We’ll write your first <span className="font-mono text-zinc-100">loom.toml</span>,
              check the model endpoint, and make the desktop app ready to use right away.
            </p>

            <div className="mt-6 space-y-3">
              {[
                { key: "provider", label: "Choose provider" },
                { key: "connection", label: "Connect and discover" },
                { key: "roles", label: "Assign roles" },
                { key: "review", label: "Review and save" },
              ].map((item) => {
                const active = (
                  item.key === "provider" && step === "provider"
                ) || (
                  item.key === "connection" && step === "connection"
                ) || (
                  item.key === "roles" && (step === "roles" || step === "utility_prompt")
                ) || (
                  item.key === "review" && step === "review"
                );
                return (
                  <div
                    key={item.key}
                    className={`rounded-xl border px-3 py-3 ${
                      active ? "border-[#8a9a7b]/60 bg-black/20" : "border-white/5 bg-black/10"
                    }`}
                  >
                    <p className={`text-sm ${active ? "text-zinc-100" : "text-zinc-400"}`}>{item.label}</p>
                  </div>
                );
              })}
            </div>

            <div className="mt-6 rounded-2xl border border-white/5 bg-black/15 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Config path</p>
              <p className="mt-2 break-all font-mono text-xs text-zinc-200">{status.config_path}</p>
            </div>
          </aside>

          <section className="p-6 md:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-zinc-500">
                  {titleForTarget(activeTarget)}
                </p>
                <h2 className="mt-2 text-2xl font-semibold text-zinc-50">{stepTitle(step, activeTarget)}</h2>
                <p className="mt-2 text-sm text-zinc-400">
                  {step === "provider" && "Pick the service Loom should talk to for this model."}
                  {step === "connection" && "Give Loom the endpoint details and choose the exact model to use."}
                  {step === "roles" && "Start simple: one model can do everything, or you can split primary and utility work."}
                  {step === "utility_prompt" && "Loom can keep one model for deep work and another for cheaper utility tasks."}
                  {step === "review" && "This will write your initial configuration and immediately refresh the desktop runtime."}
                </p>
              </div>
              {step !== "provider" && (
                <button
                  type="button"
                  onClick={handleBack}
                  className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 px-3 py-2 text-sm font-medium text-zinc-300 transition-colors hover:border-zinc-600 hover:text-zinc-100"
                >
                  <ChevronLeft className="h-4 w-4" />
                  Back
                </button>
              )}
            </div>

            {error && (
              <div className="mt-5 rounded-xl border border-red-500/25 bg-red-950/40 px-4 py-3 text-sm text-red-200">
                {error}
              </div>
            )}

            <div className="mt-6">
              {step === "provider" && renderProviderStep()}
              {step === "connection" && renderConnectionStep()}
              {step === "roles" && renderRolesStep()}
              {step === "utility_prompt" && renderUtilityPromptStep()}
              {step === "review" && (
                <div className="space-y-4">
                  {renderReviewCard(primaryDraft)}
                  {utilityDraft && renderReviewCard(utilityDraft)}
                </div>
              )}
            </div>

            <div className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-zinc-800 pt-6">
              <p className="text-sm text-zinc-500">
                {step === "review"
                  ? "You can revisit this later from the CLI with `loom setup`."
                  : "You can revise any of this before saving."}
              </p>
              {step !== "utility_prompt" && step !== "review" && (
                <button
                  type="button"
                  onClick={handleContinue}
                  className="rounded-lg bg-[#6b7a5e] px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#8a9a7b]"
                >
                  Continue
                </button>
              )}
              {step === "review" && (
                <button
                  type="button"
                  onClick={() => void handleSave()}
                  disabled={saving}
                  className="inline-flex items-center gap-2 rounded-lg bg-[#6b7a5e] px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-[#8a9a7b] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
                  Save and start Loom
                </button>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
