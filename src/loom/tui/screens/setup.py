"""TUI setup wizard — multi-step modal for first-run configuration."""

from __future__ import annotations

import asyncio
import logging

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from loom.setup import (
    CONFIG_DIR,
    CONFIG_PATH,
    PROVIDERS,
    ROLE_PRESETS,
    _generate_toml,
    discover_models,
)
from loom.utils.latency import log_latency_event

logger = logging.getLogger(__name__)

# Step indices
_STEP_PROVIDER = 0
_STEP_DETAILS = 1
_STEP_ROLES = 2
_STEP_UTILITY = 3
_STEP_CONFIRM = 4

_TOTAL_STEPS = 5
_MAX_VISIBLE_DISCOVERED = 6
_MAX_MODEL_LABEL_CHARS = 54


class SetupScreen(ModalScreen[list[dict] | None]):
    """Multi-step setup wizard running inside the TUI.

    Dismisses with a list of model config dicts on success, or None if
    the user cancels.
    """

    BINDINGS = [
        Binding("escape", "back_or_cancel", "Back", show=True),
    ]

    CSS = """
    SetupScreen {
        align: center middle;
    }

    #setup-dialog {
        width: 72;
        min-height: 16;
        max-height: 36;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }

    .step-container {
        height: auto;
    }

    .setup-heading {
        text-style: bold;
        margin-bottom: 1;
    }

    .setup-divider {
        color: $primary;
        margin-bottom: 1;
    }

    .setup-option {
        margin-left: 2;
    }

    .setup-hint {
        color: $primary;
        text-style: bold;
        margin-top: 1;
    }

    .setup-input {
        margin-top: 1;
        margin-bottom: 1;
    }

    .setup-summary {
        margin: 1 0;
        padding: 1;
        background: $panel;
        max-height: 10;
        overflow-y: auto;
    }

    .setup-selection {
        color: $success;
        text-style: bold;
        margin-top: 1;
    }

    #discovery-results {
        max-height: 8;
        overflow-y: auto;
        background: $panel;
        padding: 0 1;
    }

    #btn-discover {
        margin-top: 1;
        width: 100%;
    }

    #btn-next {
        margin-top: 1;
        width: 100%;
    }
    """

    _step: reactive[int] = reactive(0)

    def __init__(self) -> None:
        super().__init__()
        # Collected state
        self._provider_idx: int = -1
        self._provider_key: str = ""
        self._base_url: str = ""
        self._model_name: str = ""
        self._api_key: str = ""
        self._roles: list[str] = []
        # P0-3: Explicit primary/utility drafts instead of append-only list
        self._primary_model: dict | None = None
        self._utility_model: dict | None = None
        self._adding_utility: bool = False
        self._discovered_models: list[str] = []
        self._discover_request_id: int = 0
        self._discover_inflight: bool = False

    @property
    def _models(self) -> list[dict]:
        """Build the models list from the primary/utility drafts."""
        result = []
        if self._primary_model is not None:
            result.append(self._primary_model)
        if self._utility_model is not None:
            result.append(self._utility_model)
        return result

    def compose(self) -> ComposeResult:
        with Vertical(id="setup-dialog"):
            # -- Step 0: Provider selection --
            with Vertical(id="step-provider", classes="step-container"):
                yield Label("Loom Setup", classes="setup-heading")
                yield Static("─" * 40, classes="setup-divider")
                yield Label(
                    "Choose your model provider:",
                    id="provider-prompt",
                )
                yield Label("")
                for i, (display, _key, _needs, _url) in enumerate(PROVIDERS, 1):
                    yield Label(
                        f"  [{i}] {display}",
                        classes="setup-option",
                    )
                yield Label(
                    f"Press 1-{len(PROVIDERS)} to select",
                    classes="setup-hint",
                )

            # -- Step 1: Provider details --
            with Vertical(id="step-details", classes="step-container"):
                yield Label("Configuration", classes="setup-heading")
                yield Static("─" * 40, classes="setup-divider")
                yield Label("", id="details-provider-label")

                yield Label("Base URL:", id="lbl-url")
                yield Input(
                    placeholder="http://localhost:11434",
                    id="input-url",
                    classes="setup-input",
                )
                yield Label("API Key:", id="lbl-apikey")
                yield Input(
                    placeholder="sk-ant-...",
                    id="input-apikey",
                    password=True,
                    classes="setup-input",
                )
                yield Button(
                    "Discover Models",
                    id="btn-discover",
                    variant="default",
                )
                yield Static(
                    "Press Discover Models to query available models.",
                    id="discovery-results",
                    classes="setup-hint",
                )
                yield Static(
                    "No model selected yet.",
                    id="selection-feedback",
                    classes="setup-selection",
                )
                yield Label("Model name:", id="lbl-model")
                yield Input(
                    placeholder="e.g. qwen3:14b",
                    id="input-model",
                    classes="setup-input",
                )
                yield Button("Next", id="btn-next", variant="primary")

            # -- Step 2: Role selection --
            with Vertical(id="step-roles", classes="step-container"):
                yield Label("Model Roles", classes="setup-heading")
                yield Static("─" * 40, classes="setup-divider")
                yield Label("Which roles should this model handle?")
                yield Label("")
                yield Label(
                    "  [1] All roles (planner, executor, extractor, verifier, compactor)",
                    classes="setup-option",
                )
                yield Label(
                    "  [2] Primary (planner, executor)",
                    classes="setup-option",
                )
                yield Label(
                    "  [3] Utility (extractor, verifier, compactor)",
                    classes="setup-option",
                )
                yield Label("Press 1, 2, or 3", classes="setup-hint")

            # -- Step 3: Add utility model? --
            with Vertical(id="step-utility", classes="step-container"):
                yield Label("Add Utility Model?", classes="setup-heading")
                yield Static("─" * 40, classes="setup-divider")
                yield Label("", id="lbl-missing-roles")
                yield Label("")
                yield Label(
                    "  [y] Add a second model for uncovered roles",
                    classes="setup-option",
                )
                yield Label(
                    "  [n] Skip — use primary for everything",
                    classes="setup-option",
                )
                yield Label("Press Y to add, N to skip", classes="setup-hint")

            # -- Step 4: Confirm --
            with Vertical(id="step-confirm", classes="step-container"):
                yield Label("Ready to Save", classes="setup-heading")
                yield Static("─" * 40, classes="setup-divider")
                yield Static("", id="confirm-summary", classes="setup-summary")
                yield Label(f"Config: {CONFIG_PATH}")
                yield Label("")
                yield Label(
                    "  [Enter/Y/S] Save and start    [B/Esc] Back",
                    classes="setup-hint",
                )

    def on_mount(self) -> None:
        self._show_step(0)

    def watch__step(self, step: int) -> None:
        self._show_step(step)

    def _show_step(self, step: int) -> None:
        """Show the container for the given step, hide all others."""
        step_ids = [
            "step-provider",
            "step-details",
            "step-roles",
            "step-utility",
            "step-confirm",
        ]
        for i, sid in enumerate(step_ids):
            try:
                container = self.query_one(f"#{sid}")
                container.display = i == step
            except Exception:
                pass

        # Update dynamic labels
        if step == _STEP_PROVIDER:
            try:
                prompt = self.query_one("#provider-prompt", Label)
                if self._adding_utility:
                    prompt.update("Utility model provider:")
                else:
                    prompt.update("Choose your model provider:")
            except Exception:
                pass

        # P0-1: Disable/enable inputs based on current step to prevent
        # hidden inputs from swallowing keypresses.
        if step == _STEP_DETAILS:
            self._configure_details_step()
        else:
            self._disable_all_inputs()
            self.set_focus(None)

    def _disable_all_inputs(self) -> None:
        """Disable all setup inputs so they cannot capture focus or keys."""
        for input_id in ("input-url", "input-model", "input-apikey"):
            try:
                inp = self.query_one(f"#{input_id}", Input)
                inp.disabled = True
            except Exception:
                pass

    def _enable_input(self, input_id: str) -> Input:
        """Enable a specific input widget and return it."""
        inp = self.query_one(f"#{input_id}", Input)
        inp.disabled = False
        return inp

    def _provider_requires_api_key(self) -> bool:
        """Return whether the selected provider needs an API key."""
        return self._provider_key == "anthropic"

    def _can_discover(self) -> bool:
        """Return True when required fields are present for discovery."""
        base_url = self.query_one("#input-url", Input).value.strip()
        if not base_url:
            return False
        if self._provider_requires_api_key():
            return bool(self.query_one("#input-apikey", Input).value.strip())
        return True

    @staticmethod
    def _truncate_model_name(name: str) -> str:
        """Limit rendered model labels so discovery UI stays compact."""
        if len(name) <= _MAX_MODEL_LABEL_CHARS:
            return name
        return name[:_MAX_MODEL_LABEL_CHARS - 3] + "..."

    def _set_selection_feedback(self, message: str) -> None:
        """Update the prominent selection feedback text."""
        self.query_one("#selection-feedback", Static).update(message)

    def _update_discovery_button(self) -> None:
        """Enable discover button only when discovery prerequisites are met."""
        button = self.query_one("#btn-discover", Button)
        if self._discover_inflight:
            button.label = "Discovering..."
            button.disabled = True
            return
        button.label = "Discover Models"
        button.disabled = not self._can_discover()

    def _configure_details_step(self) -> None:
        """Show/hide detail fields based on selected provider."""
        is_anthropic = self._provider_key == "anthropic"
        is_openai = self._provider_key == "openai_compatible"

        _, _, _, default_url = PROVIDERS[self._provider_idx]
        provider_display = PROVIDERS[self._provider_idx][0]

        self.query_one("#details-provider-label", Label).update(
            f"Provider: {provider_display}"
        )
        self._discover_request_id += 1
        self._discover_inflight = False
        self._discovered_models = []
        self._set_selection_feedback("No model selected yet.")

        # P0-1: Start with all inputs disabled, then selectively enable
        self._disable_all_inputs()

        # URL input
        self.query_one("#lbl-url").display = True
        url_input = self._enable_input("input-url")
        url_input.display = True
        url_input.value = default_url

        # Model name input
        self.query_one("#lbl-model").display = True
        model_input = self._enable_input("input-model")
        model_input.display = True
        model_input.value = ""
        if is_anthropic:
            model_input.placeholder = "e.g. claude-sonnet-4-5-20250929"
        elif is_openai:
            model_input.placeholder = "e.g. gpt-4o, mistral-nemo"
        else:
            model_input.placeholder = "e.g. qwen3:14b, llama3:8b"

        # API key
        needs_key = is_anthropic
        self.query_one("#lbl-apikey").display = needs_key or is_openai
        apikey_input = self.query_one("#input-apikey", Input)
        apikey_input.display = needs_key or is_openai
        apikey_input.value = ""
        apikey_input.disabled = True
        if needs_key or is_openai:
            apikey_input.disabled = False
            if is_anthropic:
                apikey_input.placeholder = "sk-ant-..."
            else:
                apikey_input.placeholder = "optional"

        self._render_discovered_models([])
        self._update_discovery_button()
        url_input.focus()

    def _render_discovered_models(self, models: list[str]) -> None:
        """Render model discovery results in the details step."""
        panel = self.query_one("#discovery-results", Static)
        self._update_discovery_button()
        if not models:
            if self._can_discover():
                panel.update(
                    "Press Discover Models to query available models, "
                    "or type a model name manually."
                )
            elif self._provider_requires_api_key():
                panel.update("Enter Base URL + API key before discovery.")
            else:
                panel.update("Enter Base URL before discovery.")
            return

        visible = models[:_MAX_VISIBLE_DISCOVERED]
        lines = ["Discovered models:"]
        for i, model in enumerate(visible, 1):
            lines.append(
                f"  [{i}] {self._truncate_model_name(model)}"
            )
        if len(models) > len(visible):
            lines.append(
                f"  ... {len(models) - len(visible)} more "
                f"(type model name manually)"
            )
        lines.append(
            f"Press 1-{len(visible)} to pick, or edit Model name."
        )
        panel.update("\n".join(lines))

    def _discover_models(self, *, quiet: bool = False) -> list[str]:
        """Attempt endpoint model discovery and refresh the details UI."""
        base_url = self.query_one("#input-url", Input).value.strip()
        api_key = self.query_one("#input-apikey", Input).value.strip()

        if not base_url:
            self._discovered_models = []
            self._render_discovered_models([])
            if not quiet:
                self.notify("Base URL is required for discovery.", severity="warning")
            return []
        if self._provider_key == "anthropic" and not api_key:
            self._discovered_models = []
            self._render_discovered_models([])
            if not quiet:
                self.notify(
                    "Anthropic API key is required for discovery.",
                    severity="warning",
                )
            return []

        models = discover_models(self._provider_key, base_url, api_key)
        self._discovered_models = models
        self._render_discovered_models(models)

        if models:
            model_input = self.query_one("#input-model", Input)
            if not model_input.value.strip():
                model_input.value = models[0]
                self._model_name = models[0]
            selected = model_input.value.strip()
            if selected:
                self._set_selection_feedback(f"Selected model: {selected}")
            if not quiet:
                self.notify(
                    f"Discovered {len(models)} models.",
                    severity="information",
                )
        elif not quiet:
            self._set_selection_feedback("No model discovered. Use manual model name.")
            self.notify(
                "No models discovered; enter model name manually.",
                severity="warning",
            )
        return models

    def _start_discover_models(
        self,
        *,
        quiet: bool = False,
        focus_model_after: bool = False,
    ) -> None:
        """Launch model discovery in a background worker."""
        if self._discover_inflight:
            return

        base_url = self.query_one("#input-url", Input).value.strip()
        api_key = self.query_one("#input-apikey", Input).value.strip()

        if not base_url:
            self._discovered_models = []
            self._render_discovered_models([])
            if not quiet:
                self.notify("Base URL is required for discovery.", severity="warning")
            return
        if self._provider_key == "anthropic" and not api_key:
            self._discovered_models = []
            self._render_discovered_models([])
            if not quiet:
                self.notify(
                    "Anthropic API key is required for discovery.",
                    severity="warning",
                )
            return

        self._discover_request_id += 1
        request_id = self._discover_request_id
        provider_key = self._provider_key
        self._discover_inflight = True
        self._set_selection_feedback("Discovering models...")
        self._update_discovery_button()
        self.query_one("#discovery-results", Static).update(
            "Discovering models from endpoint..."
        )

        async def _discover_worker() -> None:
            started = asyncio.get_running_loop().time()
            try:
                models = await asyncio.to_thread(
                    discover_models,
                    provider_key,
                    base_url,
                    api_key,
                )
            except Exception:
                models = []
            if request_id != self._discover_request_id:
                return
            try:
                self._discovered_models = models
                self._render_discovered_models(models)
                if models:
                    model_input = self.query_one("#input-model", Input)
                    if not model_input.value.strip():
                        model_input.value = models[0]
                        self._model_name = models[0]
                    selected = model_input.value.strip()
                    if selected:
                        self._set_selection_feedback(f"Selected model: {selected}")
                    if focus_model_after:
                        model_input.focus()
                    if not quiet:
                        self.notify(
                            f"Discovered {len(models)} models.",
                            severity="information",
                        )
                else:
                    if focus_model_after:
                        self.query_one("#input-model", Input).focus()
                    if not quiet:
                        self._set_selection_feedback(
                            "No model discovered. Use manual model name."
                        )
                        self.notify(
                            "No models discovered; enter model name manually.",
                            severity="warning",
                        )
            finally:
                if request_id == self._discover_request_id:
                    self._discover_inflight = False
                    self._update_discovery_button()
                    log_latency_event(
                        logger,
                        event="setup_model_discovery",
                        duration_seconds=asyncio.get_running_loop().time() - started,
                        fields={"provider": provider_key, "models": len(self._discovered_models)},
                    )

        self.run_worker(
            _discover_worker(),
            group="setup-model-discovery",
            exclusive=False,
        )

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def on_key(self, event) -> None:
        # P2-9: Use event.key for normalized key names
        key = event.key

        if (
            self._step == _STEP_PROVIDER
            and key.isdigit()
            and 1 <= int(key) <= len(PROVIDERS)
        ):
            idx = int(key) - 1
            self._provider_idx = idx
            self._provider_key = PROVIDERS[idx][1]
            self._step = _STEP_DETAILS
            self.notify(
                f"Selected provider: {PROVIDERS[idx][0]}",
                severity="information",
            )
            event.prevent_default()
            event.stop()

        elif (
            self._step == _STEP_DETAILS
            and key.isdigit()
            and self._discovered_models
            and not isinstance(self.focused, Input)
        ):
            num = int(key)
            max_selectable = min(
                len(self._discovered_models), _MAX_VISIBLE_DISCOVERED,
            )
            if 1 <= num <= max_selectable:
                selected = self._discovered_models[num - 1]
                self._model_name = selected
                model_input = self.query_one("#input-model", Input)
                model_input.value = selected
                model_input.focus()
                self._set_selection_feedback(f"Selected [{num}] {selected}")
                self.notify(f"Selected model: {selected}", severity="information")
                event.prevent_default()
                event.stop()

        elif self._step == _STEP_ROLES and key in ("1", "2", "3"):
            presets = ["all", "primary", "utility"]
            self._roles = list(ROLE_PRESETS[presets[int(key) - 1]])
            role_labels = {
                "1": "all roles",
                "2": "primary roles",
                "3": "utility roles",
            }
            self.notify(
                f"Selected {role_labels[key]}",
                severity="information",
            )
            self._collect_model()
            event.prevent_default()
            event.stop()

        elif self._step == _STEP_UTILITY and key in ("y", "Y"):
            self._adding_utility = True
            self._provider_idx = -1
            self._provider_key = ""
            self._base_url = ""
            self._model_name = ""
            self._api_key = ""
            self._discovered_models = []
            self._step = _STEP_PROVIDER
            self.notify(
                "Adding utility model. Select provider.",
                severity="information",
            )
            event.prevent_default()
            event.stop()

        elif self._step == _STEP_UTILITY and key in ("n", "N"):
            self.notify(
                "Skipping utility model.",
                severity="information",
            )
            self._prepare_confirm()
            event.prevent_default()
            event.stop()

        elif self._step == _STEP_CONFIRM and key in ("enter", "y", "Y", "s", "S"):
            self._save_and_dismiss()
            event.prevent_default()
            event.stop()

        elif self._step == _STEP_CONFIRM and key in ("b", "B"):
            self.action_back_or_cancel()
            event.prevent_default()
            event.stop()

    @on(Button.Pressed, "#btn-next")
    def on_next_pressed(self) -> None:
        """Advance from details step to roles step."""
        if self._step != _STEP_DETAILS:
            return
        self._collect_details()

    @on(Button.Pressed, "#btn-discover")
    def on_discover_pressed(self) -> None:
        """Attempt model discovery with the current endpoint fields."""
        if self._step != _STEP_DETAILS:
            return
        self._start_discover_models()

    @on(Input.Changed, "#input-url")
    @on(Input.Changed, "#input-apikey")
    def on_discovery_fields_changed(self, _event: Input.Changed) -> None:
        """Invalidate discovery when endpoint/auth changes."""
        if self._step != _STEP_DETAILS:
            return
        if self._discover_inflight:
            self._discover_request_id += 1
            self._discover_inflight = False
        if self._discovered_models:
            self._discovered_models = []
            self._set_selection_feedback(
                "Connection changed. Discover models again.",
            )
        self._render_discovered_models([])

    @on(Input.Changed, "#input-model")
    def on_model_input_changed(self, _event: Input.Changed) -> None:
        """Show visible feedback while model name is being edited."""
        if self._step != _STEP_DETAILS:
            return
        model = self.query_one("#input-model", Input).value.strip()
        if model:
            self._set_selection_feedback(f"Using model: {model}")
        else:
            self._set_selection_feedback("No model selected yet.")

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in input fields — advance to next field or step."""
        if self._step != _STEP_DETAILS:
            return

        # Stop the event from bubbling to the app-level handler
        event.stop()

        input_id = event.input.id

        if input_id == "input-url":
            apikey_input = self.query_one("#input-apikey", Input)
            if apikey_input.display:
                apikey_input.focus()
            else:
                self._start_discover_models(quiet=True, focus_model_after=True)
        elif input_id == "input-apikey":
            self._start_discover_models(quiet=True, focus_model_after=True)
        elif input_id == "input-model":
            self._collect_details()

    def _collect_details(self) -> None:
        """Collect values from detail inputs and advance to roles."""
        self._base_url = self.query_one("#input-url", Input).value.strip()
        self._api_key = self.query_one("#input-apikey", Input).value.strip()
        self._model_name = self.query_one("#input-model", Input).value.strip()

        # Validate
        if not self._base_url:
            self.notify("Base URL is required.", severity="error")
            self.query_one("#input-url", Input).focus()
            return
        if self._provider_key == "anthropic" and not self._api_key:
            self.notify("API key is required for Anthropic.", severity="error")
            self.query_one("#input-apikey", Input).focus()
            return
        if not self._model_name:
            if self._discovered_models:
                self._model_name = self._discovered_models[0]
                self.query_one("#input-model", Input).value = self._model_name
                self._set_selection_feedback(
                    f"Auto-selected discovered model: {self._model_name}",
                )
                self.notify(
                    f"Auto-selected discovered model: {self._model_name}",
                    severity="information",
                )
            else:
                if self._discover_inflight:
                    self.notify(
                        "Model discovery is in progress. Wait for results or enter a model name.",
                        severity="warning",
                    )
                else:
                    self.notify(
                        "Model name is required (discover or type manually).",
                        severity="error",
                    )
                self.query_one("#input-model", Input).focus()
                return

        self._step = _STEP_ROLES

    def _collect_model(self) -> None:
        """Package current inputs into a model dict and decide next step."""
        name = "utility" if self._adding_utility else "primary"
        model = {
            "name": name,
            "provider": self._provider_key,
            "base_url": self._base_url,
            "model": self._model_name,
            "api_key": self._api_key,
            "roles": self._roles,
            "max_tokens": 2048 if self._adding_utility else 8192,
            "temperature": 0.0 if self._adding_utility else 0.1,
        }

        # P0-3: Replace draft instead of appending
        if self._adding_utility:
            self._utility_model = model
            self._prepare_confirm()
        else:
            self._primary_model = model
            # Check if all roles covered
            missing = set(ROLE_PRESETS["all"]) - set(self._roles)
            if missing:
                lbl = self.query_one("#lbl-missing-roles", Label)
                lbl.update(
                    f"Uncovered roles: {', '.join(sorted(missing))}"
                )
                self._step = _STEP_UTILITY
            else:
                self._prepare_confirm()

    def _prepare_confirm(self) -> None:
        """Build confirmation summary and show confirm step."""
        models = self._models

        # P0-4: Validate that at least one model has executor role
        has_executor = any("executor" in m["roles"] for m in models)
        if not has_executor:
            self.notify(
                "At least one model must have the executor role.",
                severity="error",
            )
            # Go back to roles step to fix
            self._step = _STEP_ROLES
            return

        lines = []
        for m in models:
            roles_str = ", ".join(m["roles"])
            lines.append(
                f"  {m['name']}: {m['model']} ({m['provider']})"
            )
            lines.append(f"    roles: {roles_str}")
            lines.append(f"    url: {m['base_url']}")
        summary = "\n".join(lines)

        self.query_one("#confirm-summary", Static).update(summary)
        self._step = _STEP_CONFIRM

    def _save_and_dismiss(self) -> None:
        """Write config and dismiss with the model list."""
        models = self._models

        # Final validation: ensure unique model names and executor coverage
        names = [m["name"] for m in models]
        if len(names) != len(set(names)):
            self.notify("Duplicate model names detected.", severity="error")
            return

        has_executor = any("executor" in m["roles"] for m in models)
        if not has_executor:
            self.notify(
                "At least one model must have the executor role.",
                severity="error",
            )
            return

        toml_content = _generate_toml(models)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(toml_content)

        # Create supporting dirs
        (CONFIG_DIR / "scratch").mkdir(parents=True, exist_ok=True)
        (CONFIG_DIR / "logs").mkdir(parents=True, exist_ok=True)
        (CONFIG_DIR / "processes").mkdir(parents=True, exist_ok=True)

        self.dismiss(models)

    def action_back_or_cancel(self) -> None:
        """Go back one step or cancel if on the first step."""
        if self._step == _STEP_PROVIDER:
            self.dismiss(None)
        elif self._step == _STEP_DETAILS:
            self._step = _STEP_PROVIDER
        elif self._step == _STEP_ROLES:
            self._step = _STEP_DETAILS
        elif self._step == _STEP_UTILITY:
            # P0-3: Going back from utility clears the primary draft
            # so re-selection replaces rather than duplicates
            self._step = _STEP_ROLES
        elif self._step == _STEP_CONFIRM:
            if self._utility_model is not None:
                # Go back to utility prompt, clear utility draft
                self._utility_model = None
                self._adding_utility = False
                self._step = _STEP_UTILITY
            elif self._primary_model is not None:
                missing = set(ROLE_PRESETS["all"]) - set(
                    self._primary_model["roles"]
                )
                if missing:
                    self._step = _STEP_UTILITY
                else:
                    self._primary_model = None
                    self._step = _STEP_ROLES
            else:
                self._step = _STEP_ROLES
