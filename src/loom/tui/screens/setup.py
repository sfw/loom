"""TUI setup wizard — multi-step modal for first-run configuration."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from loom.setup import (
    ANTHROPIC_MODELS,
    CONFIG_DIR,
    CONFIG_PATH,
    PROVIDERS,
    ROLE_PRESETS,
    _generate_toml,
)

# Step indices
_STEP_PROVIDER = 0
_STEP_DETAILS = 1
_STEP_ROLES = 2
_STEP_UTILITY = 3
_STEP_CONFIRM = 4

_TOTAL_STEPS = 5


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
        width: 64;
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
        color: $text-muted;
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
        self._models: list[dict] = []
        self._adding_utility: bool = False

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

                yield Label("Model:", id="lbl-model-select")
                for i, m in enumerate(ANTHROPIC_MODELS, 1):
                    yield Label(
                        f"  [{i}] {m}",
                        id=f"lbl-anthropic-{i}",
                        classes="setup-option",
                    )
                yield Label(
                    "Press 1-3 to select model",
                    id="lbl-anthropic-hint",
                    classes="setup-hint",
                )

                yield Label("Base URL:", id="lbl-url")
                yield Input(
                    placeholder="http://localhost:11434",
                    id="input-url",
                    classes="setup-input",
                )
                yield Label("Model name:", id="lbl-model")
                yield Input(
                    placeholder="e.g. qwen3:14b",
                    id="input-model",
                    classes="setup-input",
                )
                yield Label("API Key:", id="lbl-apikey")
                yield Input(
                    placeholder="sk-ant-...",
                    id="input-apikey",
                    password=True,
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
                    "  [1] All roles (planner, executor, extractor, verifier)",
                    classes="setup-option",
                )
                yield Label(
                    "  [2] Primary (planner, executor)",
                    classes="setup-option",
                )
                yield Label(
                    "  [3] Utility (extractor, verifier)",
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

            # -- Step 4: Confirm --
            with Vertical(id="step-confirm", classes="step-container"):
                yield Label("Ready to Save", classes="setup-heading")
                yield Static("─" * 40, classes="setup-divider")
                yield Static("", id="confirm-summary", classes="setup-summary")
                yield Label(f"Config: {CONFIG_PATH}")
                yield Label("")
                yield Label(
                    "  [Enter] Save and start    [Esc] Cancel",
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

        # Focus the first input on the details step
        if step == _STEP_DETAILS:
            self._configure_details_step()

    def _configure_details_step(self) -> None:
        """Show/hide detail fields based on selected provider."""
        is_anthropic = self._provider_key == "anthropic"
        is_ollama = self._provider_key == "ollama"
        is_openai = self._provider_key == "openai_compatible"

        _, _, _, default_url = PROVIDERS[self._provider_idx]
        provider_display = PROVIDERS[self._provider_idx][0]

        self.query_one("#details-provider-label", Label).update(
            f"Provider: {provider_display}"
        )

        # Anthropic model selector
        self.query_one("#lbl-model-select").display = is_anthropic
        for i in range(1, len(ANTHROPIC_MODELS) + 1):
            self.query_one(f"#lbl-anthropic-{i}").display = is_anthropic
        self.query_one("#lbl-anthropic-hint").display = is_anthropic

        # URL input
        self.query_one("#lbl-url").display = True
        url_input = self.query_one("#input-url", Input)
        url_input.display = True
        url_input.value = default_url

        # Model name input (non-Anthropic)
        self.query_one("#lbl-model").display = not is_anthropic
        model_input = self.query_one("#input-model", Input)
        model_input.display = not is_anthropic
        model_input.value = ""

        # API key
        needs_key = is_anthropic
        self.query_one("#lbl-apikey").display = needs_key or is_openai
        apikey_input = self.query_one("#input-apikey", Input)
        apikey_input.display = needs_key or is_openai
        apikey_input.value = ""

        # Focus the right field
        if is_anthropic:
            # Wait for anthropic model number key
            pass
        else:
            url_input.focus()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def on_key(self, event) -> None:
        key = event.character

        if (
            self._step == _STEP_PROVIDER
            and key
            and key.isdigit()
            and 1 <= int(key) <= len(PROVIDERS)
        ):
            idx = int(key) - 1
            self._provider_idx = idx
            self._provider_key = PROVIDERS[idx][1]
            self._step = _STEP_DETAILS
            event.prevent_default()
            event.stop()

        elif self._step == _STEP_DETAILS and self._provider_key == "anthropic":
            if key and key.isdigit():
                num = int(key)
                if 1 <= num <= len(ANTHROPIC_MODELS):
                    self._model_name = ANTHROPIC_MODELS[num - 1]
                    # Focus the API key input
                    self.query_one("#input-apikey", Input).focus()
                    event.prevent_default()
                    event.stop()

        elif self._step == _STEP_ROLES and key in ("1", "2", "3"):
            presets = ["all", "primary", "utility"]
            self._roles = list(ROLE_PRESETS[presets[int(key) - 1]])
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
            self._step = _STEP_PROVIDER
            event.prevent_default()
            event.stop()

        elif self._step == _STEP_UTILITY and key in ("n", "N"):
            self._prepare_confirm()
            event.prevent_default()
            event.stop()

        elif self._step == _STEP_CONFIRM and event.key == "enter":
            self._save_and_dismiss()
            event.prevent_default()
            event.stop()

    @on(Button.Pressed, "#btn-next")
    def on_next_pressed(self) -> None:
        """Advance from details step to roles step."""
        if self._step != _STEP_DETAILS:
            return
        self._collect_details()

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in input fields — advance to next field or step."""
        if self._step != _STEP_DETAILS:
            return

        input_id = event.input.id

        if self._provider_key == "anthropic":
            if input_id == "input-apikey":
                # API key entered, try to advance
                self._collect_details()
            elif input_id == "input-url":
                self.query_one("#input-apikey", Input).focus()
        elif self._provider_key == "openai_compatible":
            if input_id == "input-url":
                self.query_one("#input-model", Input).focus()
            elif input_id == "input-model":
                apikey_input = self.query_one("#input-apikey", Input)
                if apikey_input.display:
                    apikey_input.focus()
                else:
                    self._collect_details()
            elif input_id == "input-apikey":
                self._collect_details()
        else:  # ollama
            if input_id == "input-url":
                self.query_one("#input-model", Input).focus()
            elif input_id == "input-model":
                self._collect_details()

    def _collect_details(self) -> None:
        """Collect values from detail inputs and advance to roles."""
        self._base_url = self.query_one("#input-url", Input).value.strip()
        self._api_key = self.query_one("#input-apikey", Input).value.strip()

        if self._provider_key != "anthropic":
            self._model_name = self.query_one(
                "#input-model", Input,
            ).value.strip()

        # Validate
        if not self._model_name:
            self.notify("Model name is required.", severity="error")
            return
        if self._provider_key == "anthropic" and not self._api_key:
            self.notify("API key is required for Anthropic.", severity="error")
            self.query_one("#input-apikey", Input).focus()
            return
        if not self._base_url:
            self.notify("Base URL is required.", severity="error")
            self.query_one("#input-url", Input).focus()
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
            "max_tokens": 2048 if self._adding_utility else 4096,
            "temperature": 0.0 if self._adding_utility else 0.1,
        }
        self._models.append(model)

        if self._adding_utility:
            # Done collecting, go to confirm
            self._prepare_confirm()
        else:
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
        lines = []
        for m in self._models:
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
        toml_content = _generate_toml(self._models)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(toml_content)

        # Create supporting dirs
        (CONFIG_DIR / "scratch").mkdir(parents=True, exist_ok=True)
        (CONFIG_DIR / "logs").mkdir(parents=True, exist_ok=True)
        (CONFIG_DIR / "processes").mkdir(parents=True, exist_ok=True)

        self.dismiss(self._models)

    def action_back_or_cancel(self) -> None:
        """Go back one step or cancel if on the first step."""
        if self._step == _STEP_PROVIDER:
            self.dismiss(None)
        elif self._step == _STEP_DETAILS:
            self._step = _STEP_PROVIDER
        elif self._step == _STEP_ROLES:
            self._step = _STEP_DETAILS
        elif self._step == _STEP_UTILITY:
            self._step = _STEP_ROLES
        elif self._step == _STEP_CONFIRM:
            # If we had a utility prompt, go back there
            if len(self._models) > 1:
                self._models.pop()  # remove utility
                self._adding_utility = False
                self._step = _STEP_UTILITY
            elif len(self._models) == 1:
                missing = set(ROLE_PRESETS["all"]) - set(
                    self._models[0]["roles"]
                )
                if missing:
                    self._step = _STEP_UTILITY
                else:
                    self._models.pop()
                    self._step = _STEP_ROLES
            else:
                self._step = _STEP_ROLES
