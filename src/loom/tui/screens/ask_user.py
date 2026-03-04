"""Enhanced ask-user modal screen."""

from __future__ import annotations

import re
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label

from loom.tools.ask_user import normalize_ask_user_args


class AskUserScreen(ModalScreen[object]):
    """Display a question from the model and collect the user's answer."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    CSS = """
    AskUserScreen {
        align: center middle;
    }
    #ask-user-dialog {
        width: 92;
        height: auto;
        max-height: 94%;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
        overflow: hidden;
    }
    #ask-user-body {
        height: 1fr;
        min-height: 8;
        overflow-y: auto;
        overflow-x: hidden;
    }
    #ask-user-question {
        width: 1fr;
        text-wrap: wrap;
    }
    #ask-user-context {
        margin-top: 1;
        width: 1fr;
        text-wrap: wrap;
    }
    #ask-user-guidance {
        margin-top: 1;
        width: 1fr;
        text-wrap: wrap;
    }
    #ask-user-error {
        margin-top: 1;
        color: #f7768e;
        width: 1fr;
        text-wrap: wrap;
    }
    #ask-user-input {
        margin-top: 1;
    }
    #ask-user-options {
        margin-top: 1;
    }
    .ask-user-option-btn {
        width: 1fr;
        margin-top: 0;
    }
    .ask-user-option-btn.selected {
        background: $accent;
        color: $text;
        border: tall $accent;
    }
    .ask-user-option-description {
        margin-left: 2;
        color: $text-muted;
        width: 1fr;
        text-wrap: wrap;
    }
    #ask-user-submit-selection {
        margin-top: 1;
    }
    """

    def __init__(
        self,
        question: str,
        options: list[str] | None = None,
        *,
        question_type: str = "free_text",
        option_items: list[dict[str, str]] | None = None,
        allow_custom_response: bool = True,
        min_selections: int = 0,
        max_selections: int = 0,
        context_note: str = "",
        urgency: str = "normal",
        default_option_id: str = "",
        return_payload: bool = False,
        allow_cancel: bool = True,
    ) -> None:
        super().__init__()
        payload: dict[str, Any] = {
            "question": question,
            "question_type": question_type,
            "allow_custom_response": bool(allow_custom_response),
            "min_selections": int(min_selections or 0),
            "max_selections": int(max_selections or 0),
            "context_note": context_note,
            "urgency": urgency,
            "default_option_id": default_option_id,
        }
        if option_items is not None:
            payload["options"] = list(option_items)
        else:
            payload["options"] = list(options or [])

        normalized = normalize_ask_user_args(payload)
        self._question = str(normalized.get("question", "") or "")
        self._options = list(normalized.get("legacy_options", []) or [])
        self._option_items: list[dict[str, str]] = [
            dict(item)
            for item in list(normalized.get("options", []) or [])
            if isinstance(item, dict)
        ]
        self._question_type = str(normalized.get("question_type", "free_text") or "free_text")
        self._allow_custom_response = bool(normalized.get("allow_custom_response", True))
        self._min_selections = max(0, int(normalized.get("min_selections", 0) or 0))
        self._max_selections = max(0, int(normalized.get("max_selections", 0) or 0))
        self._context_note = str(normalized.get("context_note", "") or "")
        self._urgency = str(normalized.get("urgency", "normal") or "normal")
        self._default_option_id = str(normalized.get("default_option_id", "") or "")
        self._return_payload = bool(return_payload)
        self._allow_cancel = bool(allow_cancel)
        self._show_input = self._compute_show_input()
        self._option_id_to_label: dict[str, str] = {
            str(item.get("id", "")).strip(): str(item.get("label", "")).strip()
            for item in self._option_items
            if str(item.get("id", "")).strip()
        }
        self._button_to_option_id: dict[str, str] = {}
        self._selected_option_ids: list[str] = []
        if self._default_option_id in self._option_id_to_label:
            self._selected_option_ids = [self._default_option_id]

    def compose(self) -> ComposeResult:
        self._button_to_option_id = {}
        body_children: list[object] = [
            Label(f"[bold #e0af68]Question:[/] {self._question}", id="ask-user-question"),
        ]
        if self._context_note:
            body_children.append(Label(f"[dim]{self._context_note}[/dim]", id="ask-user-context"))
        if self._urgency and self._urgency != "normal":
            body_children.append(Label(f"[dim]Urgency: {self._urgency}[/dim]"))

        option_children: list[object] = []
        for i, opt in enumerate(self._option_items, 1):
            label = str(opt.get("label", "") or "")
            if not label:
                continue
            option_id = str(opt.get("id", "") or "")
            description = str(opt.get("description", "") or "")
            button_id = f"ask-user-option-{i}"
            self._button_to_option_id[button_id] = option_id
            suffix = f" ({option_id})" if option_id else ""
            if option_id and option_id == self._default_option_id:
                suffix += " [default]"
            button = Button(
                f"{i}. {label}{suffix}",
                id=button_id,
                classes=(
                    "ask-user-option-btn selected"
                    if option_id in self._selected_option_ids
                    else "ask-user-option-btn"
                ),
            )
            option_children.append(button)
            if description:
                option_children.append(
                    Label(description, classes="ask-user-option-description"),
                )
        if option_children:
            body_children.append(Vertical(*option_children, id="ask-user-options"))

        guidance = self._guidance_text()
        if guidance:
            body_children.append(Label(f"[dim]{guidance}[/dim]", id="ask-user-guidance"))
        children: list[object] = [VerticalScroll(*body_children, id="ask-user-body")]
        children.append(Label("", id="ask-user-error"))
        if self._show_input:
            children.append(Input(placeholder="Your answer...", id="ask-user-input"))
        if self._question_type == "multi_choice" and self._option_items:
            children.append(
                Button(
                    "Submit selection",
                    id="ask-user-submit-selection",
                    variant="primary",
                ),
            )
        yield Vertical(*children, id="ask-user-dialog")

    def on_mount(self) -> None:
        if self._show_input:
            try:
                self.query_one("#ask-user-input", Input).focus()
                return
            except Exception:
                pass
        if self._option_items:
            try:
                self.query_one(".ask-user-option-btn", Button).focus()
            except Exception:
                pass

    def _compute_show_input(self) -> bool:
        if self._question_type == "free_text":
            return True
        if not self._option_items:
            return True
        return bool(self._allow_custom_response)

    def _guidance_text(self) -> str:
        if self._question_type in {"single_choice", "multi_choice"} and not self._option_items:
            return (
                "No structured options were provided for this clarification. "
                "Type your response and press Enter."
            )

        if self._question_type == "multi_choice":
            bounds = ""
            if self._max_selections > 0:
                bounds = (
                    f" ({self._min_selections}-{self._max_selections} selections)"
                    if self._min_selections > 0
                    else f" (up to {self._max_selections})"
                )
            elif self._min_selections > 0:
                bounds = f" (at least {self._min_selections})"
            if self._allow_custom_response:
                return (
                    f"Select options with buttons{bounds}, or type a custom response."
                )
            return f"Select options with buttons{bounds}, then submit selection."

        if self._question_type == "single_choice" and self._option_items:
            if self._allow_custom_response:
                return "Select one option button, or type a custom response."
            return "Select one option button."

        if self._option_items:
            return "Enter a number or type your response."
        return "Type your response and press Enter."

    def _set_error(self, message: str) -> None:
        try:
            self.query_one("#ask-user-error", Label).update(message)
        except Exception:
            pass

    def _option_from_token(self, token: str) -> tuple[str, str] | None:
        clean = str(token or "").strip()
        if not clean:
            return None
        if clean.isdigit():
            idx = int(clean) - 1
            if 0 <= idx < len(self._option_items):
                item = self._option_items[idx]
                return str(item.get("id", "") or ""), str(item.get("label", "") or "")

        lowered = clean.lower()
        for item in self._option_items:
            option_id = str(item.get("id", "") or "")
            label = str(item.get("label", "") or "")
            if option_id.lower() == lowered or label.lower() == lowered:
                return option_id, label
        return None

    def _legacy_answer(self, answer: str) -> str:
        clean = answer.strip()
        if self._options and clean.isdigit():
            idx = int(clean) - 1
            if 0 <= idx < len(self._options):
                return self._options[idx]
        return clean

    def _selected_labels(self) -> list[str]:
        return [
            self._option_id_to_label[option_id]
            for option_id in self._selected_option_ids
            if option_id in self._option_id_to_label
        ]

    def _payload_from_selected_options(
        self,
        *,
        custom_text: str = "",
    ) -> dict[str, object] | None:
        custom = custom_text.strip()
        selected_ids = [
            option_id
            for option_id in self._selected_option_ids
            if option_id in self._option_id_to_label
        ]
        selected_labels = self._selected_labels()

        if self._question_type == "single_choice" and self._option_items:
            if selected_ids:
                chosen_id = selected_ids[0]
                return {
                    "response_type": "single_choice",
                    "selected_option_ids": [chosen_id],
                    "selected_labels": [self._option_id_to_label[chosen_id]],
                    "custom_response": "",
                    "source": "tui",
                }
            if custom and self._allow_custom_response:
                return {
                    "response_type": "text",
                    "selected_option_ids": [],
                    "selected_labels": [],
                    "custom_response": custom,
                    "source": "tui",
                }
            self._set_error("Select one listed option.")
            return None

        if self._question_type == "multi_choice" and self._option_items:
            if selected_ids:
                count = len(selected_ids)
                if count < self._min_selections:
                    self._set_error(f"Select at least {self._min_selections} option(s).")
                    return None
                if self._max_selections > 0 and count > self._max_selections:
                    self._set_error(f"Select at most {self._max_selections} option(s).")
                    return None
                return {
                    "response_type": "multi_choice",
                    "selected_option_ids": selected_ids,
                    "selected_labels": selected_labels,
                    "custom_response": "",
                    "source": "tui",
                }
            if custom and self._allow_custom_response:
                return {
                    "response_type": "text",
                    "selected_option_ids": [],
                    "selected_labels": [],
                    "custom_response": custom,
                    "source": "tui",
                }
            self._set_error("Select option(s) from the list.")
            return None

        if custom:
            return {
                "response_type": "text",
                "selected_option_ids": [],
                "selected_labels": [],
                "custom_response": custom,
                "source": "tui",
            }
        self._set_error("Response required.")
        return None

    def _submit_from_ui(self) -> None:
        self._set_error("")
        custom_text = ""
        if self._show_input:
            try:
                custom_text = self.query_one("#ask-user-input", Input).value
            except Exception:
                custom_text = ""

        if self._return_payload:
            payload = self._payload_from_selected_options(custom_text=custom_text)
            if payload is None:
                return
            self.dismiss(payload)
            return

        labels = self._selected_labels()
        if labels:
            if self._question_type == "single_choice":
                self.dismiss(labels[0])
            else:
                self.dismiss(", ".join(labels))
            return
        answer = self._legacy_answer(custom_text)
        if not answer:
            self._set_error("Response required.")
            return
        self.dismiss(answer)

    def _set_button_selected(self, option_id: str, selected: bool) -> None:
        for button in self.query(".ask-user-option-btn", Button):
            button_id = str(button.id or "")
            if self._button_to_option_id.get(button_id, "") != option_id:
                continue
            if selected:
                button.add_class("selected")
            else:
                button.remove_class("selected")
            return

    def _toggle_multi_option(self, option_id: str) -> None:
        if option_id in self._selected_option_ids:
            self._selected_option_ids = [
                item for item in self._selected_option_ids if item != option_id
            ]
            self._set_button_selected(option_id, False)
            self._set_error("")
            return
        if self._max_selections > 0 and len(self._selected_option_ids) >= self._max_selections:
            self._set_error(f"Select at most {self._max_selections} option(s).")
            return
        self._selected_option_ids.append(option_id)
        self._set_button_selected(option_id, True)
        self._set_error("")

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id or "")
        if button_id == "ask-user-submit-selection":
            self._submit_from_ui()
            return
        option_id = self._button_to_option_id.get(button_id, "")
        if not option_id:
            return

        if self._question_type == "multi_choice":
            self._toggle_multi_option(option_id)
            return

        # Single-choice / other option-based prompts: button click confirms selection.
        self._selected_option_ids = [option_id]
        for item in list(self._option_id_to_label):
            self._set_button_selected(item, item == option_id)
        self._submit_from_ui()

    def _payload_answer(self, answer: str) -> dict[str, object] | None:
        clean = answer.strip()
        if not clean:
            self._set_error("Response required.")
            return None

        if self._question_type == "single_choice" and self._option_items:
            selected = self._option_from_token(clean)
            if selected is not None:
                option_id, label = selected
                return {
                    "response_type": "single_choice",
                    "selected_option_ids": [option_id],
                    "selected_labels": [label],
                    "custom_response": "",
                    "source": "tui",
                }
            if self._allow_custom_response:
                return {
                    "response_type": "text",
                    "selected_option_ids": [],
                    "selected_labels": [],
                    "custom_response": clean,
                    "source": "tui",
                }
            self._set_error("Select one listed option.")
            return None

        if self._question_type == "multi_choice" and self._option_items:
            tokens = [
                part.strip()
                for part in re.split(r"[\\n,]", clean)
                if part.strip()
            ]
            selected_ids: list[str] = []
            selected_labels: list[str] = []
            for token in tokens:
                selected = self._option_from_token(token)
                if selected is None:
                    continue
                option_id, label = selected
                if option_id in selected_ids:
                    continue
                selected_ids.append(option_id)
                selected_labels.append(label)

            if selected_ids:
                count = len(selected_ids)
                if count < self._min_selections:
                    self._set_error(f"Select at least {self._min_selections} option(s).")
                    return None
                if self._max_selections > 0 and count > self._max_selections:
                    self._set_error(f"Select at most {self._max_selections} option(s).")
                    return None
                return {
                    "response_type": "multi_choice",
                    "selected_option_ids": selected_ids,
                    "selected_labels": selected_labels,
                    "custom_response": "",
                    "source": "tui",
                }

            if self._allow_custom_response:
                return {
                    "response_type": "text",
                    "selected_option_ids": [],
                    "selected_labels": [],
                    "custom_response": clean,
                    "source": "tui",
                }
            self._set_error("Select option(s) from the list.")
            return None

        if self._option_items:
            selected = self._option_from_token(clean)
            if selected is not None:
                option_id, label = selected
                return {
                    "response_type": "single_choice",
                    "selected_option_ids": [option_id],
                    "selected_labels": [label],
                    "custom_response": "",
                    "source": "tui",
                }

        return {
            "response_type": "text",
            "selected_option_ids": [],
            "selected_labels": [],
            "custom_response": clean,
            "source": "tui",
        }

    @on(Input.Submitted)
    def on_submit(self, event: Input.Submitted) -> None:
        self._set_error("")
        if self._return_payload:
            payload = self._payload_answer(event.value)
            if payload is None:
                return
            self.dismiss(payload)
            return

        answer = self._legacy_answer(event.value)
        if not answer:
            return
        self.dismiss(answer)

    def action_cancel(self) -> None:
        if not self._allow_cancel:
            self._set_error("Response required.")
            return
        if self._return_payload:
            self.dismiss({
                "response_type": "cancelled",
                "selected_option_ids": [],
                "selected_labels": [],
                "custom_response": "",
                "source": "tui",
            })
            return
        self.dismiss("")
