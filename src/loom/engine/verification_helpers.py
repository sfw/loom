"""Dynamic registry for development verification helpers.

This module stays outside ``loom.engine.verification`` so core schema/prompt
code can import it without triggering verification package import cycles.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from loom.runtime.capabilities import browser_addon_status


@dataclass(frozen=True)
class VerificationHelperSpec:
    """Registered helper metadata for development verification contracts."""

    name: str
    capabilities: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class VerificationHelperContext:
    """Execution context passed to a bound verification helper."""

    workspace: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationHelperResult:
    """Normalized result returned from a bound verification helper."""

    success: bool
    detail: str = ""
    reason_code: str = ""
    capability: str = ""
    data: dict[str, object] = field(default_factory=dict)


VerificationHelperExecutor = Callable[
    [dict[str, Any], VerificationHelperContext],
    Awaitable[VerificationHelperResult],
]


@dataclass(frozen=True)
class VerificationHelperRoutingDecision:
    """Normalized routing decision from one tool call to a helper-backed tool."""

    source_tool: str
    target_tool: str
    helper: str
    arguments: dict[str, object] = field(default_factory=dict)
    reason: str = ""


VerificationHelperRouter = Callable[
    [str, dict[str, Any], VerificationHelperContext],
    VerificationHelperRoutingDecision | None,
]


@dataclass
class _BrowserSessionField:
    name: str
    field_id: str
    field_type: str
    placeholder: str = ""
    value: str = ""
    form_id: str = ""
    label: str = ""


@dataclass
class _BrowserSessionButton:
    text: str
    field_type: str = "submit"
    form_id: str = ""
    name: str = ""
    value: str = ""


@dataclass
class _BrowserSessionLink:
    text: str
    href: str
    title: str = ""
    aria_label: str = ""


@dataclass
class _BrowserSessionForm:
    form_id: str
    action: str = ""
    method: str = "get"
    fields: list[_BrowserSessionField] = field(default_factory=list)
    buttons: list[_BrowserSessionButton] = field(default_factory=list)


@dataclass
class _BrowserSessionPage:
    url: str
    html: str
    text: str
    links: list[_BrowserSessionLink]
    forms: list[_BrowserSessionForm]


@dataclass
class _BrowserSessionResponse:
    ok: bool
    status_code: int
    body: str
    error: str = ""
    final_url: str = ""


def _browser_addon_metadata() -> dict[str, object]:
    status = browser_addon_status()
    return {
        "key": status.key,
        "label": status.label,
        "installed": status.installed,
        "required_for": status.required_for,
        "install_hint": status.install_hint,
        "detail": status.detail,
    }


def _with_browser_addon_metadata(
    result: VerificationHelperResult,
    *,
    engine: str,
    warning: str = "",
) -> VerificationHelperResult:
    data = dict(result.data) if isinstance(result.data, dict) else {}
    data["engine"] = engine
    data["browser_addon"] = _browser_addon_metadata()
    warning_text = str(warning or "").strip()
    if warning_text:
        warnings = [
            str(item).strip()
            for item in data.get("warnings", [])
            if str(item).strip()
        ]
        if warning_text not in warnings:
            warnings.insert(0, warning_text)
        data["warnings"] = warnings
    return VerificationHelperResult(
        success=result.success,
        detail=result.detail,
        reason_code=result.reason_code,
        capability=result.capability,
        data=data,
    )

_REGISTRY_LOCK = RLock()
_HELPER_REGISTRY: dict[str, VerificationHelperSpec] = {}
_HELPER_EXECUTORS: dict[str, VerificationHelperExecutor] = {}
_HELPER_ROUTERS: dict[str, VerificationHelperRouter] = {}
_MAX_CAPTURE_CHARS = 8000
_HTTP_PROBE_BODY_LIMIT = 16000
_LOCAL_HTTP_URL_PATTERN = re.compile(
    r"https?://(?:127\.0\.0\.1|localhost)(?::\d+)?[^\s'\"`]+",
    re.IGNORECASE,
)
_SINGLE_BACKGROUND_OPERATOR_PATTERN = re.compile(r"(?<!&)&(?!&)")
_TEXT_ASSERTION_PATTERN = re.compile(
    r"\bcontains?\s+(?:the\s+text\s+)?[`\"']([^`\"']+)[`\"']",
    re.IGNORECASE,
)
_NEGATIVE_TEXT_ASSERTION_PATTERN = re.compile(
    r"\b(?:does\s+not\s+contain|without)\s+(?:the\s+text\s+)?[`\"']([^`\"']+)[`\"']",
    re.IGNORECASE,
)
_CLICK_ACTION_PATTERN = re.compile(
    r"\bclick(?:\s+the)?\s+(?:button|link|element)?\s*[`\"']([^`\"']+)[`\"']",
    re.IGNORECASE,
)
_FILL_FIELD_WITH_VALUE_PATTERN = re.compile(
    r"\b(?:fill|type\s+into|enter\s+into)\s+(?:the\s+)?[`\"']([^`\"']+)[`\"']\s+"
    r"(?:with|as)\s+[`\"']([^`\"']+)[`\"']",
    re.IGNORECASE,
)
_ENTER_VALUE_INTO_FIELD_PATTERN = re.compile(
    r"\b(?:enter|type)\s+[`\"']([^`\"']+)[`\"']\s+(?:into|in)\s+(?:the\s+)?"
    r"[`\"']([^`\"']+)[`\"']",
    re.IGNORECASE,
)
_PROVIDER_AGENT_TOOL_NAMES = frozenset({"claude_code", "openai_codex", "opencode"})
_ADVANCED_BROWSER_PROMPT_MARKERS = (
    "playwright",
    "puppeteer",
    "headless",
    "screenshot",
    "console log",
    "network request",
    "click ",
    "submit ",
    "form ",
)


def register_verification_helper(
    *,
    name: str,
    capabilities: list[str] | tuple[str, ...],
    description: str,
) -> None:
    """Register or replace a verification helper spec."""
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        raise ValueError("verification helper name must be non-empty")
    normalized_capabilities = tuple(
        str(item or "").strip().lower()
        for item in capabilities
        if str(item or "").strip()
    )
    if not normalized_capabilities:
        raise ValueError("verification helper capabilities must be non-empty")
    with _REGISTRY_LOCK:
        _HELPER_REGISTRY[normalized_name] = VerificationHelperSpec(
            name=normalized_name,
            capabilities=normalized_capabilities,
            description=str(description or "").strip(),
        )


def get_verification_helper(name: str) -> VerificationHelperSpec | None:
    """Return one registered verification helper by normalized name."""
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        return None
    with _REGISTRY_LOCK:
        return _HELPER_REGISTRY.get(normalized_name)


def list_verification_helpers() -> list[VerificationHelperSpec]:
    """Return all registered verification helpers in deterministic order."""
    with _REGISTRY_LOCK:
        return sorted(_HELPER_REGISTRY.values(), key=lambda item: item.name)


def bind_verification_helper(
    name: str,
    executor: VerificationHelperExecutor,
) -> None:
    """Bind an async executor to a registered verification helper."""
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        raise ValueError("verification helper name must be non-empty")
    with _REGISTRY_LOCK:
        if normalized_name not in _HELPER_REGISTRY:
            raise KeyError(f"verification helper {normalized_name!r} is not registered")
        _HELPER_EXECUTORS[normalized_name] = executor


def unbind_verification_helper(name: str) -> None:
    """Remove any bound executor for a helper."""
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        return
    with _REGISTRY_LOCK:
        _HELPER_EXECUTORS.pop(normalized_name, None)


def verification_helper_is_bound(name: str) -> bool:
    """Return True when a registered helper has an executable binding."""
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        return False
    with _REGISTRY_LOCK:
        return normalized_name in _HELPER_EXECUTORS


def register_verification_helper_router(
    name: str,
    router: VerificationHelperRouter,
) -> None:
    """Register or replace one helper-routing rule by stable name."""
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        raise ValueError("verification helper router name must be non-empty")
    with _REGISTRY_LOCK:
        _HELPER_ROUTERS[normalized_name] = router


def unregister_verification_helper_router(name: str) -> None:
    """Remove one helper-routing rule by name."""
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        return
    with _REGISTRY_LOCK:
        _HELPER_ROUTERS.pop(normalized_name, None)


def list_verification_helper_routers() -> list[str]:
    """Return registered helper-routing rule names in deterministic order."""
    with _REGISTRY_LOCK:
        return sorted(_HELPER_ROUTERS)


def route_tool_to_verification_helper(
    tool_name: str,
    args: dict[str, Any] | None = None,
    *,
    ctx: VerificationHelperContext | None = None,
) -> VerificationHelperRoutingDecision | None:
    """Return the first helper-routing decision for a tool invocation, if any."""
    normalized_tool_name = str(tool_name or "").strip().lower()
    payload = dict(args) if isinstance(args, dict) else {}
    context = ctx if ctx is not None else VerificationHelperContext()
    with _REGISTRY_LOCK:
        routers = list(_HELPER_ROUTERS.values())
    for router in routers:
        decision = router(normalized_tool_name, payload, context)
        if decision is not None:
            return decision
    return None


async def execute_verification_helper(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    ctx: VerificationHelperContext | None = None,
) -> VerificationHelperResult:
    """Execute one bound verification helper by name."""
    normalized_name = str(name or "").strip().lower()
    if not normalized_name:
        raise ValueError("verification helper name must be non-empty")
    with _REGISTRY_LOCK:
        if normalized_name not in _HELPER_REGISTRY:
            raise KeyError(f"verification helper {normalized_name!r} is not registered")
        executor = _HELPER_EXECUTORS.get(normalized_name)
    if executor is None:
        raise RuntimeError(f"verification helper {normalized_name!r} is not bound")
    payload = dict(args) if isinstance(args, dict) else {}
    context = ctx if ctx is not None else VerificationHelperContext()
    return await executor(payload, context)


def _truncate_detail(text: str) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= _MAX_CAPTURE_CHARS:
        return normalized
    return normalized[: _MAX_CAPTURE_CHARS - 16] + "\n...[truncated]"


async def _run_command_executor(
    args: dict[str, Any],
    ctx: VerificationHelperContext,
    *,
    failure_reason_code: str,
    helper_name: str,
) -> VerificationHelperResult:
    command = str(args.get("command", "") or "").strip()
    if not command:
        return VerificationHelperResult(
            success=False,
            detail="verification helper requires a non-empty command",
            reason_code=failure_reason_code,
            capability="command_execution",
        )
    timeout_seconds = int(args.get("timeout_seconds", 120) or 120)
    cwd = ctx.workspace if isinstance(ctx.workspace, Path) else None
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=max(1, timeout_seconds),
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return VerificationHelperResult(
            success=False,
            detail=(
                f"{helper_name} timed out after {timeout_seconds}s while running: {command}"
            ),
            reason_code="dev_verifier_timeout",
            capability="command_execution",
        )

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    detail_parts = []
    if stdout_text.strip():
        detail_parts.append(stdout_text.strip())
    if stderr_text.strip():
        detail_parts.append(f"[stderr]\n{stderr_text.strip()}")
    detail = _truncate_detail("\n".join(detail_parts))
    if process.returncode == 0:
        return VerificationHelperResult(
            success=True,
            detail=detail or f"{helper_name} completed successfully.",
            capability="command_execution",
            data={"exit_code": 0, "command": command},
        )
    return VerificationHelperResult(
        success=False,
        detail=detail or f"{helper_name} failed with exit code {process.returncode}.",
        reason_code=failure_reason_code,
        capability="command_execution",
        data={"exit_code": int(process.returncode), "command": command},
    )


def _normalize_probe_text_list(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _normalize_expected_statuses(value: object, default: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, list):
        normalized: list[int] = []
        for item in value:
            try:
                parsed = int(item)
            except (TypeError, ValueError):
                continue
            if parsed not in normalized:
                normalized.append(parsed)
        if normalized:
            return tuple(normalized)
    return default


def _truncate_http_body(text: str) -> str:
    normalized = str(text or "")
    if len(normalized) <= _HTTP_PROBE_BODY_LIMIT:
        return normalized
    return normalized[: _HTTP_PROBE_BODY_LIMIT - 16] + "\n...[truncated]"


def _normalize_browser_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _browser_text_matches(candidate: str, target: str) -> bool:
    candidate_text = _normalize_browser_text(candidate).casefold()
    target_text = _normalize_browser_text(target).casefold()
    if not candidate_text or not target_text:
        return False
    return candidate_text == target_text or target_text in candidate_text


class _BrowserSessionHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[_BrowserSessionLink] = []
        self.forms: list[_BrowserSessionForm] = []
        self._form_by_id: dict[str, _BrowserSessionForm] = {}
        self._fields: list[_BrowserSessionField] = []
        self._visible_text: list[str] = []
        self._link_attrs: dict[str, str] | None = None
        self._link_text: list[str] = []
        self._button_attrs: dict[str, str] | None = None
        self._button_text: list[str] = []
        self._textarea_field: _BrowserSessionField | None = None
        self._textarea_text: list[str] = []
        self._label_for: str = ""
        self._label_text: list[str] | None = None
        self._label_nested_fields: list[_BrowserSessionField] = []
        self._current_form_id: str = ""
        self._form_counter = 0

    def _ensure_form(self, form_id: str) -> _BrowserSessionForm:
        normalized_form_id = form_id or "__page__"
        if normalized_form_id not in self._form_by_id:
            form = _BrowserSessionForm(form_id=normalized_form_id)
            self._form_by_id[normalized_form_id] = form
            self.forms.append(form)
        return self._form_by_id[normalized_form_id]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {key: str(value or "") for key, value in attrs}
        if tag == "form":
            self._form_counter += 1
            self._current_form_id = attrs_map.get("id") or f"__form_{self._form_counter}"
            form = self._ensure_form(self._current_form_id)
            form.action = attrs_map.get("action", "")
            form.method = str(attrs_map.get("method", "get") or "get").strip().lower() or "get"
            return
        if tag == "a":
            self._link_attrs = attrs_map
            self._link_text = []
            return
        if tag == "button":
            self._button_attrs = attrs_map
            self._button_text = []
            return
        if tag == "label":
            self._label_for = attrs_map.get("for", "")
            self._label_text = []
            self._label_nested_fields = []
            return
        if tag in {"input", "textarea"}:
            field = _BrowserSessionField(
                name=attrs_map.get("name", ""),
                field_id=attrs_map.get("id", ""),
                field_type=attrs_map.get("type", "text" if tag == "input" else "textarea"),
                placeholder=attrs_map.get("placeholder", ""),
                value=attrs_map.get("value", ""),
                form_id=attrs_map.get("form") or self._current_form_id or "__page__",
            )
            if tag == "textarea":
                self._textarea_field = field
                self._textarea_text = []
            else:
                self._fields.append(field)
                self._ensure_form(field.form_id).fields.append(field)
            if self._label_text is not None:
                self._label_nested_fields.append(field)
            input_type = field.field_type.strip().lower()
            if input_type in {"submit", "button"}:
                self._ensure_form(field.form_id).buttons.append(
                    _BrowserSessionButton(
                        text=attrs_map.get("value", ""),
                        field_type=input_type,
                        form_id=field.form_id,
                        name=field.name,
                        value=field.value,
                    ),
                )

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self._current_form_id = ""
            return
        if tag == "a" and self._link_attrs is not None:
            self.links.append(
                _BrowserSessionLink(
                    text=_normalize_browser_text("".join(self._link_text)),
                    href=self._link_attrs.get("href", ""),
                    title=self._link_attrs.get("title", ""),
                    aria_label=self._link_attrs.get("aria-label", ""),
                ),
            )
            self._link_attrs = None
            self._link_text = []
            return
        if tag == "button" and self._button_attrs is not None:
            form_id = self._button_attrs.get("form") or self._current_form_id or "__page__"
            self._ensure_form(form_id).buttons.append(
                _BrowserSessionButton(
                    text=_normalize_browser_text("".join(self._button_text)),
                    field_type=self._button_attrs.get("type", "submit"),
                    form_id=form_id,
                    name=self._button_attrs.get("name", ""),
                    value=self._button_attrs.get("value", ""),
                ),
            )
            self._button_attrs = None
            self._button_text = []
            return
        if tag == "textarea" and self._textarea_field is not None:
            self._textarea_field.value = _normalize_browser_text("".join(self._textarea_text))
            self._fields.append(self._textarea_field)
            self._ensure_form(self._textarea_field.form_id).fields.append(self._textarea_field)
            self._textarea_field = None
            self._textarea_text = []
            return
        if tag == "label":
            label_text = _normalize_browser_text("".join(self._label_text or []))
            if label_text:
                if self._label_for:
                    for field in self._fields:
                        if field.field_id and field.field_id == self._label_for:
                            field.label = label_text
                elif len(self._label_nested_fields) == 1:
                    self._label_nested_fields[0].label = label_text
            self._label_for = ""
            self._label_text = None
            self._label_nested_fields = []

    def handle_data(self, data: str) -> None:
        text = str(data or "")
        if not text.strip():
            return
        self._visible_text.append(text)
        if self._link_attrs is not None:
            self._link_text.append(text)
        if self._button_attrs is not None:
            self._button_text.append(text)
        if self._textarea_field is not None:
            self._textarea_text.append(text)
        if self._label_text is not None:
            self._label_text.append(text)


def _parse_browser_session_page(url: str, html: str) -> _BrowserSessionPage:
    parser = _BrowserSessionHtmlParser()
    parser.feed(str(html or ""))
    parser.close()
    return _BrowserSessionPage(
        url=url,
        html=str(html or ""),
        text=_normalize_browser_text(" ".join(parser._visible_text)),
        links=parser.links,
        forms=parser.forms,
    )


def _browser_session_request_sync(
    opener: urllib.request.OpenerDirector,
    *,
    url: str,
    timeout_seconds: int,
    method: str = "GET",
    data: dict[str, str] | None = None,
) -> _BrowserSessionResponse:
    normalized_method = str(method or "GET").strip().upper() or "GET"
    request_url = url
    request_data: bytes | None = None
    payload = dict(data) if isinstance(data, dict) else {}
    if payload and normalized_method == "GET":
        split = urlsplit(url)
        existing_pairs = parse_qsl(split.query, keep_blank_values=True)
        existing_pairs.extend((str(k), str(v)) for k, v in payload.items())
        request_url = urlunsplit(
            (split.scheme, split.netloc, split.path, urlencode(existing_pairs), split.fragment),
        )
    elif payload:
        request_data = urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        request_url,
        data=request_data,
        headers={"User-Agent": "LoomBrowserSession/1.0"},
        method=normalized_method,
    )
    try:
        with opener.open(request, timeout=max(1, timeout_seconds)) as response:
            body = response.read(_HTTP_PROBE_BODY_LIMIT + 1)
            return _BrowserSessionResponse(
                ok=True,
                status_code=int(getattr(response, "status", 200) or 200),
                body=_truncate_http_body(body.decode("utf-8", errors="replace")),
                final_url=str(getattr(response, "url", request_url) or request_url),
            )
    except urllib.error.HTTPError as exc:
        body = exc.read(_HTTP_PROBE_BODY_LIMIT + 1)
        return _BrowserSessionResponse(
            ok=False,
            status_code=int(getattr(exc, "code", 0) or 0),
            body=_truncate_http_body(body.decode("utf-8", errors="replace")),
            error=str(exc),
            final_url=str(getattr(exc, "url", request_url) or request_url),
        )
    except Exception as exc:
        return _BrowserSessionResponse(
            ok=False,
            status_code=0,
            body="",
            error=str(exc),
            final_url=request_url,
        )


async def _browser_session_request(
    opener: urllib.request.OpenerDirector,
    *,
    url: str,
    timeout_seconds: int,
    method: str = "GET",
    data: dict[str, str] | None = None,
) -> _BrowserSessionResponse:
    return await asyncio.to_thread(
        _browser_session_request_sync,
        opener,
        url=url,
        timeout_seconds=timeout_seconds,
        method=method,
        data=data,
    )


def _browser_session_form_payload(
    form: _BrowserSessionForm,
    filled_values: dict[tuple[str, str], str],
    submit_button: _BrowserSessionButton | None = None,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    for browser_field in form.fields:
        field_key = browser_field.name or browser_field.field_id
        if not field_key:
            continue
        normalized_type = browser_field.field_type.strip().lower()
        if normalized_type in {"submit", "button"}:
            continue
        payload[field_key] = filled_values.get(
            (form.form_id, field_key),
            browser_field.value,
        )
    if submit_button and submit_button.name:
        payload[submit_button.name] = submit_button.value
    return payload


def _resolve_browser_session_form(
    page: _BrowserSessionPage,
    *,
    active_form_id: str,
    filled_values: dict[tuple[str, str], str],
) -> _BrowserSessionForm | None:
    if active_form_id:
        for form in page.forms:
            if form.form_id == active_form_id:
                return form
    if len(page.forms) == 1:
        return page.forms[0]
    if filled_values:
        filled_form_ids = {form_id for form_id, _field_name in filled_values}
        for form in page.forms:
            if form.form_id in filled_form_ids:
                return form
    return page.forms[0] if page.forms else None


def _find_browser_session_link(
    page: _BrowserSessionPage,
    target: str,
) -> _BrowserSessionLink | None:
    for link in page.links:
        for candidate in (link.text, link.title, link.aria_label):
            if _browser_text_matches(candidate, target):
                return link
    return None


def _find_browser_session_field(
    page: _BrowserSessionPage,
    target: str,
) -> _BrowserSessionField | None:
    for form in page.forms:
        for browser_field in form.fields:
            for candidate in (
                browser_field.label,
                browser_field.name,
                browser_field.field_id,
                browser_field.placeholder,
            ):
                if _browser_text_matches(candidate, target):
                    return browser_field
    return None


def _find_browser_session_button(
    page: _BrowserSessionPage,
    target: str,
) -> tuple[_BrowserSessionForm, _BrowserSessionButton] | None:
    for form in page.forms:
        for button in form.buttons:
            if _browser_text_matches(button.text, target):
                return form, button
    return None


async def _browser_session_fallback_executor(
    args: dict[str, Any],
    ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    start_url = str(args.get("start_url", "") or "").strip()
    steps = args.get("steps", [])
    if not start_url or not isinstance(steps, list) or not steps:
        return VerificationHelperResult(
            success=False,
            detail="browser_session requires start_url and a non-empty steps list",
            reason_code="dev_verifier_capability_unavailable",
            capability="browser_runtime",
        )
    timeout_seconds = int(args.get("timeout_seconds", 20) or 20)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    request_log: list[dict[str, object]] = []
    warnings: list[str] = []
    current_url = start_url
    current_page: _BrowserSessionPage | None = None
    filled_values: dict[tuple[str, str], str] = {}
    active_form_id = ""

    async def _load_page(
        *,
        url: str,
        method: str = "GET",
        data: dict[str, str] | None = None,
    ) -> VerificationHelperResult | None:
        nonlocal current_url, current_page, active_form_id
        response = await _browser_session_request(
            opener,
            url=url,
            timeout_seconds=timeout_seconds,
            method=method,
            data=data,
        )
        request_log.append({
            "method": method,
            "url": url,
            "status_code": response.status_code,
            "final_url": response.final_url or url,
        })
        if not response.ok and response.status_code == 0:
            error_text = str(response.error or "").lower()
            reason_code = (
                "dev_verifier_timeout"
                if any(marker in error_text for marker in ("timed out", "timeout"))
                else "dev_verifier_capability_unavailable"
            )
            return VerificationHelperResult(
                success=False,
                detail=f"browser_session could not reach {url}: {response.error}",
                reason_code=reason_code,
                capability="browser_runtime",
                data={"network_requests": request_log},
            )
        current_url = response.final_url or url
        current_page = _parse_browser_session_page(current_url, response.body)
        active_form_id = ""
        return None

    for index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            return VerificationHelperResult(
                success=False,
                detail=f"browser_session step {index} is malformed",
                reason_code="dev_verifier_capability_unavailable",
                capability="browser_runtime",
                data={"network_requests": request_log},
            )
        action = str(raw_step.get("action", "") or "").strip().lower()
        if action == "open":
            url = str(raw_step.get("url", "") or start_url).strip() or start_url
            failure = await _load_page(url=url)
            if failure is not None:
                return failure
            continue
        if current_page is None:
            failure = await _load_page(url=start_url)
            if failure is not None:
                return failure
        if current_page is None:
            return VerificationHelperResult(
                success=False,
                detail="browser_session could not initialize a page context",
                reason_code="dev_verifier_capability_unavailable",
                capability="browser_runtime",
                data={"network_requests": request_log},
            )
        if action == "click_text":
            target = str(raw_step.get("target", "") or "").strip()
            link = _find_browser_session_link(current_page, target)
            if link and link.href:
                failure = await _load_page(url=urljoin(current_url, link.href))
                if failure is not None:
                    return failure
                continue
            button_match = _find_browser_session_button(current_page, target)
            if button_match is not None:
                form, button = button_match
                payload = _browser_session_form_payload(form, filled_values, submit_button=button)
                action_url = urljoin(current_url, form.action or current_url)
                failure = await _load_page(
                    url=action_url,
                    method=form.method.upper(),
                    data=payload,
                )
                if failure is not None:
                    return failure
                continue
            return VerificationHelperResult(
                success=False,
                detail=f"browser_session could not find clickable target '{target}'",
                reason_code="dev_browser_check_failed",
                capability="browser_runtime",
                data={"network_requests": request_log, "step_index": index},
            )
        if action == "fill_field":
            field_name = str(raw_step.get("field", "") or "").strip()
            value = str(raw_step.get("value", "") or "").strip()
            field = _find_browser_session_field(current_page, field_name)
            if field is None:
                return VerificationHelperResult(
                    success=False,
                    detail=f"browser_session could not find field '{field_name}'",
                    reason_code="dev_browser_check_failed",
                    capability="browser_runtime",
                    data={"network_requests": request_log, "step_index": index},
                )
            field_key = field.name or field.field_id
            if not field_key:
                return VerificationHelperResult(
                    success=False,
                    detail=f"browser_session field '{field_name}' is not submittable",
                    reason_code="dev_browser_check_failed",
                    capability="browser_runtime",
                    data={"network_requests": request_log, "step_index": index},
                )
            filled_values[(field.form_id, field_key)] = value
            active_form_id = field.form_id
            continue
        if action == "submit":
            form = _resolve_browser_session_form(
                current_page,
                active_form_id=active_form_id,
                filled_values=filled_values,
            )
            if form is None:
                return VerificationHelperResult(
                    success=False,
                    detail="browser_session could not find a form to submit",
                    reason_code="dev_browser_check_failed",
                    capability="browser_runtime",
                    data={"network_requests": request_log, "step_index": index},
                )
            payload = _browser_session_form_payload(form, filled_values)
            action_url = urljoin(current_url, form.action or current_url)
            failure = await _load_page(
                url=action_url,
                method=form.method.upper(),
                data=payload,
            )
            if failure is not None:
                return failure
            continue
        if action == "assert_text":
            expected_text = str(raw_step.get("text", "") or "").strip()
            present = bool(raw_step.get("present", True))
            haystack = current_page.text or current_page.html
            found = expected_text in haystack
            if found != present:
                expectation = "present" if present else "absent"
                return VerificationHelperResult(
                    success=False,
                    detail=(
                        f"browser_session expected text '{expected_text}' to be {expectation}"
                    ),
                    reason_code="dev_browser_check_failed",
                    capability="browser_runtime",
                    data={"network_requests": request_log, "step_index": index},
                )
            continue
        return VerificationHelperResult(
            success=False,
            detail=f"browser_session does not support action '{action}'",
            reason_code="dev_verifier_capability_unavailable",
            capability="browser_runtime",
            data={"network_requests": request_log, "step_index": index},
        )

    data: dict[str, object] = {
        "current_url": current_url,
        "network_requests": request_log,
        "step_count": len(steps),
    }
    if bool(args.get("capture_network", False)):
        data["captured_network"] = list(request_log)
    if bool(args.get("capture_console", False)):
        warnings.append(
            "browser_session uses a static HTTP/DOM fallback; "
            "JavaScript console capture is unavailable.",
        )
        data["console_logs"] = []
    if bool(args.get("capture_screenshot", False)):
        warnings.append(
            "browser_session fallback wrote a DOM snapshot instead of a rendered screenshot.",
        )
        if isinstance(ctx.workspace, Path) and current_page is not None:
            subtask_id = str(ctx.metadata.get("subtask_id", "") or "").strip() or "latest"
            snapshot_path = str(
                args.get("snapshot_path", "")
                or f"artifacts/browser-session-dom-snapshot-{subtask_id}.html"
            ).strip()
            destination = (ctx.workspace / snapshot_path).resolve()
            workspace_root = ctx.workspace.resolve()
            destination.relative_to(workspace_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(current_page.html, encoding="utf-8")
            data["output_path"] = snapshot_path
            data["dom_snapshot_path"] = snapshot_path
    if warnings:
        data["warnings"] = warnings
    return VerificationHelperResult(
        success=True,
        detail=f"browser_session completed successfully for {current_url}",
        capability="browser_runtime",
        data=data,
    )


async def _playwright_browser_session_executor(
    args: dict[str, Any],
    ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    try:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except Exception as exc:  # pragma: no cover - exercised through wrapper tests
        raise RuntimeError("Playwright addon is unavailable") from exc

    start_url = str(args.get("start_url", "") or "").strip()
    steps = args.get("steps", [])
    if not start_url or not isinstance(steps, list) or not steps:
        return VerificationHelperResult(
            success=False,
            detail="browser_session requires start_url and a non-empty steps list",
            reason_code="dev_verifier_capability_unavailable",
            capability="browser_runtime",
        )

    timeout_seconds = int(args.get("timeout_seconds", 20) or 20)
    timeout_ms = max(1000, timeout_seconds * 1000)
    capture_network = bool(args.get("capture_network", False))
    capture_console = bool(args.get("capture_console", False))
    capture_screenshot = bool(args.get("capture_screenshot", False))
    request_log: list[dict[str, object]] = []
    console_logs: list[str] = []
    warnings: list[str] = []

    def _append_network(request: Any) -> None:
        request_log.append(
            {
                "method": str(getattr(request, "method", "") or "").upper() or "GET",
                "url": str(getattr(request, "url", "") or ""),
                "resource_type": str(getattr(request, "resource_type", "") or ""),
            }
        )

    def _append_console(message: Any) -> None:
        text = ""
        try:
            text = str(message.text())
        except Exception:
            text = str(message)
        text = text.strip()
        if text:
            console_logs.append(text)

    async with async_playwright() as playwright:
        try:
            browser = await playwright.chromium.launch(headless=True)
        except Exception as exc:
            raise RuntimeError(f"Playwright browser launch failed: {exc}") from exc

        try:
            page = await browser.new_page()
            page.set_default_timeout(timeout_ms)
            if capture_network:
                page.on("request", _append_network)
            if capture_console:
                page.on("console", _append_console)

            current_url = start_url
            filled_selectors: list[str] = []

            async def _goto(url: str) -> None:
                nonlocal current_url
                await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                current_url = page.url or url

            for index, raw_step in enumerate(steps, start=1):
                if not isinstance(raw_step, dict):
                    return VerificationHelperResult(
                        success=False,
                        detail=f"browser_session step {index} is malformed",
                        reason_code="dev_verifier_capability_unavailable",
                        capability="browser_runtime",
                        data={"network_requests": request_log},
                    )
                action = str(raw_step.get("action", "") or "").strip().lower()
                try:
                    if action == "open":
                        url = str(raw_step.get("url", "") or start_url).strip() or start_url
                        await _goto(url)
                        continue
                    if action == "click_text":
                        target = str(raw_step.get("target", "") or "").strip()
                        locators = (
                            page.get_by_role("link", name=target),
                            page.get_by_role("button", name=target),
                            page.get_by_text(target, exact=False),
                        )
                        clicked = False
                        for locator in locators:
                            if await locator.count() > 0:
                                await locator.first.click(timeout=timeout_ms)
                                clicked = True
                                break
                        if not clicked:
                            return VerificationHelperResult(
                                success=False,
                                detail=(
                                    "browser_session could not find clickable target "
                                    f"'{target}'"
                                ),
                                reason_code="dev_browser_check_failed",
                                capability="browser_runtime",
                                data={
                                    "network_requests": request_log,
                                    "step_index": index,
                                },
                            )
                        await page.wait_for_load_state("networkidle", timeout=timeout_ms)
                        current_url = page.url or current_url
                        continue
                    if action == "fill_field":
                        field_name = str(raw_step.get("field", "") or "").strip()
                        value = str(raw_step.get("value", "") or "").strip()
                        candidates = (
                            (page.get_by_label(field_name), f"label:{field_name}"),
                            (
                                page.get_by_placeholder(field_name),
                                f"placeholder:{field_name}",
                            ),
                            (
                                page.locator(
                                    ",".join(
                                        (
                                            f'input[name="{field_name}"]',
                                            f'textarea[name="{field_name}"]',
                                            f'select[name="{field_name}"]',
                                            f'input[id="{field_name}"]',
                                            f'textarea[id="{field_name}"]',
                                            f'select[id="{field_name}"]',
                                        )
                                    )
                                ),
                                field_name,
                            ),
                        )
                        filled = False
                        for locator, selector_name in candidates:
                            if await locator.count() > 0:
                                await locator.first.fill(value, timeout=timeout_ms)
                                filled_selectors.append(selector_name)
                                filled = True
                                break
                        if not filled:
                            return VerificationHelperResult(
                                success=False,
                                detail=f"browser_session could not find field '{field_name}'",
                                reason_code="dev_browser_check_failed",
                                capability="browser_runtime",
                                data={
                                    "network_requests": request_log,
                                    "step_index": index,
                                },
                            )
                        continue
                    if action == "submit":
                        submit_button = page.locator(
                            "button[type=submit], input[type=submit]"
                        )
                        if await submit_button.count() > 0:
                            await submit_button.first.click(timeout=timeout_ms)
                        elif filled_selectors:
                            locator = page.get_by_label(filled_selectors[-1].split(":", 1)[-1])
                            if await locator.count() > 0:
                                await locator.first.press("Enter", timeout=timeout_ms)
                            else:
                                submit_script = (
                                    "(form) => form.requestSubmit ? "
                                    "form.requestSubmit() : form.submit()"
                                )
                                await page.locator("form").first.evaluate(
                                    submit_script
                                )
                        else:
                            form = page.locator("form")
                            if await form.count() == 0:
                                return VerificationHelperResult(
                                    success=False,
                                    detail="browser_session could not find a form to submit",
                                    reason_code="dev_browser_check_failed",
                                    capability="browser_runtime",
                                    data={
                                        "network_requests": request_log,
                                    "step_index": index,
                                },
                            )
                            submit_script = (
                                "(form) => form.requestSubmit ? "
                                "form.requestSubmit() : form.submit()"
                            )
                            await form.first.evaluate(
                                submit_script
                            )
                        await page.wait_for_load_state("networkidle", timeout=timeout_ms)
                        current_url = page.url or current_url
                        continue
                    if action == "assert_text":
                        expected_text = str(raw_step.get("text", "") or "").strip()
                        present = bool(raw_step.get("present", True))
                        body = page.locator("body")
                        page_text = await body.inner_text() if await body.count() > 0 else ""
                        page_html = await page.content()
                        found = expected_text in page_text or expected_text in page_html
                        if found != present:
                            expectation = "present" if present else "absent"
                            return VerificationHelperResult(
                                success=False,
                                detail=(
                                    "browser_session expected text "
                                    f"'{expected_text}' to be {expectation}"
                                ),
                                reason_code="dev_browser_check_failed",
                                capability="browser_runtime",
                                data={
                                    "network_requests": request_log,
                                    "step_index": index,
                                },
                            )
                        continue
                    return VerificationHelperResult(
                        success=False,
                        detail=f"browser_session does not support action '{action}'",
                        reason_code="dev_verifier_capability_unavailable",
                        capability="browser_runtime",
                        data={"network_requests": request_log, "step_index": index},
                    )
                except PlaywrightTimeoutError:
                    return VerificationHelperResult(
                        success=False,
                        detail=(
                            f"browser_session timed out during action '{action}' after "
                            f"{timeout_seconds}s"
                        ),
                        reason_code="dev_verifier_timeout",
                        capability="browser_runtime",
                        data={"network_requests": request_log, "step_index": index},
                    )
                except PlaywrightError as exc:
                    return VerificationHelperResult(
                        success=False,
                        detail=f"browser_session action '{action}' failed: {exc}",
                        reason_code="dev_browser_check_failed",
                        capability="browser_runtime",
                        data={"network_requests": request_log, "step_index": index},
                    )

            data: dict[str, object] = {
                "current_url": page.url or current_url,
                "network_requests": request_log,
                "step_count": len(steps),
            }
            if capture_network:
                data["captured_network"] = list(request_log)
            if capture_console:
                data["console_logs"] = list(console_logs)
            if capture_screenshot:
                if isinstance(ctx.workspace, Path):
                    subtask_id = (
                        str(ctx.metadata.get("subtask_id", "") or "").strip() or "latest"
                    )
                    screenshot_path = str(
                        args.get("screenshot_path", "")
                        or f"artifacts/browser-session-screenshot-{subtask_id}.png"
                    ).strip()
                    destination = (ctx.workspace / screenshot_path).resolve()
                    workspace_root = ctx.workspace.resolve()
                    destination.relative_to(workspace_root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    await page.screenshot(path=str(destination), full_page=True)
                    data["output_path"] = screenshot_path
                    data["screenshot_path"] = screenshot_path
                else:
                    warnings.append(
                        "browser_session could not persist a screenshot because "
                        "no workspace was provided."
                    )
            if warnings:
                data["warnings"] = warnings
            return VerificationHelperResult(
                success=True,
                detail=f"browser_session completed successfully for {page.url or current_url}",
                capability="browser_runtime",
                data=data,
            )
        finally:
            await browser.close()


async def _browser_session_executor(
    args: dict[str, Any],
    ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    status = browser_addon_status()
    if status.installed:
        try:
            result = await _playwright_browser_session_executor(args, ctx)
            return _with_browser_addon_metadata(result, engine="playwright")
        except RuntimeError as exc:
            fallback = await _browser_session_fallback_executor(args, ctx)
            return _with_browser_addon_metadata(
                fallback,
                engine="fallback",
                warning=(
                    "Playwright addon is installed but unavailable at runtime; "
                    f"browser_session used the static fallback instead. Detail: {exc}"
                ),
            )

    fallback = await _browser_session_fallback_executor(args, ctx)
    return _with_browser_addon_metadata(
        fallback,
        engine="fallback",
        warning=(
            "Playwright addon is not installed; browser_session used the static "
            "fallback engine. Install with `uv sync --extra browser` for full "
            "JS-capable verification."
        ),
    )


def _extract_local_http_urls(text: str) -> list[str]:
    normalized = str(text or "")
    if not normalized:
        return []
    urls: list[str] = []
    for match in _LOCAL_HTTP_URL_PATTERN.finditer(normalized):
        candidate = str(match.group(0) or "").rstrip(".,;:)]}")
        if candidate and candidate not in urls:
            urls.append(candidate)
    return urls


def _looks_like_service_start_command(command: str) -> bool:
    text = str(command or "").strip().lower()
    if not text:
        return False
    markers = (
        "python -m http.server",
        "python3 -m http.server",
        "npm run dev",
        "pnpm dev",
        "yarn dev",
        "next dev",
        "vite dev",
        "vite --host",
        "npx serve",
        "serve -s",
        "flask run",
        "uvicorn ",
    )
    return any(marker in text for marker in markers)


def _extract_serve_static_route_args(command: str) -> dict[str, object] | None:
    normalized = str(command or "").strip()
    if not normalized:
        return None
    parts = _SINGLE_BACKGROUND_OPERATOR_PATTERN.split(normalized, maxsplit=1)
    if len(parts) != 2:
        return None
    start_command = str(parts[0] or "").strip()
    if not _looks_like_service_start_command(start_command):
        return None
    urls = _extract_local_http_urls(normalized)
    if not urls:
        return None
    route_args: dict[str, object] = {
        "command": start_command,
        "ready_url": urls[0],
    }
    if len(urls) > 1:
        route_args["checks"] = [
            {
                "url": url,
                "capability": "service_runtime",
            }
            for url in urls
        ]
    elif urls[-1] != urls[0]:
        route_args["probe_url"] = urls[-1]
    return route_args


def _extract_prompt_text_assertions(prompt: str) -> tuple[list[str], list[str]]:
    contains_text: list[str] = []
    not_contains_text: list[str] = []
    normalized = str(prompt or "")
    for pattern, target in (
        (_TEXT_ASSERTION_PATTERN, contains_text),
        (_NEGATIVE_TEXT_ASSERTION_PATTERN, not_contains_text),
    ):
        for match in pattern.finditer(normalized):
            snippet = str(match.group(1) or "").strip()
            if snippet and snippet not in target:
                target.append(snippet)
    return contains_text, not_contains_text


def _select_prompt_probe_helper(prompt: str) -> str:
    text = str(prompt or "").strip().lower()
    if not text:
        return ""
    ui_markers = ("browser", "page", "ui", "render", "loads", "load")
    if any(marker in text for marker in ui_markers):
        return "browser_assert"
    return "http_assert"


def _normalize_probe_suite_checks(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    normalized_checks: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "") or item.get("probe_url", "") or "").strip()
        if not url:
            continue
        check: dict[str, object] = {"url": url}
        capability = str(item.get("capability", "") or "").strip().lower()
        if capability in {"service_runtime", "browser_runtime"}:
            check["capability"] = capability
        name = str(item.get("name", "") or "").strip()
        if name:
            check["name"] = name
        contains_text = _normalize_probe_text_list(item.get("contains_text"))
        if contains_text:
            check["contains_text"] = contains_text
        not_contains_text = _normalize_probe_text_list(item.get("not_contains_text"))
        if not_contains_text:
            check["not_contains_text"] = not_contains_text
        expect_status = item.get("expect_status")
        if isinstance(expect_status, int | list):
            check["expect_status"] = expect_status
        normalized_checks.append(check)
    return normalized_checks


def _extract_provider_prompt_probe_args(prompt: str) -> tuple[str, dict[str, object]] | None:
    normalized = str(prompt or "").strip()
    if not normalized:
        return None
    urls = _extract_local_http_urls(normalized)
    if not urls:
        return None
    lower_text = normalized.lower()
    if not any(
        marker in lower_text
        for marker in ("verify", "check", "assert", "confirm", "respond", "load", "render")
    ):
        return None
    if any(marker in lower_text for marker in _ADVANCED_BROWSER_PROMPT_MARKERS):
        return None
    if len(urls) > 1:
        contains_text, not_contains_text = _extract_prompt_text_assertions(normalized)
        if contains_text or not_contains_text:
            return None
        checks = []
        for url in urls:
            checks.append({
                "url": url,
                "capability": (
                    "browser_runtime"
                    if _select_prompt_probe_helper(f"page {url}") == "browser_assert"
                    else "service_runtime"
                ),
            })
        return "probe_suite", {"checks": checks}
    helper_name = _select_prompt_probe_helper(normalized)
    contains_text, not_contains_text = _extract_prompt_text_assertions(normalized)
    helper_args: dict[str, object] = {"url": urls[0]}
    if contains_text:
        helper_args["contains_text"] = contains_text
    if not_contains_text:
        helper_args["not_contains_text"] = not_contains_text
    return helper_name, helper_args


def _extract_browser_session_prompt_args(prompt: str) -> dict[str, object] | None:
    normalized = str(prompt or "").strip()
    if not normalized:
        return None
    lower_text = normalized.lower()
    if not any(marker in lower_text for marker in _ADVANCED_BROWSER_PROMPT_MARKERS):
        return None
    urls = _extract_local_http_urls(normalized)
    if not urls:
        return None
    steps: list[dict[str, object]] = []
    for url in urls:
        steps.append({"action": "open", "url": url})
    for match in _CLICK_ACTION_PATTERN.finditer(normalized):
        target = str(match.group(1) or "").strip()
        if target:
            steps.append({"action": "click_text", "target": target})
    for match in _FILL_FIELD_WITH_VALUE_PATTERN.finditer(normalized):
        field = str(match.group(1) or "").strip()
        value = str(match.group(2) or "").strip()
        if field and value:
            steps.append({"action": "fill_field", "field": field, "value": value})
    for match in _ENTER_VALUE_INTO_FIELD_PATTERN.finditer(normalized):
        value = str(match.group(1) or "").strip()
        field = str(match.group(2) or "").strip()
        if field and value:
            steps.append({"action": "fill_field", "field": field, "value": value})
    if "submit" in lower_text:
        steps.append({"action": "submit"})
    contains_text, not_contains_text = _extract_prompt_text_assertions(normalized)
    for text in contains_text:
        steps.append({"action": "assert_text", "text": text, "present": True})
    for text in not_contains_text:
        steps.append({"action": "assert_text", "text": text, "present": False})
    capture_console = any(
        marker in lower_text for marker in ("console log", "console logs")
    )
    capture_network = any(
        marker in lower_text for marker in ("network request", "network requests")
    )
    capture_screenshot = "screenshot" in lower_text
    return {
        "start_url": urls[0],
        "steps": steps,
        "capture_console": capture_console,
        "capture_network": capture_network,
        "capture_screenshot": capture_screenshot,
        "prompt": normalized,
    }


def _fetch_url_sync(url: str, timeout_seconds: int) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LoomVerificationHelper/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(_HTTP_PROBE_BODY_LIMIT + 1)
            return {
                "ok": True,
                "status_code": int(getattr(response, "status", 200) or 200),
                "body": _truncate_http_body(
                    body.decode("utf-8", errors="replace"),
                ),
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(_HTTP_PROBE_BODY_LIMIT + 1)
        return {
            "ok": False,
            "status_code": int(getattr(exc, "code", 0) or 0),
            "body": _truncate_http_body(
                body.decode("utf-8", errors="replace"),
            ),
            "error": str(exc),
        }
    except Exception as exc:  # pragma: no cover - exercised through async wrappers
        return {
            "ok": False,
            "status_code": 0,
            "body": "",
            "error": str(exc),
        }


async def _fetch_url(url: str, timeout_seconds: int) -> dict[str, object]:
    return await asyncio.to_thread(_fetch_url_sync, url, timeout_seconds)


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows fallback
            process.terminate()
        await asyncio.wait_for(process.wait(), timeout=1.0)
        return
    except Exception:
        pass
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows fallback
            process.kill()
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except Exception:
        pass


def _build_probe_failure_detail(
    *,
    helper_name: str,
    url: str,
    expected_statuses: tuple[int, ...],
    probe: dict[str, object],
    contains_text: list[str],
    not_contains_text: list[str],
) -> str:
    parts = [
        f"{helper_name} assertion failed for {url}",
        f"expected_status={list(expected_statuses)}",
        f"actual_status={int(probe.get('status_code', 0) or 0)}",
    ]
    body = str(probe.get("body", "") or "").strip()
    if contains_text:
        parts.append(f"contains_text={contains_text}")
    if not_contains_text:
        parts.append(f"not_contains_text={not_contains_text}")
    if body:
        parts.append(f"body_excerpt={_truncate_detail(body)}")
    error = str(probe.get("error", "") or "").strip()
    if error:
        parts.append(f"error={error}")
    return "; ".join(parts)


def _probe_matches_expectations(
    *,
    body: str,
    status_code: int,
    expected_statuses: tuple[int, ...],
    contains_text: list[str],
    not_contains_text: list[str],
) -> bool:
    if expected_statuses and status_code not in expected_statuses:
        return False
    for expected in contains_text:
        if expected not in body:
            return False
    for forbidden in not_contains_text:
        if forbidden in body:
            return False
    return True


async def _browser_assert_executor(
    args: dict[str, Any],
    _ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    return await _http_assert_executor(
        args,
        helper_name="browser_assert",
        capability="browser_runtime",
        semantic_reason_code="dev_browser_check_failed",
    )


async def _http_assert_executor(
    args: dict[str, Any],
    *,
    helper_name: str,
    capability: str,
    semantic_reason_code: str,
) -> VerificationHelperResult:
    url = str(args.get("url", "") or args.get("probe_url", "") or "").strip()
    if not url:
        return VerificationHelperResult(
            success=False,
            detail=f"{helper_name} requires a non-empty url",
            reason_code="dev_verifier_capability_unavailable",
            capability=capability,
        )
    timeout_seconds = int(args.get("timeout_seconds", 15) or 15)
    expected_statuses = _normalize_expected_statuses(
        args.get("expect_status"),
        default=(200,),
    )
    contains_text = _normalize_probe_text_list(args.get("contains_text"))
    not_contains_text = _normalize_probe_text_list(args.get("not_contains_text"))
    probe = await _fetch_url(url, max(1, timeout_seconds))
    if not bool(probe.get("ok")) and int(probe.get("status_code", 0) or 0) == 0:
        error_text = str(probe.get("error", "") or "").lower()
        reason_code = (
            "dev_verifier_timeout"
            if any(marker in error_text for marker in ("timed out", "timeout"))
            else "dev_verifier_capability_unavailable"
        )
        return VerificationHelperResult(
            success=False,
            detail=f"{helper_name} could not reach {url}: {probe.get('error', '')}",
            reason_code=reason_code,
            capability=capability,
            data={"url": url},
        )
    status_code = int(probe.get("status_code", 0) or 0)
    body = str(probe.get("body", "") or "")
    if not _probe_matches_expectations(
        body=body,
        status_code=status_code,
        expected_statuses=expected_statuses,
        contains_text=contains_text,
        not_contains_text=not_contains_text,
    ):
        return VerificationHelperResult(
            success=False,
            detail=_build_probe_failure_detail(
                helper_name=helper_name,
                url=url,
                expected_statuses=expected_statuses,
                probe=probe,
                contains_text=contains_text,
                not_contains_text=not_contains_text,
            ),
            reason_code=semantic_reason_code,
            capability=capability,
            data={"url": url, "status_code": status_code},
        )
    return VerificationHelperResult(
        success=True,
        detail=f"{helper_name} completed successfully for {url}",
        capability=capability,
        data={"url": url, "status_code": status_code},
    )


async def _local_http_assert_executor(
    args: dict[str, Any],
    _ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    return await _http_assert_executor(
        args,
        helper_name="http_assert",
        capability="service_runtime",
        semantic_reason_code="dev_contract_failed",
    )


async def _probe_suite_executor(
    args: dict[str, Any],
    _ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    checks = _normalize_probe_suite_checks(args.get("checks"))
    if not checks:
        return VerificationHelperResult(
            success=False,
            detail="probe_suite requires a non-empty checks list",
            reason_code="dev_verifier_capability_unavailable",
            capability="service_runtime",
        )
    timeout_seconds = int(args.get("timeout_seconds", 15) or 15)
    return await _run_probe_checks(
        checks=checks,
        timeout_seconds=timeout_seconds,
        helper_name="probe_suite",
    )


async def _run_probe_checks(
    *,
    checks: list[dict[str, object]],
    timeout_seconds: int,
    helper_name: str,
) -> VerificationHelperResult:
    completed: list[dict[str, object]] = []
    seen_capabilities: list[str] = []
    for index, check in enumerate(checks, start=1):
        capability = str(check.get("capability", "") or "service_runtime").strip().lower()
        probe_helper_name = (
            "browser_assert" if capability == "browser_runtime" else "http_assert"
        )
        probe_result = await _http_assert_executor(
            {
                "url": check["url"],
                "contains_text": check.get("contains_text", []),
                "not_contains_text": check.get("not_contains_text", []),
                "expect_status": check.get("expect_status"),
                "timeout_seconds": timeout_seconds,
            },
            helper_name=probe_helper_name,
            capability=capability,
            semantic_reason_code=(
                "dev_browser_check_failed"
                if capability == "browser_runtime"
                else "dev_contract_failed"
            ),
        )
        if capability not in seen_capabilities:
            seen_capabilities.append(capability)
        completed.append(
            {
                "index": index,
                "name": str(check.get("name", "") or "").strip(),
                "url": check["url"],
                "capability": capability,
                "success": bool(probe_result.success),
                "reason_code": str(probe_result.reason_code or "").strip(),
                "detail": str(probe_result.detail or "").strip(),
                "status_code": probe_result.data.get("status_code")
                if isinstance(probe_result.data, dict)
                else None,
            },
        )
        if not probe_result.success:
            return VerificationHelperResult(
                success=False,
                detail=(
                    f"{helper_name} check {index} failed: "
                    f"{probe_result.detail or probe_result.reason_code or 'unknown error'}"
                ),
                reason_code=str(probe_result.reason_code or "").strip(),
                capability=capability,
                data={
                    "checks": completed,
                    "helper_capabilities": seen_capabilities,
                    "failed_check_index": index,
                },
            )
    return VerificationHelperResult(
        success=True,
        detail=f"{helper_name} completed successfully for {len(completed)} checks.",
        capability=seen_capabilities[0] if len(seen_capabilities) == 1 else "service_runtime",
        data={
            "checks": completed,
            "helper_capabilities": seen_capabilities,
            "check_count": len(completed),
        },
    )


async def _serve_static_executor(
    args: dict[str, Any],
    ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    command = str(args.get("command", "") or "").strip()
    ready_url = str(args.get("ready_url", "") or args.get("url", "") or "").strip()
    probe_url = str(args.get("probe_url", "") or ready_url).strip()
    if not command or not ready_url:
        return VerificationHelperResult(
            success=False,
            detail="serve_static requires non-empty command and ready_url",
            reason_code="dev_verifier_capability_unavailable",
            capability="service_runtime",
        )
    timeout_seconds = int(args.get("timeout_seconds", 20) or 20)
    ready_timeout_seconds = int(args.get("ready_timeout_seconds", 8) or 8)
    expected_statuses = _normalize_expected_statuses(
        args.get("expect_status"),
        default=(200,),
    )
    contains_text = _normalize_probe_text_list(args.get("contains_text"))
    not_contains_text = _normalize_probe_text_list(args.get("not_contains_text"))
    probe_checks = _normalize_probe_suite_checks(args.get("checks"))
    cwd = ctx.workspace if isinstance(ctx.workspace, Path) else None
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + max(1, ready_timeout_seconds)
        ready_probe: dict[str, object] = {}
        while time.monotonic() < deadline:
            if process.returncode is not None:
                stdout_bytes, stderr_bytes = await process.communicate()
                output = _truncate_detail(
                    "\n".join(
                        item
                        for item in (
                            stdout_bytes.decode("utf-8", errors="replace").strip(),
                            stderr_bytes.decode("utf-8", errors="replace").strip(),
                        )
                        if item
                    )
                )
                return VerificationHelperResult(
                    success=False,
                    detail=(
                        f"serve_static exited before readiness probe succeeded: {output}"
                        if output
                        else "serve_static exited before readiness probe succeeded."
                    ),
                    reason_code="dev_verifier_capability_unavailable",
                    capability="service_runtime",
                    data={"command": command, "ready_url": ready_url},
                )
            ready_probe = await _fetch_url(ready_url, 2)
            if (
                bool(ready_probe.get("ok"))
                or int(ready_probe.get("status_code", 0) or 0) in expected_statuses
            ):
                break
            await asyncio.sleep(0.25)
        else:
            return VerificationHelperResult(
                success=False,
                detail=(
                    f"serve_static timed out waiting for readiness at {ready_url}"
                ),
                reason_code="dev_verifier_timeout",
                capability="service_runtime",
                data={"command": command, "ready_url": ready_url},
            )

        if probe_checks:
            suite_result = await _run_probe_checks(
                checks=probe_checks,
                timeout_seconds=max(1, timeout_seconds),
                helper_name="serve_static",
            )
            data = dict(suite_result.data) if isinstance(suite_result.data, dict) else {}
            data["command"] = command
            data["ready_url"] = ready_url
            return VerificationHelperResult(
                success=suite_result.success,
                detail=suite_result.detail,
                reason_code=suite_result.reason_code,
                capability="service_runtime",
                data=data,
            )

        probe = await _fetch_url(probe_url, max(1, timeout_seconds))
        if not bool(probe.get("ok")) and int(probe.get("status_code", 0) or 0) == 0:
            error_text = str(probe.get("error", "") or "").lower()
            reason_code = (
                "dev_verifier_timeout"
                if any(marker in error_text for marker in ("timed out", "timeout"))
                else "dev_verifier_capability_unavailable"
            )
            return VerificationHelperResult(
                success=False,
                detail=f"serve_static probe failed for {probe_url}: {probe.get('error', '')}",
                reason_code=reason_code,
                capability="service_runtime",
                data={"command": command, "url": probe_url},
            )
        status_code = int(probe.get("status_code", 0) or 0)
        body = str(probe.get("body", "") or "")
        if not _probe_matches_expectations(
            body=body,
            status_code=status_code,
            expected_statuses=expected_statuses,
            contains_text=contains_text,
            not_contains_text=not_contains_text,
        ):
            return VerificationHelperResult(
                success=False,
                detail=_build_probe_failure_detail(
                    helper_name="serve_static",
                    url=probe_url,
                    expected_statuses=expected_statuses,
                    probe=probe,
                    contains_text=contains_text,
                    not_contains_text=not_contains_text,
                ),
                reason_code="dev_contract_failed",
                capability="service_runtime",
                data={"command": command, "url": probe_url, "status_code": status_code},
            )
        return VerificationHelperResult(
            success=True,
            detail=f"serve_static completed successfully for {probe_url}",
            capability="service_runtime",
            data={"command": command, "url": probe_url, "status_code": status_code},
        )
    finally:
        await _terminate_process_group(process)


async def _run_test_suite_executor(
    args: dict[str, Any],
    ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    return await _run_command_executor(
        args,
        ctx,
        failure_reason_code="dev_test_failed",
        helper_name="run_test_suite",
    )


async def _run_build_check_executor(
    args: dict[str, Any],
    ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    return await _run_command_executor(
        args,
        ctx,
        failure_reason_code="dev_build_failed",
        helper_name="run_build_check",
    )


def _render_verification_markdown(
    *,
    title: str,
    canonical_result: dict[str, Any],
) -> str:
    passed = int(canonical_result.get("passed", 0) or 0)
    total = int(canonical_result.get("total", 0) or 0)
    failed = int(canonical_result.get("failed", max(0, total - passed)) or 0)
    percent = 0
    if total > 0:
        percent = round((passed / total) * 100)
    lines = [
        f"# {title}",
        "",
        f"**Score: {passed}/{total} tests passed ({percent}%)**",
        "",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        "",
        "Generated from the canonical structured verification result.",
    ]
    return "\n".join(lines).strip() + "\n"


async def _render_verification_report_executor(
    args: dict[str, Any],
    ctx: VerificationHelperContext,
) -> VerificationHelperResult:
    canonical_result = args.get("canonical_result", {})
    if not isinstance(canonical_result, dict):
        canonical_result = {}
    if not canonical_result:
        return VerificationHelperResult(
            success=False,
            detail="render_verification_report requires canonical_result data",
            reason_code="dev_report_contract_violation",
            capability="report_rendering",
        )
    title = str(args.get("title", "Validation Report") or "Validation Report").strip()
    markdown = _render_verification_markdown(
        title=title,
        canonical_result=canonical_result,
    )
    output_path = str(args.get("output_path", "") or "").strip()
    data: dict[str, object] = {"markdown": markdown}
    if output_path and isinstance(ctx.workspace, Path):
        workspace = ctx.workspace.resolve()
        destination = (workspace / output_path).resolve()
        destination.relative_to(workspace)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(markdown, encoding="utf-8")
        data["output_path"] = output_path
    return VerificationHelperResult(
        success=True,
        detail="render_verification_report completed successfully.",
        capability="report_rendering",
        data=data,
    )


def _route_shell_execute_command_helper(
    tool_name: str,
    args: dict[str, Any],
    _ctx: VerificationHelperContext,
) -> VerificationHelperRoutingDecision | None:
    if str(tool_name or "").strip().lower() != "shell_execute":
        return None
    command = str(args.get("command", "") or "").strip()
    if not command:
        return None
    from loom.engine.verification.development import development_product_reason_code

    product_reason = development_product_reason_code(command)
    helper_name = ""
    if product_reason == "dev_test_failed":
        helper_name = "run_test_suite"
    elif product_reason in {"dev_build_failed", "dev_contract_failed"}:
        helper_name = "run_build_check"
    if not helper_name or not verification_helper_is_bound(helper_name):
        return None
    helper_args: dict[str, object] = {"command": command}
    timeout_seconds = args.get("timeout_seconds")
    if timeout_seconds is not None:
        helper_args["timeout_seconds"] = timeout_seconds
    return VerificationHelperRoutingDecision(
        source_tool="shell_execute",
        target_tool="verification_helper",
        helper=helper_name,
        arguments={
            "helper": helper_name,
            "args": helper_args,
        },
        reason="shell_command_category",
    )


def _route_shell_service_probe_helper(
    tool_name: str,
    args: dict[str, Any],
    _ctx: VerificationHelperContext,
) -> VerificationHelperRoutingDecision | None:
    if str(tool_name or "").strip().lower() != "shell_execute":
        return None
    if not verification_helper_is_bound("serve_static"):
        return None
    command = str(args.get("command", "") or "").strip()
    helper_args = _extract_serve_static_route_args(command)
    if helper_args is None:
        return None
    timeout_seconds = args.get("timeout_seconds")
    if timeout_seconds is not None:
        helper_args["timeout_seconds"] = timeout_seconds
    return VerificationHelperRoutingDecision(
        source_tool="shell_execute",
        target_tool="verification_helper",
        helper="serve_static",
        arguments={
            "helper": "serve_static",
            "args": helper_args,
        },
        reason="local_service_probe",
    )


def _route_shell_http_probe_helper(
    tool_name: str,
    args: dict[str, Any],
    _ctx: VerificationHelperContext,
) -> VerificationHelperRoutingDecision | None:
    if str(tool_name or "").strip().lower() != "shell_execute":
        return None
    if not verification_helper_is_bound("http_assert"):
        return None
    command = str(args.get("command", "") or "").strip()
    if not command:
        return None
    if _extract_serve_static_route_args(command) is not None:
        return None
    normalized_command = command.lower()
    if "curl " not in normalized_command and "wget " not in normalized_command:
        return None
    urls = _extract_local_http_urls(command)
    if not urls:
        return None
    if len(urls) > 1:
        if not verification_helper_is_bound("probe_suite"):
            return None
        helper_name = "probe_suite"
        helper_args: dict[str, object] = {
            "checks": [
                {
                    "url": url,
                    "capability": "service_runtime",
                }
                for url in urls
            ],
        }
    else:
        helper_name = "http_assert"
        helper_args = {"url": urls[-1]}
    timeout_seconds = args.get("timeout_seconds")
    if timeout_seconds is not None:
        helper_args["timeout_seconds"] = timeout_seconds
    return VerificationHelperRoutingDecision(
        source_tool="shell_execute",
        target_tool="verification_helper",
        helper=helper_name,
        arguments={
            "helper": helper_name,
            "args": helper_args,
        },
        reason="local_http_probe",
    )


def _route_provider_agent_local_probe_helper(
    tool_name: str,
    args: dict[str, Any],
    _ctx: VerificationHelperContext,
) -> VerificationHelperRoutingDecision | None:
    normalized_tool_name = str(tool_name or "").strip().lower()
    if normalized_tool_name not in _PROVIDER_AGENT_TOOL_NAMES:
        return None
    prompt = str(args.get("prompt", "") or "").strip()
    extracted = _extract_provider_prompt_probe_args(prompt)
    if extracted is None:
        return None
    helper_name, helper_args = extracted
    if not verification_helper_is_bound(helper_name):
        return None
    timeout_seconds = args.get("timeout_seconds")
    if timeout_seconds is not None:
        helper_args["timeout_seconds"] = timeout_seconds
    return VerificationHelperRoutingDecision(
        source_tool=normalized_tool_name,
        target_tool="verification_helper",
        helper=helper_name,
        arguments={
            "helper": helper_name,
            "args": helper_args,
        },
        reason="provider_local_probe_prompt",
    )


def _route_provider_agent_browser_session_helper(
    tool_name: str,
    args: dict[str, Any],
    _ctx: VerificationHelperContext,
) -> VerificationHelperRoutingDecision | None:
    normalized_tool_name = str(tool_name or "").strip().lower()
    if normalized_tool_name not in _PROVIDER_AGENT_TOOL_NAMES:
        return None
    if not verification_helper_is_bound("browser_session"):
        return None
    prompt = str(args.get("prompt", "") or "").strip()
    helper_args = _extract_browser_session_prompt_args(prompt)
    if helper_args is None:
        return None
    timeout_seconds = args.get("timeout_seconds")
    if timeout_seconds is not None:
        helper_args["timeout_seconds"] = timeout_seconds
    return VerificationHelperRoutingDecision(
        source_tool=normalized_tool_name,
        target_tool="verification_helper",
        helper="browser_session",
        arguments={
            "helper": "browser_session",
            "args": helper_args,
        },
        reason="provider_browser_session_prompt",
    )


def _load_canonical_validation_result(workspace: Path | None) -> dict[str, int]:
    if not isinstance(workspace, Path):
        return {}
    runtime_results_path = workspace / "runtime-validation-results.json"
    try:
        payload = json.loads(runtime_results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, NameError):
        return {}
    try:
        passed = int(payload.get("passed", 0) or 0)
        failed = int(payload.get("failed", 0) or 0)
    except (TypeError, ValueError):
        return {}
    total = passed + failed
    if total <= 0:
        tests = payload.get("tests", [])
        if isinstance(tests, list):
            total = len(tests)
            failed = max(0, total - passed)
    if total <= 0:
        return {}
    return {"passed": passed, "failed": failed, "total": total}


def _path_looks_like_validation_report(path_text: str) -> bool:
    normalized = str(path_text or "").strip().lower()
    return normalized.endswith(".md") and "validation-report" in normalized


def _default_validation_report_title(path_text: str) -> str:
    stem = Path(str(path_text or "").strip()).stem
    if not stem:
        return "Validation Report"
    return " ".join(part.capitalize() for part in stem.replace("-", " ").split())


def _route_validation_report_write_helper(
    tool_name: str,
    args: dict[str, Any],
    ctx: VerificationHelperContext,
) -> VerificationHelperRoutingDecision | None:
    normalized_tool_name = str(tool_name or "").strip().lower()
    if normalized_tool_name not in {"write_file", "document_write"}:
        return None
    path_text = str(args.get("path", "") or "").strip()
    if not _path_looks_like_validation_report(path_text):
        return None
    if normalized_tool_name == "document_write" and bool(args.get("append", False)):
        return None
    canonical_result = _load_canonical_validation_result(ctx.workspace)
    if not canonical_result:
        return None
    title = str(args.get("title", "") or "").strip() or _default_validation_report_title(path_text)
    return VerificationHelperRoutingDecision(
        source_tool=normalized_tool_name,
        target_tool="verification_helper",
        helper="render_verification_report",
        arguments={
            "helper": "render_verification_report",
            "args": {
                "title": title,
                "canonical_result": canonical_result,
                "output_path": path_text,
            },
        },
        reason="canonical_validation_report",
    )


def _register_builtin_helpers() -> None:
    register_verification_helper(
        name="browser_assert",
        capabilities=("browser_runtime",),
        description="Run a bounded browser assertion step against a prepared UI.",
    )
    register_verification_helper(
        name="browser_session",
        capabilities=("browser_runtime",),
        description=(
            "Run a structured browser-session verification flow with page steps, "
            "console capture, network capture, and optional screenshots."
        ),
    )
    register_verification_helper(
        name="http_assert",
        capabilities=("service_runtime",),
        description="Probe one local HTTP URL with bounded status/content assertions.",
    )
    register_verification_helper(
        name="serve_static",
        capabilities=("service_runtime",),
        description="Start a bounded local service or static probe with explicit cleanup.",
    )
    register_verification_helper(
        name="probe_suite",
        capabilities=("service_runtime", "browser_runtime"),
        description="Run a bounded sequence of local HTTP/browser probe assertions.",
    )
    register_verification_helper(
        name="run_test_suite",
        capabilities=("command_execution",),
        description="Execute a test command with structured pass/fail and timeout capture.",
    )
    register_verification_helper(
        name="run_build_check",
        capabilities=("command_execution",),
        description="Execute a build, lint, or typecheck command with structured outcomes.",
    )
    register_verification_helper(
        name="render_verification_report",
        capabilities=("report_rendering",),
        description="Render human-readable validation output from a canonical result.",
    )


def _bind_builtin_helpers() -> None:
    bind_verification_helper("browser_assert", _browser_assert_executor)
    bind_verification_helper("browser_session", _browser_session_executor)
    bind_verification_helper("http_assert", _local_http_assert_executor)
    bind_verification_helper("serve_static", _serve_static_executor)
    bind_verification_helper("probe_suite", _probe_suite_executor)
    bind_verification_helper("run_test_suite", _run_test_suite_executor)
    bind_verification_helper("run_build_check", _run_build_check_executor)
    bind_verification_helper(
        "render_verification_report",
        _render_verification_report_executor,
    )


def _register_builtin_helper_routers() -> None:
    register_verification_helper_router(
        "provider_agent_browser_session_helper",
        _route_provider_agent_browser_session_helper,
    )
    register_verification_helper_router(
        "shell_execute_service_probe_helper",
        _route_shell_service_probe_helper,
    )
    register_verification_helper_router(
        "shell_execute_http_probe_helper",
        _route_shell_http_probe_helper,
    )
    register_verification_helper_router(
        "provider_agent_local_probe_helper",
        _route_provider_agent_local_probe_helper,
    )
    register_verification_helper_router(
        "shell_execute_command_helper",
        _route_shell_execute_command_helper,
    )
    register_verification_helper_router(
        "canonical_validation_report_writer",
        _route_validation_report_write_helper,
    )


_register_builtin_helpers()
_bind_builtin_helpers()
_register_builtin_helper_routers()
