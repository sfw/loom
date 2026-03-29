"""Tests for the dynamic development verification helper registry."""

from __future__ import annotations

import contextlib
import http.server
import socket
import threading
from pathlib import Path

import pytest

from loom.engine.verification_helpers import (
    VerificationHelperContext,
    VerificationHelperResult,
    _browser_session_executor,
    bind_verification_helper,
    execute_verification_helper,
    get_verification_helper,
    list_verification_helper_routers,
    list_verification_helpers,
    register_verification_helper,
    route_tool_to_verification_helper,
    unbind_verification_helper,
    verification_helper_is_bound,
)
from loom.runtime.capabilities import OptionalAddonStatus


@contextlib.contextmanager
def _serve_directory(directory: Path):
    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format, *args):  # noqa: A003
            del format, args

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_builtin_verification_helpers_are_registered() -> None:
    browser = get_verification_helper("browser_assert")
    assert browser is not None
    assert browser.capabilities == ("browser_runtime",)
    browser_session = get_verification_helper("browser_session")
    assert browser_session is not None
    assert browser_session.capabilities == ("browser_runtime",)

    helper_names = [item.name for item in list_verification_helpers()]
    assert "browser_assert" in helper_names
    assert "browser_session" in helper_names
    assert "http_assert" in helper_names
    assert "probe_suite" in helper_names
    assert "serve_static" in helper_names
    assert "provider_agent_browser_session_helper" in list_verification_helper_routers()
    assert "shell_execute_service_probe_helper" in list_verification_helper_routers()
    assert "shell_execute_http_probe_helper" in list_verification_helper_routers()
    assert "provider_agent_local_probe_helper" in list_verification_helper_routers()
    assert "shell_execute_command_helper" in list_verification_helper_routers()
    assert "canonical_validation_report_writer" in list_verification_helper_routers()
    assert verification_helper_is_bound("browser_assert") is True
    assert verification_helper_is_bound("browser_session") is True
    assert verification_helper_is_bound("http_assert") is True
    assert verification_helper_is_bound("probe_suite") is True
    assert verification_helper_is_bound("serve_static") is True
    assert verification_helper_is_bound("run_test_suite") is True
    assert verification_helper_is_bound("run_build_check") is True
    assert verification_helper_is_bound("render_verification_report") is True


def test_builtin_router_maps_test_shell_command_to_helper() -> None:
    decision = route_tool_to_verification_helper(
        "shell_execute",
        {"command": "pytest -q"},
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "run_test_suite"
    assert decision.arguments["helper"] == "run_test_suite"


def test_builtin_router_maps_build_shell_command_to_helper() -> None:
    decision = route_tool_to_verification_helper(
        "shell_execute",
        {"command": "npm run build"},
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "run_build_check"


def test_builtin_router_maps_local_service_probe_shell_command_to_helper() -> None:
    decision = route_tool_to_verification_helper(
        "shell_execute",
        {
            "command": (
                "cd dist && python3 -m http.server 8080 &\n"
                "sleep 2\n"
                "curl -s -o /dev/null -w \"%{http_code}\" "
                "http://127.0.0.1:8080/index.html"
            ),
        },
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "serve_static"
    assert decision.arguments["args"] == {
        "command": "cd dist && python3 -m http.server 8080",
        "ready_url": "http://127.0.0.1:8080/index.html",
    }


def test_builtin_router_maps_multi_url_service_probe_shell_command_to_helper() -> None:
    decision = route_tool_to_verification_helper(
        "shell_execute",
        {
            "command": (
                "cd dist && python3 -m http.server 8080 &\n"
                "sleep 2\n"
                "curl -s http://127.0.0.1:8080/index.html\n"
                "curl -s http://127.0.0.1:8080/pricing.html"
            ),
        },
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "serve_static"
    assert decision.arguments["args"] == {
        "command": "cd dist && python3 -m http.server 8080",
        "ready_url": "http://127.0.0.1:8080/index.html",
        "checks": [
            {
                "url": "http://127.0.0.1:8080/index.html",
                "capability": "service_runtime",
            },
            {
                "url": "http://127.0.0.1:8080/pricing.html",
                "capability": "service_runtime",
            },
        ],
    }


def test_builtin_router_maps_local_http_probe_shell_command_to_helper() -> None:
    decision = route_tool_to_verification_helper(
        "shell_execute",
        {
            "command": (
                "curl -s -o /dev/null -w \"%{http_code}\" "
                "http://127.0.0.1:8080/index.html"
            ),
        },
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "http_assert"
    assert decision.arguments["args"] == {
        "url": "http://127.0.0.1:8080/index.html",
    }


def test_builtin_router_maps_multi_url_http_probe_shell_command_to_probe_suite() -> None:
    decision = route_tool_to_verification_helper(
        "shell_execute",
        {
            "command": (
                "curl -s http://127.0.0.1:8080/index.html\n"
                "curl -s http://127.0.0.1:8080/pricing.html"
            ),
        },
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "probe_suite"
    assert decision.arguments["args"] == {
        "checks": [
            {
                "url": "http://127.0.0.1:8080/index.html",
                "capability": "service_runtime",
            },
            {
                "url": "http://127.0.0.1:8080/pricing.html",
                "capability": "service_runtime",
            },
        ],
    }


def test_builtin_router_maps_simple_provider_probe_prompt_to_helper() -> None:
    decision = route_tool_to_verification_helper(
        "claude_code",
        {
            "prompt": (
                "Verify that http://127.0.0.1:8080/index.html loads and "
                "contains \"hello microsite\"."
            ),
        },
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "browser_assert"
    assert decision.arguments["args"] == {
        "url": "http://127.0.0.1:8080/index.html",
        "contains_text": ["hello microsite"],
    }


def test_builtin_router_leaves_advanced_provider_browser_prompt_alone() -> None:
    unbind_verification_helper("browser_session")
    try:
        decision = route_tool_to_verification_helper(
            "claude_code",
            {
                "prompt": (
                    "Run Playwright headless browser tests against "
                    "http://127.0.0.1:8080/index.html, capture console logs, "
                    "and take a screenshot."
                ),
            },
            ctx=VerificationHelperContext(workspace=Path("/tmp")),
        )
    finally:
        bind_verification_helper("browser_session", _browser_session_executor)

    assert decision is None


def test_builtin_router_maps_advanced_provider_browser_prompt_when_helper_bound() -> None:
    decision = route_tool_to_verification_helper(
        "claude_code",
        {
            "prompt": (
                "Run Playwright headless browser tests against "
                "http://127.0.0.1:8080/index.html, capture console logs, "
                "capture network requests, click \"Start\", fill "
                "\"Email\" with \"test@example.com\", submit, and take a "
                "screenshot without \"Error\"."
            ),
        },
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "browser_session"
    assert decision.arguments["args"] == {
        "start_url": "http://127.0.0.1:8080/index.html",
        "steps": [
            {
                "action": "open",
                "url": "http://127.0.0.1:8080/index.html",
            },
            {
                "action": "click_text",
                "target": "Start",
            },
            {
                "action": "fill_field",
                "field": "Email",
                "value": "test@example.com",
            },
            {
                "action": "submit",
            },
            {
                "action": "assert_text",
                "text": "Error",
                "present": False,
            },
        ],
        "capture_console": True,
        "capture_network": True,
        "capture_screenshot": True,
        "prompt": (
            "Run Playwright headless browser tests against "
            "http://127.0.0.1:8080/index.html, capture console logs, "
            "capture network requests, click \"Start\", fill "
            "\"Email\" with \"test@example.com\", submit, and take a "
            "screenshot without \"Error\"."
        ),
    }


def test_builtin_router_maps_multi_url_provider_probe_prompt_to_probe_suite() -> None:
    decision = route_tool_to_verification_helper(
        "claude_code",
        {
            "prompt": (
                "Verify that http://127.0.0.1:8080/index.html loads and "
                "http://127.0.0.1:8080/pricing.html loads."
            ),
        },
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "probe_suite"
    assert decision.arguments["args"] == {
        "checks": [
            {
                "url": "http://127.0.0.1:8080/index.html",
                "capability": "browser_runtime",
            },
            {
                "url": "http://127.0.0.1:8080/pricing.html",
                "capability": "browser_runtime",
            },
        ],
    }


def test_builtin_router_ignores_non_verification_shell_commands() -> None:
    decision = route_tool_to_verification_helper(
        "shell_execute",
        {"command": "ls -la"},
        ctx=VerificationHelperContext(workspace=Path("/tmp")),
    )

    assert decision is None


def test_builtin_router_maps_validation_report_write_to_helper(tmp_path) -> None:
    (tmp_path / "runtime-validation-results.json").write_text(
        '{"passed": 15, "failed": 1}',
        encoding="utf-8",
    )

    decision = route_tool_to_verification_helper(
        "write_file",
        {
            "path": "reports/ui-integration-validation-report.md",
            "content": "stale manual content",
        },
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    assert decision is not None
    assert decision.target_tool == "verification_helper"
    assert decision.helper == "render_verification_report"
    assert decision.arguments["args"]["canonical_result"] == {
        "passed": 15,
        "failed": 1,
        "total": 16,
    }


def test_builtin_router_maps_document_write_validation_report_to_helper(tmp_path) -> None:
    (tmp_path / "runtime-validation-results.json").write_text(
        '{"passed": 7, "failed": 2}',
        encoding="utf-8",
    )

    decision = route_tool_to_verification_helper(
        "document_write",
        {
            "path": "ui-integration-validation-report.md",
            "title": "UI Validation",
            "content": "manual report",
        },
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    assert decision is not None
    assert decision.helper == "render_verification_report"
    assert decision.arguments["args"]["title"] == "UI Validation"


def test_builtin_router_skips_validation_report_write_without_canonical_result(tmp_path) -> None:
    decision = route_tool_to_verification_helper(
        "write_file",
        {
            "path": "reports/ui-integration-validation-report.md",
            "content": "manual report",
        },
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    assert decision is None


def test_verification_helper_registry_supports_dynamic_extension() -> None:
    register_verification_helper(
        name="plugin_runtime_probe",
        capabilities=("service_runtime", "browser_runtime"),
        description="Plugin-provided runtime probe helper.",
    )

    helper = get_verification_helper("plugin_runtime_probe")
    assert helper is not None
    assert helper.capabilities == ("service_runtime", "browser_runtime")
    assert helper.description == "Plugin-provided runtime probe helper."


@pytest.mark.asyncio
async def test_verification_helper_registry_supports_binding_and_execution() -> None:
    register_verification_helper(
        name="plugin_bound_probe",
        capabilities=("service_runtime",),
        description="Plugin-provided bound helper.",
    )

    async def _executor(args, ctx):
        return VerificationHelperResult(
            success=True,
            detail=f"workspace={ctx.workspace}; target={args.get('target', '')}",
            capability="service_runtime",
            data={"handled": True},
        )

    bind_verification_helper("plugin_bound_probe", _executor)
    assert verification_helper_is_bound("plugin_bound_probe") is True

    result = await execute_verification_helper(
        "plugin_bound_probe",
        {"target": "index.html"},
        ctx=VerificationHelperContext(workspace=Path("/tmp/plugin-workspace")),
    )

    assert result.success is True
    assert "index.html" in result.detail
    assert result.capability == "service_runtime"
    unbind_verification_helper("plugin_bound_probe")
    assert verification_helper_is_bound("plugin_bound_probe") is False


@pytest.mark.asyncio
async def test_builtin_run_build_check_executes_successfully(tmp_path) -> None:
    result = await execute_verification_helper(
        "run_build_check",
        {"command": "printf 'build ok'"},
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    assert result.success is True
    assert "build ok" in result.detail
    assert result.capability == "command_execution"
    assert result.data == {"exit_code": 0, "command": "printf 'build ok'"}


@pytest.mark.asyncio
async def test_builtin_serve_static_executes_probe_successfully(tmp_path) -> None:
    (tmp_path / "index.html").write_text("hello microsite", encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    result = await execute_verification_helper(
        "serve_static",
        {
            "command": f"python3 -m http.server {port}",
            "ready_url": f"http://127.0.0.1:{port}/index.html",
            "contains_text": "hello microsite",
        },
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.capability == "service_runtime"
    assert result.data["status_code"] == 200


@pytest.mark.asyncio
async def test_builtin_browser_assert_executes_probe_successfully(tmp_path) -> None:
    (tmp_path / "index.html").write_text("hello browser helper", encoding="utf-8")
    with _serve_directory(tmp_path) as port:
        result = await execute_verification_helper(
            "browser_assert",
            {
                "url": f"http://127.0.0.1:{port}/index.html",
                "contains_text": "hello browser helper",
            },
            ctx=VerificationHelperContext(workspace=tmp_path),
        )

    assert result.success is True
    assert result.capability == "browser_runtime"
    assert result.data["status_code"] == 200


@pytest.mark.asyncio
async def test_builtin_serve_static_executes_multiple_checks_successfully(tmp_path) -> None:
    (tmp_path / "index.html").write_text("home page", encoding="utf-8")
    (tmp_path / "pricing.html").write_text("pricing page", encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    result = await execute_verification_helper(
        "serve_static",
        {
            "command": f"python3 -m http.server {port}",
            "ready_url": f"http://127.0.0.1:{port}/index.html",
            "checks": [
                {
                    "url": f"http://127.0.0.1:{port}/index.html",
                    "capability": "service_runtime",
                    "contains_text": "home page",
                },
                {
                    "url": f"http://127.0.0.1:{port}/pricing.html",
                    "capability": "service_runtime",
                    "contains_text": "pricing page",
                },
            ],
        },
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.capability == "service_runtime"
    assert result.data["command"] == f"python3 -m http.server {port}"
    assert result.data["ready_url"] == f"http://127.0.0.1:{port}/index.html"
    assert len(result.data["checks"]) == 2


@pytest.mark.asyncio
async def test_builtin_http_assert_executes_probe_successfully(tmp_path) -> None:
    (tmp_path / "index.html").write_text("hello http helper", encoding="utf-8")
    with _serve_directory(tmp_path) as port:
        result = await execute_verification_helper(
            "http_assert",
            {
                "url": f"http://127.0.0.1:{port}/index.html",
                "contains_text": "hello http helper",
            },
            ctx=VerificationHelperContext(workspace=tmp_path),
        )

    assert result.success is True
    assert result.capability == "service_runtime"
    assert result.data["status_code"] == 200


@pytest.mark.asyncio
async def test_builtin_browser_assert_reports_semantic_failure(tmp_path) -> None:
    (tmp_path / "index.html").write_text("plain content", encoding="utf-8")
    with _serve_directory(tmp_path) as port:
        result = await execute_verification_helper(
            "browser_assert",
            {
                "url": f"http://127.0.0.1:{port}/index.html",
                "contains_text": "missing marker",
            },
            ctx=VerificationHelperContext(workspace=tmp_path),
        )

    assert result.success is False
    assert result.reason_code == "dev_browser_check_failed"
    assert result.capability == "browser_runtime"


@pytest.mark.asyncio
async def test_builtin_http_assert_reports_semantic_failure(tmp_path) -> None:
    (tmp_path / "index.html").write_text("http helper output", encoding="utf-8")
    with _serve_directory(tmp_path) as port:
        result = await execute_verification_helper(
            "http_assert",
            {
                "url": f"http://127.0.0.1:{port}/index.html",
                "contains_text": "missing text",
            },
            ctx=VerificationHelperContext(workspace=tmp_path),
        )

    assert result.success is False
    assert result.reason_code == "dev_contract_failed"
    assert result.capability == "service_runtime"


@pytest.mark.asyncio
async def test_builtin_browser_session_executes_local_form_flow(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "loom.engine.verification_helpers.browser_addon_status",
        lambda: OptionalAddonStatus(
            key="browser",
            label="Browser Addon",
            installed=False,
            required_for="Full JS-capable browser_session execution",
            install_hint="uv sync --extra browser",
            detail="Playwright package is not installed.",
        ),
    )
    (tmp_path / "index.html").write_text(
        '<a href="/form.html">Start</a>',
        encoding="utf-8",
    )
    (tmp_path / "form.html").write_text(
        (
            '<form method="get" action="/result.html">'
            '<label>Email<input name="email" /></label>'
            '<button type="submit">Submit</button>'
            "</form>"
        ),
        encoding="utf-8",
    )
    (tmp_path / "result.html").write_text(
        "Success page all clear",
        encoding="utf-8",
    )
    with _serve_directory(tmp_path) as port:
        result = await execute_verification_helper(
            "browser_session",
            {
                "start_url": f"http://127.0.0.1:{port}/index.html",
                "steps": [
                    {"action": "open", "url": f"http://127.0.0.1:{port}/index.html"},
                    {"action": "click_text", "target": "Start"},
                    {"action": "fill_field", "field": "Email", "value": "test@example.com"},
                    {"action": "submit"},
                    {"action": "assert_text", "text": "Success", "present": True},
                    {"action": "assert_text", "text": "Error", "present": False},
                ],
                "capture_network": True,
                "capture_console": True,
                "capture_screenshot": True,
            },
            ctx=VerificationHelperContext(
                workspace=tmp_path,
                metadata={"subtask_id": "browser-test"},
            ),
        )

    assert result.success is True
    assert result.capability == "browser_runtime"
    assert result.data["engine"] == "fallback"
    assert result.data["browser_addon"]["installed"] is False
    assert result.data["current_url"].endswith("/result.html?email=test%40example.com")
    assert len(result.data["network_requests"]) >= 3
    assert result.data["captured_network"] == result.data["network_requests"]
    assert result.data["console_logs"] == []
    assert result.data["dom_snapshot_path"] == (
        "artifacts/browser-session-dom-snapshot-browser-test.html"
    )
    assert any("console capture is unavailable" in item for item in result.data["warnings"])
    assert any("DOM snapshot" in item for item in result.data["warnings"])
    assert any("Playwright addon is not installed" in item for item in result.data["warnings"])
    assert (tmp_path / "artifacts" / "browser-session-dom-snapshot-browser-test.html").exists()


@pytest.mark.asyncio
async def test_browser_session_prefers_playwright_when_addon_is_available(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_playwright_executor(_args, _ctx) -> VerificationHelperResult:
        return VerificationHelperResult(
            success=True,
            detail="browser_session completed with Playwright",
            capability="browser_runtime",
            data={"current_url": "http://127.0.0.1:9999/index.html"},
        )

    async def _fake_fallback_executor(_args, _ctx) -> VerificationHelperResult:
        raise AssertionError("fallback executor should not run when Playwright succeeds")

    monkeypatch.setattr(
        "loom.engine.verification_helpers.browser_addon_status",
        lambda: OptionalAddonStatus(
            key="browser",
            label="Browser Addon",
            installed=True,
            required_for="Full JS-capable browser_session execution",
            install_hint="uv sync --extra browser",
            detail="Playwright package importable.",
        ),
    )
    monkeypatch.setattr(
        "loom.engine.verification_helpers._playwright_browser_session_executor",
        _fake_playwright_executor,
    )
    monkeypatch.setattr(
        "loom.engine.verification_helpers._browser_session_fallback_executor",
        _fake_fallback_executor,
    )

    result = await execute_verification_helper(
        "browser_session",
        {
            "start_url": "http://127.0.0.1:9999/index.html",
            "steps": [{"action": "open", "url": "http://127.0.0.1:9999/index.html"}],
        },
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.data["engine"] == "playwright"
    assert result.data["browser_addon"]["installed"] is True
    assert "warnings" not in result.data


@pytest.mark.asyncio
async def test_browser_session_falls_back_when_playwright_backend_is_unavailable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_playwright_executor(_args, _ctx) -> VerificationHelperResult:
        raise RuntimeError("chromium executable missing")

    async def _fake_fallback_executor(_args, _ctx) -> VerificationHelperResult:
        return VerificationHelperResult(
            success=True,
            detail="browser_session completed with fallback",
            capability="browser_runtime",
            data={"current_url": "http://127.0.0.1:9999/index.html"},
        )

    monkeypatch.setattr(
        "loom.engine.verification_helpers.browser_addon_status",
        lambda: OptionalAddonStatus(
            key="browser",
            label="Browser Addon",
            installed=True,
            required_for="Full JS-capable browser_session execution",
            install_hint="uv sync --extra browser",
            detail="Playwright package importable.",
        ),
    )
    monkeypatch.setattr(
        "loom.engine.verification_helpers._playwright_browser_session_executor",
        _fake_playwright_executor,
    )
    monkeypatch.setattr(
        "loom.engine.verification_helpers._browser_session_fallback_executor",
        _fake_fallback_executor,
    )

    result = await execute_verification_helper(
        "browser_session",
        {
            "start_url": "http://127.0.0.1:9999/index.html",
            "steps": [{"action": "open", "url": "http://127.0.0.1:9999/index.html"}],
        },
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    assert result.success is True
    assert result.data["engine"] == "fallback"
    assert result.data["browser_addon"]["installed"] is True
    assert any(
        "Playwright addon is installed but unavailable at runtime" in item
        for item in result.data["warnings"]
    )


@pytest.mark.asyncio
async def test_builtin_probe_suite_executes_multiple_checks_successfully(tmp_path) -> None:
    (tmp_path / "index.html").write_text("home page", encoding="utf-8")
    (tmp_path / "pricing.html").write_text("pricing page", encoding="utf-8")
    with _serve_directory(tmp_path) as port:
        result = await execute_verification_helper(
            "probe_suite",
            {
                "checks": [
                    {
                        "url": f"http://127.0.0.1:{port}/index.html",
                        "capability": "browser_runtime",
                        "contains_text": "home page",
                    },
                    {
                        "url": f"http://127.0.0.1:{port}/pricing.html",
                        "capability": "service_runtime",
                        "contains_text": "pricing page",
                    },
                ],
            },
            ctx=VerificationHelperContext(workspace=tmp_path),
        )

    assert result.success is True
    assert result.data["check_count"] == 2
    assert result.data["helper_capabilities"] == [
        "browser_runtime",
        "service_runtime",
    ]


@pytest.mark.asyncio
async def test_builtin_probe_suite_surfaces_first_failure(tmp_path) -> None:
    (tmp_path / "index.html").write_text("home page", encoding="utf-8")
    (tmp_path / "pricing.html").write_text("pricing page", encoding="utf-8")
    with _serve_directory(tmp_path) as port:
        result = await execute_verification_helper(
            "probe_suite",
            {
                "checks": [
                    {
                        "url": f"http://127.0.0.1:{port}/index.html",
                        "capability": "browser_runtime",
                        "contains_text": "home page",
                    },
                    {
                        "url": f"http://127.0.0.1:{port}/pricing.html",
                        "capability": "service_runtime",
                        "contains_text": "missing page copy",
                    },
                ],
            },
            ctx=VerificationHelperContext(workspace=tmp_path),
        )

    assert result.success is False
    assert result.reason_code == "dev_contract_failed"
    assert result.data["failed_check_index"] == 2


@pytest.mark.asyncio
async def test_builtin_run_test_suite_returns_dev_test_failure_reason(tmp_path) -> None:
    result = await execute_verification_helper(
        "run_test_suite",
        {"command": "false"},
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    assert result.success is False
    assert result.reason_code == "dev_test_failed"
    assert result.capability == "command_execution"
    assert result.data["exit_code"] != 0


@pytest.mark.asyncio
async def test_builtin_render_verification_report_writes_markdown(tmp_path) -> None:
    result = await execute_verification_helper(
        "render_verification_report",
        {
            "title": "UI Validation",
            "canonical_result": {"passed": 15, "total": 16, "failed": 1},
            "output_path": "ui-integration-validation-report.md",
        },
        ctx=VerificationHelperContext(workspace=tmp_path),
    )

    report_path = tmp_path / "ui-integration-validation-report.md"
    assert result.success is True
    assert report_path.exists()
    markdown = report_path.read_text()
    assert "# UI Validation" in markdown
    assert "**Score: 15/16 tests passed (94%)**" in markdown
