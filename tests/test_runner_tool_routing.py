"""Focused tests for process-scoped runner tool routing."""

from __future__ import annotations

from pathlib import Path

from loom.engine.runner.tool_routing import route_tool_call_for_process
from loom.engine.verification_helpers import (
    _browser_session_executor,
    bind_verification_helper,
    unbind_verification_helper,
)
from loom.processes.schema import ProcessDefinition, VerificationPolicyContract


def _make_process(tool_success_policy: str) -> ProcessDefinition:
    return ProcessDefinition(
        name="build-process",
        verification_policy=VerificationPolicyContract(
            static_checks={"tool_success_policy": tool_success_policy},
        ),
    )


def test_route_tool_call_for_process_rewrites_test_commands() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="shell_execute",
        tool_args={"command": "pytest -q"},
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
        subtask_id="subtask-1",
        execution_surface="cli",
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "run_test_suite"
    assert tool_args["args"] == {"command": "pytest -q"}
    assert metadata["routed_from_tool"] == "shell_execute"
    assert metadata["helper"] == "run_test_suite"


def test_route_tool_call_for_process_rewrites_build_commands() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="shell_execute",
        tool_args={"command": "npm run build"},
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "run_build_check"
    assert metadata["helper"] == "run_build_check"


def test_route_tool_call_for_process_rewrites_local_service_probe_commands() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="shell_execute",
        tool_args={
            "command": (
                "cd dist && python3 -m http.server 8080 &\n"
                "sleep 2\n"
                "curl -s -o /dev/null -w \"%{http_code}\" "
                "http://127.0.0.1:8080/index.html"
            ),
        },
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "serve_static"
    assert tool_args["args"] == {
        "command": "cd dist && python3 -m http.server 8080",
        "ready_url": "http://127.0.0.1:8080/index.html",
    }
    assert metadata["helper"] == "serve_static"


def test_route_tool_call_for_process_rewrites_multi_url_service_probe_commands() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="shell_execute",
        tool_args={
            "command": (
                "cd dist && python3 -m http.server 8080 &\n"
                "sleep 2\n"
                "curl -s http://127.0.0.1:8080/index.html\n"
                "curl -s http://127.0.0.1:8080/pricing.html"
            ),
        },
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "serve_static"
    assert tool_args["args"] == {
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
    assert metadata["helper"] == "serve_static"


def test_route_tool_call_for_process_rewrites_local_http_probe_commands() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="shell_execute",
        tool_args={
            "command": (
                "curl -s -o /dev/null -w \"%{http_code}\" "
                "http://127.0.0.1:8080/index.html"
            ),
        },
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "http_assert"
    assert tool_args["args"] == {
        "url": "http://127.0.0.1:8080/index.html",
    }
    assert metadata["helper"] == "http_assert"


def test_route_tool_call_for_process_rewrites_multi_url_http_probe_commands() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="shell_execute",
        tool_args={
            "command": (
                "curl -s http://127.0.0.1:8080/index.html\n"
                "curl -s http://127.0.0.1:8080/pricing.html"
            ),
        },
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "probe_suite"
    assert tool_args["args"] == {
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
    assert metadata["helper"] == "probe_suite"


def test_route_tool_call_for_process_rewrites_simple_provider_probe_prompt() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="claude_code",
        tool_args={
            "prompt": (
                "Verify that http://127.0.0.1:8080/index.html loads and "
                "contains \"hello microsite\"."
            ),
        },
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "browser_assert"
    assert tool_args["args"] == {
        "url": "http://127.0.0.1:8080/index.html",
        "contains_text": ["hello microsite"],
    }
    assert metadata["helper"] == "browser_assert"
    assert metadata["routed_from_tool"] == "claude_code"


def test_route_tool_call_for_process_leaves_advanced_provider_probe_prompt_alone() -> None:
    unbind_verification_helper("browser_session")
    try:
        tool_name, tool_args, metadata = route_tool_call_for_process(
            tool_name="claude_code",
            tool_args={
                "prompt": (
                    "Run Playwright headless browser tests against "
                    "http://127.0.0.1:8080/index.html, capture console logs, "
                    "and take a screenshot."
                ),
            },
            process=_make_process("development_balanced"),
            workspace=Path("/tmp"),
        )
    finally:
        bind_verification_helper("browser_session", _browser_session_executor)

    assert tool_name == "claude_code"
    assert tool_args["prompt"].startswith("Run Playwright headless browser tests")
    assert metadata == {}


def test_route_tool_call_for_process_rewrites_advanced_provider_probe_when_session_bound() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="claude_code",
        tool_args={
            "prompt": (
                "Run Playwright headless browser tests against "
                "http://127.0.0.1:8080/index.html, capture console logs, "
                "capture network requests, click \"Start\", fill "
                "\"Email\" with \"test@example.com\", submit, and take a "
                "screenshot without \"Error\"."
            ),
        },
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "browser_session"
    assert tool_args["args"] == {
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
    assert metadata["helper"] == "browser_session"
    assert metadata["routed_from_tool"] == "claude_code"


def test_route_tool_call_for_process_rewrites_multi_url_provider_probe_prompt() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="claude_code",
        tool_args={
            "prompt": (
                "Verify that http://127.0.0.1:8080/index.html loads and "
                "http://127.0.0.1:8080/pricing.html loads."
            ),
        },
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "probe_suite"
    assert tool_args["args"] == {
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
    assert metadata["helper"] == "probe_suite"
    assert metadata["routed_from_tool"] == "claude_code"


def test_route_tool_call_for_process_leaves_non_matching_commands_alone() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="shell_execute",
        tool_args={"command": "ls -la"},
        process=_make_process("development_balanced"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "shell_execute"
    assert tool_args == {"command": "ls -la"}
    assert metadata == {}


def test_route_tool_call_for_process_is_disabled_outside_dev_profile() -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="shell_execute",
        tool_args={"command": "pytest -q"},
        process=_make_process("all_tools_hard"),
        workspace=Path("/tmp"),
    )

    assert tool_name == "shell_execute"
    assert tool_args == {"command": "pytest -q"}
    assert metadata == {}


def test_route_tool_call_for_process_rewrites_validation_report_write(tmp_path: Path) -> None:
    (tmp_path / "runtime-validation-results.json").write_text(
        '{"passed": 15, "failed": 1}',
        encoding="utf-8",
    )

    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="write_file",
        tool_args={
            "path": "reports/ui-integration-validation-report.md",
            "content": "manual report",
        },
        process=_make_process("development_balanced"),
        workspace=tmp_path,
    )

    assert tool_name == "verification_helper"
    assert tool_args["helper"] == "render_verification_report"
    assert tool_args["args"]["output_path"] == "reports/ui-integration-validation-report.md"
    assert tool_args["args"]["canonical_result"] == {"passed": 15, "failed": 1, "total": 16}
    assert metadata["helper"] == "render_verification_report"


def test_route_tool_call_for_process_leaves_validation_report_write_without_json(
    tmp_path: Path,
) -> None:
    tool_name, tool_args, metadata = route_tool_call_for_process(
        tool_name="write_file",
        tool_args={
            "path": "reports/ui-integration-validation-report.md",
            "content": "manual report",
        },
        process=_make_process("development_balanced"),
        workspace=tmp_path,
    )

    assert tool_name == "write_file"
    assert tool_args["path"] == "reports/ui-integration-validation-report.md"
    assert metadata == {}
