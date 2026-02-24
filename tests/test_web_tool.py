"""Tests for the web fetch tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.registry import ToolContext, ToolResult
from loom.tools.web import (
    DEFAULT_WEB_USER_AGENT,
    WebFetchHtmlTool,
    WebFetchTool,
    _build_request_headers,
    _execute_web_fetch,
    _looks_like_html,
    _should_retry_status,
    _strip_html,
    is_safe_url,
)

# --- URL Safety ---


class TestURLSafety:
    def test_blocks_localhost(self):
        safe, reason = is_safe_url("http://localhost:8080/api")
        assert not safe
        assert "Blocked" in reason

    def test_blocks_127(self):
        safe, reason = is_safe_url("http://127.0.0.1/secret")
        assert not safe

    def test_blocks_10_network(self):
        safe, reason = is_safe_url("http://10.0.0.1/internal")
        assert not safe

    def test_blocks_172_network(self):
        safe, reason = is_safe_url("http://172.16.0.1/admin")
        assert not safe

    def test_blocks_192_168(self):
        safe, reason = is_safe_url("http://192.168.1.1/config")
        assert not safe

    def test_blocks_ftp(self):
        safe, reason = is_safe_url("ftp://example.com/file")
        assert not safe
        assert "http" in reason.lower()

    def test_blocks_no_scheme(self):
        safe, reason = is_safe_url("example.com/page")
        assert not safe

    def test_allows_https(self):
        safe, reason = is_safe_url("https://example.com/api")
        assert safe
        assert reason == ""

    def test_allows_http(self):
        safe, reason = is_safe_url("http://example.com/page")
        assert safe

    def test_allows_public_ip(self):
        safe, reason = is_safe_url("http://8.8.8.8/dns")
        assert safe

    def test_blocks_zero_address(self):
        safe, reason = is_safe_url("http://0.0.0.0/")
        assert not safe


# --- HTML Stripping ---


class TestStripHtml:
    def test_removes_tags(self):
        html = "<p>Hello <b>world</b></p>"
        result = _strip_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_removes_script(self):
        html = "<script>alert('xss')</script><p>Safe</p>"
        result = _strip_html(html)
        assert "alert" not in result
        assert "Safe" in result

    def test_removes_style(self):
        html = "<style>body{color:red}</style><p>Content</p>"
        result = _strip_html(html)
        assert "color" not in result
        assert "Content" in result

    def test_decodes_entities(self):
        html = "&amp; &lt; &gt; &quot; &#39;"
        result = _strip_html(html)
        assert "&" in result
        assert "<" in result
        assert ">" in result

    def test_collapses_whitespace(self):
        html = "<p>Hello</p>\n\n\n<p>World</p>"
        result = _strip_html(html)
        # Should not have excessive whitespace
        assert "\n\n\n" not in result

    def test_empty_input(self):
        assert _strip_html("") == ""

    def test_preserves_line_breaks_for_blocks(self):
        html = "<h1>Title</h1><p>Para 1</p><p>Para 2</p>"
        result = _strip_html(html)
        assert result.splitlines() == ["Title", "Para 1", "Para 2"]

    def test_removes_html_comments(self):
        html = "<!-- hidden --><p>Visible</p>"
        result = _strip_html(html)
        assert "hidden" not in result
        assert "Visible" in result


class TestHtmlDetection:
    def test_detects_by_content_type(self):
        assert _looks_like_html("plain", "text/html; charset=utf-8")

    def test_detects_doctype_when_mislabeled(self):
        content = "<!DOCTYPE html><html><body><p>Hello</p></body></html>"
        assert _looks_like_html(content, "text/plain")

    def test_detects_common_html_tags_when_mislabeled(self):
        content = "<html><head><title>T</title></head><body><main>ok</main></body></html>"
        assert _looks_like_html(content, "application/octet-stream")

    def test_does_not_flag_plain_text_with_brackets(self):
        content = "Use x < y and z > y in this plain text."
        assert not _looks_like_html(content, "text/plain")


# --- DNS-based SSRF check ---


class TestSSRFDnsResolution:
    def test_blocks_ipv6_loopback(self):
        safe, reason = is_safe_url("http://[::1]:8080/")
        assert not safe
        assert "Blocked" in reason

    def test_private_ip_helper(self):
        from loom.tools.web import _is_private_ip
        assert _is_private_ip("127.0.0.1")
        assert _is_private_ip("10.0.0.1")
        assert _is_private_ip("192.168.1.1")
        assert _is_private_ip("172.16.0.1")
        assert _is_private_ip("::1")
        assert not _is_private_ip("8.8.8.8")


class TestWebRequestDefaults:
    def test_build_request_headers_default_ua(self, monkeypatch):
        monkeypatch.delenv("LOOM_WEB_USER_AGENT", raising=False)
        headers = _build_request_headers()
        assert headers["User-Agent"] == DEFAULT_WEB_USER_AGENT
        assert "Accept" in headers
        assert "Accept-Language" in headers

    def test_build_request_headers_env_override(self, monkeypatch):
        monkeypatch.setenv("LOOM_WEB_USER_AGENT", "MyAgent/9.9")
        headers = _build_request_headers()
        assert headers["User-Agent"] == "MyAgent/9.9"

    def test_retryable_status_codes(self):
        assert _should_retry_status(403)
        assert _should_retry_status(429)
        assert _should_retry_status(503)
        assert not _should_retry_status(404)


class TestWebToolSchemas:
    def test_web_fetch_schema_is_text_first(self):
        tool = WebFetchTool()
        props = tool.parameters.get("properties", {})
        assert "url" in props
        assert "extract_text" not in props

    def test_web_fetch_html_schema(self):
        tool = WebFetchHtmlTool()
        props = tool.parameters.get("properties", {})
        assert "url" in props
        assert tool.name == "web_fetch_html"


class TestWebToolHiddenRuntimeArgs:
    @pytest.mark.asyncio
    async def test_web_fetch_forwards_retention_overrides(self, monkeypatch):
        captured: dict[str, object] = {}

        async def _fake_execute_web_fetch(
            url: str,
            *,
            extract_text: bool,
            max_download_bytes: int,
            enable_filetype_ingest_router: bool,
            artifact_retention_max_age_days: int,
            artifact_retention_max_files_per_scope: int,
            artifact_retention_max_bytes_per_scope: int,
            ctx,
        ):
            captured["url"] = url
            captured["extract_text"] = extract_text
            captured["max_download_bytes"] = max_download_bytes
            captured["enable_filetype_ingest_router"] = enable_filetype_ingest_router
            captured["artifact_retention_max_age_days"] = artifact_retention_max_age_days
            captured["artifact_retention_max_files_per_scope"] = (
                artifact_retention_max_files_per_scope
            )
            captured["artifact_retention_max_bytes_per_scope"] = (
                artifact_retention_max_bytes_per_scope
            )
            captured["ctx"] = ctx
            return ToolResult.ok("ok")

        monkeypatch.setattr("loom.tools.web._execute_web_fetch", _fake_execute_web_fetch)
        tool = WebFetchTool()
        ctx = ToolContext(workspace=None)
        result = await tool.execute(
            {
                "url": "https://example.com/report.pdf",
                "_enable_filetype_ingest_router": False,
                "_artifact_retention_max_age_days": 31,
                "_artifact_retention_max_files_per_scope": 120,
                "_artifact_retention_max_bytes_per_scope": 42_000_000,
            },
            ctx,
        )
        assert result.success
        assert captured["url"] == "https://example.com/report.pdf"
        assert captured["extract_text"] is True
        assert captured["enable_filetype_ingest_router"] is False
        assert captured["artifact_retention_max_age_days"] == 31
        assert captured["artifact_retention_max_files_per_scope"] == 120
        assert captured["artifact_retention_max_bytes_per_scope"] == 42_000_000


class _FakeResponse:
    def __init__(self, url: str, headers: dict[str, str], status_code: int = 200):
        self.url = url
        self.headers = headers
        self.status_code = status_code
        self.encoding = "utf-8"
        self.is_redirect = False

    async def aclose(self):
        return None

    def raise_for_status(self):
        return None


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class TestWebFetchBinaryRouting:
    @pytest.mark.asyncio
    async def test_pdf_fetch_returns_artifact_summary(self, monkeypatch, tmp_path: Path):
        fake_response = _FakeResponse(
            "https://example.com/report.pdf",
            headers={
                "content-type": "application/pdf",
                "content-length": "128",
            },
        )

        async def _fake_get_with_retries(client, url, headers=None, stream=False):
            del client, url, headers, stream
            return fake_response

        async def _fake_read_response_limited(response, max_bytes):
            del response, max_bytes
            # Minimal PDF-like bytes; extraction may fail, but routing still persists artifact.
            return (b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n", False)

        monkeypatch.setattr("loom.tools.web.httpx.AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr("loom.tools.web._get_with_retries", _fake_get_with_retries)
        monkeypatch.setattr("loom.tools.web._read_response_limited", _fake_read_response_limited)

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        ctx = ToolContext(
            workspace=workspace,
            read_roots=[],
            scratch_dir=tmp_path / "scratch",
            changelog=None,
            subtask_id="subtask-1",
            auth_context=None,
        )
        result = await _execute_web_fetch(
            "https://example.com/report.pdf",
            extract_text=True,
            max_download_bytes=1024,
            enable_filetype_ingest_router=True,
            ctx=ctx,
        )

        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data.get("content_kind") == "pdf"
        assert result.data.get("artifact_ref")
        assert isinstance(result.data.get("artifact_retention"), dict)
        assert set(result.data.get("artifact_retention", {})) == {
            "scopes_scanned",
            "files_deleted",
            "bytes_deleted",
        }
        artifact_path = Path(str(result.data.get("artifact_path", "")))
        assert artifact_path.exists()
        assert "Fetched PDF artifact" in result.output

    @pytest.mark.asyncio
    async def test_pdf_fetch_uses_legacy_decode_when_router_disabled(self, monkeypatch):
        fake_response = _FakeResponse(
            "https://example.com/report.pdf",
            headers={"content-type": "application/pdf"},
        )

        async def _fake_get_with_retries(client, url, headers=None, stream=False):
            del client, url, headers, stream
            return fake_response

        async def _fake_read_response_limited(response, max_bytes):
            del response, max_bytes
            return (b"%PDF-1.4\nraw-bytes", False)

        monkeypatch.setattr("loom.tools.web.httpx.AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr("loom.tools.web._get_with_retries", _fake_get_with_retries)
        monkeypatch.setattr("loom.tools.web._read_response_limited", _fake_read_response_limited)

        result = await _execute_web_fetch(
            "https://example.com/report.pdf",
            extract_text=True,
            max_download_bytes=1024,
            enable_filetype_ingest_router=False,
            ctx=None,
        )

        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data.get("content_kind") == "text"
        assert "%PDF-1.4" in result.output
