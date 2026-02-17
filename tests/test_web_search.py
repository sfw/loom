"""Tests for the web search tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.registry import ToolContext
from loom.tools.web_search import (
    DEFAULT_SEARCH_USER_AGENT,
    WebSearchTool,
    _build_search_headers,
    _clean_ddg_url,
    _is_retryable_search_status,
    _parse_ddg_html,
    _strip_tags,
)


@pytest.fixture
def tool():
    return WebSearchTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


# --- Unit tests for parsing helpers ---


class TestStripTags:
    def test_basic(self):
        assert _strip_tags("<b>hello</b>") == "hello"

    def test_entities(self):
        assert _strip_tags("&amp; &lt; &gt;") == "& < >"

    def test_nested(self):
        assert _strip_tags("<a href='x'><b>text</b></a>") == "text"

    def test_whitespace(self):
        assert _strip_tags("  hello   world  ") == "hello world"


class TestCleanDdgUrl:
    def test_direct_url(self):
        assert _clean_ddg_url("https://example.com") == "https://example.com"

    def test_protocol_relative(self):
        assert _clean_ddg_url("//example.com/path") == "https://example.com/path"

    def test_redirect_url(self):
        url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc"
        result = _clean_ddg_url(url)
        assert result == "https://example.com/page"


class TestParseDdgHtml:
    def test_parses_result_a_links(self):
        html = '''
        <a class="result__a" href="https://example.com">Example Title</a>
        <a class="result__snippet" href="#">This is a snippet about the result.</a>
        '''
        results = _parse_ddg_html(html, max_results=5)
        assert len(results) == 1
        assert results[0]["title"] == "Example Title"
        assert results[0]["url"] == "https://example.com"
        assert "snippet" in results[0]["snippet"].lower() or results[0]["snippet"] != ""

    def test_max_results_respected(self):
        html = ""
        for i in range(5):
            html += f'<a class="result__a" href="https://example.com/{i}">Title {i}</a>\n'
        results = _parse_ddg_html(html, max_results=3)
        assert len(results) <= 3

    def test_empty_html(self):
        results = _parse_ddg_html("", max_results=5)
        assert results == []

    def test_no_results(self):
        html = "<html><body>No results found</body></html>"
        results = _parse_ddg_html(html, max_results=5)
        assert results == []


class TestWebSearchTool:
    async def test_empty_query(self, tool, ctx):
        result = await tool.execute({"query": ""}, ctx)
        assert not result.success
        assert "No search query" in result.output or "No search query" in (result.error or "")

    async def test_whitespace_query(self, tool, ctx):
        result = await tool.execute({"query": "   "}, ctx)
        assert not result.success

    def test_schema(self, tool):
        schema = tool.schema()
        assert schema["name"] == "web_search"
        assert "query" in schema["parameters"]["properties"]
        assert "max_results" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]

    def test_timeout(self, tool):
        assert tool.timeout_seconds == 30


class TestSearchRequestDefaults:
    def test_build_search_headers_default_ua(self, monkeypatch):
        monkeypatch.delenv("LOOM_WEB_USER_AGENT", raising=False)
        headers = _build_search_headers()
        assert headers["User-Agent"] == DEFAULT_SEARCH_USER_AGENT
        assert "Referer" in headers

    def test_build_search_headers_env_override(self, monkeypatch):
        monkeypatch.setenv("LOOM_WEB_USER_AGENT", "MyAgent/9.9")
        headers = _build_search_headers()
        assert headers["User-Agent"] == "MyAgent/9.9"

    def test_retryable_search_status_codes(self):
        assert _is_retryable_search_status(403)
        assert _is_retryable_search_status(429)
        assert _is_retryable_search_status(503)
        assert not _is_retryable_search_status(404)
