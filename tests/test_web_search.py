"""Tests for the web search tool."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import loom.tools.web_search as web_search_module
from loom.tools.registry import ToolContext
from loom.tools.web_search import (
    _SEARCH_INFLIGHT,
    _SEARCH_RESULT_CACHE,
    BING_URL,
    DDG_COOLDOWN_SECONDS,
    DEFAULT_SEARCH_USER_AGENT,
    WebSearchTool,
    _build_search_headers,
    _clean_ddg_url,
    _ddg_cooldown_remaining,
    _is_retryable_search_status,
    _parse_bing_html,
    _parse_ddg_html,
    _record_ddg_cooldown,
    _search_ddg,
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


class TestParseBingHtml:
    def test_parses_bing_result_blocks(self):
        html = """
        <li class="b_algo">
          <h2><a href="https://example.com/page">Example Result</a></h2>
          <div><p>Example snippet from Bing.</p></div>
        </li>
        """
        results = _parse_bing_html(html, max_results=5)
        assert len(results) == 1
        assert results[0]["title"] == "Example Result"
        assert results[0]["url"] == "https://example.com/page"
        assert "Bing" in results[0]["snippet"]


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
        assert headers["User-Agent"].startswith("Mozilla/5.0")
        assert "Referer" in headers
        assert headers["Upgrade-Insecure-Requests"] == "1"
        assert headers["Sec-Fetch-Dest"] == "document"
        assert headers["Sec-Fetch-Mode"] == "navigate"
        assert headers["Sec-Fetch-Site"] == "same-origin"

    def test_build_search_headers_env_override(self, monkeypatch):
        monkeypatch.setenv("LOOM_WEB_USER_AGENT", "MyAgent/9.9")
        headers = _build_search_headers()
        assert headers["User-Agent"] == "MyAgent/9.9"

    def test_retryable_search_status_codes(self):
        assert not _is_retryable_search_status(403)
        assert not _is_retryable_search_status(429)
        assert _is_retryable_search_status(503)
        assert not _is_retryable_search_status(404)


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class TestSearchFallbacks:
    @pytest.fixture(autouse=True)
    def _reset_search_state(self):
        _SEARCH_RESULT_CACHE.clear()
        _SEARCH_INFLIGHT.clear()
        yield
        _SEARCH_RESULT_CACHE.clear()
        _SEARCH_INFLIGHT.clear()
        web_search_module._DDG_COOLDOWN_UNTIL = 0.0

    @pytest.mark.asyncio
    async def test_search_falls_back_to_bing_when_duckduckgo_unavailable(self, monkeypatch):
        calls: list[tuple[str, str]] = []
        bing_html = """
        <li class="b_algo">
          <h2><a href="https://example.com/blink49">Blink49 Studios</a></h2>
          <div><p>Studio overview page.</p></div>
        </li>
        """

        async def _fake_query_search_endpoint(
            client,
            method,
            endpoint,
            query,
            *,
            deadline,
            referer,
            provider_name,
        ):
            del client, query, deadline, referer, provider_name
            calls.append((method, endpoint))
            if "duckduckgo" in endpoint:
                raise httpx.TimeoutException("duckduckgo timed out")
            if endpoint == BING_URL:
                return bing_html
            raise AssertionError(f"Unexpected endpoint: {endpoint}")

        monkeypatch.setattr("loom.tools.web_search.httpx.AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr(
            "loom.tools.web_search._query_search_endpoint",
            _fake_query_search_endpoint,
        )

        results = await _search_ddg("blink49", 5)

        assert results
        assert results[0]["title"] == "Blink49 Studios"
        assert any("duckduckgo" in endpoint for _, endpoint in calls)
        assert any(endpoint == BING_URL for _, endpoint in calls)

    @pytest.mark.asyncio
    async def test_search_caches_identical_query_results(self, monkeypatch):
        calls: list[tuple[str, str]] = []
        bing_html = """
        <li class="b_algo">
          <h2><a href="https://example.com/cached">Cached Result</a></h2>
          <div><p>Cached snippet.</p></div>
        </li>
        """

        async def _fake_query_search_endpoint(
            client,
            method,
            endpoint,
            query,
            *,
            deadline,
            referer,
            provider_name,
        ):
            del client, query, deadline, referer, provider_name
            calls.append((method, endpoint))
            if "duckduckgo" in endpoint:
                raise httpx.TimeoutException("duckduckgo timed out")
            if endpoint == BING_URL:
                return bing_html
            raise AssertionError(f"Unexpected endpoint: {endpoint}")

        monkeypatch.setattr("loom.tools.web_search.httpx.AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr(
            "loom.tools.web_search._query_search_endpoint",
            _fake_query_search_endpoint,
        )

        first = await _search_ddg("cached query", 5)
        second = await _search_ddg("cached   query", 5)

        assert first == second
        assert len([endpoint for _, endpoint in calls if endpoint == BING_URL]) == 1

    @pytest.mark.asyncio
    async def test_search_skips_duckduckgo_during_cooldown(self, monkeypatch):
        calls: list[tuple[str, str]] = []
        bing_html = """
        <li class="b_algo">
          <h2><a href="https://example.com/bing-only">Bing Result</a></h2>
          <div><p>Bing snippet.</p></div>
        </li>
        """

        async def _fake_query_search_endpoint(
            client,
            method,
            endpoint,
            query,
            *,
            deadline,
            referer,
            provider_name,
        ):
            del client, query, deadline, referer, provider_name
            calls.append((method, endpoint))
            if endpoint == BING_URL:
                return bing_html
            raise AssertionError(f"Unexpected endpoint during cooldown: {endpoint}")

        monkeypatch.setattr("loom.tools.web_search.httpx.AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr(
            "loom.tools.web_search._query_search_endpoint",
            _fake_query_search_endpoint,
        )

        _record_ddg_cooldown()
        assert 0 < _ddg_cooldown_remaining() <= DDG_COOLDOWN_SECONDS

        results = await _search_ddg("bing during cooldown", 5)

        assert results
        assert results[0]["title"] == "Bing Result"
        assert calls == [("GET", BING_URL)]

    @pytest.mark.asyncio
    async def test_ddg_rate_limit_enters_cooldown_and_falls_back_to_bing(self, monkeypatch):
        calls: list[tuple[str, str]] = []
        bing_html = """
        <li class="b_algo">
          <h2><a href="https://example.com/after-cooldown">Recovered Result</a></h2>
          <div><p>Recovered snippet.</p></div>
        </li>
        """

        async def _fake_query_search_endpoint(
            client,
            method,
            endpoint,
            query,
            *,
            deadline,
            referer,
            provider_name,
        ):
            del client, query, deadline, referer
            calls.append((method, endpoint))
            if provider_name == "duckduckgo":
                web_search_module._record_ddg_cooldown()
                raise web_search_module.SearchProviderCooldownError(
                    "DuckDuckGo denied search request (HTTP 429)"
                )
            if endpoint == BING_URL:
                return bing_html
            raise AssertionError(f"Unexpected endpoint: {endpoint}")

        monkeypatch.setattr("loom.tools.web_search.httpx.AsyncClient", _FakeAsyncClient)
        monkeypatch.setattr(
            "loom.tools.web_search._query_search_endpoint",
            _fake_query_search_endpoint,
        )

        results = await _search_ddg("rate limited query", 5)

        assert results
        assert results[0]["title"] == "Recovered Result"
        assert any("duckduckgo" in endpoint for _, endpoint in calls)
        assert any(endpoint == BING_URL for _, endpoint in calls)
        assert _ddg_cooldown_remaining() > 0
