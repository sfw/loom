"""Tests for the auth-free web search tool."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

import loom.tools.web_search as web_search_module
from loom.state.memory import Database
from loom.state.search_provider_state import SearchProviderStateStore
from loom.tools.registry import ToolContext
from loom.tools.search_backend import (
    SearchBackend,
    SearchBackendClient,
    SearchProvider,
    SearchProviderError,
    SearchRegistry,
    _infer_search_locale,
    _parse_bing_html,
    _parse_duckduckgo_html,
)
from loom.tools.web_search import WebSearchTool

BING_HTML = """
<html>
  <body>
    <ol>
      <li class="b_algo">
        <div class="b_tpcn">
          <a class="tilk" href="https://www.bing.com/ck/a?!&&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9zaXRl">
            example.com
          </a>
        </div>
        <h2>
          <a href="https://www.bing.com/ck/a?!&&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9iaW5nLWRvY3M">
            <strong>Bing Docs</strong>
          </a>
        </h2>
        <div class="b_caption"><p>Primary provider result.</p></div>
      </li>
      <li class="b_algo">
        <h2>
          <a href="https://www.bing.com/ck/a?!&&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9zZWNvbmQ">
            Second Result
          </a>
        </h2>
        <div class="b_caption"><p>Second snippet.</p></div>
      </li>
    </ol>
  </body>
</html>
"""

DDG_HTML = """
<html>
  <body>
    <div class="results">
      <div class="result">
        <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fddg-result">
          DDG Result
        </a>
        <a class="result__snippet">Fallback provider result.</a>
      </div>
    </div>
  </body>
</html>
"""


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "search.db")
    await database.initialize()
    return database


@pytest.fixture
def tool() -> WebSearchTool:
    return WebSearchTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


class _Response:
    def __init__(self, *, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text
        self.request = httpx.Request("GET", "https://example.com/search")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )


class _FakeAsyncClient:
    def __init__(self, responses: dict[str, list[object]], call_log: list[dict[str, object]]):
        self._responses = responses
        self._call_log = call_log

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    async def get(self, url, *, params=None, headers=None, **kwargs):
        del kwargs
        provider = "bing" if "bing.com" in url else "duckduckgo"
        self._call_log.append(
            {"provider": provider, "url": url, "params": params or {}, "headers": headers or {}}
        )
        queue = self._responses[provider]
        if not queue:
            raise AssertionError(f"No queued response for provider {provider}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _DirectHttpClient:
    def __init__(self, response: _Response):
        self._response = response
        self.calls: list[dict[str, object]] = []

    async def get(self, url, *, params=None, headers=None, **kwargs):
        del kwargs
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        return self._response


class TestSearchRegistry:
    def test_registry_orders_duckduckgo_before_bing(self):
        registry = SearchRegistry()

        assert [provider.name for provider in registry.ordered_providers()] == [
            "duckduckgo",
            "bing",
        ]


class TestParsers:
    def test_parse_bing_html_extracts_ranked_results(self):
        results = _parse_bing_html(BING_HTML, max_results=1)

        assert results == [
            {
                "title": "Bing Docs",
                "url": "https://example.com/bing-docs",
                "snippet": "Primary provider result.",
            }
        ]

    def test_parse_duckduckgo_html_decodes_redirect_links(self):
        results = _parse_duckduckgo_html(DDG_HTML, max_results=1)

        assert results == [
            {
                "title": "DDG Result",
                "url": "https://example.com/ddg-result",
                "snippet": "Fallback provider result.",
            }
        ]


class TestProviderStateStore:
    @pytest.mark.asyncio
    async def test_store_enforces_interval_then_cooldown(self, db: Database):
        now = {"value": 100.0}
        store = SearchProviderStateStore(db, now_fn=lambda: now["value"])
        policy = SearchProvider(
            name="duckduckgo",
            search_url="https://html.duckduckgo.com/html/",
            priority=100,
            min_interval_seconds=3.0,
            cooldown_seconds=10.0,
        ).to_policy()
        await store.sync_policies([policy])

        first = await store.request_dispatch(
            policy,
            lease_owner="lease-1",
            lease_ttl_seconds=5.0,
        )
        assert first.status == "dispatch_now"

        await store.mark_success("duckduckgo", lease_owner="lease-1")
        second = await store.request_dispatch(
            policy,
            lease_owner="lease-2",
            lease_ttl_seconds=5.0,
        )
        assert second.status == "wait"
        assert second.retry_at == pytest.approx(103.0)

        now["value"] = 103.1
        third = await store.request_dispatch(
            policy,
            lease_owner="lease-3",
            lease_ttl_seconds=5.0,
        )
        assert third.status == "dispatch_now"

        await store.mark_failure(
            policy,
            lease_owner="lease-3",
            status_code=429,
            soft_block=False,
        )
        blocked = await store.request_dispatch(
            policy,
            lease_owner="lease-4",
            lease_ttl_seconds=5.0,
        )
        assert blocked.status == "cooldown"
        assert blocked.retry_at == pytest.approx(113.1)


class TestSearchBackendClient:
    @pytest.mark.asyncio
    async def test_client_uses_canada_locale_hints_for_canadian_queries(self):
        client = SearchBackendClient()
        provider = SearchProvider(
            name="bing",
            search_url="https://www.bing.com/search",
            priority=50,
            min_interval_seconds=0.5,
            cooldown_seconds=120.0,
        )
        http_client = _DirectHttpClient(_Response(text=BING_HTML))

        results = await client.search(
            http_client,
            provider,
            query="Canadian technology podcast top 10 2025",
            max_results=5,
        )

        assert results[0]["title"] == "Bing Docs"
        assert http_client.calls[0]["params"]["cc"] == "CA"
        assert http_client.calls[0]["params"]["mkt"] == "en-CA"
        assert http_client.calls[0]["headers"]["Accept-Language"].startswith("en-CA")

    @pytest.mark.asyncio
    async def test_client_raises_for_anti_bot_page(self):
        client = SearchBackendClient()
        provider = SearchProvider(
            name="bing",
            search_url="https://www.bing.com/search",
            priority=50,
            min_interval_seconds=0.5,
            cooldown_seconds=120.0,
        )
        http_client = _DirectHttpClient(
            _Response(text="<html><body>captcha required</body></html>")
        )

        with pytest.raises(SearchProviderError):
            await client.search(http_client, provider, query="loom", max_results=5)


class TestSearchBackend:
    @pytest.mark.asyncio
    async def test_backend_uses_duckduckgo_first(self, monkeypatch, db: Database):
        now = {"time": 100.0, "mono": 1000.0}
        store = SearchProviderStateStore(db, now_fn=lambda: now["time"])
        registry = SearchRegistry()
        await store.sync_policies(
            [provider.to_policy() for provider in registry.ordered_providers()]
        )
        backend = SearchBackend(registry=registry, store=store)
        call_log: list[dict[str, object]] = []
        responses = {
            "duckduckgo": [_Response(text=DDG_HTML)],
            "bing": [_Response(text=BING_HTML)],
        }

        monkeypatch.setattr(
            "loom.tools.search_backend.httpx.AsyncClient",
            lambda: _FakeAsyncClient(responses, call_log),
        )
        monkeypatch.setattr("loom.tools.search_backend.time.time", lambda: now["time"])
        monkeypatch.setattr(
            "loom.tools.search_backend.time.monotonic",
            lambda: now["mono"],
        )

        results = await backend.search(
            "Blink49 Studios Canada television production",
            5,
            runtime_deadline=now["mono"] + 20.0,
        )

        assert results[0]["title"] == "DDG Result"
        assert [entry["provider"] for entry in call_log] == ["duckduckgo"]

    @pytest.mark.asyncio
    async def test_backend_falls_back_to_bing_when_ddg_wait_exceeds_runtime_budget(
        self,
        monkeypatch,
        db: Database,
    ):
        now = {"time": 100.0, "mono": 1000.0}
        store = SearchProviderStateStore(db, now_fn=lambda: now["time"])
        registry = SearchRegistry()
        await store.sync_policies(
            [provider.to_policy() for provider in registry.ordered_providers()]
        )
        await db.execute(
            "UPDATE search_provider_state SET next_allowed_at = ? WHERE provider = ?",
            (111.0, "duckduckgo"),
        )
        backend = SearchBackend(registry=registry, store=store)
        call_log: list[dict[str, object]] = []
        responses = {
            "duckduckgo": [_Response(text=DDG_HTML)],
            "bing": [_Response(text=BING_HTML)],
        }

        monkeypatch.setattr(
            "loom.tools.search_backend.httpx.AsyncClient",
            lambda: _FakeAsyncClient(responses, call_log),
        )
        monkeypatch.setattr("loom.tools.search_backend.time.time", lambda: now["time"])
        monkeypatch.setattr(
            "loom.tools.search_backend.time.monotonic",
            lambda: now["mono"],
        )

        results = await backend.search(
            "banana pancakes recipe",
            5,
            runtime_deadline=now["mono"] + 8.0,
        )

        assert results[0]["title"] == "Bing Docs"
        assert [entry["provider"] for entry in call_log] == ["bing"]

    @pytest.mark.asyncio
    async def test_backend_caches_identical_queries(self, monkeypatch, db: Database):
        store = SearchProviderStateStore(db)
        registry = SearchRegistry()
        await store.sync_policies(
            [provider.to_policy() for provider in registry.ordered_providers()]
        )
        backend = SearchBackend(registry=registry, store=store)
        call_log: list[dict[str, object]] = []
        responses = {
            "duckduckgo": [_Response(text=DDG_HTML)],
            "bing": [_Response(text=BING_HTML)],
        }

        monkeypatch.setattr(
            "loom.tools.search_backend.httpx.AsyncClient",
            lambda: _FakeAsyncClient(responses, call_log),
        )

        first = await backend.search("cache me", 5, runtime_deadline=time.monotonic() + 20.0)
        second = await backend.search("cache   me", 5, runtime_deadline=time.monotonic() + 20.0)

        assert first == second
        assert len(call_log) == 1


class TestWebSearchTool:
    @pytest.fixture(autouse=True)
    def _reset_backend(self):
        previous = dict(web_search_module._SEARCH_BACKENDS)
        web_search_module._SEARCH_BACKENDS.clear()
        yield
        web_search_module._SEARCH_BACKENDS.clear()
        web_search_module._SEARCH_BACKENDS.update(previous)

    async def test_empty_query(self, tool, ctx):
        result = await tool.execute({"query": ""}, ctx)
        assert result.success is False
        assert "No search query" in (result.error or result.output)

    def test_schema(self, tool):
        schema = tool.schema()
        assert schema["name"] == "web_search"
        assert "query" in schema["parameters"]["properties"]
        assert "max_results" in schema["parameters"]["properties"]

    def test_timeout(self, tool):
        assert tool.timeout_seconds == 30

    def test_infer_search_locale_defaults_to_us(self):
        locale = _infer_search_locale("python httpx documentation")
        assert locale.country_code == "US"
        assert locale.bing_market == "en-US"

    @pytest.mark.asyncio
    async def test_tool_formats_backend_results(self, tool, ctx, monkeypatch):
        class _Backend:
            async def search(self, query: str, max_results: int, *, runtime_deadline: float):
                assert query == "loom"
                assert max_results == 2
                assert runtime_deadline > 0
                return [
                    {
                        "title": "Loom Docs",
                        "url": "https://example.com/docs",
                        "snippet": "Reference documentation.",
                    },
                    {
                        "title": "Loom GitHub",
                        "url": "https://example.com/github",
                        "snippet": "",
                    },
                ]

        monkeypatch.setattr(
            "loom.tools.web_search.get_search_backend",
            lambda config=None: _Backend(),
        )

        result = await tool.execute({"query": "loom", "max_results": 2}, ctx)

        assert result.success is True
        assert "1. Loom Docs" in result.output
        assert "https://example.com/docs" in result.output
        assert "Reference documentation." in result.output
        assert result.data == {"query": "loom", "count": 2}
