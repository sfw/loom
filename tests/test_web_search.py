"""Tests for the SearXNG-backed web search tool."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import loom.tools.web_search as web_search_module
from loom.tools.registry import ToolContext
from loom.tools.search_mesh import (
    DrifterDiscovery,
    SearchEndpoint,
    SearchEndpointError,
    SearchMesh,
    SearchMeshClient,
    SearchRegistry,
    SearchResponseValidationError,
    _normalize_searx_results,
)
from loom.tools.web_search import WebSearchTool


@pytest.fixture
def tool() -> WebSearchTool:
    return WebSearchTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _NoopDiscovery:
    def __init__(self):
        self.calls = 0

    async def ensure_fresh(self, client, *, force: bool = False) -> int:
        del client, force
        self.calls += 1
        return 0


class _ScenarioMeshClient:
    def __init__(self, outcomes: dict[str, list[object]]):
        self._outcomes = outcomes
        self.calls: list[tuple[str, str, int]] = []

    async def search(self, http_client, endpoint, *, query: str, max_results: int):
        del http_client
        self.calls.append((endpoint.url, query, max_results))
        outcome = self._outcomes[endpoint.url].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestSearchRegistry:
    @pytest.mark.asyncio
    async def test_dynamic_selection_prioritizes_analytics_false_pool(self):
        registry = SearchRegistry(rng=random.Random(7))
        await registry.add_endpoint(
            "https://tracked.example",
            endpoint_type="dynamic",
            analytics=True,
            latency=0.05,
        )
        await registry.add_endpoint(
            "https://private.example",
            endpoint_type="dynamic",
            analytics=False,
            latency=0.9,
        )

        selected = await registry.get_next("dynamic")

        assert selected is not None
        assert selected.url == "https://private.example"

    @pytest.mark.asyncio
    async def test_circuit_breaker_reopens_as_half_open_after_cooldown(self):
        now = {"value": 100.0}
        registry = SearchRegistry(
            static_endpoints=["https://internal.search.example"],
            rng=random.Random(11),
            now_fn=lambda: now["value"],
        )

        first = await registry.get_next("static")
        assert first is not None
        await registry.report_failure(first.url, status_code=429)

        blocked = await registry.get_next("static")
        assert blocked is None

        now["value"] += first.cooldown_seconds + 1
        probe = await registry.get_next("static")

        assert probe is not None
        assert probe.state == "half-open"
        snapshot = await registry.snapshot("static")
        assert snapshot[0].probe_in_flight is True

        await registry.report_success(probe.url, latency=0.2)
        snapshot = await registry.snapshot("static")
        assert snapshot[0].state == "closed"
        assert snapshot[0].probe_in_flight is False
        assert snapshot[0].failure_count == 0


class TestDrifterDiscovery:
    def test_parse_instances_filters_to_json_online_fast_google_instances(self):
        registry = SearchRegistry(rng=random.Random(3))
        discovery = DrifterDiscovery(registry, rng=random.Random(3))
        payload = {
            "instances": {
                "https://good.example": {
                    "json": True,
                    "analytics": False,
                    "http": {"status_code": 200},
                    "timing": {
                        "search": {"success_percentage": 100.0, "all": {"value": 0.8}},
                        "engines": {"google": {"all": {"value": 1.2}}},
                    },
                },
                "https://slow-google.example": {
                    "json": True,
                    "analytics": False,
                    "http": {"status_code": 200},
                    "timing": {
                        "search": {"success_percentage": 100.0, "all": {"value": 0.6}},
                        "engines": {"google": {"all": {"value": 2.2}}},
                    },
                },
                "https://offline.example": {
                    "json": True,
                    "analytics": False,
                    "http": {"status_code": 503},
                    "timing": {
                        "search": {"success_percentage": 100.0, "all": {"value": 0.4}},
                        "engines": {"google": {"all": {"value": 1.0}}},
                    },
                },
                "https://html-only.example": {
                    "json": False,
                    "analytics": False,
                    "http": {"status_code": 200},
                    "timing": {
                        "search": {"success_percentage": 100.0, "all": {"value": 0.4}},
                        "engines": {"google": {"all": {"value": 1.0}}},
                    },
                },
            }
        }

        endpoints = discovery.parse_instances(payload)

        assert [endpoint.url for endpoint in endpoints] == ["https://good.example"]
        assert endpoints[0].analytics is False
        assert endpoints[0].latency == pytest.approx(0.8)


class TestSearchMeshClient:
    def test_normalize_searx_results_validates_and_limits_results(self):
        payload = {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com/1",
                    "content": "First snippet",
                },
                {
                    "title": "Second",
                    "url": "https://example.com/2",
                    "content": "Second snippet",
                },
            ]
        }

        results = _normalize_searx_results(payload, max_results=1)

        assert results == [
            {
                "title": "Example",
                "url": "https://example.com/1",
                "snippet": "First snippet",
            }
        ]

    def test_normalize_searx_results_rejects_missing_results_list(self):
        with pytest.raises(SearchResponseValidationError):
            _normalize_searx_results({"answer": "nope"}, max_results=5)

    @pytest.mark.asyncio
    async def test_client_raises_validation_error_for_non_json_payload(self, monkeypatch):
        client = SearchMeshClient(rng=random.Random(5))
        endpoint = SearchEndpoint(url="https://mesh.example", endpoint_type="dynamic")

        class _Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                raise ValueError("not json")

        class _HttpClient:
            async def get(self, *args, **kwargs):
                del args, kwargs
                return _Response()

        with pytest.raises(SearchResponseValidationError):
            await client.search(_HttpClient(), endpoint, query="loom", max_results=5)


class TestSearchMesh:
    @pytest.mark.asyncio
    async def test_mesh_falls_back_from_static_to_dynamic_endpoint(self, monkeypatch):
        registry = SearchRegistry(
            static_endpoints=["https://internal.search.example"],
            rng=random.Random(13),
        )
        await registry.add_endpoint(
            "https://public.search.example",
            endpoint_type="dynamic",
            analytics=False,
            latency=0.6,
        )
        discovery = _NoopDiscovery()
        client = _ScenarioMeshClient(
            {
                "https://internal.search.example": [
                    SearchEndpointError(
                        "HTTP 429",
                        endpoint_url="https://internal.search.example",
                        status_code=429,
                    )
                ],
                "https://public.search.example": [
                    (
                        [
                            {
                                "title": "Recovered",
                                "url": "https://example.com/recovered",
                                "snippet": "Dynamic fallback worked.",
                            }
                        ],
                        0.45,
                    )
                ],
            }
        )
        monkeypatch.setattr("loom.tools.search_mesh.httpx.AsyncClient", _FakeAsyncClient)
        mesh = SearchMesh(registry=registry, discovery=discovery, client=client)

        results = await mesh.search("loom search mesh", 5)

        assert results[0]["title"] == "Recovered"
        assert discovery.calls == 1
        assert [call[0] for call in client.calls] == [
            "https://internal.search.example",
            "https://public.search.example",
        ]
        static_snapshot = await registry.snapshot("static")
        assert static_snapshot[0].state == "open"

    @pytest.mark.asyncio
    async def test_mesh_caches_identical_queries(self, monkeypatch):
        registry = SearchRegistry(rng=random.Random(17))
        await registry.add_endpoint(
            "https://public.search.example",
            endpoint_type="dynamic",
            analytics=False,
            latency=0.5,
        )
        discovery = _NoopDiscovery()
        client = _ScenarioMeshClient(
            {
                "https://public.search.example": [
                    (
                        [
                            {
                                "title": "Cached",
                                "url": "https://example.com/cached",
                                "snippet": "Only fetched once.",
                            }
                        ],
                        0.2,
                    )
                ]
            }
        )
        monkeypatch.setattr("loom.tools.search_mesh.httpx.AsyncClient", _FakeAsyncClient)
        mesh = SearchMesh(registry=registry, discovery=discovery, client=client)

        first = await mesh.search("cache me", 5)
        second = await mesh.search("cache   me", 5)

        assert first == second
        assert len(client.calls) == 1


class TestWebSearchTool:
    @pytest.fixture(autouse=True)
    def _reset_mesh(self):
        previous = web_search_module._SEARCH_MESH
        web_search_module._SEARCH_MESH = None
        yield
        web_search_module._SEARCH_MESH = previous

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

    @pytest.mark.asyncio
    async def test_tool_formats_mesh_results(self, tool, ctx, monkeypatch):
        class _Mesh:
            async def search(self, query: str, max_results: int):
                assert query == "loom"
                assert max_results == 2
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

        monkeypatch.setattr("loom.tools.web_search.get_search_mesh", lambda: _Mesh())

        result = await tool.execute({"query": "loom", "max_results": 2}, ctx)

        assert result.success is True
        assert "1. Loom Docs" in result.output
        assert "https://example.com/docs" in result.output
        assert "Reference documentation." in result.output
        assert result.data == {"query": "loom", "count": 2}
