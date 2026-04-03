"""Web search tool backed by auth-free search providers."""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from loom.config import Config, load_config
from loom.tools.registry import Tool, ToolContext, ToolResult
from loom.tools.search_backend import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS,
    SearchBackend,
    SearchBackendClient,
    SearchBackendError,
    SearchRegistry,
)

_SEARCH_BACKENDS: dict[str, SearchBackend] = {}


class WebSearchTool(Tool):
    """Search the internet using auth-free search providers."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config

    name = "web_search"
    description = (
        "Search the internet for information. Returns titles, URLs, and "
        "snippets from search results. Uses Bing HTML first and falls back "
        "to DuckDuckGo HTML with request throttling and caching."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
            "max_results": {
                "type": "integer",
                "description": (
                    f"Maximum number of results to return (default {DEFAULT_MAX_RESULTS})."
                ),
            },
        },
        "required": ["query"],
    }

    @property
    def timeout_seconds(self) -> int:
        return 30

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        del ctx
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult.fail("No search query provided.")

        raw_max_results = args.get("max_results", DEFAULT_MAX_RESULTS)
        try:
            max_results = int(raw_max_results)
        except (TypeError, ValueError):
            max_results = DEFAULT_MAX_RESULTS
        max_results = max(1, min(max_results, MAX_RESULTS))

        try:
            results = await get_search_backend(self._config).search(
                query,
                max_results,
                runtime_deadline=time.monotonic() + float(self.timeout_seconds),
            )
        except httpx.TimeoutException:
            return ToolResult.fail("Search timed out.")
        except SearchBackendError as e:
            return ToolResult.fail(f"Search error: {e}")
        except Exception as e:
            return ToolResult.fail(f"Search error: {e}")

        if not results:
            return ToolResult(
                success=True,
                output="No results found.",
                data={"query": query, "count": 0},
            )

        lines = []
        for idx, result in enumerate(results, 1):
            lines.append(f"{idx}. {result['title']}")
            lines.append(f"   {result['url']}")
            if result["snippet"]:
                lines.append(f"   {result['snippet']}")
            lines.append("")

        return ToolResult(
            success=True,
            output="\n".join(lines).strip(),
            data={"query": query, "count": len(results)},
        )


def get_search_backend(config: Config | None = None) -> SearchBackend:
    resolved = config or load_config()
    db_path = str(resolved.database_path)
    backend = _SEARCH_BACKENDS.get(db_path)
    if backend is not None:
        return backend
    backend = _build_search_backend(resolved.database_path)
    _SEARCH_BACKENDS[db_path] = backend
    return backend


def _build_search_backend(database_path: str | Path) -> SearchBackend:
    registry = SearchRegistry()
    client = SearchBackendClient()
    return SearchBackend(
        registry=registry,
        client=client,
        database_path=str(database_path),
    )
