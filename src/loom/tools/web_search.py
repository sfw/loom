"""Web search tool backed by the Search Mesh orchestrator."""

from __future__ import annotations

import os

import httpx

from loom.tools.registry import Tool, ToolContext, ToolResult
from loom.tools.search_mesh import (
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS,
    DrifterDiscovery,
    SearchMesh,
    SearchMeshClient,
    SearchMeshError,
    SearchRegistry,
)

_SEARCH_MESH: SearchMesh | None = None


class WebSearchTool(Tool):
    """Search the internet using the SearXNG search mesh."""

    name = "web_search"
    description = (
        "Search the internet for information. Returns titles, URLs, and "
        "snippets from search results. Uses a resilient SearXNG search mesh "
        "with static and dynamically discovered endpoints."
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
            results = await get_search_mesh().search(query, max_results)
        except httpx.TimeoutException:
            return ToolResult.fail("Search timed out.")
        except SearchMeshError as e:
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


def get_search_mesh() -> SearchMesh:
    global _SEARCH_MESH
    if _SEARCH_MESH is None:
        _SEARCH_MESH = _build_search_mesh_from_env()
    return _SEARCH_MESH


def _build_search_mesh_from_env() -> SearchMesh:
    registry = SearchRegistry(static_endpoints=_read_static_endpoint_env())
    discovery = DrifterDiscovery(registry)
    client = SearchMeshClient()
    return SearchMesh(registry=registry, discovery=discovery, client=client)


def _read_static_endpoint_env() -> list[str]:
    raw = os.environ.get("LOOM_SEARCH_STATIC_ENDPOINTS", "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
