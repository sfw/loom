"""Web search tool using DuckDuckGo.

Provides internet search capability without requiring API keys.
Parses DuckDuckGo HTML results to extract titles, URLs, and snippets.
"""

from __future__ import annotations

import asyncio
import os
import re

import httpx

from loom.tools.registry import Tool, ToolContext, ToolResult

SEARCH_TIMEOUT = 15.0
MAX_RESULTS = 10
DDG_URL = "https://html.duckduckgo.com/html/"
DDG_FALLBACK_URL = "https://duckduckgo.com/html/"
MAX_SEARCH_ATTEMPTS = 3
SEARCH_RETRY_BASE_DELAY = 0.4
RETRYABLE_SEARCH_STATUS = frozenset({403, 408, 425, 429, 500, 502, 503, 504})
DEFAULT_SEARCH_USER_AGENT = "Loom/1.0 (+https://github.com/sfw/loom)"


class WebSearchTool(Tool):
    """Search the internet using DuckDuckGo."""

    name = "web_search"
    description = (
        "Search the internet for information. Returns titles, URLs, and "
        "snippets from search results. Use this to find documentation, "
        "solutions, package info, or any public information. "
        "Does not require API keys."
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
                "description": f"Maximum number of results to return (default {MAX_RESULTS}).",
            },
        },
        "required": ["query"],
    }

    @property
    def timeout_seconds(self) -> int:
        return 30

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        query = args.get("query", "").strip()
        if not query:
            return ToolResult.fail("No search query provided.")

        max_results = min(args.get("max_results", MAX_RESULTS), 20)

        try:
            results = await _search_ddg(query, max_results)
        except httpx.TimeoutException:
            return ToolResult.fail("Search timed out.")
        except Exception as e:
            return ToolResult.fail(f"Search error: {e}")

        if not results:
            return ToolResult(
                success=True,
                output="No results found.",
                data={"query": query, "count": 0},
            )

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['url']}")
            if r["snippet"]:
                lines.append(f"   {r['snippet']}")
            lines.append("")

        output = "\n".join(lines).strip()
        return ToolResult(
            success=True,
            output=output,
            data={"query": query, "count": len(results)},
        )


async def _search_ddg(query: str, max_results: int) -> list[dict]:
    """Search DuckDuckGo HTML endpoints and parse results."""
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(SEARCH_TIMEOUT),
        headers=_build_search_headers(),
    ) as client:
        errors: list[str] = []
        endpoints = (
            ("POST", DDG_URL),
            ("GET", DDG_URL),
            ("GET", DDG_FALLBACK_URL),
        )

        for method, endpoint in endpoints:
            try:
                html = await _query_search_endpoint(
                    client=client,
                    method=method,
                    endpoint=endpoint,
                    query=query,
                )
            except Exception as e:
                errors.append(f"{endpoint}: {e}")
                continue

            parsed = _parse_ddg_html(html, max_results)
            if parsed:
                return parsed

        if errors:
            raise RuntimeError("All search endpoints failed: " + "; ".join(errors))

    return []


def _build_search_headers() -> dict[str, str]:
    """Build default search headers for web search requests."""
    user_agent = os.environ.get("LOOM_WEB_USER_AGENT", "").strip()
    if not user_agent:
        user_agent = DEFAULT_SEARCH_USER_AGENT
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://duckduckgo.com/",
        "Cache-Control": "no-cache",
    }


def _is_retryable_search_status(status_code: int) -> bool:
    """Return True if a search HTTP status is likely transient."""
    return status_code in RETRYABLE_SEARCH_STATUS


async def _query_search_endpoint(
    client: httpx.AsyncClient,
    method: str,
    endpoint: str,
    query: str,
) -> str:
    """Query a DDG endpoint with bounded retries and return HTML body."""
    method = method.upper()
    for attempt in range(MAX_SEARCH_ATTEMPTS):
        try:
            if method == "POST":
                response = await client.post(endpoint, data={"q": query, "b": ""})
            else:
                response = await client.get(endpoint, params={"q": query})

            if (
                _is_retryable_search_status(response.status_code)
                and attempt < MAX_SEARCH_ATTEMPTS - 1
            ):
                await asyncio.sleep(SEARCH_RETRY_BASE_DELAY * (2 ** attempt))
                continue

            response.raise_for_status()
            return response.text
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ):
            if attempt >= MAX_SEARCH_ATTEMPTS - 1:
                raise
            await asyncio.sleep(SEARCH_RETRY_BASE_DELAY * (2 ** attempt))
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else None
            if (
                status is not None
                and _is_retryable_search_status(status)
                and attempt < MAX_SEARCH_ATTEMPTS - 1
            ):
                await asyncio.sleep(SEARCH_RETRY_BASE_DELAY * (2 ** attempt))
                continue
            raise

    raise RuntimeError(
        f"Search failed after {MAX_SEARCH_ATTEMPTS} attempts: {endpoint}"
    )


def _parse_ddg_html(html: str, max_results: int) -> list[dict]:
    """Parse DuckDuckGo HTML search results page.

    Extracts result entries from the HTML without requiring
    an HTML parser library.
    """
    results: list[dict] = []

    # DuckDuckGo HTML results are in <a class="result__a" ...> tags
    # with snippets in <a class="result__snippet" ...> tags
    result_blocks = re.findall(
        r'<div[^>]*class="[^"]*result[^"]*"[^>]*>.*?</div>\s*</div>',
        html,
        re.DOTALL,
    )

    # If the block pattern doesn't match, try a simpler approach
    if not result_blocks:
        result_blocks = re.findall(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )
        snippet_matches = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            html,
            re.DOTALL,
        )

        for i, (url, title) in enumerate(result_blocks):
            if len(results) >= max_results:
                break
            url = _clean_ddg_url(url)
            title = _strip_tags(title).strip()
            snippet = _strip_tags(snippet_matches[i]).strip() if i < len(snippet_matches) else ""
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})

        return results

    for block in result_blocks:
        if len(results) >= max_results:
            break

        # Extract URL and title from result__a
        link_match = re.search(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        if not link_match:
            continue

        raw_url = link_match.group(1)
        title = _strip_tags(link_match.group(2)).strip()

        # Extract snippet
        snippet_match = re.search(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        snippet = _strip_tags(snippet_match.group(1)).strip() if snippet_match else ""

        url = _clean_ddg_url(raw_url)

        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})

    return results


def _clean_ddg_url(raw_url: str) -> str:
    """Clean a DuckDuckGo redirect URL to get the actual destination."""
    # DDG wraps URLs in redirect links like //duckduckgo.com/l/?uddg=<encoded_url>&...
    if "uddg=" in raw_url:
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(raw_url)
        params = parse_qs(parsed.query)
        if "uddg" in params:
            return params["uddg"][0]
    # If it starts with // make it https
    if raw_url.startswith("//"):
        return "https:" + raw_url
    return raw_url


def _strip_tags(html: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
