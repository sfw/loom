"""Web search tool using DuckDuckGo.

Provides internet search capability without requiring API keys.
Parses DuckDuckGo HTML results to extract titles, URLs, and snippets.
"""

from __future__ import annotations

import asyncio
import os
import re
import time

import httpx

from loom.tools.registry import Tool, ToolContext, ToolResult
from loom.tools.web import DEFAULT_WEB_USER_AGENT

SEARCH_TIMEOUT = 8.0
SEARCH_TOTAL_BUDGET_SECONDS = 24.0
MAX_RESULTS = 10
DDG_URL = "https://html.duckduckgo.com/html/"
DDG_FALLBACK_URL = "https://duckduckgo.com/html/"
BING_URL = "https://www.bing.com/search"
MAX_SEARCH_ATTEMPTS = 2
SEARCH_RETRY_BASE_DELAY = 0.4
RETRYABLE_SEARCH_STATUS = frozenset({408, 425, 500, 502, 503, 504})
DEFAULT_SEARCH_USER_AGENT = DEFAULT_WEB_USER_AGENT
SEARCH_CACHE_TTL_SECONDS = 5 * 60
SEARCH_CACHE_MAX_ENTRIES = 128
DDG_COOLDOWN_SECONDS = 10 * 60
DDG_ANTI_BOT_STATUS = frozenset({202, 403, 418, 429})
DDG_MAX_CONCURRENT_SEARCHES = 1

_SEARCH_RESULT_CACHE: dict[tuple[str, int], tuple[float, list[dict]]] = {}
_SEARCH_INFLIGHT: dict[tuple[str, int], asyncio.Future[list[dict]]] = {}
_SEARCH_CACHE_LOCK = asyncio.Lock()
_DDG_COOLDOWN_UNTIL = 0.0
_DDG_REQUEST_SEMAPHORE = asyncio.Semaphore(DDG_MAX_CONCURRENT_SEARCHES)


class SearchProviderCooldownError(Exception):
    """Raised when a search provider is temporarily cooling down."""


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
    """Search web providers and parse HTML results."""
    cache_key = _search_cache_key(query, max_results)
    cached = await _load_cached_search_result(cache_key)
    if cached is not None:
        return cached

    inflight, owner = await _reserve_search_inflight(cache_key)
    if not owner:
        return await inflight

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(SEARCH_TIMEOUT),
    ) as client:
        try:
            errors: list[str] = []
            any_successful_response = False
            overall_deadline = time.monotonic() + SEARCH_TOTAL_BUDGET_SECONDS
            providers = (
                {
                    "name": "duckduckgo",
                    "referer": "https://duckduckgo.com/",
                    "parser": _parse_ddg_html,
                    "endpoints": (
                        ("POST", DDG_URL),
                        ("GET", DDG_FALLBACK_URL),
                    ),
                },
                {
                    "name": "bing",
                    "referer": "https://www.bing.com/",
                    "parser": _parse_bing_html,
                    "endpoints": (("GET", BING_URL),),
                },
            )

            for provider in providers:
                if time.monotonic() >= overall_deadline:
                    raise httpx.TimeoutException("Search deadline exceeded.")
                parser = provider["parser"]
                referer = str(provider["referer"] or "")
                endpoints = tuple(provider["endpoints"])
                provider_name = str(provider["name"] or "search")

                try:
                    parsed = await _search_provider(
                        client=client,
                        provider_name=provider_name,
                        parser=parser,
                        endpoints=endpoints,
                        query=query,
                        max_results=max_results,
                        deadline=overall_deadline,
                        referer=referer,
                    )
                except SearchProviderCooldownError as e:
                    errors.append(f"{provider_name}: {e}")
                    continue
                except Exception as e:
                    errors.append(f"{provider_name}: {e}")
                    if time.monotonic() >= overall_deadline:
                        raise httpx.TimeoutException("Search deadline exceeded.") from e
                    continue

                any_successful_response = True
                if parsed:
                    await _store_cached_search_result(cache_key, parsed)
                    _resolve_search_inflight(cache_key, parsed, exc=None)
                    return parsed

            if errors and not any_successful_response:
                raise RuntimeError("All search providers failed: " + "; ".join(errors))

            result: list[dict] = []
            await _store_cached_search_result(cache_key, result)
            _resolve_search_inflight(cache_key, result, exc=None)
            return result
        except Exception as e:
            _resolve_search_inflight(cache_key, None, exc=e)
            raise


def _build_search_headers(*, referer: str = "https://duckduckgo.com/") -> dict[str, str]:
    """Build default search headers for web search requests."""
    user_agent = os.environ.get("LOOM_WEB_USER_AGENT", "").strip()
    if not user_agent:
        user_agent = DEFAULT_SEARCH_USER_AGENT
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def _is_retryable_search_status(status_code: int) -> bool:
    """Return True if a search HTTP status is likely transient."""
    return status_code in RETRYABLE_SEARCH_STATUS


def _request_timeout_remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise httpx.TimeoutException("Search deadline exceeded.")
    return max(1.0, min(SEARCH_TIMEOUT, remaining))


def _search_cache_key(query: str, max_results: int) -> tuple[str, int]:
    normalized_query = " ".join(str(query or "").casefold().split())
    return normalized_query, int(max_results)


def _copy_results(results: list[dict]) -> list[dict]:
    return [dict(item) for item in results]


async def _load_cached_search_result(
    cache_key: tuple[str, int],
) -> list[dict] | None:
    now = time.monotonic()
    async with _SEARCH_CACHE_LOCK:
        entry = _SEARCH_RESULT_CACHE.get(cache_key)
        if entry is None:
            return None
        expires_at, results = entry
        if expires_at <= now:
            _SEARCH_RESULT_CACHE.pop(cache_key, None)
            return None
        return _copy_results(results)


async def _reserve_search_inflight(
    cache_key: tuple[str, int],
) -> tuple[asyncio.Future[list[dict]], bool]:
    async with _SEARCH_CACHE_LOCK:
        inflight = _SEARCH_INFLIGHT.get(cache_key)
        if inflight is not None:
            return inflight, False
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[dict]] = loop.create_future()
        _SEARCH_INFLIGHT[cache_key] = future
        return future, True


async def _store_cached_search_result(
    cache_key: tuple[str, int],
    results: list[dict],
) -> None:
    async with _SEARCH_CACHE_LOCK:
        _SEARCH_RESULT_CACHE[cache_key] = (
            time.monotonic() + SEARCH_CACHE_TTL_SECONDS,
            _copy_results(results),
        )
        if len(_SEARCH_RESULT_CACHE) > SEARCH_CACHE_MAX_ENTRIES:
            oldest_key = min(
                _SEARCH_RESULT_CACHE.items(),
                key=lambda item: item[1][0],
            )[0]
            _SEARCH_RESULT_CACHE.pop(oldest_key, None)


def _resolve_search_inflight(
    cache_key: tuple[str, int],
    results: list[dict] | None,
    *,
    exc: Exception | None,
) -> None:
    future = _SEARCH_INFLIGHT.pop(cache_key, None)
    if future is None or future.done():
        return
    if exc is not None:
        future.set_exception(exc)
        future.exception()
        return
    future.set_result(_copy_results(results or []))


def _ddg_cooldown_remaining() -> float:
    remaining = _DDG_COOLDOWN_UNTIL - time.monotonic()
    return max(0.0, remaining)


def _record_ddg_cooldown() -> None:
    global _DDG_COOLDOWN_UNTIL
    until = time.monotonic() + DDG_COOLDOWN_SECONDS
    if until > _DDG_COOLDOWN_UNTIL:
        _DDG_COOLDOWN_UNTIL = until


def _format_ddg_cooldown_message(remaining_seconds: float) -> str:
    rounded = max(1, int(round(remaining_seconds)))
    return f"DuckDuckGo cooling down after rate limiting (retry in ~{rounded}s)"


async def _search_provider(
    *,
    client: httpx.AsyncClient,
    provider_name: str,
    parser,
    endpoints: tuple[tuple[str, str], ...],
    query: str,
    max_results: int,
    deadline: float,
    referer: str,
) -> list[dict]:
    if provider_name == "duckduckgo":
        async with _DDG_REQUEST_SEMAPHORE:
            cooldown_remaining = _ddg_cooldown_remaining()
            if cooldown_remaining > 0:
                raise SearchProviderCooldownError(
                    _format_ddg_cooldown_message(cooldown_remaining)
                )
            return await _search_provider_endpoints(
                client=client,
                provider_name=provider_name,
                parser=parser,
                endpoints=endpoints,
                query=query,
                max_results=max_results,
                deadline=deadline,
                referer=referer,
            )
    return await _search_provider_endpoints(
        client=client,
        provider_name=provider_name,
        parser=parser,
        endpoints=endpoints,
        query=query,
        max_results=max_results,
        deadline=deadline,
        referer=referer,
    )


async def _search_provider_endpoints(
    *,
    client: httpx.AsyncClient,
    provider_name: str,
    parser,
    endpoints: tuple[tuple[str, str], ...],
    query: str,
    max_results: int,
    deadline: float,
    referer: str,
) -> list[dict]:
    errors: list[str] = []
    any_successful_response = False
    for method, endpoint in endpoints:
        try:
            html = await _query_search_endpoint(
                client=client,
                method=method,
                endpoint=endpoint,
                query=query,
                deadline=deadline,
                referer=referer,
                provider_name=provider_name,
            )
        except SearchProviderCooldownError:
            raise
        except Exception as e:
            errors.append(f"{endpoint}: {e}")
            if time.monotonic() >= deadline:
                raise httpx.TimeoutException("Search deadline exceeded.") from e
            continue

        any_successful_response = True
        parsed = parser(html, max_results)
        if parsed:
            return parsed

    if errors and not any_successful_response:
        raise RuntimeError("; ".join(errors))
    return []


async def _query_search_endpoint(
    client: httpx.AsyncClient,
    method: str,
    endpoint: str,
    query: str,
    *,
    deadline: float,
    referer: str,
    provider_name: str,
) -> str:
    """Query an HTML search endpoint with a bounded overall deadline."""
    method = method.upper()
    headers = _build_search_headers(referer=referer)
    for attempt in range(MAX_SEARCH_ATTEMPTS):
        try:
            timeout = httpx.Timeout(_request_timeout_remaining(deadline))
            if method == "POST":
                response = await client.post(
                    endpoint,
                    data={"q": query, "b": ""},
                    headers=headers,
                    timeout=timeout,
                )
            else:
                response = await client.get(
                    endpoint,
                    params={"q": query},
                    headers=headers,
                    timeout=timeout,
                )

            if (
                provider_name == "duckduckgo"
                and response.status_code in DDG_ANTI_BOT_STATUS
            ):
                _record_ddg_cooldown()
                raise SearchProviderCooldownError(
                    f"DuckDuckGo denied search request (HTTP {response.status_code})"
                )

            if (
                _is_retryable_search_status(response.status_code)
                and attempt < MAX_SEARCH_ATTEMPTS - 1
                and time.monotonic() + SEARCH_RETRY_BASE_DELAY < deadline
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
            if (
                attempt >= MAX_SEARCH_ATTEMPTS - 1
                or time.monotonic() + SEARCH_RETRY_BASE_DELAY >= deadline
            ):
                raise
            await asyncio.sleep(SEARCH_RETRY_BASE_DELAY * (2 ** attempt))
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else None
            if (
                status is not None
                and _is_retryable_search_status(status)
                and attempt < MAX_SEARCH_ATTEMPTS - 1
                and time.monotonic() + SEARCH_RETRY_BASE_DELAY < deadline
            ):
                await asyncio.sleep(SEARCH_RETRY_BASE_DELAY * (2 ** attempt))
                continue
            raise

    raise RuntimeError(
        f"Search failed after {MAX_SEARCH_ATTEMPTS} attempts: {endpoint}"
    )


def _parse_bing_html(html: str, max_results: int) -> list[dict]:
    """Parse Bing HTML search results page."""
    results: list[dict] = []
    blocks = re.findall(
        r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>(.*?)</li>',
        html,
        re.DOTALL,
    )

    for block in blocks:
        if len(results) >= max_results:
            break
        link_match = re.search(
            r"<h2[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>",
            block,
            re.DOTALL,
        )
        if not link_match:
            continue
        url = _clean_ddg_url(link_match.group(1))
        title = _strip_tags(link_match.group(2)).strip()
        snippet_match = re.search(
            r"<p[^>]*>(.*?)</p>",
            block,
            re.DOTALL,
        )
        snippet = _strip_tags(snippet_match.group(1)).strip() if snippet_match else ""
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})

    return results


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
