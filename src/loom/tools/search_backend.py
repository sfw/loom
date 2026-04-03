"""Auth-free web search backend with authoritative provider pacing state."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from lxml import etree, html

from loom.state.search_provider_state import (
    SearchProviderPolicy,
    SearchProviderStateStore,
)
from loom.tools.web import DEFAULT_WEB_USER_AGENT

logger = logging.getLogger(__name__)

BING_SEARCH_URL = "https://www.bing.com/search"
DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"
SEARCH_TIMEOUT_SECONDS = 5.0
SEARCH_TOTAL_BUDGET_SECONDS = 15.0
DEFAULT_MAX_RESULTS = 10
MAX_RESULTS = 20
MAX_SEARCH_RETRIES = 2
RESULT_CACHE_TTL_SECONDS = 5 * 60
RESULT_CACHE_MAX_ENTRIES = 128
GLOBAL_SEARCH_CONCURRENCY = 4
RETRYABLE_STATUS_CODES = frozenset({403, 429, 500, 502, 503, 504})
ANTI_BOT_MARKERS = (
    "captcha",
    "detected unusual traffic",
    "verify you are human",
    "unusual activity",
)
SEARCH_USER_AGENTS = (
    DEFAULT_WEB_USER_AGENT,
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:138.0) "
        "Gecko/20100101 Firefox/138.0"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15"
    ),
)
GLOBAL_SEARCH_SEMAPHORE = asyncio.Semaphore(GLOBAL_SEARCH_CONCURRENCY)
LEASE_TTL_SECONDS = SEARCH_TIMEOUT_SECONDS + 2.0
MIN_RUNTIME_EXECUTION_RESERVE_SECONDS = SEARCH_TIMEOUT_SECONDS + 1.0
DEFAULT_DDG_MIN_INTERVAL_SECONDS = 2.5
DEFAULT_BING_MIN_INTERVAL_SECONDS = 0.5
DEFAULT_DDG_COOLDOWN_SECONDS = 10 * 60
DEFAULT_BING_COOLDOWN_SECONDS = 2 * 60


class SearchBackendError(RuntimeError):
    """Base exception for auth-free web search failures."""


class SearchProviderError(SearchBackendError):
    """Raised when a provider request fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        soft_block: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.soft_block = soft_block


class SearchResponseValidationError(SearchBackendError):
    """Raised when a provider response cannot be parsed into results."""


@dataclass(frozen=True, slots=True)
class SearchLocale:
    country_code: str
    bing_market: str
    accept_language: str
    ddg_region: str


@dataclass(frozen=True, slots=True)
class SearchProvider:
    """Static provider metadata tracked by the registry."""

    name: str
    search_url: str
    priority: int
    min_interval_seconds: float
    cooldown_seconds: float
    enabled: bool = True

    def to_policy(self) -> SearchProviderPolicy:
        return SearchProviderPolicy(
            name=self.name,
            priority=self.priority,
            min_interval_seconds=self.min_interval_seconds,
            cooldown_seconds=self.cooldown_seconds,
            enabled=self.enabled,
        )


class SearchRegistry:
    """Catalog of auth-free search providers and their static policies."""

    def __init__(
        self,
        *,
        providers: list[SearchProvider] | None = None,
    ) -> None:
        self._providers = {
            provider.name: provider
            for provider in (providers or _default_providers())
        }

    async def add_provider(self, provider: SearchProvider) -> SearchProvider:
        self._providers[provider.name] = provider
        return provider

    def get_provider(self, name: str) -> SearchProvider | None:
        return self._providers.get(name)

    def ordered_providers(self) -> list[SearchProvider]:
        return sorted(
            self._providers.values(),
            key=lambda provider: (-provider.priority, provider.name),
        )


class SearchBackendClient:
    """HTTP client for auth-free search providers."""

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        timeout_seconds: float = SEARCH_TIMEOUT_SECONDS,
    ) -> None:
        self._rng = rng or random.Random()
        self._timeout_seconds = timeout_seconds

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    async def search(
        self,
        client: httpx.AsyncClient,
        provider: SearchProvider,
        *,
        query: str,
        max_results: int,
    ) -> list[dict[str, str]]:
        locale = _infer_search_locale(query)
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": locale.accept_language,
            "User-Agent": _select_user_agent(self._rng),
            "Referer": provider.search_url,
        }
        params = _provider_params(
            provider.name,
            query=query,
            max_results=max_results,
            locale=locale,
        )
        try:
            async with GLOBAL_SEARCH_SEMAPHORE:
                response = await client.get(
                    provider.search_url,
                    params=params,
                    headers=headers,
                    timeout=httpx.Timeout(self._timeout_seconds),
                    follow_redirects=True,
                )
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ) as e:
            raise SearchProviderError(str(e), provider=provider.name) from e

        if response.status_code in RETRYABLE_STATUS_CODES:
            raise SearchProviderError(
                f"HTTP {response.status_code}",
                provider=provider.name,
                status_code=response.status_code,
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise SearchProviderError(
                f"HTTP {response.status_code}",
                provider=provider.name,
                status_code=response.status_code,
            ) from e

        if _looks_like_anti_bot_page(response.text):
            raise SearchProviderError(
                "Anti-bot challenge detected",
                provider=provider.name,
                soft_block=True,
            )

        return _parse_provider_results(
            provider.name,
            response.text,
            max_results=max_results,
        )


class SearchBackend:
    """High-level orchestrator for auth-free search providers."""

    def __init__(
        self,
        *,
        registry: SearchRegistry | None = None,
        client: SearchBackendClient | None = None,
        store: SearchProviderStateStore | None = None,
        database_path: str | Path | None = None,
    ) -> None:
        self._registry = registry or SearchRegistry()
        self._client = client or SearchBackendClient()
        self._store = store
        self._database_path = str(database_path) if database_path is not None else ""
        self._store_lock = asyncio.Lock()
        self._cache: dict[tuple[str, int], tuple[float, list[dict[str, str]]]] = {}
        self._inflight: dict[tuple[str, int], asyncio.Future[list[dict[str, str]]]] = {}
        self._cache_lock = asyncio.Lock()

    @property
    def registry(self) -> SearchRegistry:
        return self._registry

    async def _get_store(self) -> SearchProviderStateStore:
        if self._store is not None:
            return self._store
        if not self._database_path:
            raise RuntimeError("Search backend requires a database path or store.")
        async with self._store_lock:
            if self._store is None:
                self._store = await SearchProviderStateStore.from_database_path(
                    self._database_path
                )
                await self._store.sync_policies(
                    [provider.to_policy() for provider in self._registry.ordered_providers()]
                )
        return self._store

    async def search(
        self,
        query: str,
        max_results: int,
        *,
        runtime_deadline: float | None = None,
    ) -> list[dict[str, str]]:
        cache_key = _cache_key(query, max_results)
        cached = await self._load_cached_results(cache_key)
        if cached is not None:
            return cached
        inflight, owner = await self._reserve_inflight(cache_key)
        if not owner:
            return await inflight

        try:
            result = await self._search_uncached(
                query=query,
                max_results=max_results,
                runtime_deadline=runtime_deadline,
            )
            await self._store_cached_results(cache_key, result)
            self._resolve_inflight(cache_key, result, exc=None)
            return result
        except Exception as e:
            self._resolve_inflight(cache_key, None, exc=e)
            raise

    async def _search_uncached(
        self,
        *,
        query: str,
        max_results: int,
        runtime_deadline: float | None,
    ) -> list[dict[str, str]]:
        deadline = runtime_deadline or (time.monotonic() + SEARCH_TOTAL_BUDGET_SECONDS)
        errors: list[str] = []
        store = await self._get_store()

        async with httpx.AsyncClient() as http_client:
            for provider in self._registry.ordered_providers()[:MAX_SEARCH_RETRIES]:
                success, result_or_error = await self._attempt_provider(
                    http_client,
                    provider,
                    store=store,
                    query=query,
                    max_results=max_results,
                    runtime_deadline=deadline,
                )
                if success:
                    return result_or_error
                if result_or_error:
                    errors.append(result_or_error)

        if errors:
            raise SearchBackendError("All search providers failed: " + "; ".join(errors))
        return []

    async def _attempt_provider(
        self,
        http_client: httpx.AsyncClient,
        provider: SearchProvider,
        *,
        store: SearchProviderStateStore,
        query: str,
        max_results: int,
        runtime_deadline: float,
    ) -> tuple[bool, list[dict[str, str]] | str]:
        lease_owner = store.new_lease_owner()
        policy = provider.to_policy()

        while True:
            remaining_budget = runtime_deadline - time.monotonic()
            if remaining_budget <= 0:
                raise httpx.TimeoutException("Search deadline exceeded.")
            decision = await store.request_dispatch(
                policy,
                lease_owner=lease_owner,
                lease_ttl_seconds=LEASE_TTL_SECONDS,
            )
            if decision.status == "dispatch_now":
                break
            if decision.status in {"disabled", "cooldown"}:
                return False, f"{provider.name}: {decision.reason or decision.status}"
            if decision.status != "wait":
                return False, f"{provider.name}: unavailable"
            wait_seconds = max(0.0, float(decision.retry_at) - time.time())
            if wait_seconds <= 0:
                continue
            if wait_seconds >= max(0.0, remaining_budget - MIN_RUNTIME_EXECUTION_RESERVE_SECONDS):
                return False, f"{provider.name}: skipped due to runtime budget"
            await asyncio.sleep(wait_seconds)

        try:
            results = await self._client.search(
                http_client,
                provider,
                query=query,
                max_results=max_results,
            )
        except (SearchProviderError, SearchResponseValidationError) as e:
            await store.mark_failure(
                policy,
                lease_owner=lease_owner,
                status_code=getattr(e, "status_code", None),
                soft_block=bool(getattr(e, "soft_block", False)),
            )
            logger.warning("Search provider failed: %s (%s)", provider.name, e)
            return False, f"{provider.name}: {e}"

        await store.mark_success(provider.name, lease_owner=lease_owner)
        return True, results

    async def _load_cached_results(
        self,
        cache_key: tuple[str, int],
    ) -> list[dict[str, str]] | None:
        now = time.monotonic()
        async with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is None:
                return None
            expires_at, results = cached
            if expires_at <= now:
                self._cache.pop(cache_key, None)
                return None
            return _copy_results(results)

    async def _reserve_inflight(
        self,
        cache_key: tuple[str, int],
    ) -> tuple[asyncio.Future[list[dict[str, str]]], bool]:
        async with self._cache_lock:
            inflight = self._inflight.get(cache_key)
            if inflight is not None:
                return inflight, False
            loop = asyncio.get_running_loop()
            future: asyncio.Future[list[dict[str, str]]] = loop.create_future()
            self._inflight[cache_key] = future
            return future, True

    async def _store_cached_results(
        self,
        cache_key: tuple[str, int],
        results: list[dict[str, str]],
    ) -> None:
        async with self._cache_lock:
            self._cache[cache_key] = (
                time.monotonic() + RESULT_CACHE_TTL_SECONDS,
                _copy_results(results),
            )
            if len(self._cache) > RESULT_CACHE_MAX_ENTRIES:
                oldest_key = min(self._cache.items(), key=lambda item: item[1][0])[0]
                self._cache.pop(oldest_key, None)

    def _resolve_inflight(
        self,
        cache_key: tuple[str, int],
        results: list[dict[str, str]] | None,
        *,
        exc: Exception | None,
    ) -> None:
        future = self._inflight.pop(cache_key, None)
        if future is None or future.done():
            return
        if exc is not None:
            future.set_exception(exc)
            future.exception()
            return
        future.set_result(_copy_results(results or []))


def _default_providers() -> list[SearchProvider]:
    return [
        SearchProvider(
            name="duckduckgo",
            search_url=DUCKDUCKGO_SEARCH_URL,
            priority=100,
            min_interval_seconds=DEFAULT_DDG_MIN_INTERVAL_SECONDS,
            cooldown_seconds=DEFAULT_DDG_COOLDOWN_SECONDS,
        ),
        SearchProvider(
            name="bing",
            search_url=BING_SEARCH_URL,
            priority=50,
            min_interval_seconds=DEFAULT_BING_MIN_INTERVAL_SECONDS,
            cooldown_seconds=DEFAULT_BING_COOLDOWN_SECONDS,
        ),
    ]


def _provider_params(
    provider_name: str,
    *,
    query: str,
    max_results: int,
    locale: SearchLocale,
) -> dict[str, str]:
    if provider_name == "bing":
        return {
            "q": query,
            "cc": locale.country_code,
            "mkt": locale.bing_market,
            "setlang": locale.bing_market,
            "count": str(max(10, min(max_results, 50))),
        }
    if provider_name == "duckduckgo":
        return {
            "q": query,
            "kl": locale.ddg_region,
        }
    raise ValueError(f"Unknown search provider: {provider_name}")


def _select_user_agent(rng: random.Random) -> str:
    override = os.environ.get("LOOM_WEB_USER_AGENT", "").strip()
    if override:
        return override
    return rng.choice(SEARCH_USER_AGENTS)


def _infer_search_locale(query: str) -> SearchLocale:
    env_country = os.environ.get("LOOM_WEB_SEARCH_COUNTRY", "").strip().upper()
    env_market = os.environ.get("LOOM_WEB_SEARCH_MARKET", "").strip()
    env_lang = os.environ.get("LOOM_WEB_SEARCH_ACCEPT_LANGUAGE", "").strip()
    env_ddg = os.environ.get("LOOM_WEB_SEARCH_DDG_REGION", "").strip()
    if env_country or env_market or env_lang or env_ddg:
        country_code = env_country or "US"
        bing_market = env_market or f"en-{country_code}"
        accept_language = env_lang or f"{bing_market},en;q=0.9"
        ddg_region = env_ddg or ("ca-en" if country_code == "CA" else "us-en")
        return SearchLocale(
            country_code=country_code,
            bing_market=bing_market,
            accept_language=accept_language,
            ddg_region=ddg_region,
        )

    lowered = f" {str(query or '').casefold()} "
    canada_markers = (
        " canada ",
        " canadian ",
        " cbc ",
        " banff ",
        " cmpa ",
        " toronto ",
        " vancouver ",
        " montreal ",
        " ottawa ",
        " alberta ",
        " ontario ",
        " quebec ",
        " british columbia ",
    )
    if any(marker in lowered for marker in canada_markers):
        return SearchLocale(
            country_code="CA",
            bing_market="en-CA",
            accept_language="en-CA,en-US,en;q=0.9",
            ddg_region="ca-en",
        )
    return SearchLocale(
        country_code="US",
        bing_market="en-US",
        accept_language="en-US,en;q=0.9",
        ddg_region="us-en",
    )


def _cache_key(query: str, max_results: int) -> tuple[str, int]:
    normalized_query = " ".join(str(query or "").casefold().split())
    return normalized_query, int(max_results)


def _copy_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(result) for result in results]


def _looks_like_anti_bot_page(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(marker in lowered for marker in ANTI_BOT_MARKERS)


def _parse_provider_results(
    provider_name: str,
    html_text: str,
    *,
    max_results: int,
) -> list[dict[str, str]]:
    if provider_name == "bing":
        return _parse_bing_html(html_text, max_results=max_results)
    if provider_name == "duckduckgo":
        return _parse_duckduckgo_html(html_text, max_results=max_results)
    raise SearchResponseValidationError(f"Unknown search provider: {provider_name}")


def _parse_bing_html(html_text: str, *, max_results: int) -> list[dict[str, str]]:
    tree = _parse_html_document(html_text)
    nodes = tree.xpath("//li[contains(concat(' ', normalize-space(@class), ' '), ' b_algo ')]")
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for node in nodes:
        if len(results) >= max_results:
            break
        link = _first(node.xpath(".//h2[1]/a[1]"))
        if link is None:
            continue
        title = _clean_text(" ".join(link.itertext()))
        url = _normalize_bing_url(link.get("href"))
        snippet_node = _first(
            node.xpath(".//*[contains(@class, 'b_caption')]//p[1] | .//p[1]")
        )
        snippet = _clean_text(" ".join(snippet_node.itertext())) if snippet_node is not None else ""
        if not title or not url or url in seen:
            continue
        seen.add(url)
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


def _parse_duckduckgo_html(html_text: str, *, max_results: int) -> list[dict[str, str]]:
    tree = _parse_html_document(html_text)
    nodes = tree.xpath(
        "//div[contains(concat(' ', normalize-space(@class), ' '), ' result ')]"
        "[.//a[contains(@class, 'result__a')]]"
        " | //article[contains(concat(' ', normalize-space(@class), ' '), ' result ')]"
        "[.//a[contains(@class, 'result__a')]]"
    )
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for node in nodes:
        if len(results) >= max_results:
            break
        link = _first(node.xpath(".//a[contains(@class, 'result__a')][1]"))
        if link is None:
            continue
        title = _clean_text(" ".join(link.itertext()))
        url = _normalize_duckduckgo_url(link.get("href"))
        snippet_node = _first(node.xpath(".//*[contains(@class, 'result__snippet')][1]"))
        snippet = _clean_text(" ".join(snippet_node.itertext())) if snippet_node is not None else ""
        if not title or not url or url in seen:
            continue
        seen.add(url)
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


def _parse_html_document(html_text: str) -> html.HtmlElement:
    try:
        return html.fromstring(str(html_text or ""))
    except (etree.ParserError, ValueError) as e:
        raise SearchResponseValidationError("Search response was not valid HTML") from e


def _normalize_result_url(url: str | None) -> str:
    candidate = str(url or "").strip()
    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return candidate
    return ""


def _normalize_bing_url(url: str | None) -> str:
    candidate = str(url or "").strip()
    if not candidate:
        return ""
    parsed = urlparse(candidate)
    if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/a"):
        target = parse_qs(parsed.query).get("u", [""])[0]
        decoded = _decode_bing_target(target)
        normalized = _normalize_result_url(decoded)
        if normalized:
            return normalized
    return _normalize_result_url(candidate)


def _decode_bing_target(value: str) -> str:
    token = str(value or "").strip()
    if token.startswith("a1"):
        token = token[2:]
    if not token:
        return ""
    padding = "=" * (-len(token) % 4)
    try:
        decoded = base64.urlsafe_b64decode(token + padding)
    except (ValueError, TypeError):
        return ""
    try:
        return decoded.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _normalize_duckduckgo_url(url: str | None) -> str:
    candidate = str(url or "").strip()
    if not candidate:
        return ""
    parsed = urlparse(candidate)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        target = unquote(target)
        normalized = _normalize_result_url(target)
        if normalized:
            return normalized
    return _normalize_result_url(candidate)


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split())


def _first(items: list[Any]) -> Any | None:
    return items[0] if items else None
