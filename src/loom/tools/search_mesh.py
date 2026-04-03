"""Load-balanced SearXNG search mesh with discovery and circuit breaking."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from loom.tools.web import DEFAULT_WEB_USER_AGENT

logger = logging.getLogger(__name__)

DISCOVERY_URL = "https://searx.space/data/instances.json"
DISCOVERY_REFRESH_SECONDS = 4 * 60 * 60
SEARCH_TIMEOUT_SECONDS = 5.0
DISCOVERY_TIMEOUT_SECONDS = 5.0
SEARCH_TOTAL_BUDGET_SECONDS = 20.0
MAX_SEARCH_RETRIES = 3
DEFAULT_MAX_RESULTS = 10
MAX_RESULTS = 20
RESULT_CACHE_TTL_SECONDS = 5 * 60
RESULT_CACHE_MAX_ENTRIES = 128
STATIC_ENDPOINT_COOLDOWN_SECONDS = 2 * 60
DYNAMIC_ENDPOINT_COOLDOWN_SECONDS = 10 * 60
MAX_DYNAMIC_SELECTION_POOL = 20
GLOBAL_SEARCH_CONCURRENCY = 4
LATENCY_EPSILON = 0.05
DEFAULT_ENDPOINT_LATENCY = 0.8
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MIN_SEARCH_SUCCESS_PERCENTAGE = 50.0
MAX_PUBLIC_SEARCH_LATENCY_SECONDS = 1.5
BING_SEARCH_URL = "https://www.bing.com/search"
SEARX_USER_AGENTS = (
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

_GLOBAL_SEARCH_SEMAPHORE = asyncio.Semaphore(GLOBAL_SEARCH_CONCURRENCY)


class SearchMeshError(RuntimeError):
    """Base exception for search mesh failures."""


class SearchEndpointError(SearchMeshError):
    """Raised when a specific endpoint request fails."""

    def __init__(
        self,
        message: str,
        *,
        endpoint_url: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.endpoint_url = endpoint_url
        self.status_code = status_code


class SearchResponseValidationError(SearchMeshError):
    """Raised when an endpoint responds but not with valid search JSON."""


@dataclass(slots=True)
class SearchEndpoint:
    """Search endpoint metadata tracked by the registry."""

    url: str
    endpoint_type: str
    analytics: bool = False
    supports_json: bool = True
    latency: float = DEFAULT_ENDPOINT_LATENCY
    failure_count: int = 0
    last_used_at: float = 0.0
    state: str = "closed"
    cooldown_until: float = 0.0
    probe_in_flight: bool = False

    @property
    def search_url(self) -> str:
        base = self.url if self.url.endswith("/") else f"{self.url}/"
        return urljoin(base, "search")

    @property
    def cooldown_seconds(self) -> float:
        if self.endpoint_type == "static":
            return STATIC_ENDPOINT_COOLDOWN_SECONDS
        return DYNAMIC_ENDPOINT_COOLDOWN_SECONDS


def _normalize_endpoint_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid endpoint URL: {url!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/search"):
        path = path[:-7]
    normalized = parsed._replace(path=path, query="", fragment="")
    return normalized.geturl().rstrip("/")


class SearchRegistry:
    """Registry for static and dynamically discovered search endpoints."""

    def __init__(
        self,
        *,
        static_endpoints: list[str] | tuple[str, ...] | None = None,
        rng: random.Random | None = None,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._static: dict[str, SearchEndpoint] = {}
        self._dynamic: dict[str, SearchEndpoint] = {}
        self._lock = asyncio.Lock()
        self._rng = rng or random.Random()
        self._now = now_fn or time.monotonic
        for raw_url in static_endpoints or ():
            try:
                normalized_url = _normalize_endpoint_url(raw_url)
            except ValueError:
                continue
            self._static[normalized_url] = SearchEndpoint(
                url=normalized_url,
                endpoint_type="static",
                analytics=False,
                supports_json=True,
                latency=DEFAULT_ENDPOINT_LATENCY,
            )

    async def add_endpoint(
        self,
        url: str,
        *,
        endpoint_type: str,
        analytics: bool = False,
        supports_json: bool = True,
        latency: float = DEFAULT_ENDPOINT_LATENCY,
    ) -> SearchEndpoint:
        endpoint = SearchEndpoint(
            url=_normalize_endpoint_url(url),
            endpoint_type=endpoint_type,
            analytics=bool(analytics),
            supports_json=bool(supports_json),
            latency=max(float(latency or DEFAULT_ENDPOINT_LATENCY), LATENCY_EPSILON),
        )
        async with self._lock:
            self._endpoint_map(endpoint_type)[endpoint.url] = endpoint
            return endpoint

    async def replace_dynamic_endpoints(
        self,
        endpoints: list[SearchEndpoint],
    ) -> None:
        async with self._lock:
            existing = self._dynamic
            updated: dict[str, SearchEndpoint] = {}
            for endpoint in endpoints:
                normalized_url = _normalize_endpoint_url(endpoint.url)
                previous = existing.get(normalized_url)
                if previous is not None:
                    endpoint.failure_count = previous.failure_count
                    endpoint.last_used_at = previous.last_used_at
                    endpoint.state = previous.state
                    endpoint.cooldown_until = previous.cooldown_until
                    endpoint.probe_in_flight = previous.probe_in_flight
                    if previous.latency > 0:
                        endpoint.latency = previous.latency
                endpoint.url = normalized_url
                updated[normalized_url] = endpoint
            self._dynamic = updated

    async def blacklist(self, url: str) -> None:
        await self.report_failure(url, status_code=429)

    async def get_next(
        self,
        pool: str,
        *,
        exclude: set[str] | None = None,
    ) -> SearchEndpoint | None:
        excluded = exclude or set()
        async with self._lock:
            candidates = self._eligible_candidates_locked(pool=pool, exclude=excluded)
            if not candidates:
                return None
            endpoint = self._choose_weighted_candidate_locked(candidates)
            endpoint.last_used_at = float(self._now())
            if endpoint.state == "half-open":
                endpoint.probe_in_flight = True
            return endpoint

    async def report_success(self, url: str, *, latency: float) -> None:
        normalized_url = _normalize_endpoint_url(url)
        async with self._lock:
            endpoint = self._find_locked(normalized_url)
            if endpoint is None:
                return
            endpoint.failure_count = 0
            endpoint.state = "closed"
            endpoint.cooldown_until = 0.0
            endpoint.probe_in_flight = False
            observed = max(float(latency or DEFAULT_ENDPOINT_LATENCY), LATENCY_EPSILON)
            endpoint.latency = observed if endpoint.latency <= 0 else (
                endpoint.latency * 0.7 + observed * 0.3
            )

    async def report_failure(
        self,
        url: str,
        *,
        status_code: int | None = None,
    ) -> None:
        normalized_url = _normalize_endpoint_url(url)
        async with self._lock:
            endpoint = self._find_locked(normalized_url)
            if endpoint is None:
                return
            endpoint.failure_count += 1
            endpoint.probe_in_flight = False
            endpoint.latency = min(10.0, max(endpoint.latency, LATENCY_EPSILON) * 1.5)
            should_open = (
                endpoint.state == "half-open"
                or status_code in RETRYABLE_STATUS_CODES
            )
            if should_open:
                endpoint.state = "open"
                endpoint.cooldown_until = float(self._now()) + endpoint.cooldown_seconds

    async def snapshot(self, pool: str) -> list[SearchEndpoint]:
        async with self._lock:
            return [
                self._copy_endpoint(endpoint)
                for endpoint in self._endpoint_map(pool).values()
            ]

    def _endpoint_map(self, pool: str) -> dict[str, SearchEndpoint]:
        if pool == "static":
            return self._static
        if pool == "dynamic":
            return self._dynamic
        raise ValueError(f"Unknown endpoint pool: {pool}")

    def _find_locked(self, url: str) -> SearchEndpoint | None:
        return self._static.get(url) or self._dynamic.get(url)

    def _eligible_candidates_locked(
        self,
        *,
        pool: str,
        exclude: set[str],
    ) -> list[SearchEndpoint]:
        now = float(self._now())
        endpoints = list(self._endpoint_map(pool).values())
        eligible: list[SearchEndpoint] = []
        for endpoint in endpoints:
            self._advance_state_locked(endpoint, now=now)
            if endpoint.url in exclude or not endpoint.supports_json:
                continue
            if endpoint.state == "open":
                continue
            if endpoint.state == "half-open" and endpoint.probe_in_flight:
                continue
            eligible.append(endpoint)
        if pool != "dynamic":
            return eligible
        privacy_first = [endpoint for endpoint in eligible if not endpoint.analytics]
        if privacy_first:
            eligible = privacy_first
        eligible.sort(key=lambda endpoint: endpoint.latency)
        return eligible[:MAX_DYNAMIC_SELECTION_POOL]

    def _advance_state_locked(self, endpoint: SearchEndpoint, *, now: float) -> None:
        if endpoint.state != "open":
            return
        if endpoint.cooldown_until > now:
            return
        endpoint.state = "half-open"
        endpoint.probe_in_flight = False

    def _choose_weighted_candidate_locked(
        self,
        candidates: list[SearchEndpoint],
    ) -> SearchEndpoint:
        now = float(self._now())
        weights: list[float] = []
        for endpoint in candidates:
            age_seconds = max(0.0, now - float(endpoint.last_used_at or 0.0))
            age_bonus = 1.0 + min(1.0, age_seconds / 30.0)
            jitter = self._rng.uniform(0.85, 1.15)
            weight = (1.0 / (max(endpoint.latency, LATENCY_EPSILON) + LATENCY_EPSILON))
            weights.append(max(weight * age_bonus * jitter, LATENCY_EPSILON))
        total = sum(weights)
        roll = self._rng.uniform(0.0, total)
        cursor = 0.0
        for endpoint, weight in zip(candidates, weights, strict=False):
            cursor += weight
            if roll <= cursor:
                return endpoint
        return candidates[-1]

    @staticmethod
    def _copy_endpoint(endpoint: SearchEndpoint) -> SearchEndpoint:
        return SearchEndpoint(
            url=endpoint.url,
            endpoint_type=endpoint.endpoint_type,
            analytics=endpoint.analytics,
            supports_json=endpoint.supports_json,
            latency=endpoint.latency,
            failure_count=endpoint.failure_count,
            last_used_at=endpoint.last_used_at,
            state=endpoint.state,
            cooldown_until=endpoint.cooldown_until,
            probe_in_flight=endpoint.probe_in_flight,
        )


class DrifterDiscovery:
    """Fetch and filter public SearXNG instances from searx.space."""

    def __init__(
        self,
        registry: SearchRegistry,
        *,
        discovery_url: str = DISCOVERY_URL,
        refresh_seconds: float = DISCOVERY_REFRESH_SECONDS,
        rng: random.Random | None = None,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._registry = registry
        self._discovery_url = discovery_url
        self._refresh_seconds = refresh_seconds
        self._rng = rng or random.Random()
        self._now = now_fn or time.monotonic
        self._last_refresh_at = 0.0
        self._refresh_lock = asyncio.Lock()

    async def ensure_fresh(
        self,
        client: httpx.AsyncClient,
        *,
        force: bool = False,
    ) -> int:
        now = float(self._now())
        if (
            not force
            and self._last_refresh_at
            and (now - self._last_refresh_at) < self._refresh_seconds
        ):
            return 0
        async with self._refresh_lock:
            now = float(self._now())
            if (
                not force
                and self._last_refresh_at
                and (now - self._last_refresh_at) < self._refresh_seconds
            ):
                return 0
            headers = {
                "Accept": "application/json",
                "User-Agent": _select_user_agent(self._rng),
            }
            async with _GLOBAL_SEARCH_SEMAPHORE:
                response = await client.get(
                    self._discovery_url,
                    headers=headers,
                    timeout=httpx.Timeout(DISCOVERY_TIMEOUT_SECONDS),
                )
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError:
                return 0
            endpoints = self.parse_instances(payload)
            await self._registry.replace_dynamic_endpoints(endpoints)
            self._last_refresh_at = now
            return len(endpoints)

    def parse_instances(self, payload: Any) -> list[SearchEndpoint]:
        if not isinstance(payload, dict):
            return []
        instances = payload.get("instances")
        if not isinstance(instances, dict):
            return []
        endpoints: list[SearchEndpoint] = []
        for raw_url, details in instances.items():
            if not isinstance(details, dict):
                continue
            if details.get("json") is False:
                continue
            if _extract_http_status(details) != 200:
                continue
            search_success = _extract_search_success_percentage(details)
            if search_success is None or search_success < MIN_SEARCH_SUCCESS_PERCENTAGE:
                continue
            google_latency = _extract_google_search_latency(details)
            search_latency = _extract_search_latency(details)
            if search_latency is None:
                continue
            if google_latency is not None and google_latency >= MAX_PUBLIC_SEARCH_LATENCY_SECONDS:
                continue
            if search_latency >= MAX_PUBLIC_SEARCH_LATENCY_SECONDS * 2:
                continue
            try:
                normalized_url = _normalize_endpoint_url(str(raw_url))
            except ValueError:
                continue
            endpoints.append(
                SearchEndpoint(
                    url=normalized_url,
                    endpoint_type="dynamic",
                    analytics=bool(details.get("analytics")),
                    supports_json=True,
                    latency=max(search_latency, LATENCY_EPSILON),
                )
            )
        endpoints.sort(key=lambda endpoint: (endpoint.analytics, endpoint.latency))
        return endpoints


class SearchMeshClient:
    """HTTP client for querying individual SearXNG instances."""

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        timeout_seconds: float = SEARCH_TIMEOUT_SECONDS,
    ) -> None:
        self._rng = rng or random.Random()
        self._timeout_seconds = timeout_seconds

    async def search(
        self,
        client: httpx.AsyncClient,
        endpoint: SearchEndpoint,
        *,
        query: str,
        max_results: int,
    ) -> tuple[list[dict[str, str]], float]:
        headers = {
            "Accept": "application/json",
            "User-Agent": _select_user_agent(self._rng),
        }
        params = {"q": query, "format": "json"}
        started_at = time.monotonic()
        try:
            async with _GLOBAL_SEARCH_SEMAPHORE:
                response = await client.get(
                    endpoint.search_url,
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
            raise SearchEndpointError(
                str(e),
                endpoint_url=endpoint.url,
            ) from e

        if response.status_code in RETRYABLE_STATUS_CODES:
            raise SearchEndpointError(
                f"HTTP {response.status_code}",
                endpoint_url=endpoint.url,
                status_code=response.status_code,
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise SearchEndpointError(
                f"HTTP {response.status_code}",
                endpoint_url=endpoint.url,
                status_code=response.status_code,
            ) from e

        try:
            payload = response.json()
        except ValueError as e:
            raise SearchResponseValidationError(
                f"Invalid JSON from {endpoint.url}"
            ) from e

        results = _normalize_searx_results(payload, max_results=max_results)
        latency = max(time.monotonic() - started_at, LATENCY_EPSILON)
        return results, latency


class SearchMesh:
    """High-level orchestrator for static + dynamic SearXNG search."""

    def __init__(
        self,
        *,
        registry: SearchRegistry | None = None,
        discovery: DrifterDiscovery | None = None,
        client: SearchMeshClient | None = None,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self._registry = registry or SearchRegistry(now_fn=now_fn)
        self._client = client or SearchMeshClient()
        self._discovery = discovery or DrifterDiscovery(
            self._registry,
            now_fn=now_fn,
        )
        self._cache: dict[tuple[str, int], tuple[float, list[dict[str, str]]]] = {}
        self._inflight: dict[tuple[str, int], asyncio.Future[list[dict[str, str]]]] = {}
        self._cache_lock = asyncio.Lock()

    @property
    def registry(self) -> SearchRegistry:
        return self._registry

    @property
    def discovery(self) -> DrifterDiscovery:
        return self._discovery

    async def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        cache_key = _cache_key(query, max_results)
        cached = await self._load_cached_results(cache_key)
        if cached is not None:
            return cached
        inflight, owner = await self._reserve_inflight(cache_key)
        if not owner:
            return await inflight

        try:
            result = await self._search_uncached(query=query, max_results=max_results)
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
    ) -> list[dict[str, str]]:
        errors: list[str] = []
        tried: set[str] = set()
        attempt_count = 0
        deadline = time.monotonic() + SEARCH_TOTAL_BUDGET_SECONDS

        async with httpx.AsyncClient() as http_client:
            static_endpoint = await self._registry.get_next("static", exclude=tried)
            if static_endpoint is not None and attempt_count < MAX_SEARCH_RETRIES:
                attempt_count += 1
                tried.add(static_endpoint.url)
                success, result_or_error = await self._attempt_endpoint(
                    http_client,
                    static_endpoint,
                    query=query,
                    max_results=max_results,
                    deadline=deadline,
                )
                if success:
                    return result_or_error
                errors.append(result_or_error)

            try:
                await self._discovery.ensure_fresh(http_client)
            except Exception as e:
                logger.warning("Search mesh discovery refresh failed: %s", e)
                errors.append(f"discovery: {e}")

            while attempt_count < MAX_SEARCH_RETRIES:
                if time.monotonic() >= deadline:
                    raise httpx.TimeoutException("Search deadline exceeded.")
                dynamic_endpoint = await self._registry.get_next("dynamic", exclude=tried)
                if dynamic_endpoint is None:
                    break
                attempt_count += 1
                tried.add(dynamic_endpoint.url)
                success, result_or_error = await self._attempt_endpoint(
                    http_client,
                    dynamic_endpoint,
                    query=query,
                    max_results=max_results,
                    deadline=deadline,
                )
                if success:
                    return result_or_error
                errors.append(result_or_error)

            fallback_results = await _search_bing_fallback(
                http_client,
                query=query,
                max_results=max_results,
                deadline=deadline,
            )
            if fallback_results:
                logger.info("Search mesh used Bing fallback after SearXNG failures")
                return fallback_results

        if errors:
            raise SearchMeshError("All search mesh endpoints failed: " + "; ".join(errors))
        return []

    async def _attempt_endpoint(
        self,
        http_client: httpx.AsyncClient,
        endpoint: SearchEndpoint,
        *,
        query: str,
        max_results: int,
        deadline: float,
    ) -> tuple[bool, list[dict[str, str]] | str]:
        if time.monotonic() >= deadline:
            raise httpx.TimeoutException("Search deadline exceeded.")
        try:
            results, latency = await self._client.search(
                http_client,
                endpoint,
                query=query,
                max_results=max_results,
            )
        except (SearchEndpointError, SearchResponseValidationError) as e:
            status_code = getattr(e, "status_code", None)
            await self._registry.report_failure(endpoint.url, status_code=status_code)
            logger.warning(
                "Search mesh endpoint failed: %s (%s)",
                endpoint.url,
                e,
            )
            return False, f"{endpoint.url}: {e}"
        await self._registry.report_success(endpoint.url, latency=latency)
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
                oldest_key = min(
                    self._cache.items(),
                    key=lambda item: item[1][0],
                )[0]
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


def _select_user_agent(rng: random.Random) -> str:
    override = os.environ.get("LOOM_WEB_USER_AGENT", "").strip()
    if override:
        return override
    return rng.choice(SEARX_USER_AGENTS)


def _cache_key(query: str, max_results: int) -> tuple[str, int]:
    normalized_query = " ".join(str(query or "").casefold().split())
    return normalized_query, int(max_results)


def _copy_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    return [dict(result) for result in results]


def _extract_http_status(details: dict[str, Any]) -> int | None:
    http_data = details.get("http")
    if not isinstance(http_data, dict):
        return None
    status = http_data.get("status_code")
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _extract_search_latency(details: dict[str, Any]) -> float | None:
    timing = details.get("timing")
    if isinstance(timing, dict):
        search = timing.get("search")
        latency = _extract_latency_metric(search)
        if latency is not None:
            return latency
    for path in (
        ("timing", "search", "all", "value"),
        ("timing", "search", "all", "median"),
        ("timing", "search", "all", "mean"),
        ("timing", "search", "value"),
        ("timing_search",),
    ):
        value = _get_numeric_path(details, path)
        if value is not None:
            return value
    return _find_first_metric(
        details,
        include=("search",),
        exclude=("google",),
        keys=("value", "median", "mean"),
    )


def _extract_google_search_latency(details: dict[str, Any]) -> float | None:
    timing = details.get("timing")
    if isinstance(timing, dict):
        for key in ("search_go", "search_google", "google"):
            latency = _extract_latency_metric(timing.get(key))
            if latency is not None:
                return latency
    for path in (
        ("timing", "search_go", "all", "value"),
        ("timing", "search_go", "all", "median"),
        ("timing", "search_go", "all", "mean"),
        ("timing", "search_google", "all", "value"),
        ("timing", "search_google", "all", "median"),
        ("timing", "search_google", "all", "mean"),
        ("timing", "search", "google", "all", "value"),
        ("timing", "google", "all", "value"),
        ("timing", "engines", "google", "all", "value"),
        ("timing_search_google",),
    ):
        value = _get_numeric_path(details, path)
        if value is not None:
            return value
    return _find_first_metric(
        details,
        include=("google", "search_go"),
        exclude=(),
        keys=("value", "median", "mean"),
    )


def _extract_search_success_percentage(details: dict[str, Any]) -> float | None:
    timing = details.get("timing")
    if not isinstance(timing, dict):
        return None
    search = timing.get("search")
    if not isinstance(search, dict):
        return None
    value = search.get("success_percentage")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_latency_metric(data: Any) -> float | None:
    if not isinstance(data, dict):
        return None
    for nested_key in ("all", "server", "load"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            for metric_key in ("value", "median", "mean"):
                metric = nested.get(metric_key)
                try:
                    return float(metric)
                except (TypeError, ValueError):
                    continue
    for metric_key in ("value", "median", "mean"):
        metric = data.get(metric_key)
        try:
            return float(metric)
        except (TypeError, ValueError):
            continue
    return None


def _get_numeric_path(data: Any, path: tuple[str, ...]) -> float | None:
    current = data
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _find_first_metric(
    data: Any,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    keys: tuple[str, ...] = ("value",),
    path: tuple[str, ...] = (),
) -> float | None:
    lowered_path = tuple(segment.casefold() for segment in path)
    if isinstance(data, dict):
        for key in keys:
            if key not in data:
                continue
            if any(token in ".".join(lowered_path) for token in include) and not any(
                token in ".".join(lowered_path) for token in exclude
            ):
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    continue
        for key, value in data.items():
            metric = _find_first_metric(
                value,
                include=include,
                exclude=exclude,
                keys=keys,
                path=path + (str(key),),
            )
            if metric is not None:
                return metric
        return None
    if isinstance(data, list):
        for idx, value in enumerate(data):
            metric = _find_first_metric(
                value,
                include=include,
                exclude=exclude,
                keys=keys,
                path=path + (str(idx),),
            )
            if metric is not None:
                return metric
    return None


def _normalize_searx_results(
    payload: Any,
    *,
    max_results: int,
) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        raise SearchResponseValidationError("Search payload was not a JSON object")
    raw_results = payload.get("results")
    if raw_results is None:
        raise SearchResponseValidationError("Search payload missing 'results'")
    if not isinstance(raw_results, list):
        raise SearchResponseValidationError("Search payload 'results' was not a list")

    cleaned: list[dict[str, str]] = []
    for raw_result in raw_results:
        if len(cleaned) >= max_results:
            break
        if not isinstance(raw_result, dict):
            continue
        title = str(raw_result.get("title", "") or "").strip()
        url = str(raw_result.get("url", "") or "").strip()
        snippet = str(
            raw_result.get("content")
            or raw_result.get("snippet")
            or raw_result.get("description")
            or ""
        ).strip()
        if not title or not url:
            continue
        cleaned.append({"title": title, "url": url, "snippet": snippet})
    return cleaned


async def _search_bing_fallback(
    client: httpx.AsyncClient,
    *,
    query: str,
    max_results: int,
    deadline: float,
) -> list[dict[str, str]]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise httpx.TimeoutException("Search deadline exceeded.")
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": _select_user_agent(random.Random()),
        "Referer": "https://www.bing.com/",
    }
    async with _GLOBAL_SEARCH_SEMAPHORE:
        response = await client.get(
            BING_SEARCH_URL,
            params={"q": query, "format": "rss", "mkt": "en-US", "setlang": "en-US"},
            headers=headers,
            timeout=httpx.Timeout(min(SEARCH_TIMEOUT_SECONDS, max(1.0, remaining))),
            follow_redirects=True,
        )
    response.raise_for_status()
    return _parse_bing_rss(response.text, max_results=max_results)


def _parse_bing_rss(xml_text: str, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    channel = root.find("channel")
    if channel is None:
        return results

    for item in channel.findall("item"):
        if len(results) >= max_results:
            break
        title = str(item.findtext("title", "") or "").strip()
        url = str(item.findtext("link", "") or "").strip()
        snippet = str(item.findtext("description", "") or "").strip()
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results
