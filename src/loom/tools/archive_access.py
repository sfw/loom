"""Public archive discovery tool with normalized metadata output."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from loom.research.models import ArchiveResult
from loom.research.text import parse_year
from loom.tools.registry import Tool, ToolContext, ToolResult

SEARCH_TIMEOUT = 20.0
DEFAULT_MAX_RESULTS = 12
MAX_RESULTS = 40
DEFAULT_USER_AGENT = "Loom/1.0 (+https://github.com/sfw/loom)"

SUPPORTED_ARCHIVES = {
    "internet_archive",
    "wikimedia",
    "openverse",
}

SUPPORTED_MEDIA = {
    "text",
    "image",
    "audio",
    "video",
    "mixed",
}


class ArchiveAccessTool(Tool):
    """Discover public archival sources and return normalized records."""

    @property
    def name(self) -> str:
        return "archive_access"

    @property
    def description(self) -> str:
        return (
            "Search public archives (Internet Archive, Wikimedia, Openverse) "
            "for historical sources and return normalized record metadata."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Archive search query.",
                },
                "archive_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of archive sources.",
                },
                "date_from": {
                    "type": "string",
                    "description": "Lower date/year bound (inclusive).",
                },
                "date_to": {
                    "type": "string",
                    "description": "Upper date/year bound (inclusive).",
                },
                "media_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional media filter.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max normalized results (default 12, max 40).",
                },
            },
            "required": ["query"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 35

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult.fail("No query provided")

        archive_sources = _normalize_archives(args.get("archive_sources", []))
        if archive_sources is None:
            return ToolResult.fail(
                "archive_sources must contain only internet_archive/wikimedia/openverse"
            )
        if not archive_sources:
            archive_sources = ["internet_archive", "wikimedia", "openverse"]

        media_types = _normalize_media_types(args.get("media_types", []))
        if media_types is None:
            return ToolResult.fail(
                "media_types must contain only text/image/audio/video/mixed"
            )

        max_results = _clamp_int(args.get("max_results"), DEFAULT_MAX_RESULTS, 1, MAX_RESULTS)
        date_from = str(args.get("date_from", "")).strip()
        date_to = str(args.get("date_to", "")).strip()
        year_from = parse_year(date_from) if date_from else None
        year_to = parse_year(date_to) if date_to else None

        if year_from and year_to and year_from > year_to:
            return ToolResult.fail("date_from must be <= date_to")

        provider_errors: dict[str, str] = {}
        grouped: list[list[ArchiveResult]] = []
        headers = _build_headers()

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(SEARCH_TIMEOUT),
            headers=headers,
        ) as client:
            tasks = [
                _query_archive(
                    source,
                    query=query,
                    max_results=max_results,
                    client=client,
                )
                for source in archive_sources
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for source, response in zip(archive_sources, responses, strict=False):
            if isinstance(response, Exception):
                provider_errors[source] = f"{type(response).__name__}: {response}"
                continue
            grouped.append(response)

        merged = _merge_results(
            grouped,
            media_types=media_types,
            year_from=year_from,
            year_to=year_to,
            max_results=max_results,
        )

        if not merged:
            note = ""
            if provider_errors:
                note = " Provider errors: " + "; ".join(
                    f"{k} ({v})" for k, v in sorted(provider_errors.items())
                )
            return ToolResult.ok(
                f"No archive results found for '{query}'.{note}",
                data={
                    "query": query,
                    "count": 0,
                    "results": [],
                    "archive_sources": archive_sources,
                    "provider_errors": provider_errors,
                },
            )

        lines: list[str] = []
        for i, row in enumerate(merged, start=1):
            lines.append(f"{i}. {row.title}")
            meta = " | ".join(
                part
                for part in [row.date, row.creator, row.repository, row.media_type]
                if part
            )
            if meta:
                lines.append(f"   {meta}")
            if row.rights:
                lines.append(f"   Rights: {row.rights}")
            if row.record_url:
                lines.append(f"   Record: {row.record_url}")
            if row.access_url and row.access_url != row.record_url:
                lines.append(f"   Access: {row.access_url}")
            if row.snippet:
                lines.append(f"   Notes: {row.snippet[:240]}")
            lines.append("")

        return ToolResult.ok(
            "\n".join(lines).strip(),
            data={
                "query": query,
                "count": len(merged),
                "results": [row.to_dict() for row in merged],
                "archive_sources": archive_sources,
                "provider_errors": provider_errors,
                "date_filter": {
                    "date_from": date_from,
                    "date_to": date_to,
                },
            },
        )


def _build_headers() -> dict[str, str]:
    user_agent = os.environ.get("LOOM_WEB_USER_AGENT", "").strip() or DEFAULT_USER_AGENT
    return {
        "User-Agent": user_agent,
        "Accept": "application/json,text/html,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp_int(value: object, default: int, low: int, high: int) -> int:
    parsed = _to_int(value)
    if parsed is None:
        return default
    return max(low, min(high, parsed))


def _normalize_archives(raw: object) -> list[str] | None:
    if raw is None or raw == []:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None

    out: list[str] = []
    for item in raw:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if text not in SUPPORTED_ARCHIVES:
            return None
        if text not in out:
            out.append(text)
    return out


def _normalize_media_types(raw: object) -> set[str] | None:
    if raw is None or raw == []:
        return set()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None

    out: set[str] = set()
    for item in raw:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if text not in SUPPORTED_MEDIA:
            return None
        out.add(text)
    return out


async def _query_archive(
    source: str,
    *,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[ArchiveResult]:
    if source == "internet_archive":
        return await _search_internet_archive(query=query, max_results=max_results, client=client)
    if source == "wikimedia":
        return await _search_wikimedia(query=query, max_results=max_results, client=client)
    if source == "openverse":
        return await _search_openverse(query=query, max_results=max_results, client=client)
    raise ValueError(f"Unsupported archive source: {source}")


async def _search_internet_archive(
    *,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[ArchiveResult]:
    response = await client.get(
        "https://archive.org/advancedsearch.php",
        params={
            "q": query,
            "fl[]": ["identifier", "title", "creator", "date", "mediatype", "licenseurl"],
            "rows": max_results,
            "output": "json",
        },
    )
    response.raise_for_status()
    payload = response.json() if response.content else {}
    docs = payload.get("response", {}).get("docs", []) if isinstance(payload, dict) else []

    out: list[ArchiveResult] = []
    for item in docs:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("identifier", "")).strip()
        title = str(item.get("title", "")).strip() or identifier
        if not title:
            continue

        creator = _string_from_maybe_list(item.get("creator"))
        date_text = _string_from_maybe_list(item.get("date"))
        media = _normalize_media(_string_from_maybe_list(item.get("mediatype")))
        rights = _string_from_maybe_list(item.get("licenseurl"))

        record_url = f"https://archive.org/details/{identifier}" if identifier else ""
        out.append(
            ArchiveResult(
                title=title,
                creator=creator,
                date=date_text,
                repository="internet_archive",
                record_url=record_url,
                access_url=record_url,
                rights=rights,
                snippet="",
                media_type=media,
            )
        )
    return out


async def _search_wikimedia(
    *,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[ArchiveResult]:
    response = await client.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
            "utf8": 1,
        },
    )
    response.raise_for_status()
    payload = response.json() if response.content else {}
    items = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []

    out: list[ArchiveResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        pageid = str(item.get("pageid", "")).strip()
        timestamp = str(item.get("timestamp", "")).strip()
        snippet = str(item.get("snippet", "")).replace("<span class=\"searchmatch\">", "")
        snippet = snippet.replace("</span>", "")

        record_url = f"https://commons.wikimedia.org/?curid={pageid}" if pageid else ""
        out.append(
            ArchiveResult(
                title=title,
                creator="",
                date=timestamp,
                repository="wikimedia",
                record_url=record_url,
                access_url=record_url,
                rights="",
                snippet=" ".join(snippet.split()),
                media_type="image",
            )
        )
    return out


async def _search_openverse(
    *,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[ArchiveResult]:
    response = await client.get(
        "https://api.openverse.org/v1/images/",
        params={"q": query, "page_size": max_results},
    )
    response.raise_for_status()
    payload = response.json() if response.content else {}
    items = payload.get("results", []) if isinstance(payload, dict) else []

    out: list[ArchiveResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip() or str(item.get("id", "")).strip()
        if not title:
            continue
        creator = str(item.get("creator", "")).strip()
        date_text = str(item.get("created_on", "")).strip()
        rights = str(item.get("license", "")).strip()
        record_url = str(item.get("foreign_landing_url", "")).strip()
        access_url = str(item.get("url", "")).strip() or record_url

        out.append(
            ArchiveResult(
                title=title,
                creator=creator,
                date=date_text,
                repository="openverse",
                record_url=record_url,
                access_url=access_url,
                rights=rights,
                snippet="",
                media_type="image",
            )
        )
    return out


def _string_from_maybe_list(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _normalize_media(raw: str) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return ""
    if "image" in text:
        return "image"
    if "audio" in text:
        return "audio"
    if "video" in text:
        return "video"
    if "text" in text or "book" in text:
        return "text"
    return text


def _year_from_archive_date(text: str) -> int | None:
    return parse_year(text)


def _merge_results(
    grouped: list[list[ArchiveResult]],
    *,
    media_types: set[str],
    year_from: int | None,
    year_to: int | None,
    max_results: int,
) -> list[ArchiveResult]:
    deduped: dict[str, ArchiveResult] = {}
    for rows in grouped:
        for row in rows:
            if media_types and row.media_type and row.media_type.lower() not in media_types:
                if "mixed" not in media_types:
                    continue

            year = _year_from_archive_date(row.date)
            if year_from is not None and (year is None or year < year_from):
                continue
            if year_to is not None and (year is None or year > year_to):
                continue

            key = row.dedupe_key()
            if key not in deduped:
                deduped[key] = row
                continue

            current = deduped[key]
            if len(row.snippet) > len(current.snippet):
                deduped[key] = row

    rows = list(deduped.values())
    rows.sort(
        key=lambda item: (
            _year_from_archive_date(item.date) or 0,
            item.repository,
            item.title,
        ),
        reverse=True,
    )
    return rows[:max_results]
