"""Academic source discovery tool with normalized metadata output."""

from __future__ import annotations

import asyncio
import os
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import httpx

from loom.research.models import AcademicResult
from loom.research.text import parse_year
from loom.tools.registry import Tool, ToolContext, ToolResult

SEARCH_TIMEOUT = 18.0
DEFAULT_MAX_RESULTS = 10
MAX_RESULTS = 30
DEFAULT_USER_AGENT = "Loom/1.0 (+https://github.com/sfw/loom)"

SUPPORTED_PROVIDERS = {
    "crossref",
    "arxiv",
    "semantic_scholar",
}

SUPPORTED_SOURCE_TYPES = {
    "journal",
    "conference",
    "preprint",
    "archive",
    "book",
    "thesis",
    "article",
    "paper",
}


class AcademicSearchTool(Tool):
    """Search academic/publication sources for historical research workflows."""

    @property
    def name(self) -> str:
        return "academic_search"

    @property
    def description(self) -> str:
        return (
            "Search academic sources (Crossref, arXiv, Semantic Scholar) and "
            "return normalized publication metadata with optional year/source "
            "filters."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "year_from": {
                    "type": "integer",
                    "description": "Lower publication year bound (inclusive).",
                },
                "year_to": {
                    "type": "integer",
                    "description": "Upper publication year bound (inclusive).",
                },
                "source_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional source type filter.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max normalized results (default 10, max 30).",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "year", "citations"],
                    "description": "Sort mode for normalized output.",
                },
                "providers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional provider subset: crossref, arxiv, "
                        "semantic_scholar."
                    ),
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

        year_from = _to_int(args.get("year_from"))
        year_to = _to_int(args.get("year_to"))
        if year_from and year_to and year_from > year_to:
            return ToolResult.fail("year_from must be <= year_to")

        source_types = _normalize_source_types(args.get("source_types", []))
        if source_types is None:
            return ToolResult.fail(
                "source_types must be a list of strings from the supported set"
            )

        max_results = _clamp_int(args.get("max_results"), DEFAULT_MAX_RESULTS, 1, MAX_RESULTS)
        sort_by = str(args.get("sort_by", "relevance")).strip().lower() or "relevance"
        if sort_by not in {"relevance", "year", "citations"}:
            return ToolResult.fail("sort_by must be one of: relevance, year, citations")

        providers = _normalize_providers(args.get("providers", []))
        if providers is None:
            return ToolResult.fail(
                "providers must be a list containing only crossref/arxiv/semantic_scholar"
            )
        if not providers:
            providers = ["crossref", "arxiv", "semantic_scholar"]

        provider_errors: dict[str, str] = {}
        gathered: list[list[AcademicResult]] = []
        headers = _build_headers()

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(SEARCH_TIMEOUT),
            headers=headers,
        ) as client:
            tasks = [
                _query_provider(
                    provider,
                    query=query,
                    max_results=max_results,
                    client=client,
                )
                for provider in providers
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for provider, response in zip(providers, responses, strict=False):
            if isinstance(response, Exception):
                provider_errors[provider] = f"{type(response).__name__}: {response}"
                continue
            gathered.append(response)

        merged = _merge_results(
            gathered,
            source_types=source_types,
            year_from=year_from,
            year_to=year_to,
            sort_by=sort_by,
            max_results=max_results,
        )

        if not merged:
            error_note = ""
            if provider_errors:
                error_note = " Providers: " + "; ".join(
                    f"{name} ({msg})" for name, msg in sorted(provider_errors.items())
                )
            return ToolResult.ok(
                f"No academic results found for '{query}'.{error_note}",
                data={
                    "query": query,
                    "count": 0,
                    "results": [],
                    "providers": providers,
                    "provider_errors": provider_errors,
                },
            )

        lines: list[str] = []
        for i, row in enumerate(merged, start=1):
            author_text = ", ".join(row.authors[:3])
            if len(row.authors) > 3:
                author_text += ", et al."
            meta_parts = [
                p
                for p in [
                    str(row.year) if row.year else "",
                    row.venue,
                    row.source_db,
                ]
                if p
            ]
            meta = " | ".join(meta_parts)
            lines.append(f"{i}. {row.title}")
            if author_text:
                lines.append(f"   Authors: {author_text}")
            if meta:
                lines.append(f"   {meta}")
            if row.doi:
                lines.append(f"   DOI: {row.doi}")
            if row.url:
                lines.append(f"   URL: {row.url}")
            if row.abstract:
                lines.append(f"   Abstract: {row.abstract[:240]}")
            lines.append("")

        return ToolResult.ok(
            "\n".join(lines).strip(),
            data={
                "query": query,
                "count": len(merged),
                "results": [entry.to_dict() for entry in merged],
                "providers": providers,
                "provider_errors": provider_errors,
            },
        )


def _build_headers() -> dict[str, str]:
    user_agent = os.environ.get("LOOM_WEB_USER_AGENT", "").strip() or DEFAULT_USER_AGENT
    return {
        "User-Agent": user_agent,
        "Accept": "application/json,application/xml,text/xml,*/*",
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


def _normalize_source_types(raw: object) -> set[str] | None:
    if raw is None or raw == []:
        return set()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None

    cleaned: set[str] = set()
    for item in raw:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if text not in SUPPORTED_SOURCE_TYPES:
            return None
        cleaned.add(text)
    return cleaned


def _normalize_providers(raw: object) -> list[str] | None:
    if raw is None or raw == []:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None

    cleaned: list[str] = []
    for item in raw:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if text not in SUPPORTED_PROVIDERS:
            return None
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


async def _query_provider(
    provider: str,
    *,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[AcademicResult]:
    if provider == "crossref":
        return await _search_crossref(query=query, max_results=max_results, client=client)
    if provider == "arxiv":
        return await _search_arxiv(query=query, max_results=max_results, client=client)
    if provider == "semantic_scholar":
        return await _search_semantic_scholar(
            query=query,
            max_results=max_results,
            client=client,
        )
    raise ValueError(f"Unsupported provider: {provider}")


async def _search_crossref(
    *,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[AcademicResult]:
    response = await client.get(
        "https://api.crossref.org/works",
        params={"query": query, "rows": max_results},
    )
    response.raise_for_status()
    payload = response.json() if response.content else {}
    items = payload.get("message", {}).get("items", []) if isinstance(payload, dict) else []

    out: list[AcademicResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        titles = item.get("title") or []
        title = str(titles[0]).strip() if isinstance(titles, list) and titles else ""
        if not title:
            continue

        authors: list[str] = []
        for author in item.get("author", []) or []:
            if not isinstance(author, dict):
                continue
            given = str(author.get("given", "")).strip()
            family = str(author.get("family", "")).strip()
            name = " ".join(part for part in [given, family] if part)
            if name:
                authors.append(name)

        year = None
        issued = item.get("issued", {})
        if isinstance(issued, dict):
            parts = issued.get("date-parts", [])
            if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                year = _to_int(parts[0][0])

        venue = ""
        container = item.get("container-title")
        if isinstance(container, list) and container:
            venue = str(container[0]).strip()

        doi = str(item.get("DOI", "")).strip()
        url = str(item.get("URL", "")).strip()
        abstract = str(item.get("abstract", "")).strip()
        source_type = _crossref_type_to_source(str(item.get("type", "")).strip())
        citations = _to_int(item.get("is-referenced-by-count"))

        out.append(
            AcademicResult(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                url=url,
                doi=doi,
                abstract=_strip_jats(abstract),
                source_db="crossref",
                source_type=source_type,
                citation_count=citations,
                confidence=0.83,
            )
        )
    return out


async def _search_arxiv(
    *,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[AcademicResult]:
    encoded = quote_plus(query)
    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{encoded}&start=0&max_results={max_results}"
    )
    response = await client.get(url)
    response.raise_for_status()
    xml_text = response.text or ""
    if not xml_text.strip():
        return []

    root = ET.fromstring(xml_text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    out: list[AcademicResult] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        if not title:
            continue

        authors = [
            (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for author in entry.findall("atom:author", ns)
        ]
        authors = [a for a in authors if a]

        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        year = parse_year(published)
        abstract = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        link = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        doi = (entry.findtext("arxiv:doi", default="", namespaces=ns) or "").strip()

        out.append(
            AcademicResult(
                title=" ".join(title.split()),
                authors=authors,
                year=year,
                venue="arXiv",
                url=link,
                doi=doi,
                abstract=" ".join(abstract.split()),
                source_db="arxiv",
                source_type="preprint",
                citation_count=None,
                confidence=0.8,
            )
        )
    return out


async def _search_semantic_scholar(
    *,
    query: str,
    max_results: int,
    client: httpx.AsyncClient,
) -> list[AcademicResult]:
    response = await client.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,abstract,url,citationCount,venue,externalIds",
        },
    )
    response.raise_for_status()
    payload = response.json() if response.content else {}
    papers = payload.get("data", []) if isinstance(payload, dict) else []

    out: list[AcademicResult] = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title", "")).strip()
        if not title:
            continue

        authors: list[str] = []
        for author in paper.get("authors", []) or []:
            if not isinstance(author, dict):
                continue
            name = str(author.get("name", "")).strip()
            if name:
                authors.append(name)

        doi = ""
        ext_ids = paper.get("externalIds", {})
        if isinstance(ext_ids, dict):
            doi = str(ext_ids.get("DOI", "")).strip()

        year = _to_int(paper.get("year"))
        venue = str(paper.get("venue", "")).strip()
        abstract = str(paper.get("abstract", "")).strip()
        url = str(paper.get("url", "")).strip()
        citations = _to_int(paper.get("citationCount"))

        out.append(
            AcademicResult(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                url=url,
                doi=doi,
                abstract=abstract,
                source_db="semantic_scholar",
                source_type="article",
                citation_count=citations,
                confidence=0.78,
            )
        )
    return out


def _crossref_type_to_source(raw_type: str) -> str:
    mapping = {
        "journal-article": "journal",
        "proceedings-article": "conference",
        "book-chapter": "book",
        "book": "book",
        "posted-content": "preprint",
        "dissertation": "thesis",
        "reference-entry": "archive",
    }
    return mapping.get(raw_type.strip().lower(), "article")


def _strip_jats(text: str) -> str:
    """Best-effort cleanup for Crossref JATS abstract fragments."""
    if not text:
        return ""
    cleaned = text.replace("<jats:p>", " ").replace("</jats:p>", " ")
    while "<" in cleaned and ">" in cleaned:
        start = cleaned.find("<")
        end = cleaned.find(">", start)
        if end <= start:
            break
        cleaned = cleaned[:start] + " " + cleaned[end + 1 :]
    return " ".join(cleaned.split())


def _merge_results(
    grouped: list[list[AcademicResult]],
    *,
    source_types: set[str],
    year_from: int | None,
    year_to: int | None,
    sort_by: str,
    max_results: int,
) -> list[AcademicResult]:
    merged: list[AcademicResult] = []
    key_to_index: dict[str, int] = {}
    for bucket in grouped:
        for row in bucket:
            if source_types and row.source_type.lower() not in source_types:
                continue
            if year_from is not None and (row.year is None or row.year < year_from):
                continue
            if year_to is not None and (row.year is None or row.year > year_to):
                continue

            matched_index: int | None = None
            for key in _identity_keys(row):
                if key in key_to_index:
                    matched_index = key_to_index[key]
                    break

            if matched_index is None:
                matched_index = len(merged)
                merged.append(row)
                for key in _identity_keys(row):
                    key_to_index[key] = matched_index
                continue

            current = merged[matched_index]
            merged[matched_index] = _prefer_result(current, row)
            for key in _identity_keys(merged[matched_index]):
                key_to_index[key] = matched_index

    if sort_by == "year":
        merged.sort(
            key=lambda item: (
                item.year or 0,
                item.citation_count or 0,
                item.confidence,
            ),
            reverse=True,
        )
    elif sort_by == "citations":
        merged.sort(
            key=lambda item: (
                item.citation_count or 0,
                item.year or 0,
                item.confidence,
            ),
            reverse=True,
        )
    else:
        merged.sort(
            key=lambda item: (
                item.confidence,
                item.citation_count or 0,
                item.year or 0,
            ),
            reverse=True,
        )
    return merged[:max_results]


def _identity_keys(row: AcademicResult) -> list[str]:
    keys: list[str] = []
    if row.doi.strip():
        keys.append(f"doi:{row.doi.strip().lower()}")
    if row.url.strip():
        keys.append(f"url:{row.url.strip().lower()}")
    title = " ".join(row.title.lower().split())
    keys.append(f"title:{title}|{row.year or 0}")
    return keys


def _prefer_result(current: AcademicResult, candidate: AcademicResult) -> AcademicResult:
    better = candidate
    worse = current
    if (current.citation_count or 0) > (candidate.citation_count or 0):
        better = current
        worse = candidate
    elif (
        (current.citation_count or 0) == (candidate.citation_count or 0)
        and current.confidence >= candidate.confidence
    ):
        better = current
        worse = candidate

    return AcademicResult(
        title=better.title or worse.title,
        authors=better.authors or worse.authors,
        year=better.year or worse.year,
        venue=better.venue or worse.venue,
        url=better.url or worse.url,
        doi=better.doi or worse.doi,
        abstract=better.abstract or worse.abstract,
        source_db=better.source_db or worse.source_db,
        source_type=better.source_type or worse.source_type,
        citation_count=better.citation_count
        if better.citation_count is not None
        else worse.citation_count,
        confidence=max(current.confidence, candidate.confidence),
    )
