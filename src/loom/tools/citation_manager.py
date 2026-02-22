"""Citation and bibliography management tool."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from loom.research.models import CitationRecord
from loom.research.text import jaccard_similarity, tokenize
from loom.tools.registry import Tool, ToolContext, ToolResult

_ID_RE = re.compile(r"[^a-z0-9]+")
_BIB_ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,]+)\s*,(.*?)\n\}\s*", re.DOTALL)
_BIB_FIELD_RE = re.compile(r"(\w+)\s*=\s*\{(.*?)\}\s*,?", re.DOTALL)
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


class CitationManagerTool(Tool):
    """Manage citations and bibliography outputs for research workflows."""

    @property
    def name(self) -> str:
        return "citation_manager"

    @property
    def description(self) -> str:
        return (
            "Manage bibliography records: add, dedupe, validate, format, and "
            "map claims to citations."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "dedupe", "validate", "format", "map_claims"],
                },
                "path": {
                    "type": "string",
                    "description": "Citation database path (.json or .bib).",
                },
                "citation": {
                    "type": "object",
                    "description": "Citation payload for add operation.",
                },
                "style": {
                    "type": "string",
                    "enum": ["apa", "mla", "chicago", "ieee", "bibtex"],
                    "description": "Formatting style.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional output path for format/map_claims.",
                },
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional inline claims for map_claims.",
                },
                "claims_path": {
                    "type": "string",
                    "description": "Path to claims file for map_claims.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Top citation matches per claim (default 2, max 5).",
                },
            },
            "required": ["operation"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 30

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        operation = str(args.get("operation", "")).strip().lower()
        if operation not in {"add", "dedupe", "validate", "format", "map_claims"}:
            return ToolResult.fail("Unsupported operation")

        raw_path = str(args.get("path", "references.json")).strip() or "references.json"
        try:
            path = self._resolve_path(raw_path, ctx.workspace)
        except Exception as e:
            return ToolResult.fail(str(e))

        try:
            records = _load_citations(path)
        except ValueError as e:
            return ToolResult.fail(str(e))

        if operation == "add":
            citation = args.get("citation")
            if not isinstance(citation, dict) or not citation:
                return ToolResult.fail("add operation requires a non-empty citation object")
            record = _coerce_record(citation)
            added, updated = _upsert_record(records, record)
            _write_citations(path, records, ctx)
            rel = path.relative_to(ctx.workspace)
            action = "added" if added else "updated"
            return ToolResult.ok(
                f"Citation {action}: {record.id} in {rel}",
                files_changed=[str(rel)],
                data={
                    "action": action,
                    "citation": record.to_dict(),
                    "count": len(records),
                    "updated": updated,
                },
            )

        if operation == "dedupe":
            before = len(records)
            records = _dedupe_records(records)
            removed = before - len(records)
            _write_citations(path, records, ctx)
            rel = path.relative_to(ctx.workspace)
            return ToolResult.ok(
                f"Removed {removed} duplicate citation(s) from {rel}",
                files_changed=[str(rel)],
                data={
                    "removed": removed,
                    "count": len(records),
                },
            )

        if operation == "validate":
            report = _validate_records(records)
            issues = report["issues"]
            if not issues:
                return ToolResult.ok(
                    f"Validation passed for {len(records)} citation(s)",
                    data=report,
                )
            lines = [f"Validation found {len(issues)} issue group(s):"]
            for issue in issues[:20]:
                lines.append(f"- {issue['id']}: {', '.join(issue['problems'])}")
            if len(issues) > 20:
                lines.append(f"... and {len(issues) - 20} more")
            return ToolResult.ok("\n".join(lines), data=report)

        if operation == "format":
            style = str(args.get("style", "apa")).strip().lower() or "apa"
            if style not in {"apa", "mla", "chicago", "ieee", "bibtex"}:
                return ToolResult.fail("Unsupported style")
            output_path = _resolve_output_path(
                args.get("output_path"),
                default_name=f"references-{style}.{'bib' if style == 'bibtex' else 'md'}",
                tool=self,
                workspace=ctx.workspace,
            )
            formatted = _format_records(records, style=style)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(output_path), subtask_id=ctx.subtask_id)
            output_path.write_text(formatted, encoding="utf-8")
            rel = output_path.relative_to(ctx.workspace)
            return ToolResult.ok(
                f"Formatted {len(records)} citation(s) to {rel}",
                files_changed=[str(rel)],
                data={
                    "style": style,
                    "count": len(records),
                    "output_path": str(rel),
                },
            )

        claims = args.get("claims", [])
        claims_path = args.get("claims_path")
        top_k = _clamp_int(args.get("top_k"), default=2, low=1, high=5)

        claim_texts = _load_claims(
            claims=claims,
            claims_path=claims_path,
            tool=self,
            workspace=ctx.workspace,
        )
        if not claim_texts:
            return ToolResult.fail("map_claims needs claims or claims_path with at least one claim")

        mappings = _map_claims_to_citations(claim_texts, records, top_k=top_k)
        output_path = _resolve_output_path(
            args.get("output_path"),
            default_name="claim_citation_map.csv",
            tool=self,
            workspace=ctx.workspace,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(str(output_path), subtask_id=ctx.subtask_id)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "claim",
                    "citation_id",
                    "score",
                    "title",
                    "url",
                    "doi",
                ],
            )
            writer.writeheader()
            writer.writerows(mappings)

        rel = output_path.relative_to(ctx.workspace)
        linked_claims = len({row["claim"] for row in mappings})
        return ToolResult.ok(
            f"Mapped {linked_claims}/{len(claim_texts)} claim(s) to citations in {rel}",
            files_changed=[str(rel)],
            data={
                "claims": len(claim_texts),
                "linked_claims": linked_claims,
                "rows": len(mappings),
                "output_path": str(rel),
            },
        )


def _clamp_int(value: object, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _resolve_output_path(
    raw: object,
    *,
    default_name: str,
    tool: Tool,
    workspace: Path,
) -> Path:
    text = str(raw or "").strip()
    if not text:
        text = default_name
    return tool._resolve_path(text, workspace)


def _slug(text: str, fallback: str = "citation") -> str:
    lowered = _ID_RE.sub("-", (text or "").strip().lower()).strip("-")
    return lowered or fallback


def _coerce_authors(raw: object) -> list[str]:
    if isinstance(raw, list):
        out = [str(item or "").strip() for item in raw]
        return [item for item in out if item]
    if isinstance(raw, str):
        chunks = [part.strip() for part in raw.split(";")]
        if len(chunks) <= 1:
            chunks = [part.strip() for part in raw.split(",")]
        return [part for part in chunks if part]
    return []


def _coerce_year(raw: object) -> int | None:
    try:
        if raw is None:
            return None
        value = int(raw)
        if 1000 <= value <= 2100:
            return value
        return None
    except (TypeError, ValueError):
        return None


def _citation_id_from_raw(data: dict[str, Any]) -> str:
    doi = str(data.get("doi", "")).strip()
    if doi:
        return _slug(f"doi-{doi}")
    url = str(data.get("url", "")).strip()
    if url:
        return _slug(f"url-{url}")

    title = str(data.get("title", "")).strip()
    year = _coerce_year(data.get("year")) or 0
    author = ""
    authors = _coerce_authors(data.get("authors", data.get("author", [])))
    if authors:
        author = authors[0]
    return _slug(f"{author}-{year}-{title}")


def _coerce_record(raw: dict[str, Any]) -> CitationRecord:
    record = CitationRecord(
        id=str(raw.get("id", "")).strip() or _citation_id_from_raw(raw),
        title=str(raw.get("title", "")).strip(),
        authors=_coerce_authors(raw.get("authors", raw.get("author", []))),
        year=_coerce_year(raw.get("year")),
        venue=str(raw.get("venue", raw.get("journal", ""))).strip(),
        url=str(raw.get("url", "")).strip(),
        doi=str(raw.get("doi", "")).strip(),
        publisher=str(raw.get("publisher", "")).strip(),
        type=str(raw.get("type", "")).strip(),
        accessed_on=str(raw.get("accessed_on", "")).strip(),
        notes=str(raw.get("notes", "")).strip(),
        tags=[str(item or "").strip() for item in raw.get("tags", []) if str(item or "").strip()],
    )
    return record


def _load_citations(path: Path) -> list[CitationRecord]:
    if not path.exists():
        return []

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("citations", [])
        else:
            items = payload
        if not isinstance(items, list):
            raise ValueError("Citation JSON must be a list or {citations: [...]} object")
        return [_coerce_record(item) for item in items if isinstance(item, dict)]

    if suffix == ".bib":
        text = path.read_text(encoding="utf-8")
        out: list[CitationRecord] = []
        for match in _BIB_ENTRY_RE.finditer(text):
            key = match.group(1).strip()
            body = match.group(2)
            fields = {
                fm.group(1).strip().lower(): " ".join(fm.group(2).split())
                for fm in _BIB_FIELD_RE.finditer(body)
            }
            out.append(
                CitationRecord(
                    id=key or _slug(fields.get("title", "citation")),
                    title=fields.get("title", ""),
                    authors=_coerce_authors(fields.get("author", "")),
                    year=_coerce_year(fields.get("year")),
                    venue=fields.get("journal", fields.get("booktitle", "")),
                    url=fields.get("url", ""),
                    doi=fields.get("doi", ""),
                    publisher=fields.get("publisher", ""),
                    type=fields.get("entrytype", ""),
                )
            )
        return out

    raise ValueError("Unsupported citation path extension. Use .json or .bib")


def _record_to_bibtex(record: CitationRecord) -> str:
    entry_type = record.type.strip().lower() or "article"
    fields: list[tuple[str, str]] = [("title", record.title)]
    if record.authors:
        fields.append(("author", " and ".join(record.authors)))
    if record.year:
        fields.append(("year", str(record.year)))
    if record.venue:
        fields.append(("journal", record.venue))
    if record.publisher:
        fields.append(("publisher", record.publisher))
    if record.doi:
        fields.append(("doi", record.doi))
    if record.url:
        fields.append(("url", record.url))
    body = ",\n".join(f"  {key} = {{{value}}}" for key, value in fields if value)
    return f"@{entry_type}{{{record.id},\n{body}\n}}"


def _write_citations(path: Path, records: list[CitationRecord], ctx: ToolContext) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ctx.changelog is not None:
        ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = [record.to_dict() for record in records]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    if suffix == ".bib":
        text = "\n\n".join(_record_to_bibtex(record) for record in records)
        if text and not text.endswith("\n"):
            text += "\n"
        path.write_text(text, encoding="utf-8")
        return
    raise ValueError("Unsupported citation path extension. Use .json or .bib")


def _merge_records(preferred: CitationRecord, other: CitationRecord) -> CitationRecord:
    return CitationRecord(
        id=preferred.id,
        title=preferred.title or other.title,
        authors=preferred.authors or other.authors,
        year=preferred.year or other.year,
        venue=preferred.venue or other.venue,
        url=preferred.url or other.url,
        doi=preferred.doi or other.doi,
        publisher=preferred.publisher or other.publisher,
        type=preferred.type or other.type,
        accessed_on=preferred.accessed_on or other.accessed_on,
        notes=preferred.notes or other.notes,
        tags=sorted(set(preferred.tags) | set(other.tags)),
    )


def _record_weight(record: CitationRecord) -> int:
    score = 0
    for value in [
        record.title,
        record.authors,
        record.year,
        record.venue,
        record.url,
        record.doi,
        record.publisher,
        record.notes,
        record.tags,
    ]:
        if value:
            score += 1
    return score


def _dedupe_records(records: list[CitationRecord]) -> list[CitationRecord]:
    deduped: dict[str, CitationRecord] = {}
    for record in records:
        key = record.dedupe_key()
        current = deduped.get(key)
        if current is None:
            deduped[key] = record
            continue
        if _record_weight(record) >= _record_weight(current):
            deduped[key] = _merge_records(record, current)
        else:
            deduped[key] = _merge_records(current, record)
    return sorted(deduped.values(), key=lambda item: item.id)


def _upsert_record(records: list[CitationRecord], record: CitationRecord) -> tuple[bool, bool]:
    key = record.dedupe_key()
    for idx, existing in enumerate(records):
        if existing.dedupe_key() == key or existing.id == record.id:
            records[idx] = _merge_records(record, existing)
            return (False, True)
    records.append(record)
    records[:] = sorted(records, key=lambda item: item.id)
    return (True, False)


def _validate_records(records: list[CitationRecord]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for record in records:
        problems: list[str] = []
        if not record.title.strip():
            problems.append("missing title")
        if not record.url.strip() and not record.doi.strip():
            problems.append("missing url/doi")
        if record.year is not None and not (1000 <= record.year <= 2100):
            problems.append("invalid year")
        if record.url and not _URL_RE.match(record.url):
            problems.append("url must start with http:// or https://")
        if record.id in seen_ids:
            problems.append("duplicate id")
        seen_ids.add(record.id)

        if problems:
            issues.append({"id": record.id, "problems": problems})

    return {
        "count": len(records),
        "issue_count": len(issues),
        "issues": issues,
    }


def _format_authors(authors: list[str], style: str) -> str:
    if not authors:
        return ""
    if style == "ieee":
        return ", ".join(authors)
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return ", ".join(authors[:-1]) + f", and {authors[-1]}"


def _format_record(record: CitationRecord, style: str, index: int) -> str:
    authors = _format_authors(record.authors, style)
    year = str(record.year) if record.year else "n.d."
    venue = record.venue or ""

    if style == "apa":
        parts = [
            f"{authors} ({year})." if authors else f"({year}).",
            f"{record.title}.",
            f"{venue}." if venue else "",
            record.url or (f"https://doi.org/{record.doi}" if record.doi else ""),
        ]
        return " ".join(part for part in parts if part).strip()

    if style == "mla":
        parts = [
            f"{authors}." if authors else "",
            f'"{record.title}."',
            f"{venue}," if venue else "",
            year + ",",
            record.url or (f"https://doi.org/{record.doi}" if record.doi else ""),
        ]
        return " ".join(part for part in parts if part).strip().rstrip(",") + "."

    if style == "chicago":
        parts = [
            f"{authors}." if authors else "",
            f'"{record.title}."',
            venue + "," if venue else "",
            year + ".",
            record.url or (f"https://doi.org/{record.doi}" if record.doi else ""),
        ]
        return " ".join(part for part in parts if part).strip()

    if style == "ieee":
        parts = [
            f"[{index}]",
            f"{authors}," if authors else "",
            f'"{record.title},"',
            venue + "," if venue else "",
            year + ".",
            record.url or (f"https://doi.org/{record.doi}" if record.doi else ""),
        ]
        return " ".join(part for part in parts if part).strip()

    if style == "bibtex":
        return _record_to_bibtex(record)

    raise ValueError(f"Unsupported style: {style}")


def _format_records(records: list[CitationRecord], *, style: str) -> str:
    if style == "bibtex":
        text = "\n\n".join(_record_to_bibtex(record) for record in records)
        if text and not text.endswith("\n"):
            text += "\n"
        return text

    lines = [f"# References ({style.upper()})", ""]
    for i, record in enumerate(records, start=1):
        entry = _format_record(record, style, i)
        marker = f"{i}." if style != "ieee" else ""
        lines.append(f"{marker} {entry}".strip())
    lines.append("")
    return "\n".join(lines)


def _read_claims_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("claims", [])
        else:
            values = payload
        if not isinstance(values, list):
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            out: list[str] = []
            for row in reader:
                if not isinstance(row, dict):
                    continue
                claim = str(row.get("claim", "")).strip()
                if not claim:
                    for value in row.values():
                        text = str(value or "").strip()
                        if text:
                            claim = text
                            break
                if claim:
                    out.append(claim)
            return out

    text = path.read_text(encoding="utf-8")
    return [line.strip(" -\t") for line in text.splitlines() if line.strip()]


def _load_claims(
    *,
    claims: object,
    claims_path: object,
    tool: Tool,
    workspace: Path,
) -> list[str]:
    out: list[str] = []
    if isinstance(claims, list):
        out.extend(str(item).strip() for item in claims if str(item).strip())

    path_text = str(claims_path or "").strip()
    if path_text:
        path = tool._resolve_read_path(path_text, workspace, read_roots=None)
        if path.exists() and path.is_file():
            out.extend(_read_claims_file(path))

    deduped: list[str] = []
    seen: set[str] = set()
    for claim in out:
        norm = " ".join(claim.lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(claim)
    return deduped


def _map_claims_to_citations(
    claims: list[str],
    records: list[CitationRecord],
    *,
    top_k: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    indexed = [
        (
            record,
            tokenize(" ".join([record.title, record.venue, record.notes, " ".join(record.tags)])),
        )
        for record in records
    ]

    for claim in claims:
        claim_tokens = tokenize(claim)
        scored: list[tuple[float, CitationRecord]] = []
        for record, ref_tokens in indexed:
            score = jaccard_similarity(claim_tokens, ref_tokens)
            if score <= 0:
                continue
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        for score, record in scored[:top_k]:
            rows.append(
                {
                    "claim": claim,
                    "citation_id": record.id,
                    "score": f"{score:.4f}",
                    "title": record.title,
                    "url": record.url,
                    "doi": record.doi,
                }
            )
    return rows
