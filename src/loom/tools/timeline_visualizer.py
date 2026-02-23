"""Timeline visualization tool for historical event analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult

_ALLOWED_FORMATS = {"markdown", "mermaid", "csv", "json"}
_ALLOWED_GROUPS = {"none", "entity", "region", "topic"}
_ALLOWED_GRANULARITY = {"auto", "day", "month", "year"}


@dataclass(frozen=True)
class TimelineEvent:
    title: str
    date_raw: str
    normalized_date: str
    year: int
    month: int
    day: int
    precision: str
    description: str = ""
    entity: str = ""
    region: str = ""
    topic: str = ""
    source: str = ""

    def sort_key(self) -> tuple[int, int, int, str, str]:
        return (self.year, self.month, self.day, self.title.lower(), self.entity.lower())

    def group_value(self, group_by: str) -> str:
        if group_by == "entity":
            return self.entity.strip() or "Unspecified Entity"
        if group_by == "region":
            return self.region.strip() or "Unspecified Region"
        if group_by == "topic":
            return self.topic.strip() or "Unspecified Topic"
        return "Timeline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "date": self.normalized_date,
            "date_raw": self.date_raw,
            "precision": self.precision,
            "description": self.description,
            "entity": self.entity,
            "region": self.region,
            "topic": self.topic,
            "source": self.source,
        }


class TimelineVisualizerTool(Tool):
    """Build chronology artifacts from structured event inputs."""

    @property
    def name(self) -> str:
        return "timeline_visualizer"

    @property
    def description(self) -> str:
        return (
            "Render timeline artifacts from events (markdown, mermaid, csv, json) "
            "and flag chronological conflicts."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Inline event objects.",
                },
                "events_path": {
                    "type": "string",
                    "description": "Path to CSV/JSON events file.",
                },
                "title": {
                    "type": "string",
                    "description": "Timeline title.",
                },
                "granularity": {
                    "type": "string",
                    "enum": ["auto", "day", "month", "year"],
                    "description": "Display granularity.",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["none", "entity", "region", "topic"],
                    "description": "Grouping field.",
                },
                "output_formats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Output formats to generate.",
                },
                "output_prefix": {
                    "type": "string",
                    "description": "Output filename prefix.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Output directory path.",
                },
            },
        }

    @property
    def timeout_seconds(self) -> int:
        return 30

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        granularity = str(args.get("granularity", "auto")).strip().lower() or "auto"
        if granularity not in _ALLOWED_GRANULARITY:
            return ToolResult.fail("granularity must be one of auto/day/month/year")

        group_by = str(args.get("group_by", "none")).strip().lower() or "none"
        if group_by not in _ALLOWED_GROUPS:
            return ToolResult.fail("group_by must be one of none/entity/region/topic")

        output_formats = _coerce_formats(args.get("output_formats", ["markdown", "mermaid"]))
        if output_formats is None:
            return ToolResult.fail("output_formats must contain only markdown/mermaid/csv/json")

        events = _load_events(
            events=args.get("events"),
            events_path=args.get("events_path"),
            tool=self,
            ctx=ctx,
        )
        if not events:
            return ToolResult.fail("No valid events provided")

        events.sort(key=lambda item: item.sort_key())
        title = str(args.get("title", "")).strip() or "Historical Timeline"
        effective_granularity = _choose_granularity(events, requested=granularity)
        conflicts = _detect_conflicts(events)

        rendered: dict[str, str] = {}
        if "markdown" in output_formats:
            rendered["markdown"] = _render_markdown(
                events,
                title=title,
                group_by=group_by,
                granularity=effective_granularity,
            )
        if "mermaid" in output_formats:
            rendered["mermaid"] = _render_mermaid(
                events,
                title=title,
                group_by=group_by,
                granularity=effective_granularity,
            )
        if "csv" in output_formats:
            rendered["csv"] = _render_csv(events)
        if "json" in output_formats:
            rendered["json"] = json.dumps([item.to_dict() for item in events], indent=2)

        files_changed: list[str] = []
        output_prefix = str(args.get("output_prefix", "timeline")).strip() or "timeline"
        output_dir_raw = str(args.get("output_dir", ".")).strip() or "."

        if ctx.workspace is not None:
            output_dir = self._resolve_path(output_dir_raw, ctx.workspace)
            output_dir.mkdir(parents=True, exist_ok=True)
            for fmt, text in rendered.items():
                ext = "md" if fmt == "markdown" else fmt
                path = output_dir / f"{output_prefix}.{ext}"
                if ctx.changelog is not None:
                    ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
                path.write_text(text, encoding="utf-8")
                files_changed.append(str(path.relative_to(ctx.workspace)))

        lines = [
            (
                f"Built timeline with {len(events)} event(s), {len(conflicts)} conflict(s), "
                f"granularity={effective_granularity}, group_by={group_by}."
            )
        ]
        if files_changed:
            lines.append("Artifacts: " + ", ".join(files_changed))
        if conflicts:
            preview = "; ".join(conflicts[:3])
            lines.append(f"Conflict preview: {preview}")

        return ToolResult.ok(
            "\n".join(lines),
            files_changed=files_changed,
            data={
                "title": title,
                "events": [item.to_dict() for item in events],
                "event_count": len(events),
                "conflicts": conflicts,
                "conflict_count": len(conflicts),
                "group_by": group_by,
                "granularity": effective_granularity,
                "rendered": rendered,
            },
        )


def _coerce_formats(raw: object) -> list[str] | None:
    if raw is None:
        return ["markdown", "mermaid"]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None

    out: list[str] = []
    for item in raw:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if text not in _ALLOWED_FORMATS:
            return None
        if text not in out:
            out.append(text)
    return out or ["markdown", "mermaid"]


def _load_events(
    *,
    events: object,
    events_path: object,
    tool: Tool,
    ctx: ToolContext,
) -> list[TimelineEvent]:
    rows: list[dict[str, Any]] = []

    if isinstance(events, list):
        for item in events:
            if isinstance(item, dict):
                rows.append(item)

    path_text = str(events_path or "").strip()
    if path_text:
        if ctx.workspace is None:
            return []
        path = tool._resolve_read_path(path_text, ctx.workspace, ctx.read_roots)
        if path.exists() and path.is_file():
            rows.extend(_read_events_file(path))

    out: list[TimelineEvent] = []
    for row in rows:
        event = _coerce_event(row)
        if event is not None:
            out.append(event)
    return out


def _read_events_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("events", [])
        else:
            items = payload
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []

    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [row for row in reader if isinstance(row, dict)]

    text = path.read_text(encoding="utf-8")
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append({"title": line, "date": ""})
    return out


def _coerce_event(row: dict[str, Any]) -> TimelineEvent | None:
    title = str(row.get("title", "")).strip()
    if not title:
        return None

    date_raw = str(row.get("date", row.get("when", ""))).strip()
    parsed = _parse_event_date(date_raw)
    if parsed is None:
        return None

    normalized, year, month, day, precision = parsed
    return TimelineEvent(
        title=title,
        date_raw=date_raw,
        normalized_date=normalized,
        year=year,
        month=month,
        day=day,
        precision=precision,
        description=str(row.get("description", "")).strip(),
        entity=str(row.get("entity", "")).strip(),
        region=str(row.get("region", "")).strip(),
        topic=str(row.get("topic", "")).strip(),
        source=str(row.get("source", "")).strip(),
    )


def _parse_event_date(raw: str) -> tuple[str, int, int, int, str] | None:
    text = (raw or "").strip()
    if not text:
        return None

    text = text.replace("/", "-")
    parts = text.split("-")

    try:
        if len(parts) == 1:
            year = int(parts[0])
            if year < 1:
                return None
            return (f"{year:04d}", year, 1, 1, "year")

        if len(parts) == 2:
            year = int(parts[0])
            month = int(parts[1])
            date(year, month, 1)
            return (f"{year:04d}-{month:02d}", year, month, 1, "month")

        if len(parts) >= 3:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            date(year, month, day)
            return (f"{year:04d}-{month:02d}-{day:02d}", year, month, day, "day")
    except (TypeError, ValueError):
        return None

    return None


def _choose_granularity(events: list[TimelineEvent], *, requested: str) -> str:
    if requested != "auto":
        return requested
    if any(event.precision == "day" for event in events):
        return "day"
    if any(event.precision == "month" for event in events):
        return "month"
    return "year"


def _display_date(event: TimelineEvent, granularity: str) -> str:
    if granularity == "year":
        return f"{event.year:04d}"
    if granularity == "month":
        return f"{event.year:04d}-{event.month:02d}"
    return f"{event.year:04d}-{event.month:02d}-{event.day:02d}"


def _group_events(events: list[TimelineEvent], group_by: str) -> dict[str, list[TimelineEvent]]:
    groups: dict[str, list[TimelineEvent]] = {}
    for event in events:
        key = event.group_value(group_by)
        groups.setdefault(key, []).append(event)
    return groups


def _render_markdown(
    events: list[TimelineEvent],
    *,
    title: str,
    group_by: str,
    granularity: str,
) -> str:
    lines = [f"# {title}", ""]
    groups = _group_events(events, group_by)

    for group_name in sorted(groups):
        group_events = groups[group_name]
        if group_by != "none":
            lines.append(f"## {group_name}")
            lines.append("")
        for event in group_events:
            stamp = _display_date(event, granularity)
            lines.append(f"- **{stamp}**: {event.title}")
            details: list[str] = []
            if event.description:
                details.append(event.description)
            if event.entity:
                details.append(f"Entity: {event.entity}")
            if event.region:
                details.append(f"Region: {event.region}")
            if event.topic:
                details.append(f"Topic: {event.topic}")
            if event.source:
                details.append(f"Source: {event.source}")
            for item in details:
                lines.append(f"  - {item}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_mermaid(
    events: list[TimelineEvent],
    *,
    title: str,
    group_by: str,
    granularity: str,
) -> str:
    lines = ["timeline", f"    title {title}"]
    for event in events:
        stamp = _display_date(event, granularity)
        group_label = ""
        if group_by != "none":
            group_label = f"[{event.group_value(group_by)}] "
        label = f"{group_label}{event.title}".replace("\n", " ")
        lines.append(f"    {stamp} : {label}")
    return "\n".join(lines) + "\n"


def _render_csv(events: list[TimelineEvent]) -> str:
    lines = ["date,title,description,entity,region,topic,source"]
    for event in events:
        row = [
            event.normalized_date,
            event.title,
            event.description,
            event.entity,
            event.region,
            event.topic,
            event.source,
        ]
        escaped = []
        for value in row:
            text = value.replace('"', '""')
            escaped.append(f'"{text}"')
        lines.append(",".join(escaped))
    return "\n".join(lines) + "\n"


def _detect_conflicts(events: list[TimelineEvent]) -> list[str]:
    seen: dict[tuple[str, str], str] = {}
    conflicts: list[str] = []
    for event in events:
        key = (event.title.strip().lower(), event.entity.strip().lower())
        prior = seen.get(key)
        if prior is None:
            seen[key] = event.normalized_date
            continue
        if prior != event.normalized_date:
            conflicts.append(

                    f"{event.title} ({event.entity or 'unspecified'}) has "
                    f"conflicting dates: {prior} vs {event.normalized_date}"

            )
    return conflicts
