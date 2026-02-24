"""Correspondence analysis tool (local deterministic compute)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult

_FORMATS = {"markdown", "json", "csv"}


class CorrespondenceAnalysisTool(Tool):
    """Run correspondence analysis on contingency tables."""

    @property
    def name(self) -> str:
        return "correspondence_analysis"

    @property
    def description(self) -> str:
        return (
            "Run correspondence analysis (CA) on contingency tables and return "
            "dimensions, inertia, coordinates, and contribution tables."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "table": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Inline contingency matrix.",
                },
                "table_path": {
                    "type": "string",
                    "description": "Path to CSV/JSON contingency table.",
                },
                "records": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Record format for auto-contingency building.",
                },
                "row_field": {
                    "type": "string",
                    "description": "Row key for record-mode contingency.",
                },
                "column_field": {
                    "type": "string",
                    "description": "Column key for record-mode contingency.",
                },
                "value_field": {
                    "type": "string",
                    "description": "Optional numeric value field (default weight=1).",
                },
                "row_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional row labels for inline matrix.",
                },
                "column_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional column labels for inline matrix.",
                },
                "dimensions": {
                    "type": "integer",
                    "description": "Number of dimensions to retain (default 2).",
                },
                "output_formats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Formats to write: markdown/json/csv.",
                },
                "output_prefix": {
                    "type": "string",
                    "description": "Artifact filename prefix.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Artifact output directory.",
                },
            },
        }

    @property
    def timeout_seconds(self) -> int:
        return 45

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            loaded = _load_table(args=args, tool=self, ctx=ctx)
        except ValueError as e:
            return ToolResult.fail(str(e))
        if loaded is None:
            return ToolResult.fail("Provide table, table_path, or records input")

        table, row_labels, column_labels = loaded
        if len(table) < 2 or len(table[0]) < 2:
            return ToolResult.fail("Contingency table must be at least 2x2")

        for row in table:
            for value in row:
                if value < 0:
                    return ToolResult.fail("Contingency table values must be non-negative")
        total = sum(sum(row) for row in table)
        if total <= 0:
            return ToolResult.fail("Contingency table total must be > 0")

        max_dims = max(1, min(len(table) - 1, len(table[0]) - 1))
        dimensions = _clamp_int(args.get("dimensions"), default=min(2, max_dims), lo=1, hi=max_dims)

        result = _compute_correspondence_analysis(table=table, dimensions=dimensions)
        payload = _build_payload(
            result=result,
            row_labels=row_labels,
            column_labels=column_labels,
        )

        formats = _normalize_formats(args.get("output_formats"))
        if formats is None:
            return ToolResult.fail("output_formats must contain only markdown/json/csv")
        output_prefix = str(args.get("output_prefix", "correspondence-analysis")).strip()
        if not output_prefix:
            output_prefix = "correspondence-analysis"
        output_dir_raw = str(args.get("output_dir", ".")).strip() or "."

        files_changed: list[str] = []
        if ctx.workspace is not None and formats:
            output_dir = self._resolve_path(output_dir_raw, ctx.workspace)
            output_dir.mkdir(parents=True, exist_ok=True)
            if "markdown" in formats:
                path = output_dir / f"{output_prefix}.md"
                _write_text(path, _render_markdown(payload), ctx=ctx)
                files_changed.append(str(path.relative_to(ctx.workspace)))
            if "json" in formats:
                path = output_dir / f"{output_prefix}.json"
                _write_text(path, json.dumps(payload, indent=2), ctx=ctx)
                files_changed.append(str(path.relative_to(ctx.workspace)))
            if "csv" in formats:
                path = output_dir / f"{output_prefix}.csv"
                _write_csv(path, payload, ctx=ctx)
                files_changed.append(str(path.relative_to(ctx.workspace)))

        lines = [
            (
                f"Correspondence analysis complete: rows={len(row_labels)}, "
                f"cols={len(column_labels)}, dimensions={payload['dimensions_retained']}."
            ),
            "Explained inertia: "
            + ", ".join(
                f"dim{i + 1}={dim['explained_inertia']:.4f}"
                for i, dim in enumerate(payload["dimensions"])
            ),
        ]
        if files_changed:
            lines.append("Artifacts: " + ", ".join(files_changed))

        return ToolResult.ok(
            "\n".join(lines),
            data=payload,
            files_changed=files_changed,
        )


def _write_text(path: Path, content: str, *, ctx: ToolContext) -> None:
    if ctx.changelog is not None:
        ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
    path.write_text(content, encoding="utf-8")


def _write_csv(path: Path, payload: dict[str, Any], *, ctx: ToolContext) -> None:
    if ctx.changelog is not None:
        ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity_type", "label", "mass", "dimension", "coordinate", "contribution"])
        for row in payload.get("rows", []):
            label = row.get("label", "")
            mass = row.get("mass", 0.0)
            coords = row.get("coordinates", {})
            contribs = row.get("contributions", {})
            for dim, coord in coords.items():
                writer.writerow(
                    ["row", label, mass, dim, coord, contribs.get(dim)]
                )
        for col in payload.get("columns", []):
            label = col.get("label", "")
            mass = col.get("mass", 0.0)
            coords = col.get("coordinates", {})
            contribs = col.get("contributions", {})
            for dim, coord in coords.items():
                writer.writerow(
                    ["column", label, mass, dim, coord, contribs.get(dim)]
                )


def _clamp_int(value: object, *, default: int, lo: int, hi: int) -> int:
    try:
        parsed = default if value is None else int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(hi, parsed))


def _normalize_formats(raw: object) -> list[str] | None:
    if raw is None:
        return ["markdown", "json"]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for item in raw:
        fmt = str(item or "").strip().lower()
        if not fmt:
            continue
        if fmt not in _FORMATS:
            return None
        if fmt not in out:
            out.append(fmt)
    return out or ["markdown", "json"]


def _load_table(
    *,
    args: dict[str, Any],
    tool: Tool,
    ctx: ToolContext,
) -> tuple[list[list[float]], list[str], list[str]] | None:
    inline = args.get("table")
    if isinstance(inline, list) and inline and all(isinstance(row, list) for row in inline):
        matrix = []
        for row in inline:
            matrix.append([float(val) for val in row])
        row_labels = _labels(args.get("row_labels"), prefix="row", count=len(matrix))
        col_labels = _labels(args.get("column_labels"), prefix="col", count=len(matrix[0]))
        if any(len(row) != len(col_labels) for row in matrix):
            raise ValueError("All rows in table must have the same length")
        return matrix, row_labels, col_labels

    records = args.get("records")
    if isinstance(records, list) and records:
        row_field = str(args.get("row_field", "")).strip()
        col_field = str(args.get("column_field", "")).strip()
        value_field = str(args.get("value_field", "")).strip()
        if not row_field or not col_field:
            raise ValueError("records mode requires row_field and column_field")
        return _records_to_contingency(
            records,
            row_field=row_field,
            col_field=col_field,
            value_field=value_field,
        )

    table_path = str(args.get("table_path", "")).strip()
    if table_path and ctx.workspace is not None:
        path = tool._resolve_read_path(table_path, ctx.workspace, ctx.read_roots)
        if not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                rows = payload.get("table") or payload.get("matrix")
                if isinstance(rows, list):
                    matrix = [[float(val) for val in row] for row in rows if isinstance(row, list)]
                    if not matrix:
                        return None
                    row_labels = _labels(payload.get("row_labels"), prefix="row", count=len(matrix))
                    col_labels = _labels(
                        payload.get("column_labels"),
                        prefix="col",
                        count=len(matrix[0]),
                    )
                    return matrix, row_labels, col_labels
            return None

        return _load_csv_table(path)

    return None


def _load_csv_table(path: Path) -> tuple[list[list[float]], list[str], list[str]] | None:
    with open(path, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    if len(reader) < 2 or len(reader[0]) < 2:
        return None

    header = reader[0]
    if header and not _looks_numeric(header[0]):
        col_labels = [str(item).strip() or f"col{i + 1}" for i, item in enumerate(header[1:])]
        matrix: list[list[float]] = []
        row_labels: list[str] = []
        for idx, row in enumerate(reader[1:], start=1):
            if not row:
                continue
            row_labels.append(str(row[0]).strip() or f"row{idx}")
            numeric = [float(val) for val in row[1:1 + len(col_labels)]]
            matrix.append(numeric)
        return matrix, row_labels, col_labels

    matrix = [[float(val) for val in row] for row in reader if row]
    if not matrix:
        return None
    row_labels = [f"row{i + 1}" for i in range(len(matrix))]
    col_labels = [f"col{i + 1}" for i in range(len(matrix[0]))]
    return matrix, row_labels, col_labels


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _labels(raw: object, *, prefix: str, count: int) -> list[str]:
    if isinstance(raw, list):
        out = [str(item or "").strip() for item in raw]
        out = [item or f"{prefix}{idx + 1}" for idx, item in enumerate(out)]
        if len(out) >= count:
            return out[:count]
    return [f"{prefix}{idx + 1}" for idx in range(count)]


def _records_to_contingency(
    records: list[object],
    *,
    row_field: str,
    col_field: str,
    value_field: str,
) -> tuple[list[list[float]], list[str], list[str]]:
    row_order: list[str] = []
    col_order: list[str] = []
    row_index: dict[str, int] = {}
    col_index: dict[str, int] = {}
    table: list[list[float]] = []

    def ensure_row(label: str) -> int:
        if label in row_index:
            return row_index[label]
        idx = len(row_order)
        row_order.append(label)
        row_index[label] = idx
        table.append([0.0 for _ in col_order])
        return idx

    def ensure_col(label: str) -> int:
        if label in col_index:
            return col_index[label]
        idx = len(col_order)
        col_order.append(label)
        col_index[label] = idx
        for row in table:
            row.append(0.0)
        return idx

    for item in records:
        if not isinstance(item, dict):
            continue
        row_label = str(item.get(row_field, "")).strip()
        col_label = str(item.get(col_field, "")).strip()
        if not row_label or not col_label:
            continue
        val = 1.0
        if value_field:
            try:
                val = float(item.get(value_field, 1.0))
            except (TypeError, ValueError):
                val = 1.0
        r_idx = ensure_row(row_label)
        c_idx = ensure_col(col_label)
        table[r_idx][c_idx] += val

    if not table or not col_order:
        raise ValueError("records produced an empty contingency table")
    return table, row_order, col_order


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _norm(v: list[float]) -> float:
    return math.sqrt(_dot(v, v))


def _matvec(matrix: list[list[float]], v: list[float]) -> list[float]:
    return [_dot(row, v) for row in matrix]


def _outer(v: list[float]) -> list[list[float]]:
    return [[v[i] * v[j] for j in range(len(v))] for i in range(len(v))]


def _top_eigenpairs(
    *,
    matrix: list[list[float]],
    components: int,
    max_iter: int = 250,
    tol: float = 1e-10,
) -> tuple[list[float], list[list[float]]]:
    n = len(matrix)
    if n == 0:
        return [], []
    work = [[float(x) for x in row] for row in matrix]
    eigenvalues: list[float] = []
    eigenvectors: list[list[float]] = []
    for comp in range(components):
        v = [float((idx + 1) * (comp + 1)) for idx in range(n)]
        v_norm = _norm(v)
        if v_norm <= tol:
            break
        v = [x / v_norm for x in v]
        for _ in range(max_iter):
            w = _matvec(work, v)
            w_norm = _norm(w)
            if w_norm <= tol:
                break
            next_v = [x / w_norm for x in w]
            diff = _norm([next_v[i] - v[i] for i in range(n)])
            v = next_v
            if diff <= tol:
                break
        av = _matvec(work, v)
        eig = _dot(v, av)
        if eig <= tol:
            break
        eigenvalues.append(eig)
        eigenvectors.append(v)
        ov = _outer(v)
        for i in range(n):
            for j in range(n):
                work[i][j] -= eig * ov[i][j]
    return eigenvalues, eigenvectors


def _compute_correspondence_analysis(
    *,
    table: list[list[float]],
    dimensions: int,
) -> dict[str, Any]:
    rows = len(table)
    cols = len(table[0])
    total = sum(sum(row) for row in table)
    p = [[cell / total for cell in row] for row in table]

    r = [sum(row) for row in p]
    c = [sum(p[i][j] for i in range(rows)) for j in range(cols)]

    s = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            expected = r[i] * c[j]
            if expected > 0:
                s[i][j] = (p[i][j] - expected) / math.sqrt(expected)

    a = [[0.0 for _ in range(rows)] for _ in range(rows)]
    for i in range(rows):
        for k in range(rows):
            a[i][k] = sum(s[i][j] * s[k][j] for j in range(cols))

    eigenvalues, eigenvectors = _top_eigenpairs(matrix=a, components=dimensions)
    singular = [math.sqrt(max(0.0, val)) for val in eigenvalues]
    total_inertia = sum(eigenvalues)
    if total_inertia <= 0:
        total_inertia = 1.0

    row_coords = [[0.0 for _ in range(len(singular))] for _ in range(rows)]
    col_coords = [[0.0 for _ in range(len(singular))] for _ in range(cols)]
    row_contrib = [[0.0 for _ in range(len(singular))] for _ in range(rows)]
    col_contrib = [[0.0 for _ in range(len(singular))] for _ in range(cols)]

    for k, sigma in enumerate(singular):
        if sigma <= 0:
            continue
        u = [eigenvectors[k][i] for i in range(rows)]
        v = []
        for j in range(cols):
            num = sum(s[i][j] * u[i] for i in range(rows))
            v.append(num / sigma)

        eig = eigenvalues[k]
        for i in range(rows):
            if r[i] > 0:
                row_coords[i][k] = (u[i] * sigma) / math.sqrt(r[i])
                row_contrib[i][k] = (r[i] * row_coords[i][k] ** 2) / eig if eig > 0 else 0.0
        for j in range(cols):
            if c[j] > 0:
                col_coords[j][k] = (v[j] * sigma) / math.sqrt(c[j])
                col_contrib[j][k] = (c[j] * col_coords[j][k] ** 2) / eig if eig > 0 else 0.0

    return {
        "row_masses": r,
        "col_masses": c,
        "eigenvalues": eigenvalues,
        "explained": [val / total_inertia for val in eigenvalues],
        "row_coords": row_coords,
        "col_coords": col_coords,
        "row_contrib": row_contrib,
        "col_contrib": col_contrib,
    }


def _build_payload(
    *,
    result: dict[str, Any],
    row_labels: list[str],
    column_labels: list[str],
) -> dict[str, Any]:
    dims = len(result["eigenvalues"])
    dim_rows = []
    for idx in range(dims):
        dim_rows.append(
            {
                "dimension": idx + 1,
                "eigenvalue": result["eigenvalues"][idx],
                "explained_inertia": result["explained"][idx],
            }
        )

    row_items = []
    for i, label in enumerate(row_labels):
        row_items.append(
            {
                "label": label,
                "mass": result["row_masses"][i],
                "coordinates": {
                    f"dim{d + 1}": result["row_coords"][i][d]
                    for d in range(dims)
                },
                "contributions": {
                    f"dim{d + 1}": result["row_contrib"][i][d]
                    for d in range(dims)
                },
            }
        )

    col_items = []
    for j, label in enumerate(column_labels):
        col_items.append(
            {
                "label": label,
                "mass": result["col_masses"][j],
                "coordinates": {
                    f"dim{d + 1}": result["col_coords"][j][d]
                    for d in range(dims)
                },
                "contributions": {
                    f"dim{d + 1}": result["col_contrib"][j][d]
                    for d in range(dims)
                },
            }
        )

    return {
        "dimensions_retained": dims,
        "dimensions": dim_rows,
        "rows": row_items,
        "columns": col_items,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Correspondence Analysis",
        "",
        f"- **Dimensions Retained**: {payload.get('dimensions_retained', 0)}",
        "",
        "## Inertia",
        "",
        "| Dimension | Eigenvalue | Explained Inertia |",
        "|---|---:|---:|",
    ]
    for row in payload.get("dimensions", []):
        lines.append(
            "| "
            f"{row.get('dimension')} | "
            f"{float(row.get('eigenvalue', 0.0)):.6f} | "
            f"{float(row.get('explained_inertia', 0.0)):.6f} |"
        )

    lines.extend(["", "## Rows", ""])
    for row in payload.get("rows", []):
        coords = ", ".join(
            f"{key}={float(val):.4f}" for key, val in row.get("coordinates", {}).items()
        )
        lines.append(f"- {row.get('label')}: {coords}")

    lines.extend(["", "## Columns", ""])
    for col in payload.get("columns", []):
        coords = ", ".join(
            f"{key}={float(val):.4f}" for key, val in col.get("coordinates", {}).items()
        )
        lines.append(f"- {col.get('label')}: {coords}")

    lines.append("")
    return "\n".join(lines)
