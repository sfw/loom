"""Social network mapping tool (deterministic graph analytics)."""

from __future__ import annotations

import csv
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult

_FORMATS = {"markdown", "json", "csv", "graphml"}
_RELATION_RE = re.compile(
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b"
    r"\s+(?:met(?:\s+with)?|wrote\s+to|emailed|called|visited|contacted|"
    r"allied\s+with|worked\s+with|spoke\s+with)\s+"
    r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b"
)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    weight: float = 1.0
    relation: str = ""

    def key(self, *, directed: bool) -> tuple[str, str, str]:
        if directed:
            return (self.source, self.target, self.relation)
        a, b = sorted([self.source, self.target])
        return (a, b, self.relation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "relation": self.relation,
        }


class SocialNetworkMapperTool(Tool):
    """Build graph maps and compute network metrics."""

    @property
    def name(self) -> str:
        return "social_network_mapper"

    @property
    def description(self) -> str:
        return (
            "Build and analyze social-network graphs from structured edges and/or "
            "heuristically extracted text relations."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional node list (strings or objects with id).",
                },
                "edges": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Edge list with source/target/weight/relation.",
                },
                "directed": {
                    "type": "boolean",
                    "description": "Treat graph as directed (default false).",
                },
                "text": {
                    "type": "string",
                    "description": "Optional text for heuristic relation extraction.",
                },
                "text_path": {
                    "type": "string",
                    "description": "Optional text file path for extraction.",
                },
                "extract_relations_from_text": {
                    "type": "boolean",
                    "description": "Enable heuristic extraction from text/text_path.",
                },
                "output_formats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Output formats: markdown/json/csv/graphml.",
                },
                "output_prefix": {
                    "type": "string",
                    "description": "Artifact filename prefix.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Output directory path.",
                },
            },
        }

    @property
    def timeout_seconds(self) -> int:
        return 45

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        directed = bool(args.get("directed", False))
        formats = _normalize_formats(args.get("output_formats"))
        if formats is None:
            return ToolResult.fail("output_formats must contain only markdown/json/csv/graphml")

        load = _load_nodes_and_edges(args=args, tool=self, ctx=ctx)
        if load is None:
            return ToolResult.fail("Provide nodes/edges or text input for extraction")
        node_ids, edges, extraction_warnings = load

        if not node_ids:
            return ToolResult.fail("No nodes found")
        if not edges:
            return ToolResult.fail("No edges found")

        graph = _build_adjacency(node_ids=node_ids, edges=edges, directed=directed)
        degree = _degree_centrality(graph["adj"], directed=directed)
        closeness = _closeness_centrality(graph["adj"], directed=directed)
        betweenness = _betweenness_centrality(graph["adj"], directed=directed)
        components = _connected_components(graph["adj_undirected"])

        payload = {
            "directed": directed,
            "node_count": len(node_ids),
            "edge_count": len(edges),
            "nodes": [{"id": node} for node in node_ids],
            "edges": [edge.to_dict() for edge in edges],
            "metrics": {
                "degree_centrality": degree,
                "closeness_centrality": closeness,
                "betweenness_centrality": betweenness,
                "components": components,
                "component_count": len(components),
                "top_degree_nodes": _top_items(degree, n=5),
                "top_betweenness_nodes": _top_items(betweenness, n=5),
            },
            "warnings": extraction_warnings,
        }

        files_changed: list[str] = []
        if ctx.workspace is not None and formats:
            output_prefix = str(args.get("output_prefix", "social-network")).strip()
            if not output_prefix:
                output_prefix = "social-network"
            output_dir_raw = str(args.get("output_dir", ".")).strip() or "."
            output_dir = self._resolve_path(output_dir_raw, ctx.workspace)
            output_dir.mkdir(parents=True, exist_ok=True)
            files_changed.extend(
                _write_outputs(
                    output_dir=output_dir,
                    output_prefix=output_prefix,
                    formats=formats,
                    payload=payload,
                    ctx=ctx,
                )
            )

        lines = [
            (
                f"Social network mapped: nodes={payload['node_count']}, "
                f"edges={payload['edge_count']}, components={len(components)}."
            ),
            "Top degree: "
            + ", ".join(f"{name} ({score:.3f})" for name, score in _top_items(degree, n=3)),
        ]
        if files_changed:
            lines.append("Artifacts: " + ", ".join(files_changed))
        if extraction_warnings:
            lines.append("Warnings: " + "; ".join(extraction_warnings))

        return ToolResult.ok(
            "\n".join(lines),
            data=payload,
            files_changed=files_changed,
        )


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


def _normalize_node(item: object) -> str:
    if isinstance(item, dict):
        return str(item.get("id", item.get("name", ""))).strip()
    return str(item or "").strip()


def _normalize_edge(item: object) -> Edge | None:
    if not isinstance(item, dict):
        return None
    source = str(item.get("source", "")).strip()
    target = str(item.get("target", "")).strip()
    if not source or not target:
        return None
    try:
        weight = float(item.get("weight", 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    relation = str(item.get("relation", "")).strip()
    return Edge(source=source, target=target, weight=weight, relation=relation)


def _load_nodes_and_edges(
    *,
    args: dict[str, Any],
    tool: Tool,
    ctx: ToolContext,
) -> tuple[list[str], list[Edge], list[str]] | None:
    warnings: list[str] = []
    nodes_raw = args.get("nodes")
    edges_raw = args.get("edges")

    nodes: list[str] = []
    if isinstance(nodes_raw, list):
        for item in nodes_raw:
            node = _normalize_node(item)
            if node:
                nodes.append(node)

    edges: list[Edge] = []
    if isinstance(edges_raw, list):
        for item in edges_raw:
            edge = _normalize_edge(item)
            if edge is not None:
                edges.append(edge)

    extract_from_text = bool(args.get("extract_relations_from_text", False))
    text = str(args.get("text", "")).strip()
    text_path = str(args.get("text_path", "")).strip()
    if (extract_from_text or (not edges and (text or text_path))) and (text or text_path):
        text_blob = text
        if text_path and ctx.workspace is not None:
            path = tool._resolve_read_path(text_path, ctx.workspace, ctx.read_roots)
            if path.exists() and path.is_file():
                text_blob = path.read_text(encoding="utf-8")
        extracted = _extract_edges_from_text(text_blob)
        if extracted:
            edges.extend(extracted)
        else:
            warnings.append("No relations extracted from provided text.")

    if not nodes and edges:
        seen = set()
        for edge in edges:
            if edge.source not in seen:
                nodes.append(edge.source)
                seen.add(edge.source)
            if edge.target not in seen:
                nodes.append(edge.target)
                seen.add(edge.target)

    if not nodes and not edges:
        return None

    # Deduplicate edges while preserving order.
    directed = bool(args.get("directed", False))
    deduped: list[Edge] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = edge.key(directed=directed)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped.append(edge)

    nodes = list(dict.fromkeys(nodes))
    return nodes, deduped, warnings


def _extract_edges_from_text(text: str) -> list[Edge]:
    out: list[Edge] = []
    for match in _RELATION_RE.finditer(text or ""):
        source = match.group(1).strip()
        target = match.group(2).strip()
        if source and target and source != target:
            out.append(Edge(source=source, target=target, relation="text_relation"))
    return out


def _build_adjacency(
    *,
    node_ids: list[str],
    edges: list[Edge],
    directed: bool,
) -> dict[str, Any]:
    adj: dict[str, set[str]] = {node: set() for node in node_ids}
    adj_undirected: dict[str, set[str]] = {node: set() for node in node_ids}
    for edge in edges:
        adj.setdefault(edge.source, set()).add(edge.target)
        if not directed:
            adj.setdefault(edge.target, set()).add(edge.source)
        adj_undirected.setdefault(edge.source, set()).add(edge.target)
        adj_undirected.setdefault(edge.target, set()).add(edge.source)
    return {"adj": adj, "adj_undirected": adj_undirected}


def _degree_centrality(adj: dict[str, set[str]], *, directed: bool) -> dict[str, float]:
    nodes = list(adj.keys())
    n = len(nodes)
    if n <= 1:
        return {node: 0.0 for node in nodes}
    if directed:
        in_counts = {node: 0 for node in nodes}
        for src, neighbors in adj.items():
            for dst in neighbors:
                in_counts[dst] = in_counts.get(dst, 0) + 1
        return {
            node: (len(adj.get(node, set())) + in_counts.get(node, 0)) / (2 * (n - 1))
            for node in nodes
        }
    return {node: len(adj.get(node, set())) / (n - 1) for node in nodes}


def _bfs_distances(adj: dict[str, set[str]], source: str) -> dict[str, int]:
    dist: dict[str, int] = {source: 0}
    queue: deque[str] = deque([source])
    while queue:
        node = queue.popleft()
        for nxt in adj.get(node, set()):
            if nxt in dist:
                continue
            dist[nxt] = dist[node] + 1
            queue.append(nxt)
    return dist


def _closeness_centrality(adj: dict[str, set[str]], *, directed: bool) -> dict[str, float]:
    # For directed graphs we use outgoing path reachability.
    del directed
    nodes = list(adj.keys())
    n = len(nodes)
    out: dict[str, float] = {}
    for node in nodes:
        dist = _bfs_distances(adj, node)
        if len(dist) <= 1:
            out[node] = 0.0
            continue
        total_dist = float(sum(dist.values()))
        if total_dist <= 0:
            out[node] = 0.0
            continue
        reach = len(dist) - 1
        out[node] = (reach / (n - 1)) * (reach / total_dist)
    return out


def _betweenness_centrality(adj: dict[str, set[str]], *, directed: bool) -> dict[str, float]:
    nodes = list(adj.keys())
    cb: dict[str, float] = {node: 0.0 for node in nodes}
    for source in nodes:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {v: [] for v in nodes}
        sigma: dict[str, float] = {v: 0.0 for v in nodes}
        sigma[source] = 1.0
        dist: dict[str, int] = {v: -1 for v in nodes}
        dist[source] = 0
        queue: deque[str] = deque([source])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj.get(v, set()):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        delta: dict[str, float] = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != source:
                cb[w] += delta[w]

    if not directed:
        for node in cb:
            cb[node] /= 2.0

    n = len(nodes)
    if n <= 2:
        return {node: 0.0 for node in nodes}
    scale = 1.0 / ((n - 1) * (n - 2))
    if not directed:
        scale *= 2.0
    return {node: score * scale for node, score in cb.items()}


def _connected_components(adj_undirected: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in adj_undirected:
        if node in seen:
            continue
        comp: list[str] = []
        queue: deque[str] = deque([node])
        seen.add(node)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in adj_undirected.get(cur, set()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        components.append(sorted(comp))
    components.sort(key=lambda c: (-len(c), c[0] if c else ""))
    return components


def _top_items(values: dict[str, float], *, n: int) -> list[tuple[str, float]]:
    return sorted(values.items(), key=lambda item: (-item[1], item[0]))[:n]


def _write_outputs(
    *,
    output_dir: Path,
    output_prefix: str,
    formats: list[str],
    payload: dict[str, Any],
    ctx: ToolContext,
) -> list[str]:
    files_changed: list[str] = []
    if "markdown" in formats:
        path = output_dir / f"{output_prefix}.md"
        _write_text(path, _render_markdown(payload), ctx=ctx)
        files_changed.append(str(path.relative_to(ctx.workspace)))
    if "json" in formats:
        path = output_dir / f"{output_prefix}.json"
        _write_text(path, json.dumps(payload, indent=2), ctx=ctx)
        files_changed.append(str(path.relative_to(ctx.workspace)))
    if "csv" in formats:
        nodes_path = output_dir / f"{output_prefix}-nodes.csv"
        edges_path = output_dir / f"{output_prefix}-edges.csv"
        _write_nodes_csv(nodes_path, payload, ctx=ctx)
        _write_edges_csv(edges_path, payload, ctx=ctx)
        files_changed.extend(
            [
                str(nodes_path.relative_to(ctx.workspace)),
                str(edges_path.relative_to(ctx.workspace)),
            ]
        )
    if "graphml" in formats:
        graphml_path = output_dir / f"{output_prefix}.graphml"
        _write_text(graphml_path, _render_graphml(payload), ctx=ctx)
        files_changed.append(str(graphml_path.relative_to(ctx.workspace)))
    return files_changed


def _write_text(path: Path, content: str, *, ctx: ToolContext) -> None:
    if ctx.changelog is not None:
        ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
    path.write_text(content, encoding="utf-8")


def _write_nodes_csv(path: Path, payload: dict[str, Any], *, ctx: ToolContext) -> None:
    if ctx.changelog is not None:
        ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
    degree = payload.get("metrics", {}).get("degree_centrality", {})
    closeness = payload.get("metrics", {}).get("closeness_centrality", {})
    between = payload.get("metrics", {}).get("betweenness_centrality", {})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "degree_centrality",
                "closeness_centrality",
                "betweenness_centrality",
            ]
        )
        for node in payload.get("nodes", []):
            node_id = str(node.get("id", ""))
            writer.writerow(
                [
                    node_id,
                    degree.get(node_id, 0.0),
                    closeness.get(node_id, 0.0),
                    between.get(node_id, 0.0),
                ]
            )


def _write_edges_csv(path: Path, payload: dict[str, Any], *, ctx: ToolContext) -> None:
    if ctx.changelog is not None:
        ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight", "relation"])
        for edge in payload.get("edges", []):
            writer.writerow(
                [
                    edge.get("source", ""),
                    edge.get("target", ""),
                    edge.get("weight", 1.0),
                    edge.get("relation", ""),
                ]
            )


def _render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    lines = [
        "# Social Network Map",
        "",
        f"- **Directed**: {payload.get('directed')}",
        f"- **Nodes**: {payload.get('node_count')}",
        f"- **Edges**: {payload.get('edge_count')}",
        f"- **Components**: {metrics.get('component_count', 0)}",
        "",
        "## Top Degree Nodes",
        "",
    ]
    for node, score in metrics.get("top_degree_nodes", []):
        lines.append(f"- {node}: {float(score):.4f}")
    lines.extend(["", "## Top Betweenness Nodes", ""])
    for node, score in metrics.get("top_betweenness_nodes", []):
        lines.append(f"- {node}: {float(score):.4f}")
    return "\n".join(lines) + "\n"


def _render_graphml(payload: dict[str, Any]) -> str:
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    directed = bool(payload.get("directed", False))
    edgedefault = "directed" if directed else "undirected"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        f'  <graph id="G" edgedefault="{edgedefault}">',
    ]
    for node in nodes:
        node_id = str(node.get("id", "")).replace('"', "'")
        lines.append(f'    <node id="{node_id}" />')
    for idx, edge in enumerate(edges, start=1):
        src = str(edge.get("source", "")).replace('"', "'")
        dst = str(edge.get("target", "")).replace('"', "'")
        lines.append(f'    <edge id="e{idx}" source="{src}" target="{dst}" />')
    lines.extend(["  </graph>", "</graphml>", ""])
    return "\n".join(lines)
