"""Document write tool — structured Markdown document generation.

Creates well-structured Markdown documents with sections, tables,
lists, and citations. Designed for strategy documents, reports,
memos, and analysis deliverables.
"""

from __future__ import annotations

from loom.tools.registry import Tool, ToolContext, ToolResult


class DocumentWriteTool(Tool):
    """Create structured Markdown documents."""

    MAX_CONTENT_SIZE = 1024 * 1024  # 1 MB

    @property
    def name(self) -> str:
        return "document_write"

    @property
    def is_mutating(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Create a structured Markdown document. Provide the file path "
            "and either 'content' (raw Markdown string) or 'sections' "
            "(structured list of {heading, level, body} objects). "
            "Sections are assembled into a well-formatted document with "
            "an optional title and metadata header."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Output file path relative to workspace "
                        "(e.g., 'reports/analysis.md')."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Document title (rendered as H1).",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Raw Markdown content. Use this OR 'sections', "
                        "not both."
                    ),
                },
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {
                                "type": "string",
                                "description": "Section heading text.",
                            },
                            "level": {
                                "type": "integer",
                                "description": (
                                    "Heading level (2-4). Default 2."
                                ),
                            },
                            "body": {
                                "type": "string",
                                "description": "Section body in Markdown.",
                            },
                        },
                        "required": ["heading", "body"],
                    },
                    "description": "Structured sections for the document.",
                },
                "metadata": {
                    "type": "object",
                    "description": (
                        "Optional metadata (author, date, etc.) "
                        "rendered as a YAML frontmatter block."
                    ),
                },
                "append": {
                    "type": "boolean",
                    "description": (
                        "If true, append to existing file instead of "
                        "overwriting. Default false."
                    ),
                },
            },
            "required": ["path"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        raw_path = args.get("path", "")
        if not raw_path:
            return ToolResult.fail("No path provided")

        try:
            filepath = self._resolve_path(raw_path, ctx.workspace)
        except Exception as e:
            return ToolResult.fail(str(e))

        title = args.get("title", "")
        content = args.get("content", "")
        sections = args.get("sections", [])
        metadata = args.get("metadata", {})
        append = args.get("append", False)

        if not content and not sections and not title:
            return ToolResult.fail(
                "Provide at least one of: title, content, or sections",
            )

        # Build document
        parts: list[str] = []

        # Metadata frontmatter (use yaml.dump for safe serialization)
        if metadata and isinstance(metadata, dict) and not append:
            import yaml as _yaml
            parts.append("---")
            # yaml.dump handles quoting, escaping, and multiline values
            rendered = _yaml.dump(
                metadata, default_flow_style=False, allow_unicode=True,
            ).rstrip("\n")
            parts.append(rendered)
            parts.append("---")
            parts.append("")

        # Title
        if title:
            parts.append(f"# {title}")
            parts.append("")

        # Content (raw) or sections (structured)
        if content:
            parts.append(content)
        elif sections:
            for i, section in enumerate(sections):
                if not isinstance(section, dict):
                    return ToolResult.fail(
                        f"Section {i} must be a dict with 'heading' and 'body'",
                    )
                if "heading" not in section or "body" not in section:
                    return ToolResult.fail(
                        f"Section {i} missing required 'heading' or 'body' key",
                    )
                heading = section["heading"]
                level = min(max(section.get("level", 2), 2), 4)
                body = section["body"]
                prefix = "#" * level
                parts.append(f"{prefix} {heading}")
                parts.append("")
                if body:
                    parts.append(body)
                    parts.append("")

        document = "\n".join(parts)
        if not document.endswith("\n"):
            document += "\n"

        if len(document) > self.MAX_CONTENT_SIZE:
            return ToolResult.fail("Document too large (max 1 MB)")

        # Write
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(
                str(filepath), subtask_id=ctx.subtask_id,
            )

        mode = "a" if append else "w"
        with open(filepath, mode, encoding="utf-8") as f:
            f.write(document)

        rel = filepath.relative_to(ctx.workspace)
        action = "Appended to" if append else "Created"
        word_count = len(document.split())
        section_count = len(sections) if sections else 0

        summary = f"{action} {rel}"
        if section_count:
            summary += f" ({section_count} sections, ~{word_count} words)"
        else:
            summary += f" (~{word_count} words)"

        return ToolResult.ok(summary, files_changed=[str(rel)])
