"""Code analysis tool with tree-sitter and fallback extractors.

Parses source and markup files and returns structural information:
imports, elements, classes, functions, exports, and ids. Uses
tree-sitter when available (via ``tree-sitter-language-pack``),
falling back to built-in extractors for Python, JavaScript/
TypeScript, Go, Rust, and HTML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from loom.tools.registry import Tool, ToolContext, ToolResult
from loom.utils.concurrency import run_blocking_io

# --- Extractors per language ---

@dataclass
class CodeStructure:
    """Parsed code structure of a single file."""

    file_path: str = ""
    language: str = ""
    imports: list[str] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)

    def to_summary(self) -> str:
        lines = [f"File: {self.file_path} ({self.language})"]
        if self.imports:
            lines.append(f"  Imports: {', '.join(self.imports)}")
        if self.elements:
            lines.append(f"  Elements: {', '.join(self.elements)}")
        if self.ids:
            lines.append(f"  IDs: {', '.join(self.ids)}")
        if self.classes:
            lines.append(f"  Classes: {', '.join(self.classes)}")
        if self.functions:
            lines.append(f"  Functions: {', '.join(self.functions)}")
        if self.exports:
            lines.append(f"  Exports: {', '.join(self.exports)}")
        return "\n".join(lines)


def _append_unique(items: list[str], value: str) -> None:
    value = value.strip()
    if value and value not in items:
        items.append(value)


_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".html": "html",
    ".htm": "html",
}


def detect_language(path: str) -> str:
    """Detect language from file extension."""
    suffix = Path(path).suffix.lower()
    return _LANG_MAP.get(suffix, "unknown")


def extract_python(source: str) -> CodeStructure:
    """Extract structure from Python source."""
    structure = CodeStructure(language="python")

    # Imports
    for m in re.finditer(r"^(?:from\s+(\S+)\s+)?import\s+(.+)", source, re.MULTILINE):
        module = m.group(1) or m.group(2).split(",")[0].split(" as ")[0].strip()
        structure.imports.append(module)

    # Classes
    for m in re.finditer(r"^class\s+(\w+)", source, re.MULTILINE):
        structure.classes.append(m.group(1))

    # Top-level functions (not indented)
    for m in re.finditer(r"^def\s+(\w+)", source, re.MULTILINE):
        structure.functions.append(m.group(1))

    # Methods (indented defs)
    for m in re.finditer(r"^\s+def\s+(\w+)", source, re.MULTILINE):
        name = m.group(1)
        if name not in structure.functions and not name.startswith("_"):
            structure.functions.append(name)

    return structure


def extract_javascript(source: str) -> CodeStructure:
    """Extract structure from JavaScript/TypeScript source."""
    structure = CodeStructure(language="javascript")

    # Imports
    for m in re.finditer(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", source):
        structure.imports.append(m.group(1))
    for m in re.finditer(r"(?:const|let|var)\s+\w+\s*=\s*require\(['\"]([^'\"]+)['\"]\)", source):
        structure.imports.append(m.group(1))

    # Classes
    for m in re.finditer(r"class\s+(\w+)", source):
        structure.classes.append(m.group(1))

    # Functions (function declarations, arrow functions, methods)
    for m in re.finditer(r"(?:function|async\s+function)\s+(\w+)", source):
        structure.functions.append(m.group(1))
    for m in re.finditer(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", source):
        structure.functions.append(m.group(1))

    # Exports
    export_re = r"export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)"
    for m in re.finditer(export_re, source):
        structure.exports.append(m.group(1))

    return structure


def extract_go(source: str) -> CodeStructure:
    """Extract structure from Go source."""
    structure = CodeStructure(language="go")

    # Imports
    for m in re.finditer(r'"([^"]+)"', source):
        structure.imports.append(m.group(1))

    # Structs (Go's "classes")
    for m in re.finditer(r"type\s+(\w+)\s+struct", source):
        structure.classes.append(m.group(1))

    # Functions
    for m in re.finditer(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", source):
        structure.functions.append(m.group(1))

    return structure


def extract_rust(source: str) -> CodeStructure:
    """Extract structure from Rust source."""
    structure = CodeStructure(language="rust")

    # Use statements
    for m in re.finditer(r"use\s+([\w:]+)", source):
        structure.imports.append(m.group(1))

    # Structs and enums
    for m in re.finditer(r"(?:pub\s+)?struct\s+(\w+)", source):
        structure.classes.append(m.group(1))
    for m in re.finditer(r"(?:pub\s+)?enum\s+(\w+)", source):
        structure.classes.append(m.group(1))

    # Functions
    for m in re.finditer(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", source):
        structure.functions.append(m.group(1))

    return structure


_HTML_SKIPPED_TAGS = {"html", "head", "body"}


class _HTMLStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.structure = CodeStructure(language="html")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._record_tag(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._record_tag(tag, attrs)

    def _record_tag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.lower().strip()
        if normalized_tag and normalized_tag not in _HTML_SKIPPED_TAGS:
            _append_unique(self.structure.elements, normalized_tag)

        attr_map = {
            name.lower().strip(): (value or "").strip()
            for name, value in attrs
            if name
        }

        for key in ("src", "href"):
            value = attr_map.get(key, "")
            if value:
                _append_unique(self.structure.imports, value)

        id_value = attr_map.get("id", "")
        if id_value:
            _append_unique(self.structure.ids, id_value)

        class_value = attr_map.get("class", "")
        for class_name in class_value.split():
            _append_unique(self.structure.classes, class_name)


def extract_html(source: str) -> CodeStructure:
    """Extract structure from HTML source."""
    parser = _HTMLStructureParser()
    parser.feed(source)
    parser.close()
    return parser.structure


_EXTRACTORS = {
    "python": extract_python,
    "javascript": extract_javascript,
    "typescript": extract_javascript,  # same patterns work
    "go": extract_go,
    "rust": extract_rust,
    "html": extract_html,
}


def analyze_file(file_path: str, source: str) -> CodeStructure:
    """Analyze a source file and return its structure.

    Tries tree-sitter first (when installed), falls back to regex extractors.
    """
    lang = detect_language(file_path)

    # Try tree-sitter backend first
    from loom.tools.treesitter import extract_with_treesitter, is_available

    if is_available():
        ts_result = extract_with_treesitter(source, lang)
        if ts_result is not None:
            ts_result.file_path = file_path
            ts_result.language = lang
            return ts_result

    # Fallback to regex extractors
    extractor = _EXTRACTORS.get(lang)
    if extractor is None:
        return CodeStructure(file_path=file_path, language=lang or "unknown")

    structure = extractor(source)
    structure.file_path = file_path
    structure.language = lang
    return structure


def analyze_directory(
    workspace: Path,
    focus_dirs: list[str] | None = None,
    max_files: int = 30,
) -> list[CodeStructure]:
    """Analyze key source files in a workspace directory.

    If focus_dirs provided, only scans those directories.
    Otherwise, scans the workspace root for source files.
    """
    results: list[CodeStructure] = []
    extensions = set(_LANG_MAP.keys())

    dirs_to_scan: list[Path] = []
    if focus_dirs:
        for d in focus_dirs:
            p = workspace / d
            if p.is_dir():
                dirs_to_scan.append(p)
    else:
        dirs_to_scan.append(workspace)

    file_count = 0
    for scan_dir in dirs_to_scan:
        for path in sorted(scan_dir.rglob("*")):
            if file_count >= max_files:
                break
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            # Skip hidden dirs, node_modules, etc.
            parts = path.relative_to(workspace).parts
            skip = ("node_modules", "__pycache__", "venv", ".venv")
            if any(p.startswith(".") or p in skip for p in parts):
                continue

            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                rel = str(path.relative_to(workspace))
                structure = analyze_file(rel, source)
                results.append(structure)
                file_count += 1
            except OSError:
                continue

    return results


class AnalyzeCodeTool(Tool):
    @property
    def name(self) -> str:
        return "analyze_code"

    @property
    def description(self) -> str:
        return (
            "Analyze source and markup structure (imports, elements, classes, "
            "functions). Supports Python, JS/TS, Go, Rust, HTML."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace",
                },
            },
            "required": ["path"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 15

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return await run_blocking_io(self._execute_sync, args, ctx)

    def _execute_sync(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        path = self._resolve_path(args["path"], ctx.workspace)
        if not path.exists():
            return ToolResult.fail(f"File not found: {args['path']}")
        if not path.is_file():
            return ToolResult.fail(f"Not a file: {args['path']}")

        lang = detect_language(args["path"])
        if lang == "unknown":
            return ToolResult.fail(
                f"Unsupported language for {args['path']}. "
                "Supported: .py, .js, .ts, .tsx, .jsx, .go, .rs, .html, .htm"
            )

        source = path.read_text(encoding="utf-8", errors="replace")
        structure = analyze_file(args["path"], source)
        return ToolResult.ok(structure.to_summary())
