"""Code analysis tool with tree-sitter and regex backends.

Parses source files and returns structural information:
classes, functions, imports. Uses tree-sitter when available
(via ``tree-sitter-language-pack``), falling back to regex
extractors for Python, JavaScript/TypeScript, Go, and Rust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)

    def to_summary(self) -> str:
        lines = [f"File: {self.file_path} ({self.language})"]
        if self.imports:
            lines.append(f"  Imports: {', '.join(self.imports)}")
        if self.classes:
            lines.append(f"  Classes: {', '.join(self.classes)}")
        if self.functions:
            lines.append(f"  Functions: {', '.join(self.functions)}")
        if self.exports:
            lines.append(f"  Exports: {', '.join(self.exports)}")
        return "\n".join(lines)


_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
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


_EXTRACTORS = {
    "python": extract_python,
    "javascript": extract_javascript,
    "typescript": extract_javascript,  # same patterns work
    "go": extract_go,
    "rust": extract_rust,
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
            "Analyze source file structure (classes, functions, imports). "
            "Supports Python, JS/TS, Go, Rust."
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
                "Supported: .py, .js, .ts, .tsx, .jsx, .go, .rs"
            )

        source = path.read_text(encoding="utf-8", errors="replace")
        structure = analyze_file(args["path"], source)
        return ToolResult.ok(structure.to_summary())
