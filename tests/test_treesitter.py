"""Tests for tree-sitter integration.

Tests both the tree-sitter extraction backend (Phase A) and
structural candidate finding for edit_file (Phase B).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.treesitter import (
    extract_with_treesitter,
    find_structural_candidates,
    is_available,
)

# Skip all tests if tree-sitter is not installed
pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="tree-sitter-language-pack not installed",
)


# ============================================================================
# Phase A: Code structure extraction
# ============================================================================


class TestTreeSitterAvailability:
    def test_is_available(self):
        assert is_available() is True


# --- Python ---


class TestExtractPython:
    def test_imports(self):
        source = "import os\nfrom pathlib import Path\nimport json\n"
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "os" in result.imports
        assert "pathlib" in result.imports

    def test_classes(self):
        source = "class MyClass:\n    pass\n\nclass Another:\n    pass\n"
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "MyClass" in result.classes
        assert "Another" in result.classes

    def test_functions(self):
        source = "def hello():\n    pass\n\ndef world():\n    pass\n"
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "hello" in result.functions
        assert "world" in result.functions

    def test_methods(self):
        source = "class Foo:\n    def bar(self):\n        pass\n"
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "bar" in result.functions

    def test_skips_private_methods(self):
        source = "class Foo:\n    def _private(self):\n        pass\n"
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "_private" not in result.functions

    def test_async_function(self):
        source = "async def fetch():\n    pass\n"
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "fetch" in result.functions

    def test_decorated_function(self):
        source = "@app.route('/hello')\ndef hello():\n    return 'hi'\n"
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "hello" in result.functions

    def test_decorated_class(self):
        source = "@dataclass\nclass Config:\n    name: str\n"
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "Config" in result.classes

    def test_nested_class_methods(self):
        source = (
            "class Outer:\n"
            "    def public_method(self):\n"
            "        pass\n"
            "    def _internal(self):\n"
            "        pass\n"
        )
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "public_method" in result.functions
        assert "_internal" not in result.functions

    def test_from_import(self):
        source = "from os.path import join, dirname\n"
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "os.path" in result.imports

    def test_multiline_function(self):
        source = (
            "def complex_func(\n"
            "    arg1: int,\n"
            "    arg2: str = 'default',\n"
            ") -> bool:\n"
            "    return True\n"
        )
        result = extract_with_treesitter(source, "python")
        assert result is not None
        assert "complex_func" in result.functions


# --- JavaScript ---


class TestExtractJavaScript:
    def test_imports(self):
        source = "import React from 'react';\nimport { useState } from 'react';\n"
        result = extract_with_treesitter(source, "javascript")
        assert result is not None
        assert "react" in result.imports

    def test_require(self):
        source = "const fs = require('fs');\n"
        result = extract_with_treesitter(source, "javascript")
        assert result is not None
        assert "fs" in result.imports

    def test_classes(self):
        source = "class Component extends React.Component {}\n"
        result = extract_with_treesitter(source, "javascript")
        assert result is not None
        assert "Component" in result.classes

    def test_functions(self):
        source = "function hello() {}\nasync function fetchData() {}\n"
        result = extract_with_treesitter(source, "javascript")
        assert result is not None
        assert "hello" in result.functions
        assert "fetchData" in result.functions

    def test_arrow_functions(self):
        source = "const add = (a, b) => a + b;\n"
        result = extract_with_treesitter(source, "javascript")
        assert result is not None
        assert "add" in result.functions

    def test_exports(self):
        source = "export default class App {}\nexport function helper() {}\n"
        result = extract_with_treesitter(source, "javascript")
        assert result is not None
        assert "App" in result.exports
        assert "helper" in result.exports


# --- TypeScript ---


class TestExtractTypeScript:
    def test_typescript_uses_js_extractor(self):
        source = "function greet(): void {}\nclass Service {}\n"
        result = extract_with_treesitter(source, "typescript")
        assert result is not None
        assert "greet" in result.functions
        assert "Service" in result.classes


# --- Go ---


class TestExtractGo:
    def test_structs(self):
        source = 'package main\n\ntype Server struct {\n\tPort int\n}\n'
        result = extract_with_treesitter(source, "go")
        assert result is not None
        assert "Server" in result.classes

    def test_functions(self):
        source = "package main\n\nfunc main() {\n}\n\nfunc (s *Server) Start() {\n}\n"
        result = extract_with_treesitter(source, "go")
        assert result is not None
        assert "main" in result.functions
        assert "Start" in result.functions

    def test_imports(self):
        source = 'package main\n\nimport (\n\t"fmt"\n\t"os"\n)\n'
        result = extract_with_treesitter(source, "go")
        assert result is not None
        assert "fmt" in result.imports
        assert "os" in result.imports


# --- Rust ---


class TestExtractRust:
    def test_structs(self):
        source = "pub struct Config {\n    pub name: String,\n}\n"
        result = extract_with_treesitter(source, "rust")
        assert result is not None
        assert "Config" in result.classes

    def test_enums(self):
        source = "pub enum Status {\n    Active,\n    Inactive,\n}\n"
        result = extract_with_treesitter(source, "rust")
        assert result is not None
        assert "Status" in result.classes

    def test_functions(self):
        source = "pub fn new() -> Self {}\nfn helper() {}\nasync fn process() {}\n"
        result = extract_with_treesitter(source, "rust")
        assert result is not None
        assert "new" in result.functions
        assert "helper" in result.functions
        assert "process" in result.functions

    def test_use_statements(self):
        source = "use std::io;\nuse crate::config;\n"
        result = extract_with_treesitter(source, "rust")
        assert result is not None
        assert "std::io" in result.imports
        assert "crate::config" in result.imports


# --- Fallback ---


class TestFallback:
    def test_unsupported_language_returns_none(self):
        result = extract_with_treesitter("# markdown", "unknown")
        assert result is None

    def test_empty_source(self):
        result = extract_with_treesitter("", "python")
        assert result is not None
        assert result.classes == []
        assert result.functions == []


# ============================================================================
# Phase A: Integration with analyze_file
# ============================================================================


class TestAnalyzeFileWithTreeSitter:
    """Verify analyze_file() uses tree-sitter when available."""

    def test_python_through_analyze_file(self):
        from loom.tools.code_analysis import analyze_file

        source = "@decorator\ndef my_func():\n    pass\n\nclass MyClass:\n    pass\n"
        result = analyze_file("test.py", source)
        assert result.language == "python"
        assert "my_func" in result.functions
        assert "MyClass" in result.classes

    def test_javascript_through_analyze_file(self):
        from loom.tools.code_analysis import analyze_file

        source = "import x from 'mod';\nfunction foo() {}\n"
        result = analyze_file("app.js", source)
        assert result.language == "javascript"
        assert "foo" in result.functions
        assert "mod" in result.imports


# ============================================================================
# Phase B: Structural candidate finding
# ============================================================================


class TestStructuralCandidates:
    def test_python_candidates(self):
        source = (
            "import os\n\n"
            "class Foo:\n"
            "    def bar(self):\n"
            "        pass\n\n"
            "def baz():\n"
            "    return 1\n"
        )
        candidates = find_structural_candidates(source, "python")
        assert len(candidates) >= 2  # class + function at minimum
        # Each candidate is a (start_byte, end_byte) tuple
        for start, end in candidates:
            assert start < end
            segment = source.encode("utf-8")[start:end]
            # Should contain actual code
            assert len(segment) > 0

    def test_javascript_candidates(self):
        source = (
            "function hello() {}\n"
            "class App {}\n"
            "const x = () => null;\n"
        )
        candidates = find_structural_candidates(source, "javascript")
        assert len(candidates) >= 2

    def test_go_candidates(self):
        source = (
            "package main\n\n"
            "type Server struct { Port int }\n\n"
            "func main() {}\n"
        )
        candidates = find_structural_candidates(source, "go")
        assert len(candidates) >= 2

    def test_rust_candidates(self):
        source = (
            "pub struct Config { name: String }\n\n"
            "pub fn new() -> Config { Config { name: String::new() } }\n"
        )
        candidates = find_structural_candidates(source, "rust")
        assert len(candidates) >= 2

    def test_unsupported_language_empty(self):
        assert find_structural_candidates("some text", "unknown") == []


# ============================================================================
# Phase B: Structural fuzzy matching in EditFileTool
# ============================================================================


class TestStructuralFuzzyMatch:
    """Test that EditFileTool uses structural matching for supported languages."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.fixture
    def ctx(self, workspace: Path):
        from loom.tools.registry import ToolContext
        return ToolContext(workspace=workspace)

    async def test_structural_match_indentation_drift(self, ctx, workspace: Path):
        """Tree-sitter structural matching should handle indentation drift."""
        from loom.tools.file_ops import EditFileTool

        # File with tabs
        (workspace / "code.py").write_text(
            "class Handler:\n"
            "\tdef process(self, data):\n"
            "\t\treturn data.strip()\n"
            "\n"
            "\tdef validate(self, data):\n"
            "\t\treturn bool(data)\n"
        )
        tool = EditFileTool()
        # Model uses spaces instead of tabs
        result = await tool.execute(
            {
                "path": "code.py",
                "old_str": "    def process(self, data):\n        return data.strip()",
                "new_str": "\tdef process(self, data):\n\t\treturn data.upper()",
            },
            ctx,
        )
        assert result.success
        assert "fuzzy" in result.output.lower()
        content = (workspace / "code.py").read_text()
        assert "upper()" in content

    async def test_structural_match_decorator_reorder(self, ctx, workspace: Path):
        """Test matching a function even when decorators differ slightly."""
        from loom.tools.file_ops import EditFileTool

        (workspace / "app.py").write_text(
            "from flask import Flask\n\n"
            "app = Flask(__name__)\n\n"
            "@app.route('/hello')\n"
            "def hello():  \n"
            "    return 'Hello, World!'\n\n"
            "@app.route('/bye')\n"
            "def bye():\n"
            "    return 'Goodbye'\n"
        )
        tool = EditFileTool()
        # Model omits trailing whitespace on hello():
        result = await tool.execute(
            {
                "path": "app.py",
                "old_str": "def hello():\n    return 'Hello, World!'",
                "new_str": "def hello():\n    return 'Hello, Flask!'",
            },
            ctx,
        )
        assert result.success
        content = (workspace / "app.py").read_text()
        assert "Hello, Flask!" in content

    async def test_non_source_file_still_uses_sliding_window(self, ctx, workspace: Path):
        """Non-source files should fall back to the sliding-window approach."""
        from loom.tools.file_ops import EditFileTool

        (workspace / "config.txt").write_text("key = old_value  \nother = thing\n")
        tool = EditFileTool()
        result = await tool.execute(
            {
                "path": "config.txt",
                "old_str": "key = old_value\nother = thing",
                "new_str": "key = new_value\nother = thing",
            },
            ctx,
        )
        assert result.success
        assert "new_value" in (workspace / "config.txt").read_text()
