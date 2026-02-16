"""Tests for code analysis tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.code_analysis import (
    AnalyzeCodeTool,
    CodeStructure,
    analyze_directory,
    analyze_file,
    detect_language,
    extract_go,
    extract_javascript,
    extract_python,
    extract_rust,
)
from loom.tools.registry import ToolContext

# --- Language Detection ---


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("app.py") == "python"

    def test_javascript(self):
        assert detect_language("index.js") == "javascript"

    def test_typescript(self):
        assert detect_language("component.tsx") == "typescript"

    def test_go(self):
        assert detect_language("main.go") == "go"

    def test_rust(self):
        assert detect_language("lib.rs") == "rust"

    def test_unknown(self):
        assert detect_language("readme.md") == "unknown"

    def test_jsx(self):
        assert detect_language("App.jsx") == "javascript"


# --- Python Extractor ---


class TestExtractPython:
    def test_extracts_imports(self):
        source = "import os\nfrom pathlib import Path\nimport json\n"
        result = extract_python(source)
        assert "os" in result.imports
        assert "pathlib" in result.imports

    def test_extracts_classes(self):
        source = "class MyClass:\n    pass\n\nclass Another:\n    pass\n"
        result = extract_python(source)
        assert "MyClass" in result.classes
        assert "Another" in result.classes

    def test_extracts_functions(self):
        source = "def hello():\n    pass\n\ndef world():\n    pass\n"
        result = extract_python(source)
        assert "hello" in result.functions
        assert "world" in result.functions

    def test_extracts_methods(self):
        source = "class Foo:\n    def bar(self):\n        pass\n"
        result = extract_python(source)
        assert "bar" in result.functions

    def test_skips_private_methods(self):
        source = "class Foo:\n    def _private(self):\n        pass\n"
        result = extract_python(source)
        assert "_private" not in result.functions

    def test_language_set(self):
        result = extract_python("")
        assert result.language == "python"


# --- JavaScript Extractor ---


class TestExtractJavaScript:
    def test_extracts_imports(self):
        source = "import React from 'react';\nimport { useState } from 'react';\n"
        result = extract_javascript(source)
        assert "react" in result.imports

    def test_extracts_require(self):
        source = "const fs = require('fs');\n"
        result = extract_javascript(source)
        assert "fs" in result.imports

    def test_extracts_classes(self):
        source = "class Component extends React.Component {}\n"
        result = extract_javascript(source)
        assert "Component" in result.classes

    def test_extracts_functions(self):
        source = "function hello() {}\nasync function fetchData() {}\n"
        result = extract_javascript(source)
        assert "hello" in result.functions
        assert "fetchData" in result.functions

    def test_extracts_arrow_functions(self):
        source = "const add = (a, b) => a + b;\n"
        result = extract_javascript(source)
        assert "add" in result.functions

    def test_extracts_exports(self):
        source = "export default class App {}\nexport function helper() {}\n"
        result = extract_javascript(source)
        assert "App" in result.exports
        assert "helper" in result.exports


# --- Go Extractor ---


class TestExtractGo:
    def test_extracts_structs(self):
        source = 'package main\n\ntype Server struct {\n\tPort int\n}\n'
        result = extract_go(source)
        assert "Server" in result.classes

    def test_extracts_functions(self):
        source = "func main() {\n}\n\nfunc (s *Server) Start() {\n}\n"
        result = extract_go(source)
        assert "main" in result.functions
        assert "Start" in result.functions


# --- Rust Extractor ---


class TestExtractRust:
    def test_extracts_structs(self):
        source = "pub struct Config {\n    pub name: String,\n}\n"
        result = extract_rust(source)
        assert "Config" in result.classes

    def test_extracts_enums(self):
        source = "pub enum Status {\n    Active,\n    Inactive,\n}\n"
        result = extract_rust(source)
        assert "Status" in result.classes

    def test_extracts_functions(self):
        source = "pub fn new() -> Self {}\nfn helper() {}\nasync fn process() {}\n"
        result = extract_rust(source)
        assert "new" in result.functions
        assert "helper" in result.functions
        assert "process" in result.functions

    def test_extracts_use(self):
        source = "use std::io;\nuse crate::config;\n"
        result = extract_rust(source)
        assert "std::io" in result.imports
        assert "crate::config" in result.imports


# --- CodeStructure ---


class TestCodeStructure:
    def test_to_summary(self):
        cs = CodeStructure(
            file_path="src/main.py",
            language="python",
            imports=["os", "sys"],
            classes=["App"],
            functions=["main", "run"],
        )
        summary = cs.to_summary()
        assert "src/main.py" in summary
        assert "python" in summary
        assert "os" in summary
        assert "App" in summary
        assert "main" in summary

    def test_empty_summary(self):
        cs = CodeStructure(file_path="empty.py", language="python")
        summary = cs.to_summary()
        assert "empty.py" in summary
        # Should not contain section headers for empty lists
        assert "Imports" not in summary


# --- analyze_file ---


class TestAnalyzeFile:
    def test_python_file(self):
        source = "import os\n\nclass Foo:\n    pass\n\ndef bar():\n    pass\n"
        result = analyze_file("app.py", source)
        assert result.language == "python"
        assert "Foo" in result.classes
        assert "bar" in result.functions

    def test_unknown_language(self):
        result = analyze_file("readme.md", "# Hello")
        assert result.language == "unknown"
        assert result.classes == []
        assert result.functions == []

    def test_file_path_preserved(self):
        result = analyze_file("src/utils.py", "x = 1")
        assert result.file_path == "src/utils.py"


# --- analyze_directory ---


class TestAnalyzeDirectory:
    def test_scans_python_files(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("def hello():\n    pass\n")
        (tmp_path / "utils.py").write_text("class Helper:\n    pass\n")
        results = analyze_directory(tmp_path)
        assert len(results) == 2
        funcs = [f for r in results for f in r.functions]
        classes = [c for r in results for c in r.classes]
        assert "hello" in funcs
        assert "Helper" in classes

    def test_respects_max_files(self, tmp_path: Path):
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"def f{i}(): pass\n")
        results = analyze_directory(tmp_path, max_files=3)
        assert len(results) == 3

    def test_skips_hidden_dirs(self, tmp_path: Path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("def secret(): pass\n")
        (tmp_path / "visible.py").write_text("def visible(): pass\n")
        results = analyze_directory(tmp_path)
        assert len(results) == 1
        assert results[0].file_path == "visible.py"

    def test_skips_node_modules(self, tmp_path: Path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "dep.js").write_text("function dep() {}")
        (tmp_path / "app.js").write_text("function app() {}")
        results = analyze_directory(tmp_path)
        assert len(results) == 1

    def test_focus_dirs(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("def main(): pass\n")
        (tmp_path / "setup.py").write_text("def setup(): pass\n")
        results = analyze_directory(tmp_path, focus_dirs=["src"])
        assert len(results) == 1
        assert "src/app.py" in results[0].file_path

    def test_skips_non_source_files(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text("# Hello")
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "app.py").write_text("x = 1")
        results = analyze_directory(tmp_path)
        assert len(results) == 1


# --- AnalyzeCodeTool ---


class TestAnalyzeCodeTool:
    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        (tmp_path / "main.py").write_text("class App:\n    def run(self):\n        pass\n")
        return tmp_path

    @pytest.fixture
    def ctx(self, workspace: Path) -> ToolContext:
        return ToolContext(workspace=workspace)

    async def test_analyze_python_file(self, ctx: ToolContext):
        tool = AnalyzeCodeTool()
        result = await tool.execute({"path": "main.py"}, ctx)
        assert result.success
        assert "App" in result.output
        assert "run" in result.output

    async def test_analyze_missing_file(self, ctx: ToolContext):
        tool = AnalyzeCodeTool()
        result = await tool.execute({"path": "nope.py"}, ctx)
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_analyze_unsupported_language(self, ctx: ToolContext, workspace: Path):
        (workspace / "data.csv").write_text("a,b,c")
        tool = AnalyzeCodeTool()
        result = await tool.execute({"path": "data.csv"}, ctx)
        assert not result.success
        assert "Unsupported" in result.error

    async def test_no_workspace(self):
        tool = AnalyzeCodeTool()
        result = await tool.execute({"path": "x.py"}, ToolContext(workspace=None))
        assert not result.success
        assert "No workspace" in result.error
