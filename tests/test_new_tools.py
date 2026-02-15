"""Tests for calculator, spreadsheet, and document_write tools.

Covers:
- CalculatorTool: arithmetic, math functions, financial functions, safety, errors
- SpreadsheetTool: create, read, add_rows, add_column, update_cell, summary, errors
- DocumentWriteTool: title, content, sections, metadata, append, errors
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
import yaml

from loom.tools.calculator import CalculatorTool, _safe_eval, cagr, npv, pmt, wacc
from loom.tools.document_write import DocumentWriteTool
from loom.tools.registry import ToolContext
from loom.tools.spreadsheet import SpreadsheetTool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def ctx(workspace):
    return ToolContext(workspace=workspace)


@pytest.fixture
def no_workspace_ctx():
    return ToolContext(workspace=None)


@pytest.fixture
def calc():
    return CalculatorTool()


@pytest.fixture
def sheet():
    return SpreadsheetTool()


@pytest.fixture
def doc():
    return DocumentWriteTool()


def _read_csv(path: Path) -> list[list[str]]:
    """Helper to read CSV content as list of rows."""
    content = path.read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(content))
    return list(reader)


# ===================================================================
# CalculatorTool Tests
# ===================================================================


class TestCalculatorBasicArithmetic:
    """Test basic arithmetic operations."""

    async def test_addition(self, calc, ctx):
        result = await calc.execute({"expression": "2+3"}, ctx)
        assert result.success
        assert "= 5" in result.output

    async def test_multiplication(self, calc, ctx):
        result = await calc.execute({"expression": "10*5"}, ctx)
        assert result.success
        assert "= 50" in result.output

    async def test_division(self, calc, ctx):
        result = await calc.execute({"expression": "100/3"}, ctx)
        assert result.success
        assert result.data["result"] == pytest.approx(100 / 3)

    async def test_exponentiation(self, calc, ctx):
        result = await calc.execute({"expression": "2**10"}, ctx)
        assert result.success
        assert "= 1024" in result.output

    async def test_floor_division(self, calc, ctx):
        result = await calc.execute({"expression": "17//5"}, ctx)
        assert result.success
        assert "= 3" in result.output

    async def test_modulo(self, calc, ctx):
        result = await calc.execute({"expression": "17%5"}, ctx)
        assert result.success
        assert "= 2" in result.output

    async def test_subtraction(self, calc, ctx):
        result = await calc.execute({"expression": "100-37"}, ctx)
        assert result.success
        assert "= 63" in result.output

    async def test_negative_numbers(self, calc, ctx):
        result = await calc.execute({"expression": "-5 + 3"}, ctx)
        assert result.success
        assert "= -2" in result.output

    async def test_complex_expression(self, calc, ctx):
        result = await calc.execute({"expression": "(2+3)*4-1"}, ctx)
        assert result.success
        assert "= 19" in result.output


class TestCalculatorMathFunctions:
    """Test math function support."""

    async def test_sqrt(self, calc, ctx):
        result = await calc.execute({"expression": "sqrt(144)"}, ctx)
        assert result.success
        assert "= 12" in result.output

    async def test_log(self, calc, ctx):
        result = await calc.execute({"expression": "log(100)"}, ctx)
        assert result.success
        assert result.data["result"] == pytest.approx(4.605170185988092)

    async def test_round(self, calc, ctx):
        result = await calc.execute({"expression": "round(3.14159, 2)"}, ctx)
        assert result.success
        assert "3.14" in result.output

    async def test_abs(self, calc, ctx):
        result = await calc.execute({"expression": "abs(-42)"}, ctx)
        assert result.success
        assert "= 42" in result.output

    async def test_min_max(self, calc, ctx):
        result = await calc.execute({"expression": "max(10, 20, 30)"}, ctx)
        assert result.success
        assert "= 30" in result.output

    async def test_sum(self, calc, ctx):
        result = await calc.execute({"expression": "sum([1, 2, 3, 4, 5])"}, ctx)
        assert result.success
        assert "= 15" in result.output

    async def test_ceil_floor(self, calc, ctx):
        r1 = await calc.execute({"expression": "ceil(3.2)"}, ctx)
        r2 = await calc.execute({"expression": "floor(3.8)"}, ctx)
        assert r1.success
        assert r2.success
        assert "= 4" in r1.output
        assert "= 3" in r2.output

    async def test_pi_constant(self, calc, ctx):
        result = await calc.execute({"expression": "pi"}, ctx)
        assert result.success
        assert result.data["result"] == pytest.approx(3.14159265358979)

    async def test_e_constant(self, calc, ctx):
        result = await calc.execute({"expression": "e"}, ctx)
        assert result.success
        assert result.data["result"] == pytest.approx(2.71828182845904)

    async def test_exp(self, calc, ctx):
        result = await calc.execute({"expression": "exp(1)"}, ctx)
        assert result.success
        assert result.data["result"] == pytest.approx(2.71828182845904)

    async def test_log10(self, calc, ctx):
        result = await calc.execute({"expression": "log10(1000)"}, ctx)
        assert result.success
        assert "= 3" in result.output

    async def test_log2(self, calc, ctx):
        result = await calc.execute({"expression": "log2(256)"}, ctx)
        assert result.success
        assert "= 8" in result.output


class TestCalculatorFinancialFunctions:
    """Test financial formula support."""

    async def test_npv(self, calc, ctx):
        result = await calc.execute(
            {"expression": "npv(0.1, [-1000, 300, 400, 500])"},
            ctx,
        )
        assert result.success
        expected = npv(0.1, [-1000, 300, 400, 500])
        assert result.data["result"] == pytest.approx(expected)

    async def test_cagr(self, calc, ctx):
        result = await calc.execute(
            {"expression": "cagr(100, 200, 5)"},
            ctx,
        )
        assert result.success
        expected = cagr(100, 200, 5)
        assert result.data["result"] == pytest.approx(expected)

    async def test_wacc(self, calc, ctx):
        result = await calc.execute(
            {"expression": "wacc(1000000, 500000, 0.12, 0.06, 0.21)"},
            ctx,
        )
        assert result.success
        expected = wacc(1000000, 500000, 0.12, 0.06, 0.21)
        assert result.data["result"] == pytest.approx(expected)

    async def test_pmt(self, calc, ctx):
        result = await calc.execute(
            {"expression": "pmt(0.05/12, 360, 200000)"},
            ctx,
        )
        assert result.success
        expected = pmt(0.05 / 12, 360, 200000)
        assert result.data["result"] == pytest.approx(expected)

    async def test_npv_positive_cashflows(self, calc, ctx):
        result = await calc.execute(
            {"expression": "npv(0.05, [0, 100, 100, 100])"},
            ctx,
        )
        assert result.success
        assert result.data["result"] > 0

    async def test_cagr_doubling(self, calc, ctx):
        """Doubling in ~7 years at 10% CAGR (rule of 72)."""
        result = await calc.execute(
            {"expression": "cagr(100, 200, 7.2)"},
            ctx,
        )
        assert result.success
        assert result.data["result"] == pytest.approx(0.1006, abs=0.01)


class TestCalculatorErrorHandling:
    """Test error cases for the calculator."""

    async def test_division_by_zero(self, calc, ctx):
        result = await calc.execute({"expression": "1/0"}, ctx)
        assert not result.success
        assert "error" in result.error.lower()

    async def test_unknown_variable(self, calc, ctx):
        result = await calc.execute({"expression": "xyz + 1"}, ctx)
        assert not result.success
        assert "Unknown variable" in result.error

    async def test_too_long_expression(self, calc, ctx):
        result = await calc.execute({"expression": "1+" * 600}, ctx)
        assert not result.success
        assert "too long" in result.error.lower()

    async def test_empty_expression(self, calc, ctx):
        result = await calc.execute({"expression": ""}, ctx)
        assert not result.success
        assert "No expression" in result.error

    async def test_no_expression_key(self, calc, ctx):
        result = await calc.execute({}, ctx)
        assert not result.success

    async def test_syntax_error(self, calc, ctx):
        result = await calc.execute({"expression": "2 +"}, ctx)
        assert not result.success
        assert "error" in result.error.lower()

    async def test_string_literal_rejected(self, calc, ctx):
        result = await calc.execute({"expression": "'hello'"}, ctx)
        assert not result.success


class TestCalculatorSafety:
    """Test that the calculator is sandboxed."""

    async def test_no_builtins_access(self, calc, ctx):
        result = await calc.execute({"expression": "__import__('os')"}, ctx)
        assert not result.success

    async def test_no_import(self, calc, ctx):
        result = await calc.execute({"expression": "import os"}, ctx)
        assert not result.success

    async def test_no_exec(self, calc, ctx):
        result = await calc.execute({"expression": "exec('print(1)')"}, ctx)
        assert not result.success

    async def test_no_eval(self, calc, ctx):
        result = await calc.execute({"expression": "eval('2+2')"}, ctx)
        assert not result.success

    async def test_no_open(self, calc, ctx):
        result = await calc.execute({"expression": "open('/etc/passwd')"}, ctx)
        assert not result.success

    def test_safe_eval_rejects_attribute_access(self):
        with pytest.raises(ValueError):
            _safe_eval("().__class__.__bases__")

    def test_safe_eval_rejects_lambda(self):
        with pytest.raises(ValueError):
            _safe_eval("(lambda: 1)()")


class TestCalculatorMetadata:
    """Test tool metadata properties."""

    def test_name(self, calc):
        assert calc.name == "calculator"

    def test_description(self, calc):
        assert "mathematical" in calc.description.lower()

    def test_parameters_schema(self, calc):
        params = calc.parameters
        assert params["type"] == "object"
        assert "expression" in params["properties"]
        assert "expression" in params["required"]


# ===================================================================
# SpreadsheetTool Tests
# ===================================================================


class TestSpreadsheetCreate:
    """Test spreadsheet creation."""

    async def test_create_with_headers_and_rows(self, sheet, ctx, workspace):
        result = await sheet.execute(
            {
                "operation": "create",
                "path": "data.csv",
                "headers": ["Name", "Age", "City"],
                "rows": [
                    ["Alice", "30", "NYC"],
                    ["Bob", "25", "LA"],
                ],
            },
            ctx,
        )
        assert result.success
        assert "3 columns" in result.output
        assert "2 rows" in result.output

        rows = _read_csv(workspace / "data.csv")
        assert rows[0] == ["Name", "Age", "City"]
        assert rows[1] == ["Alice", "30", "NYC"]
        assert rows[2] == ["Bob", "25", "LA"]

    async def test_create_headers_only(self, sheet, ctx, workspace):
        result = await sheet.execute(
            {
                "operation": "create",
                "path": "empty.csv",
                "headers": ["A", "B", "C"],
            },
            ctx,
        )
        assert result.success
        rows = _read_csv(workspace / "empty.csv")
        assert len(rows) == 1
        assert rows[0] == ["A", "B", "C"]

    async def test_create_missing_headers(self, sheet, ctx):
        result = await sheet.execute(
            {"operation": "create", "path": "bad.csv"},
            ctx,
        )
        assert not result.success
        assert "headers" in result.error.lower()

    async def test_create_in_subdirectory(self, sheet, ctx, workspace):
        result = await sheet.execute(
            {
                "operation": "create",
                "path": "reports/data.csv",
                "headers": ["X"],
            },
            ctx,
        )
        assert result.success
        assert (workspace / "reports" / "data.csv").exists()

    async def test_create_records_files_changed(self, sheet, ctx):
        result = await sheet.execute(
            {
                "operation": "create",
                "path": "data.csv",
                "headers": ["X"],
            },
            ctx,
        )
        assert result.success
        assert "data.csv" in result.files_changed


class TestSpreadsheetRead:
    """Test spreadsheet reading."""

    async def test_read_csv(self, sheet, ctx, workspace):
        # Create a CSV first
        csv_path = workspace / "test.csv"
        csv_path.write_text("Name,Score\nAlice,95\nBob,87\n")

        result = await sheet.execute(
            {"operation": "read", "path": "test.csv"},
            ctx,
        )
        assert result.success
        assert "Alice" in result.output
        assert "Bob" in result.output
        assert "|" in result.output  # table format

    async def test_read_formatted_output(self, sheet, ctx, workspace):
        csv_path = workspace / "test.csv"
        csv_path.write_text("A,B\n1,2\n")
        result = await sheet.execute(
            {"operation": "read", "path": "test.csv"},
            ctx,
        )
        assert result.success
        lines = result.output.strip().split("\n")
        # Header, separator, data
        assert len(lines) >= 3
        assert "---" in lines[1]  # separator line

    async def test_read_file_not_found(self, sheet, ctx):
        result = await sheet.execute(
            {"operation": "read", "path": "nonexistent.csv"},
            ctx,
        )
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_read_empty_csv(self, sheet, ctx, workspace):
        csv_path = workspace / "empty.csv"
        csv_path.write_text("")
        result = await sheet.execute(
            {"operation": "read", "path": "empty.csv"},
            ctx,
        )
        assert result.success
        assert "Empty" in result.output


class TestSpreadsheetAddRows:
    """Test adding rows to existing CSV."""

    async def test_add_rows(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name,Score\nAlice,95\n")

        result = await sheet.execute(
            {
                "operation": "add_rows",
                "path": "data.csv",
                "rows": [["Bob", "87"], ["Carol", "92"]],
            },
            ctx,
        )
        assert result.success
        assert "2 rows" in result.output

        rows = _read_csv(workspace / "data.csv")
        assert len(rows) == 4  # header + 3 data rows
        assert rows[2] == ["Bob", "87"]
        assert rows[3] == ["Carol", "92"]

    async def test_add_rows_file_not_found(self, sheet, ctx):
        result = await sheet.execute(
            {
                "operation": "add_rows",
                "path": "missing.csv",
                "rows": [["x"]],
            },
            ctx,
        )
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_add_rows_no_rows_provided(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("A\n1\n")
        result = await sheet.execute(
            {"operation": "add_rows", "path": "data.csv"},
            ctx,
        )
        assert not result.success
        assert "rows" in result.error.lower()


class TestSpreadsheetAddColumn:
    """Test adding a column to existing CSV."""

    async def test_add_column_with_default(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name,Score\nAlice,95\nBob,87\n")

        result = await sheet.execute(
            {
                "operation": "add_column",
                "path": "data.csv",
                "column_name": "Grade",
                "default_value": "A",
            },
            ctx,
        )
        assert result.success
        assert "Grade" in result.output

        rows = _read_csv(workspace / "data.csv")
        assert rows[0] == ["Name", "Score", "Grade"]
        assert rows[1][2] == "A"
        assert rows[2][2] == "A"

    async def test_add_column_no_default(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name\nAlice\n")

        result = await sheet.execute(
            {
                "operation": "add_column",
                "path": "data.csv",
                "column_name": "Extra",
            },
            ctx,
        )
        assert result.success
        rows = _read_csv(workspace / "data.csv")
        assert rows[0] == ["Name", "Extra"]
        assert rows[1][1] == ""  # empty default

    async def test_add_column_file_not_found(self, sheet, ctx):
        result = await sheet.execute(
            {
                "operation": "add_column",
                "path": "missing.csv",
                "column_name": "X",
            },
            ctx,
        )
        assert not result.success

    async def test_add_column_no_name(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("A\n1\n")
        result = await sheet.execute(
            {"operation": "add_column", "path": "data.csv"},
            ctx,
        )
        assert not result.success
        assert "column_name" in result.error.lower()


class TestSpreadsheetUpdateCell:
    """Test updating a specific cell."""

    async def test_update_cell(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name,Score\nAlice,95\nBob,87\n")

        result = await sheet.execute(
            {
                "operation": "update_cell",
                "path": "data.csv",
                "column_name": "Score",
                "row_index": 1,
                "value": "99",
            },
            ctx,
        )
        assert result.success
        assert "'87' -> '99'" in result.output

        rows = _read_csv(workspace / "data.csv")
        assert rows[2][1] == "99"  # row_index 1 is second data row

    async def test_update_cell_first_row(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name,Score\nAlice,95\nBob,87\n")

        result = await sheet.execute(
            {
                "operation": "update_cell",
                "path": "data.csv",
                "column_name": "Name",
                "row_index": 0,
                "value": "Alicia",
            },
            ctx,
        )
        assert result.success
        rows = _read_csv(workspace / "data.csv")
        assert rows[1][0] == "Alicia"

    async def test_update_cell_column_not_found(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name,Score\nAlice,95\n")

        result = await sheet.execute(
            {
                "operation": "update_cell",
                "path": "data.csv",
                "column_name": "NonExistent",
                "row_index": 0,
                "value": "x",
            },
            ctx,
        )
        assert not result.success
        assert "not found" in result.error.lower()
        assert "Name" in result.error  # should show available columns

    async def test_update_cell_row_out_of_range(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name\nAlice\n")

        result = await sheet.execute(
            {
                "operation": "update_cell",
                "path": "data.csv",
                "column_name": "Name",
                "row_index": 5,
                "value": "x",
            },
            ctx,
        )
        assert not result.success
        assert "out of range" in result.error.lower()

    async def test_update_cell_no_data_rows(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name\n")

        result = await sheet.execute(
            {
                "operation": "update_cell",
                "path": "data.csv",
                "column_name": "Name",
                "row_index": 0,
                "value": "x",
            },
            ctx,
        )
        assert not result.success
        assert "no data rows" in result.error.lower()

    async def test_update_cell_missing_column_name(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name\nAlice\n")
        result = await sheet.execute(
            {
                "operation": "update_cell",
                "path": "data.csv",
                "row_index": 0,
                "value": "x",
            },
            ctx,
        )
        assert not result.success

    async def test_update_cell_missing_row_index(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name\nAlice\n")
        result = await sheet.execute(
            {
                "operation": "update_cell",
                "path": "data.csv",
                "column_name": "Name",
                "value": "x",
            },
            ctx,
        )
        assert not result.success


class TestSpreadsheetSummary:
    """Test spreadsheet summary operation."""

    async def test_summary(self, sheet, ctx, workspace):
        csv_path = workspace / "data.csv"
        csv_path.write_text("Name,Score,Grade\nAlice,95,A\nBob,87,B\nCarol,92,A\n")

        result = await sheet.execute(
            {"operation": "summary", "path": "data.csv"},
            ctx,
        )
        assert result.success
        assert "Columns (3)" in result.output
        assert "Rows: 3" in result.output
        assert "Name" in result.output
        assert "Score" in result.output
        assert "Sample data" in result.output

    async def test_summary_file_not_found(self, sheet, ctx):
        result = await sheet.execute(
            {"operation": "summary", "path": "missing.csv"},
            ctx,
        )
        assert not result.success
        assert "not found" in result.error.lower()

    async def test_summary_empty_csv(self, sheet, ctx, workspace):
        csv_path = workspace / "empty.csv"
        csv_path.write_text("")
        result = await sheet.execute(
            {"operation": "summary", "path": "empty.csv"},
            ctx,
        )
        assert result.success
        assert "Empty" in result.output


class TestSpreadsheetErrors:
    """Test spreadsheet error handling."""

    async def test_no_workspace(self, sheet, no_workspace_ctx):
        result = await sheet.execute(
            {"operation": "read", "path": "data.csv"},
            no_workspace_ctx,
        )
        assert not result.success
        assert "No workspace" in result.error

    async def test_no_path(self, sheet, ctx):
        result = await sheet.execute(
            {"operation": "read", "path": ""},
            ctx,
        )
        assert not result.success
        assert "No path" in result.error

    async def test_unknown_operation(self, sheet, ctx):
        result = await sheet.execute(
            {"operation": "pivot_table", "path": "data.csv"},
            ctx,
        )
        assert not result.success
        assert "Unknown operation" in result.error


class TestSpreadsheetMetadata:
    """Test tool metadata properties."""

    def test_name(self, sheet):
        assert sheet.name == "spreadsheet"

    def test_description(self, sheet):
        assert "CSV" in sheet.description

    def test_parameters_schema(self, sheet):
        params = sheet.parameters
        assert params["type"] == "object"
        assert "operation" in params["properties"]
        assert "path" in params["properties"]
        assert "operation" in params["required"]
        assert "path" in params["required"]


# ===================================================================
# DocumentWriteTool Tests
# ===================================================================


class TestDocumentCreateTitle:
    """Test creating documents with title."""

    async def test_title_only(self, doc, ctx, workspace):
        result = await doc.execute(
            {"path": "doc.md", "title": "My Document"},
            ctx,
        )
        assert result.success
        content = (workspace / "doc.md").read_text()
        assert content.startswith("# My Document")

    async def test_title_word_count(self, doc, ctx):
        result = await doc.execute(
            {"path": "doc.md", "title": "Test"},
            ctx,
        )
        assert result.success
        assert "words" in result.output


class TestDocumentCreateContent:
    """Test creating documents with raw content."""

    async def test_raw_content(self, doc, ctx, workspace):
        result = await doc.execute(
            {
                "path": "doc.md",
                "content": "This is **raw** markdown content.\n\n- Item 1\n- Item 2",
            },
            ctx,
        )
        assert result.success
        content = (workspace / "doc.md").read_text()
        assert "**raw**" in content
        assert "- Item 1" in content

    async def test_title_and_content(self, doc, ctx, workspace):
        result = await doc.execute(
            {
                "path": "doc.md",
                "title": "Report",
                "content": "Some body text.",
            },
            ctx,
        )
        assert result.success
        content = (workspace / "doc.md").read_text()
        assert "# Report" in content
        assert "Some body text." in content


class TestDocumentCreateSections:
    """Test creating documents with structured sections."""

    async def test_sections(self, doc, ctx, workspace):
        result = await doc.execute(
            {
                "path": "doc.md",
                "title": "Analysis Report",
                "sections": [
                    {
                        "heading": "Introduction",
                        "level": 2,
                        "body": "This report covers...",
                    },
                    {
                        "heading": "Findings",
                        "level": 2,
                        "body": "We found that...",
                    },
                    {
                        "heading": "Detail A",
                        "level": 3,
                        "body": "Specific detail about A.",
                    },
                ],
            },
            ctx,
        )
        assert result.success
        assert "3 sections" in result.output

        content = (workspace / "doc.md").read_text()
        assert "# Analysis Report" in content
        assert "## Introduction" in content
        assert "## Findings" in content
        assert "### Detail A" in content
        assert "This report covers..." in content

    async def test_section_default_level(self, doc, ctx, workspace):
        """Default heading level should be 2."""
        result = await doc.execute(
            {
                "path": "doc.md",
                "sections": [
                    {"heading": "Section", "body": "Content"},
                ],
            },
            ctx,
        )
        assert result.success
        content = (workspace / "doc.md").read_text()
        assert "## Section" in content

    async def test_section_level_clamped(self, doc, ctx, workspace):
        """Heading levels should be clamped to 2-4."""
        result = await doc.execute(
            {
                "path": "doc.md",
                "sections": [
                    {"heading": "Too High", "level": 6, "body": "Content"},
                    {"heading": "Too Low", "level": 1, "body": "Content"},
                ],
            },
            ctx,
        )
        assert result.success
        content = (workspace / "doc.md").read_text()
        assert "#### Too High" in content  # clamped to 4
        assert "## Too Low" in content  # clamped to 2


class TestDocumentMetadata:
    """Test creating documents with YAML frontmatter."""

    async def test_metadata_frontmatter(self, doc, ctx, workspace):
        result = await doc.execute(
            {
                "path": "doc.md",
                "title": "Report",
                "metadata": {
                    "author": "Test User",
                    "date": "2025-01-15",
                    "status": "draft",
                },
            },
            ctx,
        )
        assert result.success
        content = (workspace / "doc.md").read_text()
        assert content.startswith("---\n")
        assert "author: Test User" in content
        # yaml.dump may quote date-like strings for safety
        assert "date:" in content and "2025-01-15" in content
        assert "status: draft" in content
        # Frontmatter should be closed
        parts = content.split("---")
        assert len(parts) >= 3  # before ---, frontmatter, after ---

    async def test_metadata_not_added_in_append_mode(self, doc, ctx, workspace):
        """Metadata frontmatter should not be added when appending."""
        # First create a document
        await doc.execute(
            {"path": "doc.md", "title": "Original", "content": "Original content."},
            ctx,
        )
        # Now append with metadata - metadata should be ignored
        result = await doc.execute(
            {
                "path": "doc.md",
                "title": "Appendix",
                "metadata": {"author": "Should Not Appear"},
                "append": True,
            },
            ctx,
        )
        assert result.success
        content = (workspace / "doc.md").read_text()
        assert "Should Not Appear" not in content


class TestDocumentAppend:
    """Test appending to existing documents."""

    async def test_append_mode(self, doc, ctx, workspace):
        # Create initial document
        await doc.execute(
            {"path": "doc.md", "title": "Original", "content": "First content."},
            ctx,
        )
        # Append
        result = await doc.execute(
            {
                "path": "doc.md",
                "title": "Addendum",
                "content": "Additional content.",
                "append": True,
            },
            ctx,
        )
        assert result.success
        assert "Appended to" in result.output

        content = (workspace / "doc.md").read_text()
        assert "First content." in content
        assert "Additional content." in content
        assert "# Addendum" in content

    async def test_append_to_nonexistent_file(self, doc, ctx, workspace):
        """Appending to a file that doesn't exist should create it."""
        result = await doc.execute(
            {
                "path": "new.md",
                "content": "Appended content.",
                "append": True,
            },
            ctx,
        )
        assert result.success
        assert (workspace / "new.md").exists()


class TestDocumentErrors:
    """Test document_write error handling."""

    async def test_no_workspace(self, doc, no_workspace_ctx):
        result = await doc.execute(
            {"path": "doc.md", "title": "Test"},
            no_workspace_ctx,
        )
        assert not result.success
        assert "No workspace" in result.error

    async def test_empty_document(self, doc, ctx):
        """Providing no title, content, or sections should fail."""
        result = await doc.execute(
            {"path": "doc.md"},
            ctx,
        )
        assert not result.success
        assert "title" in result.error.lower() or "content" in result.error.lower()

    async def test_no_path(self, doc, ctx):
        result = await doc.execute(
            {"path": "", "title": "Test"},
            ctx,
        )
        assert not result.success
        assert "No path" in result.error

    async def test_creates_parent_directories(self, doc, ctx, workspace):
        result = await doc.execute(
            {"path": "deep/nested/doc.md", "title": "Deep"},
            ctx,
        )
        assert result.success
        assert (workspace / "deep" / "nested" / "doc.md").exists()

    async def test_files_changed_returned(self, doc, ctx):
        result = await doc.execute(
            {"path": "doc.md", "title": "Test"},
            ctx,
        )
        assert result.success
        assert "doc.md" in result.files_changed

    async def test_document_ends_with_newline(self, doc, ctx, workspace):
        """Generated documents should end with a newline."""
        await doc.execute(
            {"path": "doc.md", "title": "Test"},
            ctx,
        )
        content = (workspace / "doc.md").read_text()
        assert content.endswith("\n")


class TestDocumentMetadataProperties:
    """Test tool metadata properties."""

    def test_name(self, doc):
        assert doc.name == "document_write"

    def test_description(self, doc):
        assert "Markdown" in doc.description

    def test_parameters_schema(self, doc):
        params = doc.parameters
        assert params["type"] == "object"
        assert "path" in params["properties"]
        assert "title" in params["properties"]
        assert "content" in params["properties"]
        assert "sections" in params["properties"]
        assert "metadata" in params["properties"]
        assert "append" in params["properties"]
        assert "path" in params["required"]


# ===================================================================
# Direct function tests (not through the tool interface)
# ===================================================================


class TestFinancialFunctions:
    """Direct tests of financial functions outside the tool wrapper."""

    def test_npv_basic(self):
        result = npv(0.1, [-1000, 300, 400, 500])
        assert result == pytest.approx(-1000 + 300 / 1.1 + 400 / 1.21 + 500 / 1.331)

    def test_npv_zero_rate(self):
        result = npv(0, [-100, 50, 50])
        assert result == pytest.approx(0)

    def test_cagr_basic(self):
        result = cagr(100, 200, 5)
        assert result == pytest.approx((200 / 100) ** (1 / 5) - 1)

    def test_cagr_invalid_beginning_value(self):
        with pytest.raises(ValueError):
            cagr(0, 200, 5)

    def test_cagr_invalid_years(self):
        with pytest.raises(ValueError):
            cagr(100, 200, 0)

    def test_wacc_basic(self):
        result = wacc(1000000, 500000, 0.12, 0.06, 0.21)
        total = 1500000
        expected = (1000000 / total) * 0.12 + (500000 / total) * 0.06 * (1 - 0.21)
        assert result == pytest.approx(expected)

    def test_wacc_invalid_total(self):
        with pytest.raises(ValueError):
            wacc(0, 0, 0.1, 0.05, 0.2)

    def test_pmt_basic(self):
        result = pmt(0.05 / 12, 360, 200000)
        assert result < 0  # payments are negative (outflow)
        assert abs(result) > 0

    def test_pmt_zero_rate(self):
        result = pmt(0, 12, 1200)
        assert result == pytest.approx(-100)


class TestSafeEval:
    """Direct tests of the _safe_eval function."""

    def test_integer_result(self):
        assert _safe_eval("2 + 3") == 5

    def test_float_result(self):
        assert _safe_eval("1.5 * 2") == pytest.approx(3.0)

    def test_list_literal(self):
        result = _safe_eval("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_tuple_literal(self):
        result = _safe_eval("(1, 2, 3)")
        assert result == (1, 2, 3)

    def test_nested_call(self):
        result = _safe_eval("round(sqrt(2), 4)")
        assert result == pytest.approx(1.4142)

    def test_bool_constants(self):
        assert _safe_eval("true") is True
        assert _safe_eval("false") is False


class TestCalculatorExponentSafety:
    """Test that large exponents are rejected."""

    async def test_large_exponent_rejected(self, calc, ctx):
        result = await calc.execute(
            {"expression": "2 ** 100000000"},
            ctx,
        )
        assert not result.success
        assert "exponent" in result.error.lower() or "too large" in result.error.lower()

    async def test_large_negative_exponent_rejected(self, calc, ctx):
        result = await calc.execute(
            {"expression": "10 ** -50000"},
            ctx,
        )
        assert not result.success

    async def test_safe_exponent_allowed(self, calc, ctx):
        result = await calc.execute(
            {"expression": "2 ** 100"},
            ctx,
        )
        assert result.success
        assert result.data["result"] == 2**100

    async def test_pow_function_safe(self, calc, ctx):
        """pow() function should still work for reasonable values."""
        result = await calc.execute(
            {"expression": "pow(2, 20)"},
            ctx,
        )
        assert result.success
        assert "= 1048576" in result.output

    def test_safe_eval_exponent_limit(self):
        """Direct test of the exponent limit in _safe_eval."""
        with pytest.raises(ValueError, match="Exponent too large"):
            _safe_eval("2 ** 20000")


class TestSpreadsheetFileSizeGuards:
    """Test that file size limits are enforced on modify operations."""

    async def test_add_column_rejects_large_file(self, sheet, ctx, workspace):
        large = workspace / "big.csv"
        # Write a file slightly over 5MB
        large.write_text("A\n" + "x\n" * (5 * 1024 * 1024 // 2))
        result = await sheet.execute(
            {
                "operation": "add_column",
                "path": "big.csv",
                "column_name": "B",
            },
            ctx,
        )
        assert not result.success
        assert "too large" in result.error.lower()

    async def test_update_cell_rejects_large_file(self, sheet, ctx, workspace):
        large = workspace / "big.csv"
        large.write_text("A\n" + "x\n" * (5 * 1024 * 1024 // 2))
        result = await sheet.execute(
            {
                "operation": "update_cell",
                "path": "big.csv",
                "column_name": "A",
                "row_index": 0,
                "value": "y",
            },
            ctx,
        )
        assert not result.success
        assert "too large" in result.error.lower()


# ===================================================================
# Financial function input validation (Round 2 safety fixes)
# ===================================================================


class TestFinancialFunctionValidation:
    """Tests for input validation in financial functions."""

    def test_npv_rejects_rate_leq_neg1(self):
        with pytest.raises(ValueError, match="greater than -1"):
            npv(-1.0, [100, 200])
        with pytest.raises(ValueError, match="greater than -1"):
            npv(-2.0, [100, 200])

    def test_npv_accepts_valid_rate(self):
        assert npv(0.1, [100, 200]) > 0
        assert npv(0.0, [100, 200]) == 300.0

    def test_cagr_rejects_negative_ending_value(self):
        with pytest.raises(ValueError, match="negative"):
            cagr(100, -50, 5)

    def test_cagr_accepts_zero_ending(self):
        result = cagr(100, 0, 5)
        assert result == -1.0

    def test_pmt_rejects_nonpositive_nper(self):
        with pytest.raises(ValueError, match="positive"):
            pmt(0.05, 0, 1000)
        with pytest.raises(ValueError, match="positive"):
            pmt(0.05, -5, 1000)

    def test_pmt_accepts_valid_nper(self):
        result = pmt(0.05, 12, 1000)
        assert isinstance(result, float)


# ===================================================================
# Document metadata YAML safety (Round 2 safety fix)
# ===================================================================


class TestDocumentMetadataYamlSafety:
    """Test that metadata frontmatter is safely serialized."""

    @pytest.fixture
    def doc(self):
        return DocumentWriteTool()

    @pytest.fixture
    def ctx(self, workspace):
        return ToolContext(workspace=workspace, subtask_id="s1")

    async def test_metadata_with_colons_is_safe(self, doc, ctx, workspace):
        """Values with colons should not break YAML structure."""
        result = await doc.execute(
            {
                "path": "doc.md",
                "title": "Test",
                "metadata": {"description": "a: b: c"},
            },
            ctx,
        )
        assert result.success
        content = (workspace / "doc.md").read_text()
        parts = content.split("---")
        assert len(parts) >= 3
        parsed = yaml.safe_load(parts[1])
        assert parsed["description"] == "a: b: c"

    async def test_metadata_with_newlines_is_safe(self, doc, ctx, workspace):
        """Newlines in values should not escape the frontmatter block."""
        result = await doc.execute(
            {
                "path": "doc.md",
                "title": "Test",
                "metadata": {"note": "line1\nline2"},
            },
            ctx,
        )
        assert result.success
        content = (workspace / "doc.md").read_text()
        parts = content.split("---")
        assert len(parts) >= 3
        parsed = yaml.safe_load(parts[1])
        assert parsed["note"] == "line1\nline2"

    async def test_section_validation_rejects_missing_body(self, doc, ctx, workspace):
        """Sections missing required keys should be rejected."""
        result = await doc.execute(
            {
                "path": "doc.md",
                "sections": [{"heading": "Test"}],
            },
            ctx,
        )
        assert not result.success
        assert "body" in result.error.lower()

    async def test_section_validation_rejects_non_dict(self, doc, ctx, workspace):
        result = await doc.execute(
            {
                "path": "doc.md",
                "sections": ["not a dict"],
            },
            ctx,
        )
        assert not result.success
        assert "dict" in result.error.lower()
