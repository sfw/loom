"""Tests for the error categorization system."""

from __future__ import annotations

from loom.recovery.errors import (
    CategorizedError,
    ErrorCategory,
    categorize_error,
    categorize_tool_failure,
)


class TestCategorizeError:
    def test_syntax_error(self):
        result = categorize_error("SyntaxError: invalid syntax at line 42")
        assert result.category == ErrorCategory.SYNTAX_ERROR
        assert "syntax" in result.recovery_hint.lower()

    def test_indentation_error(self):
        result = categorize_error("IndentationError: unexpected indent")
        assert result.category == ErrorCategory.SYNTAX_ERROR

    def test_file_not_found(self):
        result = categorize_error("FileNotFoundError: /path/to/file.py")
        assert result.category == ErrorCategory.FILE_NOT_FOUND
        assert "exist" in result.recovery_hint.lower()

    def test_file_not_found_variant(self):
        result = categorize_error("File not found: src/missing.py")
        assert result.category == ErrorCategory.FILE_NOT_FOUND

    def test_permission_error(self):
        result = categorize_error("PermissionError: [Errno 13] Permission denied")
        assert result.category == ErrorCategory.PERMISSION_ERROR

    def test_safety_violation(self):
        result = categorize_error("Safety violation: path escapes workspace")
        assert result.category == ErrorCategory.SAFETY_VIOLATION
        assert "safety" in result.recovery_hint.lower()

    def test_timeout(self):
        result = categorize_error("Tool 'shell_execute' timed out after 60s")
        assert result.category == ErrorCategory.TIMEOUT
        assert "timeout" in result.recovery_hint.lower() or "simpler" in result.recovery_hint.lower()

    def test_tool_error(self):
        result = categorize_error("Unknown tool: foobar")
        assert result.category == ErrorCategory.TOOL_ERROR

    def test_validation_error(self):
        result = categorize_error("Invalid JSON response from model")
        assert result.category == ErrorCategory.VALIDATION_ERROR

    def test_model_error(self):
        result = categorize_error("ConnectError: failed to reach model server")
        assert result.category == ErrorCategory.MODEL_ERROR
        assert "transient" in result.recovery_hint.lower()

    def test_runtime_error(self):
        result = categorize_error("RuntimeError: some unexpected thing")
        assert result.category == ErrorCategory.RUNTIME_ERROR

    def test_generic_exception(self):
        result = categorize_error("ValueError: bad value")
        assert result.category == ErrorCategory.RUNTIME_ERROR

    def test_unknown_error(self):
        result = categorize_error("something completely novel happened")
        assert result.category == ErrorCategory.UNKNOWN
        assert "different approach" in result.recovery_hint.lower()

    def test_empty_error(self):
        result = categorize_error("")
        assert result.category == ErrorCategory.UNKNOWN
        assert result.original_error == ""

    def test_priority_order(self):
        # "SyntaxError" should match syntax_error, not runtime_error
        result = categorize_error("SyntaxError: this is also an Error")
        assert result.category == ErrorCategory.SYNTAX_ERROR


class TestCategorizeToolFailure:
    def test_file_not_found_enriched(self):
        result = categorize_tool_failure(
            "read_file", "File not found: missing.py", ""
        )
        assert result.category == ErrorCategory.FILE_NOT_FOUND
        assert "read_file" in result.recovery_hint
        assert "list the directory" in result.recovery_hint.lower()

    def test_tool_error_enriched(self):
        result = categorize_tool_failure(
            "foobar", "Unknown tool: foobar", ""
        )
        assert result.category == ErrorCategory.TOOL_ERROR
        assert "foobar" in result.recovery_hint

    def test_generic_failure(self):
        result = categorize_tool_failure(
            "shell_execute", None, "command not found"
        )
        # Should still categorize even without explicit error
        assert isinstance(result, CategorizedError)
