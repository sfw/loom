"""Error categorization for intelligent retry.

Classifies errors into categories so the retry manager can provide
specific, actionable feedback instead of generic "try again" messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ErrorCategory(Enum):
    """Categories of errors with different recovery strategies."""

    SYNTAX_ERROR = "syntax_error"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    TOOL_ERROR = "tool_error"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    SAFETY_VIOLATION = "safety_violation"
    VALIDATION_ERROR = "validation_error"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


@dataclass
class CategorizedError:
    """An error with its category, details, and recovery hint."""

    category: ErrorCategory
    original_error: str
    detail: str
    recovery_hint: str


# Patterns for error classification — order matters (first match wins)
_PATTERNS: list[tuple[re.Pattern, ErrorCategory, str]] = [
    (
        re.compile(r"SyntaxError|IndentationError|TabError", re.IGNORECASE),
        ErrorCategory.SYNTAX_ERROR,
        "Fix the syntax error. Check indentation, brackets, and quotes.",
    ),
    (
        re.compile(r"File not found|FileNotFoundError|No such file|not found:", re.IGNORECASE),
        ErrorCategory.FILE_NOT_FOUND,
        "The file doesn't exist. Verify the path, or create the file first.",
    ),
    (
        re.compile(r"PermissionError|Permission denied|not writable", re.IGNORECASE),
        ErrorCategory.PERMISSION_ERROR,
        "Permission denied. Check file permissions or try a different path.",
    ),
    (
        re.compile(r"Safety violation|escapes workspace|Blocked dangerous", re.IGNORECASE),
        ErrorCategory.SAFETY_VIOLATION,
        "This operation was blocked for safety. Use a different approach.",
    ),
    (
        re.compile(r"timed out|TimeoutError|timeout", re.IGNORECASE),
        ErrorCategory.TIMEOUT,
        "The operation timed out. Try a simpler approach or break it into smaller steps.",
    ),
    (
        re.compile(r"Unknown tool|TOOL CALL ERROR|not allowed", re.IGNORECASE),
        ErrorCategory.TOOL_ERROR,
        "Tool call failed. Check tool name and arguments.",
    ),
    (
        re.compile(r"Invalid JSON|JSONDecodeError|expected_keys|validate", re.IGNORECASE),
        ErrorCategory.VALIDATION_ERROR,
        "Response validation failed. Ensure output matches the expected format.",
    ),
    (
        re.compile(r"ConnectError|HTTPStatusError|model.*error|503|502|500", re.IGNORECASE),
        ErrorCategory.MODEL_ERROR,
        "Model or API error. This may be transient — retry should help.",
    ),
    (
        re.compile(r"Error|Exception|Traceback|Failed", re.IGNORECASE),
        ErrorCategory.RUNTIME_ERROR,
        "A runtime error occurred. Review the error message and adjust your approach.",
    ),
]


def categorize_error(error_text: str) -> CategorizedError:
    """Categorize an error string into a structured CategorizedError.

    Scans the error text against known patterns and returns
    the first matching category with a recovery hint.
    """
    if not error_text:
        return CategorizedError(
            category=ErrorCategory.UNKNOWN,
            original_error="",
            detail="No error information provided",
            recovery_hint="Review what happened and try again.",
        )

    for pattern, category, hint in _PATTERNS:
        match = pattern.search(error_text)
        if match:
            return CategorizedError(
                category=category,
                original_error=error_text,
                detail=match.group(0),
                recovery_hint=hint,
            )

    return CategorizedError(
        category=ErrorCategory.UNKNOWN,
        original_error=error_text,
        detail=error_text[:200],
        recovery_hint="Review the error and try a different approach.",
    )


def categorize_tool_failure(
    tool_name: str, tool_error: str | None, tool_output: str
) -> CategorizedError:
    """Categorize a tool execution failure with tool-specific context."""
    combined = f"Tool '{tool_name}': {tool_error or ''} {tool_output}"
    result = categorize_error(combined)

    # Enrich with tool-specific hints
    if result.category == ErrorCategory.FILE_NOT_FOUND and tool_name in (
        "read_file", "edit_file", "delete_file",
    ):
        result.recovery_hint = (
            f"The file path passed to {tool_name} doesn't exist. "
            "List the directory first to find the correct path, "
            "or create the file with write_file."
        )
    elif result.category == ErrorCategory.TOOL_ERROR:
        result.recovery_hint = (
            f"The call to '{tool_name}' failed. "
            "Check the tool's parameter schema and try again with valid arguments."
        )

    return result
