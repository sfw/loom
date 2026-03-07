"""TUI tool call preview formatting tests."""

from __future__ import annotations

from loom.tui.widgets.tool_call import (
    _escape,
    _style_diff_output,
    _trunc,
    tool_args_preview,
    tool_output_preview,
)


class TestToolArgsPreview:
    def test_file_tool(self):
        assert tool_args_preview("read_file", {"path": "foo.py"}) == "foo.py"

    def test_shell(self):
        result = tool_args_preview(
            "shell_execute", {"command": "ls -la"},
        )
        assert result == "ls -la"

    def test_git(self):
        result = tool_args_preview(
            "git_command", {"args": ["push", "origin"]},
        )
        assert result == "push origin"

    def test_ripgrep(self):
        result = tool_args_preview(
            "ripgrep_search", {"pattern": "TODO"},
        )
        assert result == "/TODO/"

    def test_glob(self):
        result = tool_args_preview(
            "glob_find", {"pattern": "**/*.py"},
        )
        assert result == "**/*.py"

    def test_web_search(self):
        result = tool_args_preview(
            "web_search", {"query": "python docs"},
        )
        assert result == "python docs"

    def test_web_fetch(self):
        result = tool_args_preview(
            "web_fetch", {"url": "https://example.com"},
        )
        assert result == "https://example.com"

    def test_web_fetch_html(self):
        result = tool_args_preview(
            "web_fetch_html", {"url": "https://example.com"},
        )
        assert result == "https://example.com"

    def test_task_tracker(self):
        result = tool_args_preview(
            "task_tracker", {"action": "add", "content": "Fix bug"},
        )
        assert result == "add: Fix bug"

    def test_task_tracker_no_content(self):
        result = tool_args_preview("task_tracker", {"action": "list"})
        assert result == "list"

    def test_ask_user(self):
        result = tool_args_preview(
            "ask_user", {"question": "Which DB?"},
        )
        assert result == "Which DB?"

    def test_analyze_code(self):
        result = tool_args_preview("analyze_code", {"path": "src/"})
        assert result == "src/"

    def test_generic_fallback(self):
        assert tool_args_preview("unknown", {"x": "hello"}) == "hello"

    def test_empty(self):
        assert tool_args_preview("unknown", {}) == ""

class TestToolOutputPreview:
    def test_empty(self):
        assert tool_output_preview("read_file", "") == ""

    def test_read_file(self):
        result = tool_output_preview(
            "read_file", "line1\nline2\nline3\n",
        )
        assert result == "3 lines"

    def test_search_no_matches(self):
        result = tool_output_preview(
            "ripgrep_search", "No matches found.",
        )
        assert result == "No matches found."

    def test_search_results(self):
        output = "file1.py:10:match\nfile2.py:20:match"
        assert tool_output_preview("ripgrep_search", output) == "2 results"

    def test_shell(self):
        result = tool_output_preview(
            "shell_execute", "hello world\nmore output",
        )
        assert result == "hello world"

    def test_web_search(self):
        output = (
            "1. Result one\n   url\n"
            "2. Result two\n   url\n"
            "3. Result three\n   url"
        )
        assert "3 results" in tool_output_preview("web_search", output)

    def test_edit_file_summary(self):
        output = "Edited foo.py: replaced 2 lines with 3 lines\n\n--- a/foo.py\n+++ b/foo.py"
        result = tool_output_preview("edit_file", output)
        assert "Edited foo.py" in result
        # Should NOT contain diff markers
        assert "---" not in result

    def test_unknown_tool(self):
        assert tool_output_preview("unknown", "whatever") == ""

class TestTrunc:
    def test_short(self):
        assert _trunc("hello", 10) == "hello"

    def test_exact(self):
        assert _trunc("hello", 5) == "hello"

    def test_long(self):
        assert _trunc("hello world", 8) == "hello..."

class TestEscape:
    def test_escapes_brackets(self):
        assert _escape("list[int]") == "list\\[int]"

    def test_no_brackets(self):
        assert _escape("hello world") == "hello world"

class TestStyleDiffOutput:
    def test_summary_line_is_dim(self):
        output = "Edited foo.py: replaced 1 lines with 1 lines"
        styled = _style_diff_output(output)
        assert "[dim]" in styled

    def test_additions_are_green(self):
        output = "summary\n\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"
        styled = _style_diff_output(output)
        assert "[#9ece6a]" in styled  # green for additions
        assert "[#f7768e]" in styled  # red for removals
        assert "[#7dcfff]" in styled  # cyan for hunk headers

    def test_headers_are_bold(self):
        output = "--- a/foo.py\n+++ b/foo.py"
        styled = _style_diff_output(output)
        assert "[bold]" in styled

    def test_brackets_in_code_escaped(self):
        output = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x: list[int]\n+x: list[str]"
        styled = _style_diff_output(output)
        # Brackets should be escaped to prevent Rich markup interpretation
        assert "\\[int]" in styled
        assert "\\[str]" in styled
