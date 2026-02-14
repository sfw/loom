# Spec 05: Tool System

## Overview

Tools are what make Loom agentic. Without tools, the model can only produce text. With tools, it can read files, edit code, run commands, and interact with the workspace. The tool system provides registration, dispatch, security, and result formatting.

## Tool Registry

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises if name conflicts."""

    async def execute(self, name: str, arguments: dict, workspace: Path | None = None) -> ToolResult:
        """Execute a tool with argument validation, workspace context, and timeout."""

    def tools_for_subtask(self, subtask: Subtask) -> list[dict]:
        """Return OpenAI-format tool schemas relevant to a subtask."""

    def all_schemas(self) -> list[dict]:
        """Return all tool schemas for model consumption."""
```

## Tool Base Class

```python
@dataclass
class ToolResult:
    success: bool
    output: str                      # Human-readable output (truncated to 10KB)
    data: dict | None = None         # Structured data (optional)
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> str: ...

@dataclass
class ToolContext:
    workspace: Path | None
    scratch_dir: Path | None = None

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for parameters."""

    @property
    def timeout_seconds(self) -> int:
        return 30

    @abstractmethod
    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult: ...

    def schema(self) -> dict:
        """Return OpenAI-format function definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
```

## Built-In Tools

| Tool | Description | Timeout |
|------|-------------|---------|
| `read_file` | Read file contents (text, PDF via pypdf, image metadata), optional line range | 10s |
| `write_file` | Write/create file, records changelog | 10s |
| `edit_file` | Find-and-replace unique string in file, records changelog | 10s |
| `shell_execute` | Run shell command in workspace directory | 120s |
| `search_files` | Grep for patterns across workspace files | 30s |
| `list_directory` | List files and dirs up to 2 levels deep | 10s |
| `delete_file` | Delete file or empty directory | 10s |
| `move_file` | Move or rename a file within workspace | 10s |
| `git_command` | Git operations (status, diff, log, add, commit, push, etc.) | 60s |
| `analyze_code` | Parse file structure (classes, functions, imports) | 15s |
| `web_fetch` | Fetch URL content for documentation/specs | 45s |
| `web_search` | Internet search via DuckDuckGo (no API key required) | 30s |
| `ripgrep_search` | Fast content search via ripgrep with regex, context lines, type filters | 30s |
| `glob_find` | Fast file discovery by glob pattern, sorted by mtime, skips junk dirs | 10s |
| `ask_user` | Ask the developer questions mid-execution (free-text or multiple choice) | 300s |
| `task_tracker` | In-memory progress tracking (add/update/list/clear tasks) | 10s |

### read_file
- Parameters: `path` (required, string), `line_range` (optional, [start, end])
- Resolves path relative to workspace
- Returns file content as text

### write_file
- Parameters: `path` (required), `content` (required)
- Creates parent directories if needed
- Records previous content in changelog before overwriting
- Returns bytes written

### edit_file
- Parameters: `path` (required), `old_str` (required), `new_str` (required)
- Validates `old_str` appears exactly once in file
- Records changelog before edit
- Returns confirmation with context around change

### shell_execute
- Parameters: `command` (required)
- Runs in workspace as cwd
- Captures stdout and stderr
- Truncates output to 10KB
- Blocks dangerous commands (see Security section)

### search_files
- Parameters: `pattern` (required, regex), `path` (optional, subdirectory), `file_pattern` (optional, glob like `*.py`)
- Uses grep under the hood
- Returns matching lines with file paths and line numbers

### list_directory
- Parameters: `path` (optional, defaults to workspace root)
- Returns tree-style listing, 2 levels deep, ignoring `.git`, `node_modules`, `__pycache__`

## Plugin Discovery

Tools are auto-discovered via `Tool.__init_subclass__`. Any concrete `Tool` subclass
in the `loom.tools` package is automatically collected when its module is imported.
The `discover_tools()` function scans all modules in the package and returns the
collected classes. `create_default_registry()` calls `discover_tools()` and
instantiates each class.

To add a new built-in tool:
1. Create a new module in `src/loom/tools/`
2. Define a class that subclasses `Tool` and implements all abstract methods
3. That's it — no manual registration code needed

For tools outside the package, use manual registration:

```python
from loom.tools import create_default_registry
registry = create_default_registry()
registry.register(MyCustomTool())
```

## Path Security

All file tools MUST validate resolved paths stay within the workspace:

```python
def _verify_within_workspace(self, path: Path, workspace: Path) -> None:
    try:
        path.resolve().relative_to(workspace.resolve())
    except ValueError:
        raise ToolSafetyViolation(f"Path '{path}' escapes workspace")
```

## Shell Command Safety

Block obviously destructive patterns:
- `rm -rf /`, `rm -rf ~`, `rm -rf /*`
- `mkfs`, `dd if=`, `> /dev/`
- `chmod -R 777 /`
- `curl | sh`, `wget | bash` (arbitrary remote execution)

This is a basic blocklist. Phase 2 adds Docker sandboxing for full isolation.

## Changelog Integration

Every file-modifying tool (write_file, edit_file) records changes via the workspace changelog (Spec 11) before making modifications. This enables full undo capability.

## Acceptance Criteria

- [x] All built-in tools register and produce valid OpenAI-format schemas
- [x] File tools resolve paths relative to workspace
- [x] Path traversal attacks are blocked
- [x] Dangerous shell commands are blocked
- [x] Tool execution respects timeout limits
- [x] Changelog entries are recorded before file modifications
- [x] ToolResult serializes to JSON for model consumption
- [x] Tools work gracefully when workspace is None (report error, don't crash)
- [x] Tools auto-discovered via `__init_subclass__` — no manual registration needed
