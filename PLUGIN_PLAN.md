# Plugin-Based Tool Auto-Registration Plan

Replace the manual `create_default_registry()` with auto-discovery so adding a tool
is as simple as creating a file in `tools/` that subclasses `Tool`.

---

## Step 1: Add `__init_subclass__` Tracking to `Tool` ABC

**File:** `src/loom/tools/registry.py`

Add a class-level registry set on `Tool` that collects all concrete subclasses:

```python
class Tool(ABC):
    _registered_classes: ClassVar[set[type[Tool]]] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only collect concrete classes (all 3 abstract properties implemented)
        if not getattr(cls, '__abstractmethods__', None):
            Tool._registered_classes.add(cls)
```

This fires automatically when Python imports any module that defines a `Tool` subclass.
Abstract intermediate classes (if any) are skipped since they still have `__abstractmethods__`.

---

## Step 2: Add `discover_tools()` Module Scanner

**File:** `src/loom/tools/__init__.py`

Replace the manual import list + `create_default_registry()` with:

```python
import importlib
import pkgutil

import loom.tools as _pkg

def discover_tools() -> list[type[Tool]]:
    """Import all modules in loom.tools and return discovered Tool subclasses."""
    for finder, module_name, is_pkg in pkgutil.walk_packages(
        _pkg.__path__, prefix=_pkg.__name__ + "."
    ):
        # Skip registry itself and workspace (not a tool)
        if module_name.endswith(".registry") or module_name.endswith(".workspace"):
            continue
        importlib.import_module(module_name)
    return sorted(Tool._registered_classes, key=lambda cls: cls.__name__)


def create_default_registry() -> ToolRegistry:
    """Create a registry with all discovered built-in tools."""
    registry = ToolRegistry()
    for tool_cls in discover_tools():
        registry.register(tool_cls())
    return registry
```

This preserves the existing `create_default_registry()` API so **zero changes** are needed
in `api/engine.py`, tests, or integration code. But now it's automatic.

Keep the public re-exports (`Tool`, `ToolContext`, `ToolResult`, `ToolRegistry`) for
backward compatibility.

---

## Step 3: Verify Skip Logic

The `workspace.py` module contains `ChangeLog`, `DiffGenerator` etc. — utility classes,
not tools. The scanner skips it explicitly. The `registry.py` module defines the base
classes and is also skipped.

If someone later adds a module with helper classes that subclass `Tool` as abstract
intermediaries, those are automatically excluded by the `__abstractmethods__` check.

---

## Step 4: Update Tests

**File:** `tests/test_tools.py`

- Change `test_all_schemas` from `assert len(schemas) == 11` to a dynamic check:
  `assert len(schemas) >= 11` — or better, validate against `discover_tools()` count
- Add a new test: `test_auto_discovery` that verifies `discover_tools()` finds all
  expected tool classes
- Add a test that verifies adding a new `Tool` subclass in a test module gets auto-collected
- Keep existing `create_default_registry()` fixture — it still works

---

## Step 5: Update Documentation

### 5a. `README.md`

**Architecture section** (line 217-222): Update the `tools/` listing to include the
new modules and mention plugin auto-discovery:

```
  tools/
    registry.py          Tool ABC with auto-discovery via __init_subclass__
    file_ops.py          Read, write, edit, delete, move files
    shell.py             Shell execution with safety
    git.py               Git operations with allowlist
    search.py            File search and directory listing
    code_analysis.py     Regex-based code structure analysis
    web.py               Web fetch with URL safety
    workspace.py         Changelog, diff, revert
```

**Features > Tool system** (line 39): Update to reflect all 11 tools and plugin model:

```
- **Tool system** — 11 built-in tools (file ops, shell, git, search, code analysis, web fetch)
  with plugin auto-discovery
```

**API Endpoints table** (lines 110-127): Add the new endpoints:

```
| `GET` | `/tasks/{id}/tokens` | SSE token stream |
| `POST` | `/tasks/{id}/message` | Send conversation message |
| `GET` | `/tasks/{id}/conversation` | Conversation history |
```

### 5b. `docs/tutorial.html`

**TUI Keyboard Shortcuts table** (lines 863-871): Add the new keybindings:

```html
<tr><td><code>m</code></td><td>Open memory inspector</td></tr>
<tr><td><code>f</code></td><td>Submit feedback</td></tr>
<tr><td><code>t</code></td><td>Open conversation/chat</td></tr>
```

**Source Layout section** (lines 1156-1202): Add the new tool modules:

```
    git.py               # Git operations
    code_analysis.py     # Code structure analysis
    web.py               # Web fetch
```

**Streaming Events table** (lines 799-816): Add new event types:

```html
<tr><td><code>token_streamed</code></td><td>A model token was generated (streaming mode)</td></tr>
<tr><td><code>conversation_message</code></td><td>A conversation message was sent/received</td></tr>
```

**Endpoints section** (lines 1015-1037): Add the new endpoints:

```html
<!-- Under Task Lifecycle -->
<div class="endpoint"><span class="method method-get">GET</span><span class="endpoint-path">/tasks/{id}/tokens</span><span class="endpoint-desc">SSE token stream (streaming mode)</span></div>

<!-- New section: Conversation -->
<h3>Conversation</h3>
<div class="endpoint"><span class="method method-post">POST</span><span class="endpoint-path">/tasks/{id}/message</span><span class="endpoint-desc">Send a message to a running task</span></div>
<div class="endpoint"><span class="method method-get">GET</span><span class="endpoint-path">/tasks/{id}/conversation</span><span class="endpoint-desc">Conversation history</span></div>
```

### 5c. `docs/agent-integration.md`

**Custom Tools section** (lines 345-419): Update the registration example to show both
approaches — manual registration still works, but now there's auto-discovery too:

```python
# Option 1: Auto-discovery (just subclass Tool — it's found automatically)
# Place your tool in src/loom/tools/my_tool.py and it registers on import

# Option 2: Manual registration (for tools outside the tools/ package)
from loom.tools import create_default_registry
registry = create_default_registry()
registry.register(RunTestsTool())
```

**Event Types Reference** (line 563+): Add new events:

```
| `token_streamed` | Model token generated (streaming mode) |
| `conversation_message` | Conversation message sent to task |
```

### 5d. `planning/05-TOOL-SYSTEM.md`

**Built-In Tools table** (lines 80-87): Add the 5 new tools:

```
| `delete_file` | Delete file or empty directory | 10s |
| `move_file` | Move or rename a file within workspace | 10s |
| `git_command` | Git operations (status, diff, log, add, commit, etc.) | 60s |
| `analyze_code` | Parse file structure (classes, functions, imports) | 15s |
| `web_fetch` | Fetch URL content for documentation/specs | 45s |
```

Add a new **Plugin Discovery** section after the Tool Base Class section:

```markdown
## Plugin Discovery

Tools are auto-discovered via `__init_subclass__`. Any concrete `Tool` subclass
in the `loom.tools` package is automatically collected when its module is imported.
The `create_default_registry()` function scans all modules in the package and
instantiates discovered tools.

To add a new built-in tool:
1. Create a new module in `src/loom/tools/`
2. Define a class that subclasses `Tool` and implements all abstract methods
3. That's it — no registration code needed

For tools outside the package, use manual registration:
    registry = create_default_registry()
    registry.register(MyCustomTool())
```

Update **Acceptance Criteria** to add:
- `[x]` marks for completed items
- New item: `[ ] Tools auto-discovered via __init_subclass__ — no manual registration needed`

### 5e. `PLAN.md`

Mark all 6 phases as completed. Add a note about the plugin refactor as a follow-up improvement.

---

## Step 6: Run Tests, Commit, Push

- Run full test suite (`pytest tests/ -x -q`)
- Fix any failures
- Commit with message: "Refactor tool registration to plugin auto-discovery model"
- Push to `claude/loom-foundation-sEWVU`

---

## Files Changed Summary

| File | Change |
|------|--------|
| `src/loom/tools/registry.py` | Add `__init_subclass__` and `_registered_classes` to `Tool` |
| `src/loom/tools/__init__.py` | Replace manual list with `discover_tools()` + module scanning |
| `tests/test_tools.py` | Update schema count, add auto-discovery tests |
| `README.md` | Update architecture, features, API endpoints |
| `docs/tutorial.html` | Add keybindings, events, endpoints, tool modules |
| `docs/agent-integration.md` | Update custom tool example, add events |
| `planning/05-TOOL-SYSTEM.md` | Add new tools, plugin discovery section |
| `PLAN.md` | Mark phases complete |

**Backward compatibility:** `create_default_registry()` keeps the same signature and
behavior. All existing callers (engine.py, tests, integrations) work unchanged.
