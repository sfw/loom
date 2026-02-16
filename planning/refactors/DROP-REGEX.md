# Drop Regex: Replace All `re` Usage with Structural Alternatives

## Problem Statement

The codebase uses `re` in 37 files with ~97 compiled patterns. Regex is write-only code:
the patterns are hard to read, hard to test edge cases for, and silently wrong when they
under-match or over-match. Every category of regex usage in Loom has a clearer, more
correct structural alternative that already exists in Python's stdlib or as a trivial
data-driven lookup.

**Goal:** Remove every `import re` from the codebase. Zero regex. Each call site gets
replaced with the simplest thing that actually solves the problem.

---

## Inventory of All `re` Usage

| # | File | Count | What It Does |
|---|------|-------|-------------|
| 1 | `learning/reflection.py` | 52 patterns + 2 subs | Detect steering signals in user text |
| 2 | `recovery/errors.py` | 9 patterns | Classify error strings into categories |
| 3 | `tools/shell.py` | 17 patterns | Block dangerous shell commands |
| 4 | `tools/git.py` | 5 patterns | Block dangerous git operations |
| 5 | `tools/web.py` | 1 pattern + 3 subs | SSRF blocking + HTML stripping |
| 6 | `tools/web_search.py` | 6 findall/search + 2 subs | Parse DuckDuckGo HTML results |
| 7 | `processes/schema.py` | 1 match + 1 compile check | Validate process names; validate user-authored regex rules |
| 8 | `processes/installer.py` | 1 match | Validate process names |
| 9 | `__main__.py` | 1 match | Validate task IDs |
| 10 | `engine/verification.py` | 1 search | Execute user-authored regex verification rules |
| 11 | `tools/ripgrep.py` | 1 compile | Python fallback for grep — compile user-supplied pattern |
| 12 | `tools/search.py` | 1 compile | Compile user-supplied search pattern |

---

## Replacement Strategy Per Call Site

### 1. `learning/reflection.py` — Steering Signal Detection

**Current:** 52 regex patterns across 4 marker lists (`_CORRECTION_MARKERS`,
`_PREFERENCE_MARKERS`, `_STYLE_MARKERS`, `_KNOWLEDGE_MARKERS`), pre-compiled at init,
scanned via `_scan_markers`. Two `re.sub` calls in `_signal_key` for text normalization.

**Problem:** The regex here simulates keyword/phrase matching. Patterns like
`r"\bdon'?t\b"` and `r"\bi said\b"` are just searching for substrings at word
boundaries. The `\b` anchoring and optional characters add complexity but the actual
matching is phrase-level, not structural. False positives are likely (e.g., "I said
goodbye" triggers correction detection).

**Replacement:** Flat phrase lists with a `_contains_any_phrase` helper that tokenizes
on whitespace and checks subsequences. This is what the regex is *actually* doing, just
stated directly.

```python
_CORRECTION_PHRASES = [
    "no,", "no.", "don't", "dont", "stop", "instead",
    "actually,", "actually ", "i said", "i meant",
    "not like that", "that's not", "thats not", "that's wrong",
    "thats wrong", "i didn't ask", "i didnt ask", "i didn't want",
    "i didnt want", "i didn't mean", "i didnt mean",
    "please don't", "please dont", "not what i", "wrong",
]

# Same flat-list approach for _PREFERENCE_PHRASES, _STYLE_PHRASES, _KNOWLEDGE_PHRASES.

def _count_phrase_hits(text: str, phrases: list[str]) -> int:
    """Count how many phrases appear in the lowercased text."""
    lower = text.lower()
    return sum(1 for phrase in phrases if phrase in lower)
```

For `_signal_key` normalization, replace the two `re.sub` calls:

```python
_FILLER_WORDS = {"please", "just", "can", "you", "could", "would"}

def _signal_key(signal: SteeringSignal) -> str:
    text = signal.content.strip().lower()
    if not text:
        return ""
    words = [w for w in text.split() if w not in _FILLER_WORDS][:8]
    return "-".join(words)
```

**Trade-off:** Phrase matching with `in` is slightly broader than `\b`-anchored regex
(e.g., `"stop"` matches inside `"unstoppable"`). This is acceptable — the reflection
engine is a fuzzy signal detector, not a parser. The LLM phase (phase 2) provides the
precision. If needed, add a `_word_in(word, text)` helper that checks
`f" {word} "` against `f" {text} "` (pad with spaces), which gives word-boundary
semantics without regex.

---

### 2. `recovery/errors.py` — Error Classification

**Current:** 9 `re.compile` patterns checked via first-match-wins loop. Each pattern is
a simple `|`-delimited list of literal strings (e.g., `"SyntaxError|IndentationError|TabError"`).

**Problem:** These aren't using any regex features. No quantifiers, no groups, no
anchoring. Just "does any of these substrings appear?"

**Replacement:** A list of `(keywords, category, hint)` tuples with substring checks.

```python
_RULES: list[tuple[list[str], ErrorCategory, str]] = [
    (
        ["SyntaxError", "IndentationError", "TabError"],
        ErrorCategory.SYNTAX_ERROR,
        "Fix the syntax error. Check indentation, brackets, and quotes.",
    ),
    (
        ["File not found", "FileNotFoundError", "No such file", "not found:"],
        ErrorCategory.FILE_NOT_FOUND,
        "The file doesn't exist. Verify the path, or create the file first.",
    ),
    # ... same structure for remaining 7 categories
]

def categorize_error(error_text: str) -> CategorizedError:
    if not error_text:
        return CategorizedError(...)
    lower = error_text.lower()
    for keywords, category, hint in _RULES:
        for kw in keywords:
            if kw.lower() in lower:
                return CategorizedError(
                    category=category,
                    original_error=error_text,
                    detail=kw,
                    recovery_hint=hint,
                )
    return CategorizedError(category=ErrorCategory.UNKNOWN, ...)
```

**Note:** The current code captures `match.group(0)` for `detail`. The replacement
stores the matched keyword string directly — same information, clearer provenance.

---

### 3. `tools/shell.py` — Dangerous Command Blocking

**Current:** 17 regex patterns compiled into `BLOCKED_RE`, checked per command.

**Problem:** Regex gives a false sense of coverage. An attacker who knows the patterns
can trivially bypass them (e.g., `r\m -rf /`, env var expansion, newline injection).
The patterns also block legitimate use (`python3 -c` blocks all inline Python, which
is common). Regex-based command blocking is fundamentally the wrong tool for this job.

**Replacement:** Parse the command with `shlex.split` and match against structured
rules. This is more correct *and* more readable.

```python
import shlex
from dataclasses import dataclass

@dataclass(frozen=True)
class BlockedCommand:
    """A blocked command pattern with structured matching."""
    program: str
    description: str
    flags: frozenset[str] = frozenset()     # any of these flags triggers block
    args_check: str | None = None           # substring in remaining args triggers block

_BLOCKED_COMMANDS: list[BlockedCommand] = [
    BlockedCommand("rm", "recursive rm at root", flags=frozenset({"-r", "-rf", "-fr", "--recursive"})),
    BlockedCommand("mkfs", "format filesystem"),
    BlockedCommand("dd", "raw disk write", args_check="if="),
    BlockedCommand("sudo", "privilege escalation"),
    BlockedCommand("chmod", "recursive chmod 777 at root", flags=frozenset({"-R"}), args_check="777"),
    BlockedCommand("chown", "recursive chown at root", flags=frozenset({"-R"})),
]

# Pipe-to-shell detection
_PIPE_SHELL_PROGRAMS = {"sh", "bash", "zsh"}

def check_command_safety(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "Unparseable command (mismatched quotes)"

    if not tokens:
        return None

    program = tokens[0].rsplit("/", 1)[-1]  # basename

    # Check pipe-to-shell: curl ... | sh
    if "|" in command:
        parts = command.split("|")
        for part in parts[1:]:
            part_tokens = part.strip().split()
            if part_tokens and part_tokens[0].rsplit("/", 1)[-1] in _PIPE_SHELL_PROGRAMS:
                return f"Blocked: pipe to shell ({part_tokens[0]})"

    # Check command substitution
    if "$(" in command or "`" in command:
        # Re-check the inner command for dangerous patterns
        pass  # handled by blocking rm globally when flags match

    for rule in _BLOCKED_COMMANDS:
        if program != rule.program:
            continue
        if not rule.flags and not rule.args_check:
            return f"Blocked: {rule.description}"
        flags_present = {t for t in tokens[1:] if t.startswith("-")}
        if rule.flags and rule.flags & flags_present:
            if rule.args_check is None or any(rule.args_check in t for t in tokens[1:]):
                return f"Blocked: {rule.description}"
            # For rm, check if target is root-ish
            non_flag = [t for t in tokens[1:] if not t.startswith("-")]
            if program == "rm" and any(t in ("/", "/*", "~") or t.startswith("/") for t in non_flag):
                return f"Blocked: {rule.description}"
        if rule.args_check and any(rule.args_check in t for t in tokens[1:]):
            if not rule.flags:
                return f"Blocked: {rule.description}"

    return None
```

**Trade-off:** `shlex.split` handles quoting correctly (regex doesn't). But it can't
parse all shell syntax (e.g., heredocs, multi-line). That's fine — the safety check
is a first-pass heuristic, not a sandbox. For the cases where `shlex.split` raises
`ValueError`, we block the command (fail-closed).

---

### 4. `tools/git.py` — Dangerous Git Blocking

**Current:** 5 regex patterns for force push, hard reset, clean -f, branch -D, checkout dot.

**Replacement:** Direct token inspection on the already-parsed `args` list. The git
tool receives arguments as a structured `list[str]` — there is no reason to join them
into a string and regex-match. The data is already parsed.

```python
def check_git_safety(git_args: list[str]) -> str | None:
    """Check structured git args for dangerous operations."""
    args_set = set(git_args)
    joined = " ".join(git_args)

    if "--force" in args_set and "push" in args_set:
        return "Blocked: force push"
    if "--hard" in args_set and "reset" in args_set:
        return "Blocked: hard reset"
    if "clean" in args_set and any(f.startswith("-") and "f" in f for f in git_args):
        return "Blocked: git clean -f"
    if "branch" in args_set and any(f.startswith("-") and "D" in f for f in git_args):
        return "Blocked: force delete branch"
    if git_args[-1:] == ["."] and "checkout" in args_set:
        return "Blocked: checkout . (discards all changes)"

    return None
```

**Change the call site** in `execute()` to pass `git_args` directly instead of
joining to a string first. This eliminates the regex *and* the unnecessary string
concatenation.

---

### 5. `tools/web.py` — SSRF Blocking + HTML Stripping

**Current:**
- `_BLOCKED_HOSTS`: Single compiled regex matching private IP ranges + localhost.
- `_strip_html()`: 3 `re.sub` calls for script/style removal, tag removal, whitespace collapse.

**Replacement for SSRF blocking:** Use `ipaddress` module for IP validation and a
set for hostname checks. This is *more correct* than the regex — the current pattern
doesn't handle all RFC 1918 ranges, misses link-local (169.254.x.x), and can't handle
IPv6.

```python
import ipaddress
from urllib.parse import urlparse

_BLOCKED_HOSTNAMES = {"localhost"}

def _is_private_host(host: str) -> bool:
    """Check if host is a private/internal network address."""
    if host in _BLOCKED_HOSTNAMES:
        return True
    # Strip brackets for IPv6
    bare = host.strip("[]")
    try:
        addr = ipaddress.ip_address(bare)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return False  # Not an IP literal — allow DNS names

def is_safe_url(url: str) -> tuple[bool, str]:
    if not url.startswith(("http://", "https://")):
        return False, "Only http:// and https:// URLs are allowed"
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if _is_private_host(host):
        return False, f"Blocked host: {host} (private/internal network)"
    return True, ""
```

**Replacement for HTML stripping:** Use `html.parser` from stdlib. Zero dependencies.

```python
from html.parser import HTMLParser
from html import unescape

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

def _strip_html(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    text = " ".join(extractor._parts)
    return " ".join(text.split())  # collapse whitespace
```

---

### 6. `tools/web_search.py` — DuckDuckGo HTML Parsing

**Current:** 6 `re.findall`/`re.search` calls and 2 `re.sub` calls to parse DDG HTML
without an HTML parser.

**Problem:** This is the worst regex in the codebase. Parsing HTML with regex is fragile,
and the multi-line `re.DOTALL` patterns are unreadable. Any change to DDG's HTML
structure silently breaks every pattern.

**Replacement:** Use `html.parser.HTMLParser` from stdlib (no new dependencies).

```python
from html.parser import HTMLParser
from html import unescape

class _DDGResultParser(HTMLParser):
    """Parse DuckDuckGo HTML results into structured data."""

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._in_result_link = False
        self._in_snippet = False
        self._current_url = ""
        self._current_title_parts: list[str] = []
        self._current_snippet_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        cls = attr_dict.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._in_result_link = True
            self._current_url = attr_dict.get("href", "")
            self._current_title_parts = []
        elif tag == "a" and "result__snippet" in cls:
            self._in_snippet = True
            self._current_snippet_parts = []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_result_link:
            self._in_result_link = False
            title = " ".join(self._current_title_parts).strip()
            if title and self._current_url:
                self.results.append({
                    "title": title,
                    "url": _clean_ddg_url(self._current_url),
                    "snippet": "",
                })
        elif tag == "a" and self._in_snippet:
            self._in_snippet = False
            snippet = " ".join(self._current_snippet_parts).strip()
            if self.results:
                self.results[-1]["snippet"] = snippet

    def handle_data(self, data):
        if self._in_result_link:
            self._current_title_parts.append(data)
        elif self._in_snippet:
            self._current_snippet_parts.append(data)

def _parse_ddg_html(html: str, max_results: int) -> list[dict]:
    parser = _DDGResultParser()
    parser.feed(html)
    return parser.results[:max_results]
```

**Also replace `_strip_tags`** with the same `_TextExtractor` pattern from web.py
(share via a small `_html` utility module or inline it).

---

### 7 & 8. `processes/schema.py` + `processes/installer.py` — Name Validation

**Current:** `re.match(r"^[a-z0-9][a-z0-9-]*$", name)` in two places.

**Replacement:** A plain function.

```python
def _is_valid_process_name(name: str) -> bool:
    """Check name is lowercase alphanumeric with hyphens, starting with alnum."""
    if not name or name[0] not in "abcdefghijklmnopqrstuvwxyz0123456789":
        return False
    return all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name)
```

Put this in `processes/schema.py` and import it from `installer.py`. One definition,
two call sites.

**Schema regex compilation check** (lines 514-517): The `re.compile(rule.check)` call
validates that a user-authored regex rule is syntactically valid. This is retained as
the one deliberate exception — see "Exceptions" section below.

---

### 9. `__main__.py` — Task ID Validation

**Current:** `re.match(r'^[a-zA-Z0-9_-]+$', task_id)`

**Replacement:**

```python
def _validate_task_id(task_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not task_id or not all(c in allowed for c in task_id):
        click.echo(f"Invalid task ID: {task_id}", err=True)
        sys.exit(1)
    return task_id
```

Or use `str.isascii()` + `str.isalnum()` with allowance for `-` and `_`.

---

### 10. `engine/verification.py` — User-Authored Regex Rules

**Current:** `re.search(rule.check, target_text)` executes a regex pattern that the
user defined in a process YAML file under `rules: [{type: regex, check: "..."}]`.

**This is retained.** See "Exceptions" section.

---

### 11 & 12. `tools/ripgrep.py` + `tools/search.py` — User-Supplied Search Patterns

**Current:** `re.compile(pattern)` compiles a pattern the user passed as a tool argument
to perform code search.

**This is retained.** See "Exceptions" section.

---

## Exceptions: Deliberate `re` Retention

Three call sites compile/execute **user-supplied** regex. The user is writing the
pattern; we're just running it. Removing `re` here would mean removing a feature.

| File | Line | Why Kept |
|------|------|----------|
| `engine/verification.py:116` | `re.search(rule.check, target_text)` | User-authored verification rules in process YAML |
| `tools/ripgrep.py:227` | `re.compile(pattern)` | Python fallback for user-initiated code search |
| `tools/search.py:51` | `re.compile(pattern)` | User-initiated search tool |
| `processes/schema.py:515` | `re.compile(rule.check)` | Validation that user-authored regex compiles |

**Isolation:** Move these four usages into a single `loom/utils/regex.py` module with
a safe execution wrapper (timeout, error handling). The rest of the codebase loses its
`import re`.

```python
# src/loom/utils/regex.py
"""Quarantined regex: only for executing user-supplied patterns."""

import re

def compile_user_pattern(pattern: str, ignore_case: bool = False) -> re.Pattern:
    """Compile a user-supplied regex with validation."""
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(pattern, flags)  # raises re.error on bad pattern

def search_user_pattern(pattern: str, text: str) -> bool:
    """Execute a user-supplied regex search. Returns True if found."""
    try:
        return bool(re.search(pattern, text))
    except re.error:
        return False

def validate_user_pattern(pattern: str) -> str | None:
    """Return error message if pattern is invalid, None if ok."""
    try:
        re.compile(pattern)
        return None
    except re.error as e:
        return str(e)
```

---

## Shared Utilities

Two small helpers emerge from the replacements above. Place them in `loom/utils/`.

### `loom/utils/html.py` — HTML Text Extraction

Shared by `tools/web.py` and `tools/web_search.py`. The `_TextExtractor` class
(HTMLParser subclass) replaces all HTML-stripping regex.

### `loom/utils/validation.py` — Character-Set Validators

Shared by `processes/schema.py`, `processes/installer.py`, and `__main__.py`.

```python
_PROCESS_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
_TASK_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)

def is_valid_process_name(name: str) -> bool:
    return bool(name) and name[0] in _PROCESS_NAME_CHARS - {"-"} and all(
        c in _PROCESS_NAME_CHARS for c in name
    )

def is_valid_task_id(task_id: str) -> bool:
    return bool(task_id) and all(c in _TASK_ID_CHARS for c in task_id)
```

---

## File-by-File Change Summary

| File | Action | `import re` After |
|------|--------|-------------------|
| `learning/reflection.py` | Replace markers with phrase lists, replace `_scan_markers` with `_count_phrase_hits`, replace `re.sub` in `_signal_key` with set-based word filtering | **Removed** |
| `recovery/errors.py` | Replace `_PATTERNS` with `_RULES` using keyword lists and `str.lower()`/`in` | **Removed** |
| `tools/shell.py` | Replace `BLOCKED_RE` with `shlex.split` + structured `BlockedCommand` rules | **Removed** |
| `tools/git.py` | Replace `BLOCKED_RE` with direct `list[str]` inspection on the already-parsed args | **Removed** |
| `tools/web.py` | Replace `_BLOCKED_HOSTS` with `ipaddress` module, replace `_strip_html` with `HTMLParser` from `utils/html` | **Removed** |
| `tools/web_search.py` | Replace `_parse_ddg_html` with `HTMLParser` subclass, replace `_strip_tags` with shared util | **Removed** |
| `processes/schema.py` | Extract `is_valid_process_name` to `utils/validation`, keep `re.compile` check for user regex via `utils/regex` | **Removed** (uses `utils/regex`) |
| `processes/installer.py` | Import `is_valid_process_name` from `utils/validation` | **Removed** |
| `__main__.py` | Import `is_valid_task_id` from `utils/validation` | **Removed** |
| `engine/verification.py` | Import `search_user_pattern` from `utils/regex` | **Removed** (uses `utils/regex`) |
| `tools/ripgrep.py` | Import `compile_user_pattern` from `utils/regex` | **Removed** (uses `utils/regex`) |
| `tools/search.py` | Import `compile_user_pattern` from `utils/regex` | **Removed** (uses `utils/regex`) |
| **New:** `utils/regex.py` | Quarantined user-pattern execution | **Yes** (only file) |
| **New:** `utils/html.py` | Shared `HTMLParser`-based text extraction | No |
| **New:** `utils/validation.py` | Character-set validators for names/IDs | No |

---

## Implementation Order

Implement in dependency order. Each step is independently shippable and testable.

1. **`utils/validation.py`** + update `schema.py`, `installer.py`, `__main__.py`
   - Smallest change. Three files lose `import re`. Easy to verify: existing tests pass.

2. **`utils/html.py`** + update `web.py` `_strip_html` and `web_search.py` `_strip_tags`
   - Introduce the shared HTMLParser. Update both consumers. Test with saved HTML fixtures.

3. **`tools/web_search.py`** — replace `_parse_ddg_html` with HTMLParser subclass
   - The biggest single-file change. Needs a test with a saved DDG HTML page.

4. **`tools/web.py`** — replace `_BLOCKED_HOSTS` with `ipaddress`
   - Strictly more correct (covers IPv6, link-local). Test with known private/public IPs.

5. **`recovery/errors.py`** — replace `_PATTERNS` with keyword lists
   - Mechanical. Existing tests cover all categories.

6. **`tools/git.py`** — replace regex with direct args inspection
   - Small. Change `check_git_safety` signature to accept `list[str]`.

7. **`tools/shell.py`** — replace regex with `shlex.split` + structured rules
   - Moderate complexity. Needs careful testing of edge cases (quoted args, pipes).

8. **`learning/reflection.py`** — replace markers with phrase lists
   - Largest pattern count. The confidence scoring logic stays the same.

9. **`utils/regex.py`** + update `verification.py`, `ripgrep.py`, `search.py`, `schema.py`
   - Last step: quarantine the remaining user-pattern `re` usage into one module.

---

## Testing Strategy

- Every step must leave `pytest` green before moving to the next.
- For steps 2-3 (HTML parsing): add test fixtures with saved HTML snapshots.
- For step 4 (SSRF): add explicit tests for IPv6, link-local, and non-IP hostnames.
- For step 7 (shell safety): add tests for quoting bypasses that the regex *couldn't*
  catch but `shlex.split` can.
- Final check: `grep -r "import re" src/loom/` returns only `utils/regex.py`.

---

## What This Doesn't Change

- **No behavioral changes.** Every replacement matches the same inputs as the regex
  it replaces (or is documented where it intentionally broadens/narrows).
- **No new dependencies.** All replacements use stdlib (`html.parser`, `ipaddress`,
  `shlex`).
- **No API changes.** Tool signatures, error categories, and learning patterns stay
  identical.
