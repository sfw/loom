# Tree-Sitter Integration Plan

Decisions made during design discussion. This document is the source of truth
for tree-sitter scope, approach, and constraints.

---

## Core Decision

Use tree-sitter as a **better extraction/matching backend**, not as real-time
editing infrastructure. No IDE-style incremental parsing pipeline — that's
overkill for an orchestration engine.

## Package

[`tree-sitter-language-pack`](https://pypi.org/project/tree-sitter-language-pack/)
— bundles pre-built grammars for all common languages. Single dependency,
no per-language grammar compilation.

## Two Integration Points

### 1. Replace `analyze_code` regex extractors

**File:** `src/loom/tools/code_analysis.py`

**Current state:** Six regex extractors (Python, JS, TS, Go, Rust) with ~100
lines of fragile per-language patterns. Misses edge cases: decorators, nested
classes, conditional imports, multiline signatures, etc.

**Change:** Replace regex extractors with tree-sitter S-expression queries for
symbol extraction. One query per language for classes/functions/imports, using
tree-sitter's native node types.

**Benefits:**
- Accurate, language-agnostic code structure
- Zero per-language regex maintenance
- Works correctly on syntactically complex code
- Trivial to add new languages (just a query file)

**Fallback:** Keep regex extractors as fallback when tree-sitter is not
installed (optional dependency). The `analyze_file()` function checks for
tree-sitter availability at call time.

**Interface stays the same:** `CodeStructure` dataclass, `analyze_file()`,
`analyze_directory()`, and the `AnalyzeCodeTool` API are unchanged. This is
purely a backend swap.

### 2. Enhance `edit_file` structural matching

**File:** `src/loom/tools/file_ops.py` (class `EditFileTool`, line 475+)

**Current state:** Fuzzy matching uses `difflib.SequenceMatcher` on
whitespace-normalized lines. Sliding window over line ranges. Threshold 0.85.
Works reasonably well but struggles when local models drift on:
- Indentation level (tabs vs spaces, wrong nesting depth)
- Blank line placement
- Decorator/attribute ordering

**Change:** When tree-sitter is available and the file is a supported language,
parse the file and anchor edit candidates to structural nodes (functions,
classes, blocks). Instead of sliding a line window over the entire file:

1. Parse the file into a tree-sitter tree
2. Parse `old_str` into a tree (or extract its structural signature)
3. Find candidate nodes in the file tree that match the structural signature
4. Within matched nodes, apply the existing fuzzy string matching

This narrows the search space and makes matching resilient to whitespace drift
because the structural boundaries are syntax-aware, not line-count-aware.

**Fallback:** Current `_fuzzy_find` logic remains the fallback when:
- tree-sitter not installed
- File language not supported by tree-sitter
- tree-sitter parse fails (e.g., severely broken syntax)

**No interface change.** `edit_file` tool parameters and behavior stay the
same. The improvement is internal matching quality.

---

## What We Are NOT Doing

- **No incremental parsing.** Files are parsed once per tool call, not
  maintained across calls.
- **No LSP integration.** No hover, go-to-definition, or diagnostics.
- **No type analysis.** Tree-sitter is syntax-only. No type inference or
  cross-file resolution.
- **No call graph construction.** Symbol extraction only.
- **No real-time TUI integration.** Tree-sitter runs inside tool execution,
  not in the UI layer.

---

## Dependency Strategy

`tree-sitter-language-pack` is an **optional dependency**:

```toml
# pyproject.toml
[project.optional-dependencies]
treesitter = ["tree-sitter-language-pack>=0.3"]
```

Install with `pip install loom[treesitter]`. All tree-sitter code paths
check availability at runtime and fall back to regex/difflib.

---

## Implementation Order

### Phase A: analyze_code backend swap ✓ DONE
1. Add `tree-sitter-language-pack` to optional deps ✓
2. Tree-sitter tree-walking extractors for Python, JS/TS, Go, Rust ✓
3. `extract_with_treesitter(source, language) -> CodeStructure` in `tools/treesitter.py` ✓
4. Wired into `analyze_file()` with try/fallback to regex ✓
5. Tests: 38 new tree-sitter tests + all 1272 existing tests pass ✓

### Phase B: edit_file structural anchoring ✓ DONE
1. `find_structural_candidates(source, language)` returns node byte ranges ✓
2. `_structural_fuzzy_find()` narrows search to syntax boundaries ✓
3. `_fuzzy_find()` tries structural candidates first, falls back to sliding window ✓
4. Tests: indentation-drift, decorator-reorder, and non-source fallback cases ✓

### Phase C: Language expansion (optional, later)
- Add queries for Java, C/C++, Ruby, PHP, Kotlin, Swift
- Each language is just an extractor function + entry in the language map

---

## Relation to Other Plans

- **PLAN.md Phase 4a** references this work. Update status when done.
- **FOUNDATION-REFACTOR.md** is independent (code quality). No conflicts.
- **PLAN.md Phase 4b** (planner context enhancement) builds on Phase A here —
  better `analyze_code` output means better planner context automatically.
