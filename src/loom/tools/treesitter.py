"""Tree-sitter integration for code analysis and structural matching.

Optional backend that replaces regex extractors with syntax-tree-based
extraction. Uses ``tree-sitter-language-pack`` when available, falling
back silently when it is not installed.

Two integration points:
1. Code structure extraction (classes, functions, imports) for analyze_code.
2. Structural candidate finding for edit_file fuzzy matching.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_AVAILABLE = False

try:
    import tree_sitter  # noqa: F401
    import tree_sitter_language_pack as _tslp  # noqa: F401

    _AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Return True if tree-sitter and language-pack are importable."""
    return _AVAILABLE


# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------

# Maps our internal language name -> tree-sitter-language-pack language key.
_TS_LANG_MAP: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "go": "go",
    "rust": "rust",
}


def _get_parser(language: str):
    """Return a tree-sitter Parser for *language*, or None."""
    ts_name = _TS_LANG_MAP.get(language)
    if ts_name is None or not _AVAILABLE:
        return None
    try:
        return _tslp.get_parser(ts_name)
    except Exception:
        return None


def _parse(source: str, language: str):
    """Parse *source* with tree-sitter and return the root node, or None."""
    parser = _get_parser(language)
    if parser is None:
        return None
    try:
        tree = parser.parse(source.encode("utf-8"))
        return tree.root_node
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase A: Code-structure extraction
# ---------------------------------------------------------------------------

def extract_with_treesitter(source: str, language: str):
    """Extract code structure from *source* using tree-sitter.

    Returns a ``CodeStructure`` (imported lazily to avoid circular imports)
    or *None* if tree-sitter is not available / parsing fails.
    """
    root = _parse(source, language)
    if root is None:
        return None

    from loom.tools.code_analysis import CodeStructure

    structure = CodeStructure(language=language)

    extractor = _EXTRACTORS.get(language)
    if extractor is None:
        return None

    extractor(root, structure)
    return structure


# -- Per-language extractors ------------------------------------------------

def _extract_python(root, structure) -> None:
    """Walk a Python tree and populate *structure*."""
    _python_walk_block(root, structure, top_level=True)


def _python_walk_block(node, structure, *, top_level: bool) -> None:
    """Recurse into a block of Python statements.

    Handles top-level definitions, decorated definitions, and compound
    statements (if/try/with/for) that may contain imports and definitions.
    """
    for child in node.children:
        ntype = child.type
        if ntype in ("import_statement", "import_from_statement",
                     "future_import_statement"):
            _python_import(child, structure)
        elif ntype == "class_definition":
            _python_class(child, structure)
        elif ntype == "function_definition":
            _python_function(child, structure, top_level=top_level)
        elif ntype == "decorated_definition":
            for sub in child.children:
                if sub.type == "class_definition":
                    _python_class(sub, structure)
                elif sub.type == "function_definition":
                    _python_function(sub, structure, top_level=top_level)
        elif ntype in ("if_statement", "try_statement", "with_statement",
                        "for_statement", "while_statement"):
            # Recurse into compound statement bodies (block children)
            _python_walk_compound(child, structure, top_level=top_level)


def _python_walk_compound(node, structure, *, top_level: bool) -> None:
    """Recurse into block children of compound statements (if/try/with/for)."""
    for child in node.children:
        if child.type == "block":
            _python_walk_block(child, structure, top_level=top_level)
        elif child.type in ("except_clause", "finally_clause",
                             "else_clause", "elif_clause"):
            for sub in child.children:
                if sub.type == "block":
                    _python_walk_block(sub, structure, top_level=top_level)


def _python_import(node, structure) -> None:
    if node.type == "import_statement":
        # import os / import os, sys
        for child in node.children:
            if child.type == "dotted_name":
                structure.imports.append(child.text.decode())
            elif child.type == "aliased_import":
                name_node = child.children[0] if child.children else None
                if name_node is not None:
                    structure.imports.append(name_node.text.decode())
    elif node.type == "future_import_statement":
        # from __future__ import X — the module is __future__
        structure.imports.append("__future__")
    elif node.type == "import_from_statement":
        # from X import Y — capture the module (X)
        for child in node.children:
            if child.type == "dotted_name":
                structure.imports.append(child.text.decode())
                break
            elif child.type == "relative_import":
                structure.imports.append(child.text.decode())
                break


def _python_class(node, structure) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        structure.classes.append(name_node.text.decode())
    # Extract public methods inside the class body
    body = node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            actual = child
            if child.type == "decorated_definition":
                for sub in child.children:
                    if sub.type == "function_definition":
                        actual = sub
                        break
                else:
                    continue
            if actual.type == "function_definition":
                _python_function(actual, structure, top_level=False)


def _python_function(node, structure, *, top_level: bool) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return
    name = name_node.text.decode()
    if top_level:
        structure.functions.append(name)
    elif not name.startswith("_"):
        # Method — skip private
        if name not in structure.functions:
            structure.functions.append(name)


def _extract_javascript(root, structure) -> None:
    """Walk a JavaScript / TypeScript tree and populate *structure*."""
    for node in root.children:
        ntype = node.type
        if ntype == "import_statement":
            _js_import(node, structure)
        elif ntype == "lexical_declaration":
            _js_lexical(node, structure)
        elif ntype == "class_declaration":
            _js_class(node, structure)
        elif ntype == "function_declaration":
            _js_function(node, structure)
        elif ntype == "export_statement":
            _js_export(node, structure)


def _js_import(node, structure) -> None:
    # import X from 'module' — the source is a string child
    for child in node.children:
        if child.type == "string":
            # Strip quotes
            text = child.text.decode().strip("'\"")
            structure.imports.append(text)
            break


def _js_lexical(node, structure) -> None:
    """Handle const/let/var declarations — detect arrow functions and require()."""
    for child in node.children:
        if child.type == "variable_declarator":
            _js_variable_declarator(child, structure)


def _js_variable_declarator(node, structure) -> None:
    name_node = node.child_by_field_name("name")
    value_node = node.child_by_field_name("value")
    if name_node is None or value_node is None:
        return
    name = name_node.text.decode()
    vtype = value_node.type
    # require() call → import
    if vtype == "call_expression":
        func = value_node.child_by_field_name("function")
        if func is not None and func.text == b"require":
            args = value_node.child_by_field_name("arguments")
            if args is not None:
                for arg in args.children:
                    if arg.type == "string":
                        structure.imports.append(arg.text.decode().strip("'\""))
                        return
    # Arrow function → function
    if vtype == "arrow_function":
        structure.functions.append(name)


def _js_class(node, structure) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        structure.classes.append(name_node.text.decode())


def _js_function(node, structure) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        structure.functions.append(name_node.text.decode())


def _js_export(node, structure) -> None:
    """Handle export statements: extract the exported name and delegate."""
    for child in node.children:
        ctype = child.type
        if ctype == "class_declaration":
            _js_class(child, structure)
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                structure.exports.append(name_node.text.decode())
        elif ctype == "function_declaration":
            _js_function(child, structure)
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                structure.exports.append(name_node.text.decode())
        elif ctype == "lexical_declaration":
            for sub in child.children:
                if sub.type == "variable_declarator":
                    name_node = sub.child_by_field_name("name")
                    if name_node is not None:
                        structure.exports.append(name_node.text.decode())
                    _js_variable_declarator(sub, structure)


def _extract_go(root, structure) -> None:
    """Walk a Go tree and populate *structure*."""
    for node in root.children:
        ntype = node.type
        if ntype == "import_declaration":
            _go_import(node, structure)
        elif ntype == "type_declaration":
            _go_type(node, structure)
        elif ntype == "function_declaration":
            _go_function(node, structure)
        elif ntype == "method_declaration":
            _go_method(node, structure)


def _go_import(node, structure) -> None:
    # import "fmt" or import ( "fmt"\n"os" )
    for child in node.children:
        if child.type == "import_spec":
            _go_import_spec(child, structure)
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    _go_import_spec(spec, structure)
        elif child.type == "interpreted_string_literal":
            text = child.text.decode().strip('"')
            structure.imports.append(text)


def _go_import_spec(node, structure) -> None:
    for child in node.children:
        if child.type == "interpreted_string_literal":
            text = child.text.decode().strip('"')
            structure.imports.append(text)
            break


def _go_type(node, structure) -> None:
    for child in node.children:
        if child.type == "type_spec":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                structure.classes.append(name_node.text.decode())


def _go_function(node, structure) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        structure.functions.append(name_node.text.decode())


def _go_method(node, structure) -> None:
    # Method receiver + name: the name is a field_identifier
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        structure.functions.append(name_node.text.decode())


def _extract_rust(root, structure) -> None:
    """Walk a Rust tree and populate *structure*."""
    for node in root.children:
        ntype = node.type
        if ntype == "use_declaration":
            _rust_use(node, structure)
        elif ntype == "struct_item":
            _rust_struct(node, structure)
        elif ntype == "enum_item":
            _rust_enum(node, structure)
        elif ntype == "function_item":
            _rust_function(node, structure)
        elif ntype == "impl_item":
            # Extract methods from impl blocks
            _rust_impl(node, structure)


def _rust_impl(node, structure) -> None:
    """Extract methods from a Rust impl block."""
    body = node.child_by_field_name("body")
    if body is None:
        return
    for child in body.children:
        if child.type == "function_item":
            _rust_function(child, structure)


def _rust_use(node, structure) -> None:
    for child in node.children:
        if child.type in ("scoped_identifier", "identifier", "use_as_clause",
                          "scoped_use_list", "use_wildcard"):
            structure.imports.append(child.text.decode())
            break


def _rust_struct(node, structure) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        structure.classes.append(name_node.text.decode())


def _rust_enum(node, structure) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        structure.classes.append(name_node.text.decode())


def _rust_function(node, structure) -> None:
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        structure.functions.append(name_node.text.decode())


_EXTRACTORS = {
    "python": _extract_python,
    "javascript": _extract_javascript,
    "typescript": _extract_javascript,
    "go": _extract_go,
    "rust": _extract_rust,
}


# ---------------------------------------------------------------------------
# Phase B: Structural candidate finding for edit_file
# ---------------------------------------------------------------------------

def find_structural_candidates(
    source: str,
    language: str,
) -> list[tuple[int, int]]:
    """Return **character-offset** ranges of top-level structural nodes.

    Each element is ``(start_char, end_char)`` for a function, class,
    method, struct, etc.  Used by ``EditFileTool`` to narrow fuzzy-match
    search to structural boundaries instead of a naive sliding window.

    Tree-sitter internally uses byte offsets; this function converts them
    to character offsets so callers can index directly into the Python str.

    Returns an empty list when tree-sitter is unavailable or parse fails.
    """
    root = _parse(source, language)
    if root is None:
        return []

    finder = _CANDIDATE_FINDERS.get(language)
    if finder is None:
        return []

    byte_ranges: list[tuple[int, int]] = []
    finder(root, byte_ranges)

    # Convert byte offsets → character offsets.
    # For pure ASCII this is a no-op, but for multi-byte UTF-8 we need the map.
    source_bytes = source.encode("utf-8")
    if len(source_bytes) == len(source):
        # Fast path: all ASCII, byte == char
        return byte_ranges

    b2c = _byte_to_char_map(source)
    return [(b2c[s], b2c[e]) for s, e in byte_ranges]


def _byte_to_char_map(source: str) -> list[int]:
    """Build a mapping from byte offset → character offset.

    Returns a list where ``result[byte_idx]`` is the character index
    in *source* that corresponds to that byte position in the UTF-8
    encoding.  The list has length ``len(source.encode('utf-8')) + 1``
    so that end-of-string byte offsets map correctly.
    """
    encoded = source.encode("utf-8")
    mapping = [0] * (len(encoded) + 1)
    byte_idx = 0
    for char_idx, ch in enumerate(source):
        ch_bytes = len(ch.encode("utf-8"))
        for b in range(ch_bytes):
            mapping[byte_idx + b] = char_idx
        byte_idx += ch_bytes
    mapping[byte_idx] = len(source)  # sentinel for end-of-string
    return mapping


def _python_candidates(root, ranges: list[tuple[int, int]]) -> None:
    _python_candidates_walk(root, ranges)


def _python_candidates_walk(node, ranges: list[tuple[int, int]]) -> None:
    for child in node.children:
        ntype = child.type
        if ntype in ("function_definition", "class_definition", "decorated_definition"):
            ranges.append((child.start_byte, child.end_byte))
            # Also add inner definitions for classes
            if ntype == "class_definition":
                _add_inner_defs(child, ranges)
            elif ntype == "decorated_definition":
                for sub in child.children:
                    if sub.type == "class_definition":
                        _add_inner_defs(sub, ranges)
        elif ntype in ("if_statement", "try_statement", "with_statement",
                        "for_statement", "while_statement"):
            # Recurse into compound statement bodies
            for sub in child.children:
                if sub.type == "block":
                    _python_candidates_walk(sub, ranges)
                elif sub.type in ("except_clause", "finally_clause",
                                   "else_clause", "elif_clause"):
                    for inner in sub.children:
                        if inner.type == "block":
                            _python_candidates_walk(inner, ranges)


def _add_inner_defs(class_node, ranges: list[tuple[int, int]]) -> None:
    """Add method-level nodes inside a class body."""
    body = class_node.child_by_field_name("body")
    if body is None:
        return
    for child in body.children:
        if child.type in ("function_definition", "decorated_definition"):
            ranges.append((child.start_byte, child.end_byte))


def _js_candidates(root, ranges: list[tuple[int, int]]) -> None:
    for node in root.children:
        ntype = node.type
        if ntype in ("function_declaration", "class_declaration",
                      "lexical_declaration", "export_statement"):
            ranges.append((node.start_byte, node.end_byte))


def _go_candidates(root, ranges: list[tuple[int, int]]) -> None:
    for node in root.children:
        ntype = node.type
        if ntype in ("function_declaration", "method_declaration", "type_declaration"):
            ranges.append((node.start_byte, node.end_byte))


def _rust_candidates(root, ranges: list[tuple[int, int]]) -> None:
    for node in root.children:
        ntype = node.type
        if ntype in ("function_item", "struct_item", "enum_item",
                      "impl_item", "trait_item"):
            ranges.append((node.start_byte, node.end_byte))


_CANDIDATE_FINDERS = {
    "python": _python_candidates,
    "javascript": _js_candidates,
    "typescript": _js_candidates,
    "go": _go_candidates,
    "rust": _rust_candidates,
}
