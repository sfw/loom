# Drop Regex from Learning System Signal Detection

## Problem Statement

`learning/reflection.py` uses 52 pre-compiled regex patterns across 4 marker lists
(`_CORRECTION_MARKERS`, `_PREFERENCE_MARKERS`, `_STYLE_MARKERS`, `_KNOWLEDGE_MARKERS`)
plus 2 `re.sub` calls in `_signal_key`. These patterns simulate phrase matching using
regex features that add complexity without value — `\b` word boundaries, `?` optional
characters, `(?:...)` non-capturing groups — when the actual job is "does this user
message contain any of these phrases?"

The regex is:
- **Hard to read.** `r"\bthat'?s (?:not |wrong)"` means "that's not" or "that's wrong"
  or "thats not" or "thats wrong". Four strings encoded as one opaque pattern.
- **Hard to extend.** Adding a new trigger phrase requires knowing regex syntax.
- **Falsely precise.** The `\b` boundaries suggest exactness, but the reflection engine
  is a fuzzy signal detector — Phase 2 (LLM extraction) provides the real precision.
  The rule-based phase just needs to flag "something steering-related is here."
- **Not actually using regex.** None of the 52 patterns use quantifiers on variable
  content, character classes, backreferences, or anything that plain string matching
  can't do. They are literal phrases with optional apostrophes.

**Scope:** Only `learning/reflection.py`. No other files.

---

## Current Code

```python
# 52 patterns like:
_CORRECTION_MARKERS = [
    r"\bno[,.]?\s",
    r"\bdon'?t\b",
    r"\bstop\b",
    r"\binstead\b",
    r"\bactually[,]?\s",
    r"\bi said\b",
    r"\bi meant\b",
    r"\bnot like that\b",
    r"\bthat'?s (?:not |wrong)",
    r"\bi didn'?t (?:ask|want|mean)",
    r"\bplease don'?t\b",
    r"\bnot what i\b",
    r"\bwrong\b.*\bright\b",
]

# Pre-compiled in __init__:
self._correction_re = [re.compile(p, re.IGNORECASE) for p in _CORRECTION_MARKERS]

# Scanned per message:
@staticmethod
def _scan_markers(text: str, patterns: list[re.Pattern]) -> int:
    return sum(1 for p in patterns if p.search(text))

# Two re.sub calls in _signal_key:
text = re.sub(r"\b(please|just|can you|could you|would you)\b", "", text)
text = re.sub(r"\s+", " ", text).strip()
```

---

## Replacement

### Marker Lists: Regex -> Plain Phrases

Replace each regex pattern with the literal phrase(s) it matches. Where a single regex
encodes multiple variants (apostrophe vs no apostrophe), expand them out. Readability
over compression.

```python
_CORRECTION_PHRASES: list[str] = [
    "no, ",
    "no. ",
    "no ",
    "don't",
    "dont",
    "stop",
    "instead",
    "actually, ",
    "actually ",
    "i said",
    "i meant",
    "not like that",
    "that's not",
    "thats not",
    "that's wrong",
    "thats wrong",
    "i didn't ask",
    "i didnt ask",
    "i didn't want",
    "i didnt want",
    "i didn't mean",
    "i didnt mean",
    "please don't",
    "please dont",
    "not what i",
    "wrong",
]

_PREFERENCE_PHRASES: list[str] = [
    "always",
    "never",
    "prefer",
    "i like",
    "i want",
    "from now on",
    "going forward",
    "in the future",
    "by default",
    "remember",
]

_STYLE_PHRASES: list[str] = [
    "be more",
    "less verbose",
    "less wordy",
    "less detailed",
    "more concise",
    "more brief",
    "more detail",
    "more specific",
    "too long",
    "too short",
    "too verbose",
    "too wordy",
    "too vague",
    "too detailed",
    "shorter",
    "longer",
    "simpler",
    "get to the point",
    "don't explain",
    "dont explain",
    "skip the explanation",
    "skip the intro",
    "skip the preamble",
]

_KNOWLEDGE_PHRASES: list[str] = [
    "we use",
    "in our",
    "our team",
    "our company",
    "our org",
    "our project",
    "our codebase",
    "our stack",
    "our api",
    "our database",
    "our workflow",
    "our convention",
    "our standard",
    "fyi",
    "just so you know",
    "for context",
    "for reference",
    "the way we",
    "we follow",
    "we adopt",
    "we run",
    "we deploy",
    "keep in mind",
    "keep that in mind",
    "keep this in mind",
]
```

### Scanner: Regex Search -> Substring `in`

```python
@staticmethod
def _count_phrase_hits(text: str, phrases: list[str]) -> int:
    """Count how many trigger phrases appear in the lowercased text."""
    lower = text.lower()
    return sum(1 for phrase in phrases if phrase in lower)
```

The `__init__` method no longer pre-compiles anything. The four `self._*_re` attributes
are removed. `_detect_signals_rule_based` calls `_count_phrase_hits` with the phrase
lists directly.

### `_signal_key`: `re.sub` -> Split/Filter

```python
_FILLER_WORDS = frozenset({"please", "just", "can", "you", "could", "would"})

@staticmethod
def _signal_key(signal: SteeringSignal) -> str:
    text = signal.content.strip().lower()
    if not text:
        return ""
    words = [w for w in text.split() if w not in _FILLER_WORDS][:8]
    return "-".join(words)
```

This replaces:
1. `re.sub(r"\b(please|just|can you|could you|would you)\b", "", text)` — the regex
   treats "can you" and "could you" as two-word units, but `text.split()` handles
   individual words. The effect is the same: both words get removed.
2. `re.sub(r"\s+", " ", text).strip()` — `" ".join(text.split())` does the same
   whitespace normalization.

### Remove `import re`

After these changes, `reflection.py` has zero `re` usage. Drop the import.

---

## Trade-offs

**Broader matching.** `"stop" in text` matches inside "unstoppable", while
`r"\bstop\b"` wouldn't. This is acceptable for three reasons:

1. The reflection engine is a fuzzy first-pass. Phase 2 (LLM extraction) provides
   precision. A false positive from the rule-based phase just means we pass it to
   the LLM for confirmation.
2. The confidence scores are already capped low (`min(0.7, 0.3 + score * 0.2)` for
   corrections). A single spurious phrase hit produces a 0.5 confidence signal — not
   enough to be stored (threshold is 0.3) or to dominate prompt injection.
3. The current regex has its own false positives. `r"\bwrong\b.*\bright\b"` matches
   "the wrong approach might be right" — a discussion, not a correction.

If tighter word boundary semantics are needed later, add a helper:

```python
def _phrase_in_text(phrase: str, text: str) -> bool:
    """Check if phrase appears in text with word-boundary-like semantics."""
    padded = f" {text} "
    return f" {phrase} " in padded or f" {phrase}," in padded or f" {phrase}." in padded
```

This covers the common delimiters (space, comma, period) without regex. But start
without it — the simpler `in` check is likely sufficient.

**Expanded variant coverage.** The regex `r"\bi didn'?t (?:ask|want|mean)"` matches
4 combinations. The phrase list has 6 entries (with/without apostrophe x 3 verbs).
More entries but each one is self-documenting. Adding a new variant is adding a string
to a list, not debugging a regex alternation.

---

## Changes Summary

| What | Before | After |
|------|--------|-------|
| `_CORRECTION_MARKERS` | 13 regex patterns | ~26 plain strings |
| `_PREFERENCE_MARKERS` | 13 regex patterns | ~10 plain strings |
| `_STYLE_MARKERS` | 11 regex patterns | ~23 plain strings |
| `_KNOWLEDGE_MARKERS` | 13 regex patterns | ~25 plain strings |
| `_scan_markers()` | Regex `.search()` loop | `str.__contains__` loop |
| `_signal_key()` | 2x `re.sub` | `str.split` + set filter |
| `__init__` | 4 list comprehensions compiling regex | Nothing (phrase lists are module-level constants) |
| `import re` | Yes | **Removed** |

One file changed. No new files. No behavioral changes to the rest of the system.

---

## Testing

- Existing `tests/` for the reflection engine must pass with no changes to test logic.
- Verify specifically that the test cases for each signal type still trigger detection.
- If any test relied on exact word-boundary behavior, adjust the test expectation or
  add the `_phrase_in_text` helper.
