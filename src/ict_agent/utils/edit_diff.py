"""Edit diff utilities for surgical file edits.

Ported from pi-mono packages/coding-agent/src/core/tools/edit-diff.ts.
Supports exact and fuzzy text matching for oldText/newText replacement.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


# Unicode character classes for normalization (from Pi)
_SMART_SINGLE_QUOTES = "\u2018\u2019\u201A\u201B"  # ' ' ‚ ‛
_SMART_DOUBLE_QUOTES = "\u201C\u201D\u201E\u201F"  # " " „ ‟
_UNICODE_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"  # hyphen, nbsp hyphen, figure dash, en, em, bar, minus
_SPECIAL_SPACES = "\u00A0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000"  # NBSP, various spaces


def detect_line_ending(content: str) -> str:
    """Detect line ending: '\\r\\n' or '\\n'."""
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    """Normalize all line endings to \\n."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    """Restore line endings to the original format."""
    if ending == "\r\n":
        return text.replace("\n", "\r\n")
    return text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching.

    - Strip trailing whitespace from each line
    - Smart quotes → ASCII
    - Unicode dashes/hyphens → ASCII hyphen
    - Special spaces → regular space
    """
    lines = text.split("\n")
    normalized = "\n".join(line.rstrip() for line in lines)
    # Smart single quotes → '
    for c in _SMART_SINGLE_QUOTES:
        normalized = normalized.replace(c, "'")
    # Smart double quotes → "
    for c in _SMART_DOUBLE_QUOTES:
        normalized = normalized.replace(c, '"')
    # Unicode dashes → -
    for c in _UNICODE_DASHES:
        normalized = normalized.replace(c, "-")
    # Special spaces → regular space
    for c in _SPECIAL_SPACES:
        normalized = normalized.replace(c, " ")
    return normalized


@dataclass
class FuzzyMatchResult:
    """Result of fuzzy text search."""

    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find old_text in content: exact match first, then fuzzy match.

    When fuzzy matching is used, content_for_replacement is the normalized
    version of content (used for the replacement operation).
    """
    # Try exact match first
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    # Try fuzzy match in normalized space
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)

    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def strip_bom(content: str) -> tuple[str, str]:
    """Strip UTF-8 BOM if present. Returns (bom, text_without_bom)."""
    if content.startswith("\uFEFF"):
        return "\uFEFF", content[1:]
    return "", content


def generate_diff_string(
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> tuple[str, int | None]:
    """Generate a unified diff string and first changed line number.

    Returns (diff_string, first_changed_line) where first_changed_line is
    1-indexed in the new file, or None if no changes.
    """
    old_list = old_content.splitlines(keepends=True)
    new_list = new_content.splitlines(keepends=True)
    if not old_list and not new_list:
        return "", None

    diff_gen = difflib.unified_diff(
        old_list,
        new_list,
        fromfile="a",
        tofile="b",
        lineterm="",
        n=context_lines,
    )
    diff_lines = list(diff_gen)
    if not diff_lines:
        return "", None

    # Find first changed line in new file from hunk header
    first_changed: int | None = None
    for line in diff_lines:
        if line.startswith("@@"):
            m = re.search(r"\+(\d+),?", line)
            if m:
                first_changed = int(m.group(1))
                break

    return "\n".join(diff_lines), first_changed
