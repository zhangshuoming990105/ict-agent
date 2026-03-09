"""Unit tests for edit_file tool and edit_diff utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ict_agent.tools import edit_file, execute_tool
from ict_agent.utils.edit_diff import (
    detect_line_ending,
    fuzzy_find_text,
    generate_diff_string,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)


class TestEditDiff:
    def test_normalize_to_lf(self) -> None:
        assert normalize_to_lf("a\r\nb") == "a\nb"
        assert normalize_to_lf("a\rb") == "a\nb"

    def test_detect_line_ending(self) -> None:
        assert detect_line_ending("a\nb") == "\n"
        assert detect_line_ending("a\r\nb") == "\r\n"
        assert detect_line_ending("") == "\n"

    def test_restore_line_endings(self) -> None:
        assert restore_line_endings("a\nb", "\n") == "a\nb"
        assert restore_line_endings("a\nb", "\r\n") == "a\r\nb"

    def test_strip_bom(self) -> None:
        bom, text = strip_bom("\uFEFFhello")
        assert bom == "\uFEFF"
        assert text == "hello"
        bom, text = strip_bom("hello")
        assert bom == ""
        assert text == "hello"

    def test_normalize_for_fuzzy_match(self) -> None:
        # Trailing whitespace
        assert normalize_for_fuzzy_match("a  \nb  ") == "a\nb"
        # Smart quotes
        assert normalize_for_fuzzy_match("\u2018x\u2019") == "'x'"
        assert normalize_for_fuzzy_match("\u201Cx\u201D") == '"x"'
        # Unicode dashes
        assert normalize_for_fuzzy_match("a\u2013b") == "a-b"

    def test_fuzzy_find_exact(self) -> None:
        content = "hello\nworld"
        result = fuzzy_find_text(content, "hello")
        assert result.found
        assert result.index == 0
        assert not result.used_fuzzy_match

    def test_fuzzy_find_fuzzy(self) -> None:
        # Smart quote in content: U+2019; user searches with ASCII '
        content = "hel\u2019lo\nworld"
        result = fuzzy_find_text(content, "hel'lo")
        assert result.found
        assert result.used_fuzzy_match

    def test_fuzzy_find_not_found(self) -> None:
        result = fuzzy_find_text("hello", "xyz")
        assert not result.found
        assert result.index == -1

    def test_generate_diff_string(self) -> None:
        old_c = "line1\nline2\nline3"
        new_c = "line1\nline2_modified\nline3"
        diff, first_changed = generate_diff_string(old_c, new_c)
        assert "line2" in diff and "line2_modified" in diff
        assert first_changed is not None
        assert first_changed >= 1


class TestEditFileTool:
    def test_edit_file_exact_match(self, tmp_path: Path) -> None:
        from ict_agent.tools import set_workspace_root

        set_workspace_root(tmp_path)
        f = tmp_path / "test.txt"
        f.write_text("hello\nworld\nfoo")

        result = execute_tool(
            "edit_file",
            '{"path": "test.txt", "old_text": "world", "new_text": "world_modified"}',
        )
        assert "Successfully" in result
        assert f.read_text() == "hello\nworld_modified\nfoo"

    def test_edit_file_fuzzy_match(self, tmp_path: Path) -> None:
        from ict_agent.tools import set_workspace_root

        set_workspace_root(tmp_path)
        f = tmp_path / "test.txt"
        # Smart quote U+2019; user will search with ASCII '
        f.write_text("hel\u2019lo\nworld\nfoo")

        result = execute_tool(
            "edit_file",
            '{"path": "test.txt", "old_text": "hel\'lo", "new_text": "hi"}',
        )
        assert "Successfully" in result
        assert "fuzzy" in result.lower()
        assert f.read_text() == "hi\nworld\nfoo"

    def test_edit_file_not_found(self, tmp_path: Path) -> None:
        from ict_agent.tools import set_workspace_root

        set_workspace_root(tmp_path)
        f = tmp_path / "test.txt"
        f.write_text("hello\nworld")

        result = execute_tool(
            "edit_file",
            '{"path": "test.txt", "old_text": "xyz", "new_text": "abc"}',
        )
        assert "Error" in result
        assert "could not find" in result.lower()
        assert f.read_text() == "hello\nworld"

    def test_edit_file_multiple_occurrences(self, tmp_path: Path) -> None:
        from ict_agent.tools import set_workspace_root

        set_workspace_root(tmp_path)
        f = tmp_path / "test.txt"
        f.write_text("foo\nfoo\n")

        result = execute_tool(
            "edit_file",
            '{"path": "test.txt", "old_text": "foo", "new_text": "bar"}',
        )
        assert "Error" in result
        assert "occurrences" in result.lower()
        assert f.read_text() == "foo\nfoo\n"
