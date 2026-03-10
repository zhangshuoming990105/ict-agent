"""Unit tests for CLI system prompt and input customization."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import after potential path setup
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_system_prompt_file_and_system_prompt_mutually_exclusive() -> None:
    """--system-prompt and --system-prompt-file cannot be used together."""
    from ict_agent.app.cli import main

    with patch.object(sys, "argv", ["main.py", "--system-prompt", "x", "--system-prompt-file", "y"]):
        result = main()
    assert result == 1


def test_append_system_prompt_applied_to_adapter() -> None:
    """--append-system-prompt sets domain_adapter.append_system_prompt."""
    from ict_agent.app.bootstrap import create_domain_adapter
    from ict_agent.domains.cuda.adapter import CudaDomainAdapter

    adapter = create_domain_adapter(ROOT)
    adapter.append_system_prompt = "Always use TypeScript."
    prompt = adapter.compose_system_prompt()
    assert "Always use TypeScript." in prompt
    assert "general-purpose coding" in prompt


def test_system_prompt_override_replaces_default() -> None:
    """--system-prompt-file override replaces entire default prompt."""
    from ict_agent.app.bootstrap import create_domain_adapter

    adapter = create_domain_adapter(ROOT)
    adapter.system_prompt_override = "You are a Python expert. That is all."
    prompt = adapter.compose_system_prompt()
    assert prompt == "You are a Python expert. That is all."
    assert "general-purpose" not in prompt


def test_resolve_initial_message_priority() -> None:
    """Initial message resolution: --input > --input-file > positional > task > stdin."""
    from ict_agent.app.cli import _resolve_initial_message

    class Args:
        input = None
        input_file = None
        input_positional = None

    args = Args()

    # --input wins
    args.input = "from flag"
    assert _resolve_initial_message("from task", args) == "from flag"

    # --input-file with missing file raises
    args.input = None
    args.input_file = "/nonexistent/path/12345"
    args.input_positional = None
    with pytest.raises(FileNotFoundError):
        _resolve_initial_message("from task", args)

    # positional
    args.input_file = None
    args.input_positional = "from positional"
    assert _resolve_initial_message(None, args) == "from positional"

    # task
    args.input_positional = None
    assert _resolve_initial_message("from task", args) == "from task"

    # None when nothing set
    assert _resolve_initial_message(None, args) is None
