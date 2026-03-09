"""Tests for chat(headless=True) — the unified batch/headless mode.

Uses a mock LLM client so no real API key is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from ict_agent.context import ContextManager
from ict_agent.runtime.agent_loop import (
    BatchTurnResult,
    _TurnOutcome,
    _run_single_turn,
    chat,
)
from ict_agent.tools import set_workspace_root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyLogger:
    def __init__(self):
        self.lines: list[str] = []

    def log(self, msg="", **kwargs):
        self.lines.append(msg)

    def is_live_session(self) -> bool:
        return False

    def print_user_prompt(self) -> None:
        pass

    def reset_style(self) -> None:
        pass

    def print_streaming(self, text: str) -> None:
        pass

    def end_streaming(self) -> None:
        pass


class DummyDomainAdapter:
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.task_dir = None
        self.history_prompt = ""
        self.task_prompt = ""
        self.task_context_source = ""
        self.system_prompt_override = None
        self.append_system_prompt = ""

    def compose_system_prompt(self) -> str:
        if self.system_prompt_override is not None:
            return self.system_prompt_override
        return f"You are a test agent. Workspace: {self.workspace_root}"

    def try_save_history(self, content, ctx, logger):
        pass


class DummyCommandRegistry:
    def dispatch(self, cmd, ctx):
        return False


def _make_usage(prompt=10, completion=5):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )


def _make_text_response(content: str):
    """Create a mock response with just text content (no tool calls)."""
    message = SimpleNamespace(
        tool_calls=None,
        content=content,
        model_dump=lambda: {"role": "assistant", "content": content, "tool_calls": None},
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=_make_usage(),
    )


def _make_tool_response(tool_name: str, args: dict):
    """Create a mock response with a tool call."""
    args_json = json.dumps(args)
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=tool_name, arguments=args_json),
    )
    message = SimpleNamespace(
        tool_calls=[tool_call],
        content="",
        model_dump=lambda: {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": tool_name, "arguments": args_json}}],
        },
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=_make_usage(),
    )


# ---------------------------------------------------------------------------
# Tests: _run_single_turn
# ---------------------------------------------------------------------------

class TestRunSingleTurn:
    """Unit tests for the extracted _run_single_turn helper."""

    def test_text_only_response(self, tmp_path: Path):
        """Non-streaming turn with a simple text reply."""
        set_workspace_root(tmp_path)
        ctx = ContextManager("system prompt")
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        from ict_agent.skills import load_skills
        skills = load_skills(skills_root)

        runtime_state = {
            "verbose": False, "no_truncate": False, "safe_shell": False,
            "recovery_cleanup": True, "skills": skills, "pinned_skills": set(),
            "active_skill_names": [], "active_tool_schemas": [],
            "active_skill_prompt": "", "task_dir": None,
            "preempt_shell_kill": False, "compact_client": None,
            "compact_model": None, "model": "test-model",
            "fork_result_queue": __import__("queue").Queue(),
            "fork_job_counter": 0, "fork_results": {}, "fork_threads": [],
        }

        mock_client = mock.MagicMock()
        logger = DummyLogger()

        with mock.patch(
            "ict_agent.runtime.agent_loop.request_model_response",
            return_value=_make_text_response("Hello from headless!"),
        ):
            outcome = _run_single_turn(
                client=mock_client, model="test-model", ctx=ctx,
                runtime_state=runtime_state, logger=logger,
                user_input="say hello",
                tool_schema_map={}, all_tool_schemas=[],
                max_agent_steps=10, no_truncate=False,
                recovery_cleanup=True, use_streaming=False,
                user_queue=None,
            )

        assert isinstance(outcome, _TurnOutcome)
        assert outcome.content == "Hello from headless!"
        assert outcome.steps == 1
        assert outcome.tool_called is False
        assert outcome.preempted is False

    def test_tool_then_text(self, tmp_path: Path):
        """Non-streaming turn: tool call then text reply."""
        set_workspace_root(tmp_path)
        ctx = ContextManager("system prompt")
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        from ict_agent.skills import load_skills
        skills = load_skills(skills_root)

        runtime_state = {
            "verbose": False, "no_truncate": False, "safe_shell": False,
            "recovery_cleanup": True, "skills": skills, "pinned_skills": set(),
            "active_skill_names": [], "active_tool_schemas": [],
            "active_skill_prompt": "", "task_dir": None,
            "preempt_shell_kill": False, "compact_client": None,
            "compact_model": None, "model": "test-model",
            "fork_result_queue": __import__("queue").Queue(),
            "fork_job_counter": 0, "fork_results": {}, "fork_threads": [],
        }

        mock_client = mock.MagicMock()
        logger = DummyLogger()

        # First call: tool, second call: text
        responses = [
            _make_tool_response("calculator", {"expression": "2+2"}),
            _make_text_response("The answer is 4."),
        ]

        with mock.patch(
            "ict_agent.runtime.agent_loop.request_model_response",
            side_effect=responses,
        ):
            outcome = _run_single_turn(
                client=mock_client, model="test-model", ctx=ctx,
                runtime_state=runtime_state, logger=logger,
                user_input="what is 2+2?",
                tool_schema_map={}, all_tool_schemas=[],
                max_agent_steps=10, no_truncate=False,
                recovery_cleanup=True, use_streaming=False,
                user_queue=None,
            )

        assert outcome.content == "The answer is 4."
        assert outcome.steps == 2
        assert outcome.tool_called is True


# ---------------------------------------------------------------------------
# Tests: chat(headless=True)
# ---------------------------------------------------------------------------

class TestChatHeadless:
    """Integration tests for chat() in headless mode."""

    def test_headless_returns_batch_result(self, tmp_path: Path):
        """chat(headless=True) should return a BatchTurnResult."""
        set_workspace_root(tmp_path)
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        adapter = DummyDomainAdapter(tmp_path)
        logger = DummyLogger()
        mock_client = mock.MagicMock()

        with mock.patch(
            "ict_agent.runtime.agent_loop.request_model_response",
            return_value=_make_text_response("Headless reply!"),
        ):
            result = chat(
                client=mock_client,
                model="test-model",
                max_tokens=4096,
                max_agent_steps=5,
                safe_shell=False,
                recovery_cleanup=True,
                preempt_shell_kill=False,
                initial_message="hello",
                compact_client=None,
                compact_model=None,
                logger=logger,
                command_registry=DummyCommandRegistry(),
                domain_adapter=adapter,
                skills_root=skills_root,
                headless=True,
            )

        assert isinstance(result, BatchTurnResult)
        assert result.assistant_content == "Headless reply!"
        assert result.steps == 1
        assert result.error is None

    def test_headless_empty_input(self, tmp_path: Path):
        """chat(headless=True) with empty input should return error."""
        set_workspace_root(tmp_path)
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        adapter = DummyDomainAdapter(tmp_path)
        logger = DummyLogger()
        mock_client = mock.MagicMock()

        result = chat(
            client=mock_client,
            model="test-model",
            max_tokens=4096,
            max_agent_steps=5,
            safe_shell=False,
            recovery_cleanup=True,
            preempt_shell_kill=False,
            initial_message="",
            compact_client=None,
            compact_model=None,
            logger=logger,
            command_registry=DummyCommandRegistry(),
            domain_adapter=adapter,
            skills_root=skills_root,
            headless=True,
        )

        assert isinstance(result, BatchTurnResult)
        assert result.error == "Empty user input"
        assert result.steps == 0

    def test_headless_tool_execution(self, tmp_path: Path):
        """chat(headless=True) executes tools and returns result."""
        set_workspace_root(tmp_path)
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        adapter = DummyDomainAdapter(tmp_path)
        logger = DummyLogger()
        mock_client = mock.MagicMock()

        responses = [
            _make_tool_response("calculator", {"expression": "3*7"}),
            _make_text_response("3 times 7 is 21."),
        ]

        with mock.patch(
            "ict_agent.runtime.agent_loop.request_model_response",
            side_effect=responses,
        ):
            result = chat(
                client=mock_client,
                model="test-model",
                max_tokens=4096,
                max_agent_steps=10,
                safe_shell=False,
                recovery_cleanup=True,
                preempt_shell_kill=False,
                initial_message="what is 3*7?",
                compact_client=None,
                compact_model=None,
                logger=logger,
                command_registry=DummyCommandRegistry(),
                domain_adapter=adapter,
                skills_root=skills_root,
                headless=True,
            )

        assert isinstance(result, BatchTurnResult)
        assert result.assistant_content == "3 times 7 is 21."
        assert result.tool_called is True
        assert result.steps == 2
        assert result.ctx_messages is not None
        # Verify calculator was called in the context
        tool_msgs = [m for m in result.ctx_messages if m.get("role") == "tool"]
        assert any("3*7 = 21" in m.get("content", "") for m in tool_msgs)

    def test_headless_max_steps_reached(self, tmp_path: Path):
        """chat(headless=True) with max_agent_steps=1 and tool call should hit step limit."""
        set_workspace_root(tmp_path)
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        adapter = DummyDomainAdapter(tmp_path)
        logger = DummyLogger()
        mock_client = mock.MagicMock()

        # Only return tool calls — never a text response
        with mock.patch(
            "ict_agent.runtime.agent_loop.request_model_response",
            return_value=_make_tool_response("calculator", {"expression": "1+1"}),
        ):
            result = chat(
                client=mock_client,
                model="test-model",
                max_tokens=4096,
                max_agent_steps=2,
                safe_shell=False,
                recovery_cleanup=True,
                preempt_shell_kill=False,
                initial_message="keep calculating",
                compact_client=None,
                compact_model=None,
                logger=logger,
                command_registry=DummyCommandRegistry(),
                domain_adapter=adapter,
                skills_root=skills_root,
                headless=True,
            )

        assert isinstance(result, BatchTurnResult)
        assert "max autonomous step limit" in result.assistant_content
