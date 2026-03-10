"""Tests for the ict-agent enhancement features (Tasks 1-4)."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Task 1: Workspace = cwd
# ---------------------------------------------------------------------------


class TestWorkspaceCwd:
    """Verify that workspace defaults to cwd, not the ict-agent source tree."""

    def test_workspace_root_fallback_is_cwd(self):
        """_workspace_root() with no custom root should return cwd, not source dir."""
        from ict_agent.tools import _workspace_root, set_workspace_root, _CUSTOM_WORKSPACE_ROOT

        # Save and clear custom root
        import ict_agent.tools as tools_mod
        original = tools_mod._CUSTOM_WORKSPACE_ROOT
        try:
            tools_mod._CUSTOM_WORKSPACE_ROOT = None
            result = _workspace_root()
            assert result == Path.cwd(), (
                f"Expected {Path.cwd()}, got {result}. "
                "Workspace should default to cwd, not the package source directory."
            )
        finally:
            tools_mod._CUSTOM_WORKSPACE_ROOT = original

    def test_set_workspace_root_overrides_cwd(self):
        """set_workspace_root() should override the cwd default."""
        from ict_agent.tools import set_workspace_root, _workspace_root
        import ict_agent.tools as tools_mod

        original = tools_mod._CUSTOM_WORKSPACE_ROOT
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                set_workspace_root(tmpdir)
                result = _workspace_root()
                assert result == Path(tmpdir).resolve()
        finally:
            tools_mod._CUSTOM_WORKSPACE_ROOT = original

    def test_cli_no_task_uses_cwd_or_workdir(self):
        """cli.py: when no --task, workspace should use args.workdir or os.getcwd()."""
        # We test the logic by reading the source — the actual behavior is:
        # workspace = Path(args.workdir or os.getcwd()).resolve()
        from ict_agent.app.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([])
        # Default: no task, no workdir
        assert args.task is None
        assert args.workdir is None
        # With --workdir
        args2 = parser.parse_args(["--workdir", "/tmp/mywork"])
        assert args2.workdir == "/tmp/mywork"


# ---------------------------------------------------------------------------
# Task 2: Large output persistence
# ---------------------------------------------------------------------------


class TestLargeOutputPersistence:
    """Verify that large tool outputs are persisted to disk."""

    def test_small_output_not_persisted(self):
        """Output below threshold should be returned unchanged."""
        from ict_agent.runtime.agent_loop import _maybe_persist_large_output, LARGE_OUTPUT_THRESHOLD

        class DummyLog:
            def log(self, *a, **k): pass

        small = "x" * (LARGE_OUTPUT_THRESHOLD - 1)
        result = _maybe_persist_large_output(small, "test_tool", DummyLog())
        assert result == small, "Small output should pass through unchanged"

    def test_large_output_persisted_to_file(self):
        """Output above threshold should be saved to a temp file."""
        from ict_agent.runtime.agent_loop import _maybe_persist_large_output, LARGE_OUTPUT_THRESHOLD

        log_lines = []
        class DummyLog:
            def log(self, msg="", **k):
                log_lines.append(msg)

        large = "A" * (LARGE_OUTPUT_THRESHOLD + 100)
        result = _maybe_persist_large_output(large, "run_shell", DummyLog())

        # Result should be a compact reference, not the full output
        assert len(result) < len(large), "Persisted result should be shorter than original"
        assert "[Output too large" in result
        assert "saved to" in result

        # The file should exist (workspace-relative path: .tool_outputs/xxx.txt)
        import re
        match = re.search(r"saved to (\S+\.txt)", result)
        assert match, f"Could not find file path in result: {result}"
        rel_path = match.group(1)
        assert "read_file" in result, "Should hint to use read_file"

        from ict_agent.tools import _workspace_root
        full_path = _workspace_root() / rel_path
        assert full_path.is_file(), f"Persisted file should exist: {full_path}"
        content = full_path.read_text()
        assert content == large, "Persisted file should contain full output"

        assert any("[large-output]" in line for line in log_lines)
        full_path.unlink()

    def test_large_output_contains_head_and_tail(self):
        """The persisted reference should contain first and last N chars."""
        from ict_agent.runtime.agent_loop import _maybe_persist_large_output, LARGE_OUTPUT_THRESHOLD

        class DummyLog:
            def log(self, *a, **k): pass

        # Create output with recognizable head and tail
        head = "HEAD_MARKER_" + "x" * 500
        tail = "y" * 500 + "_TAIL_MARKER"
        middle = "m" * (LARGE_OUTPUT_THRESHOLD + 100 - len(head) - len(tail))
        large = head + middle + tail
        result = _maybe_persist_large_output(large, "test", DummyLog())

        assert "HEAD_MARKER_" in result, "Result should contain head of original"
        assert "_TAIL_MARKER" in result, "Result should contain tail of original"

        # Cleanup
        import re
        match = re.search(r"saved to (\S+\.txt)", result)
        if match:
            try:
                from ict_agent.tools import _workspace_root
                (_workspace_root() / match.group(1)).unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Task 3: max_tokens control
# ---------------------------------------------------------------------------


class TestMaxTokensControl:
    """Verify that max_tokens constants exist and start_async_model_call accepts it."""

    def test_max_tokens_constants_exist(self):
        from ict_agent.runtime.agent_loop import MAX_TOKENS_TOOL_TURN, MAX_TOKENS_FINAL_TURN
        assert MAX_TOKENS_TOOL_TURN < MAX_TOKENS_FINAL_TURN, (
            "Tool turn max_tokens should be smaller than final turn"
        )
        assert MAX_TOKENS_TOOL_TURN > 0
        assert MAX_TOKENS_FINAL_TURN > 0

    def test_start_async_model_call_accepts_max_tokens(self):
        """start_async_model_call should accept a max_tokens parameter."""
        import inspect
        from ict_agent.runtime.agent_loop import start_async_model_call
        sig = inspect.signature(start_async_model_call)
        assert "max_tokens" in sig.parameters, (
            "start_async_model_call should have a max_tokens parameter"
        )

    def test_start_async_model_call_passes_max_tokens_to_api(self):
        """When max_tokens is given, it should be passed to the API call."""
        from ict_agent.runtime.agent_loop import start_async_model_call

        captured_kwargs = {}
        class FakeResponse:
            class Choice:
                class Message:
                    content = "test"
                    tool_calls = None
                message = Message()
            choices = [Choice()]
            usage = type("Usage", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()

        class FakeChat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    captured_kwargs.update(kwargs)
                    return FakeResponse()

        fake_client = type("FakeClient", (), {"chat": FakeChat()})()

        call = start_async_model_call(
            client=fake_client,
            model="test-model",
            request_messages=[{"role": "user", "content": "hi"}],
            active_tool_schemas=[],
            max_tokens=2048,
        )
        call.done.wait(timeout=5)
        assert call.error is None, f"Unexpected error: {call.error}"
        assert captured_kwargs.get("max_tokens") == 2048, (
            f"Expected max_tokens=2048, got {captured_kwargs.get('max_tokens')}"
        )

    def test_start_async_model_call_no_max_tokens_when_none(self):
        """When max_tokens is None, it should NOT be in the API call."""
        from ict_agent.runtime.agent_loop import start_async_model_call

        captured_kwargs = {}
        class FakeResponse:
            class Choice:
                class Message:
                    content = "test"
                    tool_calls = None
                message = Message()
            choices = [Choice()]
            usage = type("Usage", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()

        class FakeChat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    captured_kwargs.update(kwargs)
                    return FakeResponse()

        fake_client = type("FakeClient", (), {"chat": FakeChat()})()

        call = start_async_model_call(
            client=fake_client,
            model="test-model",
            request_messages=[{"role": "user", "content": "hi"}],
            active_tool_schemas=[],
            max_tokens=None,
        )
        call.done.wait(timeout=5)
        assert "max_tokens" not in captured_kwargs, (
            "max_tokens should not be passed when it is None"
        )


# ---------------------------------------------------------------------------
# Task 4: Dynamic tool schema
# ---------------------------------------------------------------------------


class TestDynamicToolSchema:
    """Verify that tool schemas are pruned when skills don't specify tools."""

    def test_core_tools_constant_exists(self):
        from ict_agent.runtime.agent_loop import CORE_TOOLS
        assert isinstance(CORE_TOOLS, set)
        assert "read_file" in CORE_TOOLS
        assert "write_file" in CORE_TOOLS
        assert "run_shell" in CORE_TOOLS

    def test_no_skill_tools_returns_core_only(self):
        """When skills have empty tool lists, only core tools should be returned."""
        from ict_agent.runtime.agent_loop import resolve_active_tool_schemas, CORE_TOOLS
        from ict_agent.skills import SkillSpec

        skill = SkillSpec(
            name="test", description="", tools=[], triggers=[],
            always_on=False, instructions="", context_mode="inline",
        )
        # Build a fake tool_schema_map with both core and non-core tools
        all_names = list(CORE_TOOLS) + ["fork_subagent", "calculator", "get_current_time"]
        tool_schema_map = {
            name: {"function": {"name": name}} for name in all_names
        }
        all_schemas = list(tool_schema_map.values())

        result = resolve_active_tool_schemas([skill], tool_schema_map, all_schemas)
        result_names = {s["function"]["name"] for s in result}

        # Should only contain core tools
        assert result_names == {n for n in CORE_TOOLS if n in tool_schema_map}, (
            f"Expected only core tools, got {result_names}"
        )
        assert "fork_subagent" not in result_names
        assert "calculator" not in result_names

    def test_skill_with_tools_gets_core_plus_specified(self):
        """When a skill specifies tools, result should be core + specified."""
        from ict_agent.runtime.agent_loop import resolve_active_tool_schemas, CORE_TOOLS
        from ict_agent.skills import SkillSpec

        skill = SkillSpec(
            name="qa", description="", tools=["calculator", "get_current_time"],
            triggers=[], always_on=False, instructions="", context_mode="inline",
        )
        all_names = list(CORE_TOOLS) + ["fork_subagent", "calculator", "get_current_time"]
        tool_schema_map = {name: {"function": {"name": name}} for name in all_names}
        all_schemas = list(tool_schema_map.values())

        result = resolve_active_tool_schemas([skill], tool_schema_map, all_schemas)
        result_names = {s["function"]["name"] for s in result}

        assert "calculator" in result_names
        assert "get_current_time" in result_names
        assert "read_file" in result_names  # core tool
        assert "fork_subagent" not in result_names  # not specified by skill


# ---------------------------------------------------------------------------
# Task 6: Sandbox enhancements
# ---------------------------------------------------------------------------


class TestSandboxSoftEnhancements:
    """Test safe commands, banned commands, and wildcard matching."""

    def test_safe_commands_recognized(self):
        from ict_agent.tools import _is_safe_command
        assert _is_safe_command("git status") is True
        assert _is_safe_command("pwd") is True
        assert _is_safe_command("date") is True
        assert _is_safe_command("rm -rf /tmp/foo") is False

    def test_banned_command_detected(self):
        from ict_agent.tools import _is_banned_command
        assert _is_banned_command("rm -rf /") is True
        assert _is_banned_command("rm -rf /home") is False  # not root /
        assert _is_banned_command("mkfs.ext4 /dev/sda1") is True
        assert _is_banned_command("dd if=/dev/zero of=/dev/sda") is True
        assert _is_banned_command("shutdown -h now") is True
        assert _is_banned_command("git push") is False

    def test_wildcard_allowlist_matching(self):
        from ict_agent.tools import _matches_allowlist
        allowlist = {"git *", "python test.py", "npm *"}
        assert _matches_allowlist("git status", allowlist) is True
        assert _matches_allowlist("git push origin main", allowlist) is True
        assert _matches_allowlist("git", allowlist) is True  # "git *" matches "git" itself
        assert _matches_allowlist("python test.py", allowlist) is True  # exact match
        assert _matches_allowlist("python other.py", allowlist) is False
        assert _matches_allowlist("npm install", allowlist) is True
        assert _matches_allowlist("curl http://evil.com", allowlist) is False

    def test_wildcard_denylist_matching(self):
        from ict_agent.tools import _matches_denylist
        denylist = {"curl *", "wget *"}
        assert _matches_denylist("curl http://example.com", denylist) is True
        assert _matches_denylist("wget http://example.com", denylist) is True
        assert _matches_denylist("git push", denylist) is False


class TestSandboxModule:
    """Test the sandbox module itself."""

    def test_sandbox_backend_returns_string(self):
        from ict_agent.sandbox import sandbox_backend
        result = sandbox_backend()
        assert result in ("bubblewrap", "seatbelt", "none")

    def test_build_sandboxed_command_fallback(self):
        """When no sandbox tools available, should return plain bash command."""
        from ict_agent.sandbox import build_sandboxed_command
        import unittest.mock as mock
        with mock.patch("ict_agent.sandbox.shutil.which", return_value=None):
            result = build_sandboxed_command("echo hello", "/tmp")
            assert result == ["bash", "-c", "echo hello"]

    def test_seatbelt_profile_generation(self):
        from ict_agent.sandbox import _generate_seatbelt_profile
        profile = _generate_seatbelt_profile("/workspace/project")
        assert "/workspace/project" in profile
        assert "deny file-write*" in profile
        assert "/tmp" in profile

    def test_sandbox_flag_toggle(self):
        from ict_agent.tools import set_sandbox_enabled, is_sandbox_enabled
        original = is_sandbox_enabled()
        try:
            set_sandbox_enabled(True)
            assert is_sandbox_enabled() is True
            set_sandbox_enabled(False)
            assert is_sandbox_enabled() is False
        finally:
            set_sandbox_enabled(original)

    def test_cli_sandbox_argument(self):
        from ict_agent.app.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["--sandbox"])
        assert args.sandbox is True
        args2 = parser.parse_args([])
        assert args2.sandbox is False

    def test_sandbox_blocks_system_write(self):
        """Sandbox should allow workspace writes and block writes outside."""
        import tempfile
        from ict_agent.sandbox import run_sandboxed, is_sandbox_available

        if not is_sandbox_available():
            pytest.skip("No sandbox backend (bubblewrap/seatbelt) available")

        with tempfile.TemporaryDirectory() as workspace:
            # Write inside workspace — should succeed
            code_ok, _, err_ok = run_sandboxed("touch test_ok.txt", workspace, timeout_sec=10)
            if code_ok != 0 and ("permission" in err_ok.lower() or "operation not permitted" in err_ok.lower()):
                # bwrap installed but kernel disallows user namespaces (e.g. GitHub Actions runner)
                pytest.skip(f"bwrap present but unprivileged namespaces blocked: {err_ok.strip()}")
            assert code_ok == 0, f"Write inside workspace should succeed, stderr: {err_ok}"

            # Write outside workspace — should fail
            code_fail, _, err = run_sandboxed("touch /etc/sandbox_test_xyz", workspace, timeout_sec=10)
            assert code_fail != 0, "Write to /etc should be blocked"
            err_lower = err.lower()
            # macOS seatbelt: "not permitted"/"denied"; Linux bwrap: "read-only file system"
            assert any(s in err_lower for s in ("not permitted", "denied", "read-only")), (
                f"Expected permission/read-only error, got: {err}"
            )


# ---------------------------------------------------------------------------
# Task 5: Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    """Verify streaming infrastructure."""

    def test_assemble_streaming_response_text_only(self):
        """Assembled response should correctly reconstruct text content."""
        from ict_agent.runtime.agent_loop import _assemble_streaming_response

        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})()
        resp = _assemble_streaming_response(
            content_parts=["Hello", " world"],
            tool_calls_acc={},
            finish_reason="stop",
            usage=usage,
        )
        assert resp.choices[0].message.content == "Hello world"
        assert resp.choices[0].message.tool_calls is None
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.total_tokens == 15

    def test_assemble_streaming_response_with_tool_calls(self):
        """Assembled response should reconstruct tool calls from accumulated chunks."""
        from ict_agent.runtime.agent_loop import _assemble_streaming_response

        usage = type("U", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()
        tool_calls_acc = {
            0: {"id": "call_abc", "name": "read_file", "arguments": '{"path": "foo.py"}'},
        }
        resp = _assemble_streaming_response([], tool_calls_acc, "tool_calls", usage)
        tc = resp.choices[0].message.tool_calls
        assert tc is not None
        assert len(tc) == 1
        assert tc[0].id == "call_abc"
        assert tc[0].function.name == "read_file"
        assert tc[0].function.arguments == '{"path": "foo.py"}'

    def test_assemble_streaming_response_multiple_tool_calls(self):
        """Multiple tool calls should be ordered by index."""
        from ict_agent.runtime.agent_loop import _assemble_streaming_response

        usage = type("U", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})()
        tool_calls_acc = {
            1: {"id": "call_2", "name": "write_file", "arguments": "{}"},
            0: {"id": "call_1", "name": "read_file", "arguments": "{}"},
        }
        resp = _assemble_streaming_response([], tool_calls_acc, "tool_calls", usage)
        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2
        assert tc[0].function.name == "read_file"  # index 0 first
        assert tc[1].function.name == "write_file"  # index 1 second

    def test_start_async_streaming_call_text_response(self):
        """Streaming call should accumulate text chunks into a complete response."""
        from ict_agent.runtime.agent_loop import start_async_streaming_call

        streamed_text = []

        class DummyLogger:
            def print_streaming(self, text):
                streamed_text.append(text)
            def end_streaming(self):
                pass

        # Simulate streaming chunks
        class FakeChunk:
            def __init__(self, content=None, tool_calls=None, finish_reason=None):
                self.choices = [type("C", (), {
                    "delta": type("D", (), {
                        "content": content,
                        "tool_calls": tool_calls,
                    })(),
                    "finish_reason": finish_reason,
                })()]
                self.usage = None

        chunks = [
            FakeChunk(content="Hello"),
            FakeChunk(content=" streaming"),
            FakeChunk(content=" world"),
            FakeChunk(finish_reason="stop"),
        ]

        class FakeChat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    assert kwargs.get("stream") is True
                    return iter(chunks)

        fake_client = type("C", (), {"chat": FakeChat()})()

        call = start_async_streaming_call(
            client=fake_client,
            model="test",
            request_messages=[{"role": "user", "content": "hi"}],
            active_tool_schemas=[],
            logger=DummyLogger(),
        )
        call.done.wait(timeout=5)
        assert call.error is None, f"Unexpected error: {call.error}"
        assert call.response.choices[0].message.content == "Hello streaming world"
        assert streamed_text == ["Hello", " streaming", " world"]

    def test_start_async_streaming_call_tool_call_response(self):
        """Streaming call should accumulate tool call chunks."""
        from ict_agent.runtime.agent_loop import start_async_streaming_call

        class FakeTcDelta:
            def __init__(self, index, tc_id=None, fn_name=None, fn_args=None):
                self.index = index
                self.id = tc_id
                self.function = type("F", (), {"name": fn_name, "arguments": fn_args})() if (fn_name or fn_args) else None

        class FakeChunk:
            def __init__(self, tool_calls=None, finish_reason=None):
                self.choices = [type("C", (), {
                    "delta": type("D", (), {
                        "content": None,
                        "tool_calls": tool_calls,
                    })(),
                    "finish_reason": finish_reason,
                })()]
                self.usage = None

        chunks = [
            FakeChunk(tool_calls=[FakeTcDelta(0, tc_id="call_1", fn_name="read_file", fn_args='{"pa')]),
            FakeChunk(tool_calls=[FakeTcDelta(0, fn_args='th": "x.py"}')]),
            FakeChunk(finish_reason="tool_calls"),
        ]

        class FakeChat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return iter(chunks)

        fake_client = type("C", (), {"chat": FakeChat()})()

        call = start_async_streaming_call(
            client=fake_client, model="test",
            request_messages=[], active_tool_schemas=[],
        )
        call.done.wait(timeout=5)
        assert call.error is None
        tc = call.response.choices[0].message.tool_calls
        assert tc is not None
        assert len(tc) == 1
        assert tc[0].function.name == "read_file"
        assert tc[0].function.arguments == '{"path": "x.py"}'

    def test_logger_has_streaming_methods(self):
        """RunLogger should have print_streaming and end_streaming methods."""
        from ict_agent.runtime.logging import RunLogger
        logger = RunLogger()
        assert hasattr(logger, "print_streaming")
        assert hasattr(logger, "end_streaming")
        assert callable(logger.print_streaming)
        assert callable(logger.end_streaming)
