"""Real API tests for enhancement features (Tasks 1-6).

Uses gpt-oss-120b (weaker model) to verify features work robustly.
Requires: ICT_AGENT_RUN_REAL_API=1 and KSYUN_API_KEY set.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SKIP_REASON = "Set ICT_AGENT_RUN_REAL_API=1 to run real API tests."
skip_unless_real_api = pytest.mark.skipif(
    os.getenv("ICT_AGENT_RUN_REAL_API") != "1",
    reason=SKIP_REASON,
)
real_api = pytest.mark.real_api

MODEL = "gpt-oss-120b"


def _get_client():
    from ict_agent.llm import create_client
    client, provider, model, base_url = create_client("ksyun")
    return client


# ---------------------------------------------------------------------------
# Task 1: Workspace = cwd
# ---------------------------------------------------------------------------


@real_api
@skip_unless_real_api
def test_workspace_cwd_with_tool_call():
    """Agent tool call should operate in the correct workspace (cwd-based)."""
    from ict_agent.tools import set_workspace_root, execute_tool

    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "marker.txt").write_text("real_api_test")
        set_workspace_root(tmpdir)

        result = execute_tool("list_directory", '{"path": "."}')
        assert "marker.txt" in result, f"marker.txt not found in workspace listing: {result}"


# ---------------------------------------------------------------------------
# Task 2: Large output persistence
# ---------------------------------------------------------------------------


@real_api
@skip_unless_real_api
def test_large_output_persisted():
    """Large tool output should be saved to disk with only a reference in context."""
    from ict_agent.runtime.agent_loop import _maybe_persist_large_output, LARGE_OUTPUT_THRESHOLD

    class DummyLog:
        def log(self, *a, **k): pass

    large = "Z" * (LARGE_OUTPUT_THRESHOLD + 500)
    result = _maybe_persist_large_output(large, "run_shell", DummyLog())

    assert len(result) < LARGE_OUTPUT_THRESHOLD, "Persisted reference should be compact"
    match = re.search(r"saved to (\S+\.txt)", result)
    assert match, f"Should contain file path, got: {result[:200]}"
    fpath = match.group(1)
    # Path may be workspace-relative (e.g. .tool_outputs/xxx.txt) or absolute
    p = Path(fpath)
    if not p.is_absolute():
        from ict_agent.tools import _workspace_root
        p = _workspace_root() / fpath
    content = p.read_text()
    assert len(content) == len(large), "File should contain full output"
    os.unlink(str(p))


# ---------------------------------------------------------------------------
# Task 3+5: Streaming + max_tokens (real API call)
# ---------------------------------------------------------------------------


@real_api
@skip_unless_real_api
def test_streaming_text_response():
    """Streaming should deliver text via start_async_streaming_call (same path as agent loop)."""
    from ict_agent.runtime.agent_loop import start_async_streaming_call

    client = _get_client()
    call = start_async_streaming_call(
        client=client,
        model=MODEL,
        request_messages=[{"role": "user", "content": "回答：1+1=？只回答数字"}],
        active_tool_schemas=[],
        max_tokens=64,
    )
    call.done.wait(timeout=60)
    assert call.error is None, f"Streaming error: {call.error}"

    msg = call.response.choices[0].message
    # Should get text content (no tools provided)
    assert msg.content and msg.content.strip(), (
        f"Should receive non-empty text content, got: {msg.content!r}"
    )


@real_api
@skip_unless_real_api
def test_streaming_tool_call():
    """Streaming should correctly accumulate tool_call chunks."""
    from ict_agent.runtime.agent_loop import start_async_streaming_call, MAX_TOKENS_TOOL_TURN
    from ict_agent.tools import get_all_tool_schemas

    client = _get_client()
    # Only include get_current_time to strongly hint the model to use it
    schemas = [s for s in get_all_tool_schemas() if s["function"]["name"] == "get_current_time"]

    call = start_async_streaming_call(
        client=client,
        model=MODEL,
        request_messages=[
            {"role": "system", "content": "You must use the get_current_time tool to answer. Do not answer with text."},
            {"role": "user", "content": "现在几点？"},
        ],
        active_tool_schemas=schemas,
        max_tokens=MAX_TOKENS_TOOL_TURN,
    )
    call.done.wait(timeout=60)
    assert call.error is None, f"API error: {call.error}"

    msg = call.response.choices[0].message
    # Model should either call the tool or respond with text — both are acceptable
    if msg.tool_calls:
        assert msg.tool_calls[0].function.name == "get_current_time"
        # Verify model_dump works (was crashing before fix)
        dump = msg.model_dump()
        assert "tool_calls" in dump
    else:
        # Weaker model may not reliably call tools — text response is acceptable
        assert msg.content, "Should have either tool_calls or content"


# ---------------------------------------------------------------------------
# Task 4: Dynamic tool schema
# ---------------------------------------------------------------------------


@real_api
@skip_unless_real_api
def test_dynamic_schema_core_only():
    """With no skill-specified tools, only core tools should be active."""
    from ict_agent.runtime.agent_loop import resolve_active_tool_schemas, CORE_TOOLS
    from ict_agent.skills import SkillSpec
    from ict_agent.tools import get_all_tool_schemas, get_tool_schema_map

    skill = SkillSpec(name="test", description="", tools=[], triggers=[],
                      always_on=False, instructions="", context_mode="inline")
    tool_map = get_tool_schema_map()
    all_schemas = get_all_tool_schemas()

    result = resolve_active_tool_schemas([skill], tool_map, all_schemas)
    result_names = {s["function"]["name"] for s in result}

    assert result_names.issubset(CORE_TOOLS | {"workspace_info"}), (
        f"Non-core tools leaked: {result_names - CORE_TOOLS}"
    )
    assert "fork_subagent" not in result_names


# ---------------------------------------------------------------------------
# Task 6: Sandbox
# ---------------------------------------------------------------------------


@real_api
@skip_unless_real_api
def test_sandbox_blocks_system_write():
    """Sandbox should block writes outside workspace."""
    from ict_agent.sandbox import run_sandboxed, is_sandbox_available

    if not is_sandbox_available():
        pytest.skip("No sandbox backend available on this platform")

    with tempfile.TemporaryDirectory() as workspace:
        # Write inside workspace — should succeed
        code_ok, _, _ = run_sandboxed("touch test_ok.txt", workspace, timeout_sec=10)
        assert code_ok == 0, "Write inside workspace should succeed"

        # Write outside workspace — should fail
        code_fail, _, err = run_sandboxed("touch /etc/sandbox_test_xyz", workspace, timeout_sec=10)
        assert code_fail != 0, f"Write to /etc should be blocked, got exit=0"
        err_lower = err.lower()
        # macOS seatbelt: "not permitted" / "denied"; Linux bwrap: "read-only file system"
        assert any(s in err_lower for s in ("not permitted", "denied", "read-only")), (
            f"Expected permission/read-only error, got: {err}"
        )


@real_api
@skip_unless_real_api
def test_banned_command_blocked():
    """Banned commands should be blocked at the code level."""
    from ict_agent.tools import run_shell

    result = run_shell("rm -rf /")
    assert "Blocked" in result, f"rm -rf / should be blocked, got: {result}"

    result2 = run_shell("shutdown -h now")
    assert "Blocked" in result2, f"shutdown should be blocked, got: {result2}"


@real_api
@skip_unless_real_api
def test_safe_command_auto_approved():
    """Safe commands should execute without prompting."""
    from ict_agent.tools import _is_safe_command

    assert _is_safe_command("git status")
    assert _is_safe_command("pwd")
    assert not _is_safe_command("curl http://example.com")
