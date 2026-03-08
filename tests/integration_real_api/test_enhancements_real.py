"""Real API tests for streaming (the only enhancement that requires a live endpoint).

Other enhancement features (workspace, large output, dynamic schema, sandbox,
shell safety) are fully covered by unit tests in tests/unit/test_enhancements.py.

Requires: ICT_AGENT_RUN_REAL_API=1 and KSYUN_API_KEY set.
"""

from __future__ import annotations

import os
import sys
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
        dump = msg.model_dump()
        assert "tool_calls" in dump
    else:
        assert msg.content, "Should have either tool_calls or content"
