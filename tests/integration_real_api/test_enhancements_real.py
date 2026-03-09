"""Real API tests for dual-provider streaming (Anthropic + OpenAI paths).

Tests both the Anthropic Messages API path (mcs-1, with prompt caching) and
the OpenAI Chat Completions path (gpt-oss-120b).  Also verifies that the
ModelRouter dispatches correctly when the model is switched at runtime
(simulating ``/model`` command).

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

OPENAI_MODEL = "gpt-oss-120b"
ANTHROPIC_MODEL = "mcs-1"
# gpt-oss-120b is a reasoning model: needs enough tokens for reasoning + answer
OPENAI_MAX_TOKENS = 256

try:
    import anthropic  # noqa: F401
    _has_anthropic = True
except ImportError:
    _has_anthropic = False

skip_unless_anthropic = pytest.mark.skipif(
    not _has_anthropic, reason="anthropic package not installed"
)


def _get_router():
    """Return the ModelRouter from create_client (ksyun provider)."""
    from ict_agent.llm import create_client
    router, provider, model, base_url = create_client("ksyun")
    return router


# ---------------------------------------------------------------------------
#  Anthropic path (mcs-1) — streaming + prompt caching
# ---------------------------------------------------------------------------

@real_api
@skip_unless_real_api
@skip_unless_anthropic
def test_anthropic_streaming_text():
    """Anthropic streaming should deliver text via start_anthropic_streaming_call."""
    from ict_agent.runtime.agent_loop import start_anthropic_streaming_call
    from ict_agent.llm import get_client_for_model

    router = _get_router()
    client = get_client_for_model(router, ANTHROPIC_MODEL)

    call = start_anthropic_streaming_call(
        client=client,
        model=ANTHROPIC_MODEL,
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
@skip_unless_anthropic
def test_anthropic_streaming_tool_call():
    """Anthropic streaming should correctly accumulate tool_call chunks."""
    from ict_agent.runtime.agent_loop import start_anthropic_streaming_call, MAX_TOKENS_TOOL_TURN
    from ict_agent.llm import get_client_for_model
    from ict_agent.tools import get_all_tool_schemas

    router = _get_router()
    client = get_client_for_model(router, ANTHROPIC_MODEL)
    schemas = [s for s in get_all_tool_schemas() if s["function"]["name"] == "get_current_time"]

    call = start_anthropic_streaming_call(
        client=client,
        model=ANTHROPIC_MODEL,
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
    if msg.tool_calls:
        assert msg.tool_calls[0].function.name == "get_current_time"
        dump = msg.model_dump()
        assert "tool_calls" in dump
    else:
        assert msg.content, "Should have either tool_calls or content"


@real_api
@skip_unless_real_api
@skip_unless_anthropic
def test_anthropic_prompt_caching():
    """Two consecutive calls with a long shared prefix should show cache activity."""
    from ict_agent.runtime.agent_loop import start_anthropic_streaming_call
    from ict_agent.llm import get_client_for_model

    router = _get_router()
    client = get_client_for_model(router, ANTHROPIC_MODEL)

    # System prompt must be >1024 tokens to trigger Anthropic prompt caching
    long_system = (
        "You are a helpful assistant specialized in mathematics, science, "
        "and engineering. You provide detailed, accurate answers with "
        "step-by-step explanations. "
    ) * 60  # ~2300 tokens
    messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": "Say hello in one word."},
    ]

    # First call — should write to cache
    call1 = start_anthropic_streaming_call(
        client=client, model=ANTHROPIC_MODEL,
        request_messages=messages, active_tool_schemas=[], max_tokens=32,
    )
    call1.done.wait(timeout=60)
    assert call1.error is None, f"First call error: {call1.error}"
    usage1 = call1.response.usage

    # Second call with same prefix — should read from cache
    call2 = start_anthropic_streaming_call(
        client=client, model=ANTHROPIC_MODEL,
        request_messages=messages, active_tool_schemas=[], max_tokens=32,
    )
    call2.done.wait(timeout=60)
    assert call2.error is None, f"Second call error: {call2.error}"
    usage2 = call2.response.usage

    cache_write_1 = getattr(usage1, "cache_creation_input_tokens", 0) or 0
    cache_read_1 = getattr(usage1, "cache_read_input_tokens", 0) or 0
    cache_write_2 = getattr(usage2, "cache_creation_input_tokens", 0) or 0
    cache_read_2 = getattr(usage2, "cache_read_input_tokens", 0) or 0

    # At least one call should show cache activity
    total_cache = cache_write_1 + cache_read_1 + cache_write_2 + cache_read_2
    assert total_cache > 0, (
        f"Expected prompt caching activity with {len(long_system)} char system prompt. "
        f"call1: write={cache_write_1} read={cache_read_1}, "
        f"call2: write={cache_write_2} read={cache_read_2}"
    )


# ---------------------------------------------------------------------------
#  OpenAI path (gpt-oss-120b) — reasoning model needs extra max_tokens
# ---------------------------------------------------------------------------

@real_api
@skip_unless_real_api
def test_openai_streaming_text():
    """OpenAI streaming should deliver text via start_async_streaming_call."""
    from ict_agent.runtime.agent_loop import start_async_streaming_call
    from ict_agent.llm import get_client_for_model

    router = _get_router()
    client = get_client_for_model(router, OPENAI_MODEL)

    call = start_async_streaming_call(
        client=client,
        model=OPENAI_MODEL,
        request_messages=[{"role": "user", "content": "回答：1+1=？只回答数字"}],
        active_tool_schemas=[],
        max_tokens=OPENAI_MAX_TOKENS,
    )
    call.done.wait(timeout=60)
    assert call.error is None, f"Streaming error: {call.error}"

    msg = call.response.choices[0].message
    assert msg.content and msg.content.strip(), (
        f"Should receive non-empty text content, got: {msg.content!r}"
    )


@real_api
@skip_unless_real_api
def test_openai_streaming_tool_call():
    """OpenAI streaming should correctly accumulate tool_call chunks."""
    from ict_agent.runtime.agent_loop import start_async_streaming_call, MAX_TOKENS_TOOL_TURN
    from ict_agent.llm import get_client_for_model
    from ict_agent.tools import get_all_tool_schemas

    router = _get_router()
    client = get_client_for_model(router, OPENAI_MODEL)
    schemas = [s for s in get_all_tool_schemas() if s["function"]["name"] == "get_current_time"]

    call = start_async_streaming_call(
        client=client,
        model=OPENAI_MODEL,
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
    if msg.tool_calls:
        assert msg.tool_calls[0].function.name == "get_current_time"
        dump = msg.model_dump()
        assert "tool_calls" in dump
    else:
        assert msg.content, "Should have either tool_calls or content"


# ---------------------------------------------------------------------------
#  /model switching — ModelRouter dispatches correctly
# ---------------------------------------------------------------------------

@real_api
@skip_unless_real_api
@skip_unless_anthropic
def test_model_switch_dispatch():
    """ModelRouter should return the correct SDK client when model changes.

    Simulates ``/model`` command by switching between Anthropic and OpenAI
    models and verifying each path produces a valid streaming response.
    """
    from ict_agent.runtime.agent_loop import (
        start_anthropic_streaming_call,
        start_async_streaming_call,
    )
    from ict_agent.llm import get_client_for_model, is_anthropic_model

    router = _get_router()
    messages = [{"role": "user", "content": "回答：1+1=？只回答数字"}]

    # --- Turn 1: Anthropic model (mcs-1) ---
    model_1 = ANTHROPIC_MODEL
    assert is_anthropic_model(model_1)
    client_1 = get_client_for_model(router, model_1)
    call_1 = start_anthropic_streaming_call(
        client=client_1, model=model_1,
        request_messages=messages, active_tool_schemas=[], max_tokens=64,
    )
    call_1.done.wait(timeout=60)
    assert call_1.error is None, f"Anthropic call error: {call_1.error}"
    assert call_1.response.choices[0].message.content

    # --- Turn 2: switch to OpenAI model (gpt-oss-120b) ---
    model_2 = OPENAI_MODEL
    assert not is_anthropic_model(model_2)
    client_2 = get_client_for_model(router, model_2)
    call_2 = start_async_streaming_call(
        client=client_2, model=model_2,
        request_messages=messages, active_tool_schemas=[],
        max_tokens=OPENAI_MAX_TOKENS,
    )
    call_2.done.wait(timeout=60)
    assert call_2.error is None, f"OpenAI call error: {call_2.error}"
    assert call_2.response.choices[0].message.content

    # --- Turn 3: switch back to Anthropic ---
    client_3 = get_client_for_model(router, model_1)
    call_3 = start_anthropic_streaming_call(
        client=client_3, model=model_1,
        request_messages=messages, active_tool_schemas=[], max_tokens=64,
    )
    call_3.done.wait(timeout=60)
    assert call_3.error is None, f"Switch-back error: {call_3.error}"
    assert call_3.response.choices[0].message.content
