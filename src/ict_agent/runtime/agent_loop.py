"""Core interactive agent loop."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from queue import Empty, Queue
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import OpenAI

from ict_agent.commands.common import do_compact, format_turn_usage
from ict_agent.commands.registry import CommandContext
from ict_agent.context import ContextManager
from ict_agent.domains.cuda.recovery import (
    RecoveryState,
    ToolExecutionOutcome,
    build_recovery_nudge,
    classify_cuda_failure,
    is_tool_failure,
    summarize_failure,
)
from ict_agent.runtime.preemption import clear_preempt_request, set_autonomous_turn, set_shell_interrupt_on_preempt
from ict_agent.runtime.session import (
    InputReaderThread,
    dequeue_user_input_blocking,
    dequeue_user_input_nowait,
    dequeue_user_input_with_timeout,
    to_pending_input_from_preempt_event,
)
from ict_agent.runtime.current_context import clear_current_runtime, get_current_runtime, set_current_runtime
from ict_agent.llm import is_anthropic_model, get_client_for_model
from ict_agent.skills import build_skill_prompt, load_skills, select_skills
from ict_agent.skills import SkillSpec
from ict_agent.tools import execute_tool, get_all_tool_schemas, get_tool_schema_map, set_no_truncate, set_shell_safety


CONTINUATION_REPLIES = {
    "y",
    "yes",
    "ok",
    "okay",
    "sure",
    "continue",
    "go",
    "proceed",
    "继续",
    "好的",
    "好",
}
ACTION_INTENT_RE = re.compile(
    r"\b(run|execute|check|list|read|write|search|find|show|compile|verify|profile|optimize|build|benchmark|fix|implement|ask|reply|send|wait|monitor|listen)\b|"
    r"(运行|执行|编译|验证|测速|性能|优化|实现|修改|修复|提问|回复|发送|等待|监听|读取|查看)"
)
AUTONOMY_NUDGE_TEXT = (
    "Do not provide a plan or ask the user for procedural confirmation. "
    "For this turn, immediately call the required tools and only then provide a final summary."
)
SESSION_CONTINUATION_HINTS = (
    "session",
    "session0",
    "session1",
    "agent0",
    "agent1",
    "other agent",
    "multi-agent",
    "read output",
    "read reply",
    "send message",
    "continue asking",
    "follow up",
    "sleep 5",
    "提问",
    "继续提问",
    "继续问",
    "再问",
    "回复",
    "读取输出",
    "读取回复",
    "发送消息",
    "等回复",
    "等待回复",
    "监听",
    "查看session",
    "其他agent",
    "多agent",
)

BACKSLASH_COMMAND_RE = re.compile(r"^\\([a-zA-Z][a-zA-Z0-9_-]*)(?:\s+.*)?$")


class AsyncModelCall:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.response: Any = None
        self.error: Exception | None = None


MAX_TOKENS_TOOL_TURN = 2048
MAX_TOKENS_FINAL_TURN = 8192


def _create_openai_chat_completion(
    client: "OpenAI",
    model: str,
    request_messages: list[dict],
    active_tool_schemas: list[dict],
    max_tokens: int | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        "tools": active_tool_schemas,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Log request size for 400 debugging (e.g. gpt-oss vs mco-4 tool_calls strictness)
        try:
            n_msgs = len(request_messages)
            n_tools = len(active_tool_schemas)
            err_str = str(exc)
            if "400" in err_str or "Bad Request" in err_str:
                import logging

                logging.getLogger(__name__).warning(
                    "API 400 on chat.completions.create: model=%s messages=%s tools=%s err=%s",
                    model,
                    n_msgs,
                    n_tools,
                    err_str[:200],
                )
        except Exception:  # noqa: S110
            pass
        raise


def start_async_model_call(
    client: "OpenAI",
    model: str,
    request_messages: list[dict],
    active_tool_schemas: list[dict],
    max_tokens: int | None = None,
) -> AsyncModelCall:
    call = AsyncModelCall()

    def worker() -> None:
        try:
            call.response = _create_openai_chat_completion(
                client=client,
                model=model,
                request_messages=request_messages,
                active_tool_schemas=active_tool_schemas,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            call.error = exc
        finally:
            call.done.set()

    threading.Thread(target=worker, daemon=True).start()
    return call


def _assemble_streaming_response(
    content_parts: list[str],
    tool_calls_acc: dict[int, dict],
    finish_reason: str | None,
    usage: Any,
) -> Any:
    """Assemble a complete response-like object from accumulated streaming chunks.

    The assembled objects must support ``model_dump()`` so that
    ``ContextManager._sanitize_assistant_tool_calls_message`` can serialise
    them the same way it handles real OpenAI SDK response objects.
    """
    tool_calls_list = None
    if tool_calls_acc:
        tool_calls_list = []
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            tc_id = tc.get("id", "")
            tc_name = tc.get("name", "")
            tc_args = tc.get("arguments", "")

            fn_obj = type("Function", (), {
                "name": tc_name,
                "arguments": tc_args,
            })()

            tc_obj = type("ToolCall", (), {
                "id": tc_id,
                "type": "function",
                "function": fn_obj,
                "get": lambda self, key, default=None, _d={"id": tc_id, "type": "function", "function": {"name": tc_name, "arguments": tc_args}}: _d.get(key, default),
                "model_dump": lambda self, _d={"id": tc_id, "type": "function", "function": {"name": tc_name, "arguments": tc_args}}: _d,
            })()
            tool_calls_list.append(tc_obj)

    content = "".join(content_parts) or None
    tc_dump = [
        {"id": tc.get("id", ""), "type": "function", "function": {"name": tc.get("name", ""), "arguments": tc.get("arguments", "")}}
        for tc in (tool_calls_acc[idx] for idx in sorted(tool_calls_acc.keys()))
    ] if tool_calls_acc else []

    message = type("Message", (), {
        "content": content,
        "tool_calls": tool_calls_list,
        "role": "assistant",
        "model_dump": lambda self, _c=content, _tc=tc_dump: {
            "role": "assistant",
            "content": _c or "",
            "tool_calls": _tc,
        },
    })()
    choice = type("Choice", (), {
        "message": message,
        "finish_reason": finish_reason,
    })()
    return type("Response", (), {"choices": [choice], "usage": usage})()


# ---------------------------------------------------------------------------
#  Anthropic Messages API helpers (prompt caching, format conversion)
# ---------------------------------------------------------------------------

_CACHE_CTRL = {"type": "ephemeral"}


def _openai_tools_to_anthropic(tool_schemas: list[dict]) -> list[dict]:
    """Convert OpenAI function-calling tool schemas → Anthropic tool format.

    Also adds ``cache_control`` to the **last** tool definition so the entire
    tool list is included in the cached prefix.
    """
    out: list[dict] = []
    for schema in tool_schemas:
        fn = schema.get("function", {})
        params = fn.get("parameters", {})
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": {
                "type": params.get("type", "object"),
                "properties": params.get("properties", {}),
                "required": params.get("required", []),
            },
        })
    # Cache the tool list as part of the prefix
    if out:
        out[-1]["cache_control"] = _CACHE_CTRL
    return out


# Bedrock (and some gateways) allow at most 4 blocks with cache_control.
# We use: 1 system (merged) + 1 last tool def + 1 last user block = 3.


def _openai_messages_to_anthropic(
    openai_messages: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Convert OpenAI-format messages → Anthropic (system, messages).

    Returns ``(system_blocks, messages)`` where *system_blocks* is a list of
    text content blocks (with ``cache_control``) and *messages* is the
    Anthropic ``messages`` parameter.

    Cache breakpoints:
    - Only the first system prompt block gets ``cache_control`` (stable across
      turns and keeps total cache-control blocks within Anthropic's limit).
    - The last user/tool-result message gets ``cache_control`` (caches the
      entire conversation prefix up to the current turn).
    """
    system_parts: list[str] = []
    messages: list[dict] = []

    system_block_index = 0
    for msg in openai_messages:
        role = msg.get("role")
        content = msg.get("content", "") or ""

        if role == "system":
            block = {
                "type": "text",
                "text": content,
            }
            if system_block_index == 0:
                block["cache_control"] = _CACHE_CTRL
            system_blocks.append(block)
            system_block_index += 1

        elif role == "user":
            messages.append({"role": "user", "content": content})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                # Assistant with tool_use → content blocks
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    import json as _json
                    try:
                        args = _json.loads(fn.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                messages.append({"role": "assistant", "content": blocks})
            else:
                messages.append({"role": "assistant", "content": content})

        elif role == "tool":
            # Tool results must be inside a "user" message as tool_result blocks.
            # If the previous message is already a user with tool_result blocks, append.
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content,
            }
            if (
                messages
                and messages[-1]["role"] == "user"
                and isinstance(messages[-1]["content"], list)
                and messages[-1]["content"]
                and messages[-1]["content"][0].get("type") == "tool_result"
            ):
                messages[-1]["content"].append(tool_result_block)
            else:
                messages.append({"role": "user", "content": [tool_result_block]})

    # Add cache_control to the last user message (cache conversation prefix)
    for msg in reversed(messages):
        if msg["role"] != "user":
            continue
        c = msg["content"]
        if isinstance(c, str):
            msg["content"] = [{"type": "text", "text": c, "cache_control": _CACHE_CTRL}]
        elif isinstance(c, list) and c:
            c[-1] = dict(c[-1])
            c[-1]["cache_control"] = _CACHE_CTRL
        break

    # Single system block with cache_control so total cache_control blocks stay ≤ 4
    # (1 system + 1 last tool + 1 last user). Multiple system messages are merged.
    system_blocks = []
    if system_parts:
        system_blocks = [{
            "type": "text",
            "text": "\n\n".join(system_parts),
            "cache_control": _CACHE_CTRL,
        }]
    return system_blocks, messages


def _anthropic_response_to_openai_like(anth_resp: Any) -> Any:
    """Convert a non-streaming Anthropic response to OpenAI-like shape.

    This lets ``process_tool_calls`` and the rest of the loop work unchanged.
    """
    import json as _json
    content_text = ""
    tool_calls_acc: dict[int, dict] = {}
    tc_idx = 0
    for block in anth_resp.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "tool_use":
            tool_calls_acc[tc_idx] = {
                "id": block.id,
                "name": block.name,
                "arguments": _json.dumps(block.input, ensure_ascii=False),
            }
            tc_idx += 1

    stop = anth_resp.stop_reason
    finish_reason = "tool_calls" if stop == "tool_use" else ("stop" if stop == "end_turn" else stop)

    oai_usage = _anthropic_usage_to_openai_like(anth_resp.usage)
    return _assemble_streaming_response(
        [content_text] if content_text else [],
        tool_calls_acc,
        finish_reason,
        oai_usage,
    )


def _anthropic_usage_to_openai_like(usage) -> Any:
    """Wrap Anthropic usage into an object with OpenAI-compatible attribute names.

    This allows ``ContextManager.record_usage()`` and ``format_turn_usage()``
    to work without changes.
    """
    input_tok = getattr(usage, "input_tokens", 0) or 0
    output_tok = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    # Total prompt = all input tokens (uncached + cache hit + cache miss/write)
    total_prompt = input_tok + cache_read + cache_write
    return type("Usage", (), {
        "prompt_tokens": total_prompt,
        "completion_tokens": output_tok,
        "total_tokens": total_prompt + output_tok,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    })()


def request_model_response(
    client_or_router: "OpenAI",
    model: str,
    request_messages: list[dict],
    active_tool_schemas: list[dict],
    max_tokens: int | None = None,
) -> Any:
    """Create one non-streaming model response using the provider-specific API."""
    model_client = get_client_for_model(client_or_router, model)
    if is_anthropic_model(model):
        system_blocks, anth_messages = _openai_messages_to_anthropic(request_messages)
        anth_tools = _openai_tools_to_anthropic(active_tool_schemas)
        anth_resp = model_client.messages.create(
            model=model,
            system=system_blocks,
            messages=anth_messages,
            tools=anth_tools,
            max_tokens=max_tokens or MAX_TOKENS_FINAL_TURN,
        )
        return _anthropic_response_to_openai_like(anth_resp)
    return _create_openai_chat_completion(
        client=model_client,
        model=model,
        request_messages=request_messages,
        active_tool_schemas=active_tool_schemas,
        max_tokens=max_tokens,
    )


def start_anthropic_streaming_call(
    client: Any,
    model: str,
    request_messages: list[dict],
    active_tool_schemas: list[dict],
    max_tokens: int | None = None,
    logger: Any = None,
) -> AsyncModelCall:
    """Streaming call via Anthropic Messages API (with prompt caching).

    Produces the same ``AsyncModelCall`` / response shape as
    ``start_async_streaming_call`` so the rest of agent_loop is unchanged.
    """
    call = AsyncModelCall()

    def worker() -> None:
        try:
            system_blocks, anth_messages = _openai_messages_to_anthropic(request_messages)
            anth_tools = _openai_tools_to_anthropic(active_tool_schemas)

            content_parts: list[str] = []
            tool_calls_acc: dict[int, dict] = {}
            finish_reason: str | None = None
            usage = None
            tc_index = 0  # running tool_call index for ordering

            with client.messages.stream(
                model=model,
                system=system_blocks,
                messages=anth_messages,
                tools=anth_tools,
                max_tokens=max_tokens or MAX_TOKENS_FINAL_TURN,
            ) as stream:
                for event in stream:
                    etype = event.type

                    if etype == "message_start":
                        # Initial usage (input tokens)
                        usage = event.message.usage

                    elif etype == "content_block_start":
                        block = event.content_block
                        if block.type == "text":
                            pass  # text deltas arrive in content_block_delta
                        elif block.type == "tool_use":
                            idx = tc_index
                            tc_index += 1
                            tool_calls_acc[idx] = {
                                "id": block.id,
                                "name": block.name,
                                "arguments": "",
                            }

                    elif etype == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            content_parts.append(delta.text)
                            if logger and hasattr(logger, "print_streaming"):
                                logger.print_streaming(delta.text)
                        elif delta.type == "input_json_delta":
                            # Find the latest tool_call being built
                            idx = tc_index - 1
                            if idx in tool_calls_acc:
                                tool_calls_acc[idx]["arguments"] += delta.partial_json

                    elif etype == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            # Merge final usage (output tokens etc.)
                            # Create a combined usage by merging initial + delta
                            if usage is not None:
                                final_usage = type("MergedUsage", (), {
                                    "input_tokens": getattr(usage, "input_tokens", 0),
                                    "output_tokens": getattr(event.usage, "output_tokens", 0) or getattr(usage, "output_tokens", 0),
                                    "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                                    "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
                                })()
                                usage = final_usage
                        stop = getattr(event.delta, "stop_reason", None)
                        if stop == "tool_use":
                            finish_reason = "tool_calls"
                        elif stop == "end_turn":
                            finish_reason = "stop"
                        elif stop:
                            finish_reason = stop

            # End streaming line if we printed text
            if content_parts and logger and hasattr(logger, "end_streaming"):
                logger.end_streaming()

            # Convert Anthropic usage → OpenAI-like for ContextManager
            if usage is None:
                oai_usage = type("Usage", (), {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                })()
            else:
                oai_usage = _anthropic_usage_to_openai_like(usage)

            call.response = _assemble_streaming_response(
                content_parts, tool_calls_acc, finish_reason, oai_usage,
            )
        except Exception as exc:
            call.error = exc
        finally:
            call.done.set()

    threading.Thread(target=worker, daemon=True).start()
    return call


# ---------------------------------------------------------------------------
#  OpenAI Chat Completions streaming (non-Anthropic models)
# ---------------------------------------------------------------------------

def start_async_streaming_call(
    client: "OpenAI",
    model: str,
    request_messages: list[dict],
    active_tool_schemas: list[dict],
    max_tokens: int | None = None,
    logger: Any = None,
) -> AsyncModelCall:
    """Streaming call via OpenAI Chat Completions API."""
    call = AsyncModelCall()

    def worker() -> None:
        try:
            kwargs: dict = dict(
                model=model,
                messages=request_messages,
                tools=active_tool_schemas,
                stream=True,
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            stream = client.chat.completions.create(**kwargs)

            content_parts: list[str] = []
            tool_calls_acc: dict[int, dict] = {}
            finish_reason: str | None = None
            usage = None

            for chunk in stream:
                if not chunk.choices:
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = chunk.usage
                    continue
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason or finish_reason

                if hasattr(delta, "content") and delta.content:
                    content_parts.append(delta.content)
                    if logger and hasattr(logger, "print_streaming"):
                        logger.print_streaming(delta.content)

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index if hasattr(tc_delta, "index") else 0
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if hasattr(tc_delta, "id") and tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if hasattr(tc_delta, "function") and tc_delta.function:
                            fn = tc_delta.function
                            if hasattr(fn, "name") and fn.name:
                                tool_calls_acc[idx]["name"] += fn.name
                            if hasattr(fn, "arguments") and fn.arguments:
                                tool_calls_acc[idx]["arguments"] += fn.arguments

                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage

            if content_parts and logger and hasattr(logger, "end_streaming"):
                logger.end_streaming()

            if usage is None:
                usage = type("Usage", (), {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                })()

            call.response = _assemble_streaming_response(
                content_parts, tool_calls_acc, finish_reason, usage,
            )
        except Exception as exc:
            call.error = exc
        finally:
            call.done.set()

    threading.Thread(target=worker, daemon=True).start()
    return call


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


CORE_TOOLS = {
    "read_file",
    "write_file",
    "edit_file",
    "run_shell",
    "list_directory",
    "search_files",
    "grep_text",
    "workspace_info",
}


def resolve_active_tool_schemas(selected_skills, tool_schema_map: dict[str, dict], all_tool_schemas: list[dict]) -> list[dict]:
    selected_tool_names = [
        tool_name
        for skill in selected_skills
        for tool_name in skill.tools
        if tool_name in tool_schema_map
    ]
    if selected_tool_names:
        # Skill specifies tools → use those + core tools
        combined = list(CORE_TOOLS) + selected_tool_names
        return [tool_schema_map[n] for n in unique_preserve_order(combined) if n in tool_schema_map]
    # No skill-specified tools → use core tools only (fork tools added separately in chat())
    core_schemas = [tool_schema_map[n] for n in CORE_TOOLS if n in tool_schema_map]
    return core_schemas if core_schemas else all_tool_schemas


def maybe_extend_skills_for_continuation(user_input: str, selected_skills, runtime_state: dict) -> None:
    text = user_input.lower()
    should_extend = text in CONTINUATION_REPLIES or any(
        hint in text for hint in SESSION_CONTINUATION_HINTS
    )
    if not should_extend:
        return
    existing = {skill.name for skill in selected_skills}
    for prev_name in runtime_state.get("active_skill_names", []):
        if prev_name in runtime_state["skills"] and prev_name not in existing:
            selected_skills.append(runtime_state["skills"][prev_name])


def has_action_intent(user_input: str) -> bool:
    text = user_input.strip().lower()
    if not text or text.startswith("/"):
        return False
    return bool(ACTION_INTENT_RE.search(text))


def is_procedural_confirmation(text: str) -> bool:
    low = text.lower()
    cues = (
        "proceed?",
        "reply `yes` or `no`",
        "reply yes or no",
        "wait for your go-ahead",
        "approve",
        "go-ahead",
        "allow once",
        "always allow",
    )
    return any(cue in low for cue in cues)


def normalize_command_input(user_input: str) -> str:
    """Accept common backslash-prefixed command variants like ``\\debug``.

    The live-session workflow and docs use slash commands, but in practice users
    sometimes type shell-like backslash prefixes. We normalize that form before
    command dispatch so the command is handled locally instead of being sent to
    the LLM as plain text.
    """
    text = user_input.strip()
    if text.startswith("/"):
        return text
    if BACKSLASH_COMMAND_RE.match(text):
        return "/" + text[1:]
    return user_input


LARGE_OUTPUT_THRESHOLD = 30_000


def _maybe_persist_large_output(result: str, tool_name: str, logger) -> str:
    """Persist large tool outputs to disk and return a compact reference for context.

    Files are saved inside the workspace under ``.tool_outputs/`` so that
    ``read_file`` can access them (it rejects paths outside workspace root).
    """
    if len(result) <= LARGE_OUTPUT_THRESHOLD:
        return result
    import os
    try:
        from ict_agent.tools import _workspace_root
        output_dir = _workspace_root() / ".tool_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".txt", prefix=f"{tool_name}_", dir=str(output_dir))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(result)
        # Use workspace-relative path so read_file can access it
        rel_path = f".tool_outputs/{Path(path).name}"
        half = 500
        logger.log(f"  [large-output] Saved {len(result)} chars to {rel_path}", level="system")
        return (
            f"[Output too large ({len(result)} chars), saved to {rel_path}]\n"
            f"Use read_file(path=\"{rel_path}\") to access the full content.\n"
            f"First {half} chars:\n{result[:half]}\n...\n"
            f"Last {half} chars:\n...{result[-half:]}"
        )
    except Exception as exc:
        logger.log(f"  [large-output] Failed to persist: {exc}", level="error")
        return result


def process_tool_calls(
    response, ctx: ContextManager, logger, no_truncate: bool = False
) -> ToolExecutionOutcome:
    message = response.choices[0].message
    if not message.tool_calls:
        return ToolExecutionOutcome(called=False, failures=[], failure_kinds=[])
    ctx.add_assistant_tool_calls(message)
    failures: list[str] = []
    failure_kinds: list[str] = []
    for tool_call in message.tool_calls:
        name = (tool_call.function.name or "").strip()
        args = tool_call.function.arguments
        if not name:
            result = "Error: empty tool name returned by model."
            logger.log(f"  <- Result: {result}", level="result")
            ctx.add_tool_result(tool_call.id, "unknown", result)
            failures.append("unknown: empty tool name")
            continue
        logger.log(f"  -> Calling tool: {name}({args})", level="tool")
        result = execute_tool(name, args)
        display = (
            result
            if no_truncate or len(result) <= 2000
            else result[:1000] + "\n...(truncated)...\n" + result[-500:]
        )
        logger.log(f"  <- Result: {display}", level="result")
        result_for_ctx = (
            result if no_truncate else _maybe_persist_large_output(result, name, logger)
        )
        ctx.add_tool_result(tool_call.id, name, result_for_ctx)
        if is_tool_failure(name, result):
            failures.append(summarize_failure(name, result))
            failure_kinds.append(classify_cuda_failure(name, result))
    return ToolExecutionOutcome(called=True, failures=failures, failure_kinds=failure_kinds)


class _ForkLogger:
    """Wraps a logger to prefix all log lines with [fork:skill_name] for sub-agent output."""

    def __init__(self, logger, prefix: str):
        self._logger = logger
        self._prefix = prefix

    def log(self, msg: str, level: str = "info") -> None:
        self._logger.log(f"{self._prefix} {msg}", level=level)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)


def _serialize_fork_context(messages: list[dict]) -> str:
    """Serialize fork ctx.messages into one string for return_mode=full_context."""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "system":
            parts.append(f"--- system ---\n{content}")
        elif role == "user":
            parts.append(f"--- user ---\n{content}")
        elif role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    parts.append(f"--- assistant (tool_call: {fn.get('name', '')}) ---\n{fn.get('arguments', '')}")
            if content:
                parts.append(f"--- assistant ---\n{content}")
        elif role == "tool":
            name = m.get("name", "")
            parts.append(f"--- tool ({name}) ---\n{content}")
    return "\n\n".join(parts)


def run_fork_skill(
    client: "OpenAI",
    model: str,
    skill: SkillSpec,
    task: str,
    tool_schema_map: dict[str, dict],
    all_tool_schemas: list[dict],
    logger: Any,
    max_steps: int = 20,
    max_tokens: int = 128_000,
    return_mode: str = "final",
    no_truncate: bool = False,
) -> str:
    """Run a fork (Agent as Skill) sub-agent: isolated context, skill's tools only, single task.

    return_mode: "final" = return only the last assistant reply (default; saves main context).
                 "full_context" = return entire subagent conversation as one blob (for handoff).
    Returns the final assistant message content, or full context string, or an error message.
    """
    fork_log = _ForkLogger(logger, f"[fork:{skill.name}]")
    task_preview = task if no_truncate else (task[:200] + "..." if len(task) > 200 else task)
    fork_log.log(f"Starting fork task: {task_preview}", level="system")
    ctx = ContextManager(system_prompt=skill.instructions or "Complete the user's task.", max_tokens=max_tokens)
    ctx.add_user_message(task)
    active_tool_schemas = resolve_active_tool_schemas([skill], tool_schema_map, all_tool_schemas)
    if not active_tool_schemas:
        return "Error: fork skill has no tools available (check skill's tools list and tool registry)."

    model_client = get_client_for_model(client, model)
    _use_anthropic = is_anthropic_model(model)
    last_content = ""
    for step_idx in range(max_steps):
        try:
            request_messages = list(ctx.messages)
            if _use_anthropic:
                system_blocks, anth_messages = _openai_messages_to_anthropic(request_messages)
                anth_tools = _openai_tools_to_anthropic(active_tool_schemas)
                anth_resp = model_client.messages.create(
                    model=model, system=system_blocks, messages=anth_messages,
                    tools=anth_tools, max_tokens=MAX_TOKENS_FINAL_TURN,
                )
                response = _anthropic_response_to_openai_like(anth_resp)
            else:
                async_call = start_async_model_call(
                    client=model_client,
                    model=model,
                    request_messages=request_messages,
                    active_tool_schemas=active_tool_schemas,
                )
                async_call.done.wait(timeout=300)
                if async_call.error is not None:
                    raise async_call.error
                if async_call.response is None:
                    raise RuntimeError("Model call ended without response.")
                response = async_call.response
            ctx.record_usage(response.usage, overhead_tokens=0)

            tool_outcome = process_tool_calls(response, ctx, fork_log, no_truncate)
            if tool_outcome.called:
                continue

            content = (response.choices[0].message.content or "").strip()
            last_content = content
            ctx.add_assistant_message(content)
            fork_log.log(f"Completed in {step_idx + 1} step(s).", level="system")
            content_preview = content if no_truncate else (content[:500] + "..." if len(content) > 500 else content)
            fork_log.log(content_preview, level="assistant")
            if return_mode == "full_context":
                return _serialize_fork_context(ctx.messages)
            return content
        except Exception as exc:
            err_msg = f"Error in fork step {step_idx + 1}: {exc}"
            fork_log.log(err_msg, level="error")
            return err_msg

    last_preview = last_content if no_truncate else (last_content[:300] + "..." if len(last_content) > 300 else last_content)
    msg = f"Fork reached max steps ({max_steps}) without final reply. Last content: {last_preview}"
    fork_log.log(msg, level="error")
    if return_mode == "full_context":
        return _serialize_fork_context(ctx.messages)
    return msg


def _run_fork_in_thread(
    job_id: str,
    skill: SkillSpec,
    task: str,
    client: "OpenAI",
    model: str,
    tool_schema_map: dict[str, dict],
    all_tool_schemas: list[dict],
    logger: Any,
    result_queue: "Queue[tuple[str, str, str]]",
    return_mode: str = "final",
    no_truncate: bool = False,
) -> None:
    """Background thread: run run_fork_skill and put (job_id, skill_name, result) into result_queue."""
    try:
        result = run_fork_skill(
            client=client,
            model=model,
            skill=skill,
            task=task,
            tool_schema_map=tool_schema_map,
            all_tool_schemas=all_tool_schemas,
            logger=logger,
            return_mode=return_mode,
            no_truncate=no_truncate,
        )
        result_queue.put((job_id, skill.name, result))
    except Exception as exc:
        result_queue.put((job_id, skill.name, f"Error: {exc}"))


def drain_fork_results(
    ctx: ContextManager,
    runtime_state: dict,
    *,
    inject_into_ctx: bool = True,
) -> None:
    """Drain fork_result_queue into fork_results dict; optionally inject assistant messages with [subagent xxx] into ctx.
    When called from get_subagent_result (mid-turn), use inject_into_ctx=False to avoid breaking tool_use/tool_result ordering."""
    queue = runtime_state.get("fork_result_queue")
    if not queue:
        return
    results = runtime_state.setdefault("fork_results", {})
    while True:
        try:
            job_id, skill_name, result = queue.get_nowait()
        except Empty:
            break
        results[job_id] = result
        if inject_into_ctx:
            ctx.add_assistant_message(f"[subagent {skill_name} job_id={job_id}] {result}")


def start_async_fork(
    runtime_state: dict,
    client: "OpenAI",
    model: str,
    skill: SkillSpec,
    task: str,
    logger: Any,
    return_mode: str = "final",
) -> str:
    """Start run_fork_skill in a background thread; return job_id.
    return_mode: 'final' = only last reply (default); 'full_context' = entire subagent conversation."""
    result_queue = runtime_state.get("fork_result_queue")
    if not result_queue:
        return "Error: no fork_result_queue in runtime state."
    counter = runtime_state.get("fork_job_counter", 0)
    runtime_state["fork_job_counter"] = counter + 1
    job_id = str(counter + 1)
    tool_schema_map = get_tool_schema_map()
    all_tool_schemas = get_all_tool_schemas()
    no_truncate = runtime_state.get("no_truncate", False)
    thread = threading.Thread(
        target=_run_fork_in_thread,
        args=(job_id, skill, task, client, model, tool_schema_map, all_tool_schemas, logger, result_queue, return_mode, no_truncate),
        daemon=True,
    )
    threads = runtime_state.setdefault("fork_threads", [])
    threads.append({"job_id": job_id, "skill_name": skill.name, "thread": thread})
    thread.start()
    return job_id


def _prune_fork_threads(runtime_state: dict) -> list[dict]:
    """Remove finished threads from fork_threads; return list of still-alive entries {job_id, skill_name, thread}."""
    threads = runtime_state.get("fork_threads") or []
    alive = [t for t in threads if t["thread"].is_alive()]
    runtime_state["fork_threads"] = alive
    return alive


def get_fork_threads_status(runtime_state: dict) -> list[dict]:
    """Return list of still-running fork jobs: [{"job_id": "1", "skill_name": "scout"}, ...]. Prunes finished threads."""
    alive = _prune_fork_threads(runtime_state)
    return [{"job_id": t["job_id"], "skill_name": t["skill_name"]} for t in alive]


def wait_for_fork_threads(runtime_state: dict, timeout_sec: float = 60.0) -> bool:
    """Wait for all fork threads to finish. Returns True if all ended within timeout, False if some still alive."""
    import time as _time
    alive = _prune_fork_threads(runtime_state)
    if not alive:
        return True
    deadline = _time.monotonic() + timeout_sec
    n = len(alive)
    for t in alive:
        remaining = max(0.0, deadline - _time.monotonic())
        if remaining <= 0:
            break
        t["thread"].join(timeout=min(remaining, max(0.1, timeout_sec / n)))
    _prune_fork_threads(runtime_state)
    return len(runtime_state.get("fork_threads") or []) == 0


@dataclass
class _TurnOutcome:
    """Result of a single agent turn (shared by interactive and headless modes)."""
    content: str = ""
    response_model: str = ""
    steps: int = 0
    tool_called: bool = False
    had_failure: bool = False
    preempted: bool = False
    pending_input: str | None = None  # set when preempted with queued input


@dataclass
class BatchTurnResult:
    assistant_content: str
    response_model: str
    steps: int
    tool_called: bool
    had_failure: bool
    error: str | None = None
    ctx_messages: list[dict] | None = None
    token_usage: dict[str, int] | None = None


def _token_usage_from_ctx(ctx: ContextManager) -> dict[str, int]:
    usage = {
        "prompt_tokens": ctx.stats.total_prompt_tokens,
        "completion_tokens": ctx.stats.total_completion_tokens,
        "total_tokens": ctx.stats.total_tokens,
    }
    if ctx.stats.total_cache_read_tokens:
        usage["cache_read_tokens"] = ctx.stats.total_cache_read_tokens
    if ctx.stats.total_cache_write_tokens:
        usage["cache_write_tokens"] = ctx.stats.total_cache_write_tokens
    return usage


def _run_single_turn(
    client: "OpenAI",
    model: str,
    ctx: ContextManager,
    runtime_state: dict,
    logger,
    user_input: str,
    tool_schema_map: dict,
    all_tool_schemas: list[dict],
    max_agent_steps: int,
    no_truncate: bool,
    recovery_cleanup: bool,
    use_streaming: bool = True,
    user_queue: "Queue | None" = None,
    domain_adapter=None,
    compact_client=None,
    compact_model=None,
) -> _TurnOutcome:
    """Execute one user turn: skill selection, agent step loop, tool execution, recovery.

    Shared by interactive (``use_streaming=True``, ``user_queue`` set) and
    headless/batch (``use_streaming=False``, ``user_queue=None``) modes.
    """
    turn_start_index = len(ctx.messages)
    ctx.add_user_message(user_input)

    # --- Skill & tool selection ---
    selected_skills = select_skills(user_input, runtime_state["skills"], runtime_state["pinned_skills"])
    if use_streaming:
        maybe_extend_skills_for_continuation(user_input, selected_skills, runtime_state)
    selected_skill_names = [skill.name for skill in selected_skills]
    active_tool_schemas = resolve_active_tool_schemas(selected_skills, tool_schema_map, all_tool_schemas)
    _fork_keywords = ("fork", "subagent", "子代理", "并行")
    if any(kw in user_input.lower() for kw in _fork_keywords):
        for fork_tool_name in ("fork_subagent", "get_subagent_result"):
            if fork_tool_name in tool_schema_map and not any(
                t.get("function", {}).get("name") == fork_tool_name for t in active_tool_schemas
            ):
                active_tool_schemas = active_tool_schemas + [tool_schema_map[fork_tool_name]]
    active_skill_prompt = build_skill_prompt(selected_skills)
    runtime_state["active_skill_names"] = selected_skill_names
    runtime_state["active_tool_schemas"] = active_tool_schemas
    runtime_state["active_skill_prompt"] = active_skill_prompt

    active_tool_names = [t.get("function", {}).get("name", "?") for t in active_tool_schemas]
    if selected_skill_names:
        logger.log(f"  [skills] active: {', '.join(selected_skill_names)}", level="debug")
    logger.log(f"  [tools] sending {len(active_tool_schemas)} tools: {', '.join(active_tool_names)}", level="debug")

    # --- Compaction check ---
    overhead_tokens = ctx.estimate_tokens(str(active_tool_schemas)) + ctx.estimate_tokens(active_skill_prompt)
    if ctx.needs_compaction(overhead_tokens=overhead_tokens):
        logger.log("  (Context approaching limit, auto-compacting...)", level="system")
        do_compact(
            client, runtime_state["model"], ctx, logger,
            current_overhead_tokens=overhead_tokens, level="low",
            compact_client=compact_client, compact_model=compact_model,
        )

    # --- Agent step loop ---
    likely_action_intent = has_action_intent(user_input) if use_streaming else has_action_intent(user_input)
    autonomy_nudge = ""
    did_call_tool = False
    recovery = RecoveryState()
    queued_input_while_waiting = False
    outcome = _TurnOutcome()
    set_autonomous_turn(True)

    steps = range(max_agent_steps) if max_agent_steps > 0 else itertools.count()
    for step_idx in steps:
        try:
            # --- Preemption check (interactive only) ---
            if user_queue is not None:
                pending_from_preempt = to_pending_input_from_preempt_event(
                    dequeue_user_input_nowait(user_queue), ctx,
                )
                if pending_from_preempt:
                    outcome.preempted = True
                    outcome.pending_input = pending_from_preempt
                    clear_preempt_request()
                    logger.log(
                        "  [preempt] "
                        + (
                            "Received terminal interrupt signal; finishing current turn."
                            if pending_from_preempt == "quit"
                            else "Received user input during autonomous loop; switching turns."
                        ),
                        level="system",
                    )
                    break

            # --- Build request ---
            request_messages = list(ctx.messages)
            if autonomy_nudge:
                request_messages.append({"role": "system", "content": autonomy_nudge})
            if active_skill_prompt:
                request_messages.append({"role": "system", "content": active_skill_prompt})

            response_model = runtime_state["model"]

            # --- Model call ---
            if use_streaming:
                turn_max_tokens = MAX_TOKENS_TOOL_TURN if (did_call_tool or step_idx > 0) else MAX_TOKENS_FINAL_TURN
                model_client = get_client_for_model(client, response_model)
                _streaming_fn = start_anthropic_streaming_call if is_anthropic_model(response_model) else start_async_streaming_call
                async_call = _streaming_fn(
                    client=model_client, model=response_model,
                    request_messages=request_messages,
                    active_tool_schemas=active_tool_schemas,
                    max_tokens=turn_max_tokens, logger=logger,
                )
                # Poll with preemption detection while waiting for model
                model_wait_deadline = time.monotonic() + 90
                while not async_call.done.wait(0.1):
                    if time.monotonic() > model_wait_deadline:
                        async_call.error = TimeoutError("Model call timed out after 90s")
                        async_call.done.set()
                        break
                    if user_queue is not None and outcome.pending_input is None:
                        pending_from_preempt = to_pending_input_from_preempt_event(
                            dequeue_user_input_nowait(user_queue), ctx,
                        )
                        if pending_from_preempt:
                            outcome.pending_input = pending_from_preempt
                            outcome.preempted = True
                            queued_input_while_waiting = True
                            clear_preempt_request()
                            logger.log(
                                "  [preempt] "
                                + (
                                    "Queued terminal interrupt while waiting model response; "
                                    "will switch turns after the current response."
                                    if pending_from_preempt == "quit"
                                    else "Queued new input while waiting model response; "
                                    "will switch turns after the current response."
                                ),
                                level="system",
                            )
                if async_call.error is not None:
                    raise async_call.error
                response = async_call.response
                if response is None:
                    raise RuntimeError("Model call ended without response.")
            else:
                # Headless: non-streaming, always use large max_tokens for code generation
                response = request_model_response(
                    client_or_router=client, model=response_model,
                    request_messages=request_messages,
                    active_tool_schemas=active_tool_schemas,
                    max_tokens=MAX_TOKENS_FINAL_TURN,
                )

            ctx.record_usage(response.usage, overhead_tokens=overhead_tokens)

            # --- Tool execution ---
            tool_outcome = process_tool_calls(response, ctx, logger, no_truncate)
            if tool_outcome.called:
                did_call_tool = True
                if tool_outcome.failures:
                    recovery.record_failures(tool_outcome.failures, tool_outcome.failure_kinds)
                    logger.log(
                        f"  [recovery] Detected failure ({recovery.last_failure_kind}): "
                        f"{tool_outcome.failures[-1]}",
                        level="system",
                    )
                    autonomy_nudge = build_recovery_nudge(recovery)
                else:
                    recovery.unresolved_failure = False
                    autonomy_nudge = ""
                continue

            # --- Final text response ---
            content = response.choices[0].message.content or ""
            should_continue = False
            if likely_action_intent and (max_agent_steps <= 0 or step_idx < max_agent_steps - 1):
                if not did_call_tool:
                    logger.log("  [autonomy] Suppressed no-tool reply; continuing...", level="system")
                    autonomy_nudge = AUTONOMY_NUDGE_TEXT
                    should_continue = True
                elif recovery.unresolved_failure:
                    logger.log("  [recovery] Suppressed unresolved-failure reply; continuing...", level="system")
                    autonomy_nudge = build_recovery_nudge(recovery)
                    should_continue = True
                elif is_procedural_confirmation(content):
                    logger.log("  [autonomy] Suppressed procedural-confirmation reply; continuing...", level="system")
                    autonomy_nudge = AUTONOMY_NUDGE_TEXT
                    should_continue = True
            if should_continue:
                continue

            ctx.add_assistant_message(content)
            if use_streaming:
                # Streaming already printed content; only log model tag + usage
                if content:
                    logger.log(f"  [{response_model}] {format_turn_usage(response.usage)}\n", level="info")
                else:
                    logger.log(f"\nAssistant [{response_model}]: (no content)", level="assistant")
                    logger.log(f"  {format_turn_usage(response.usage)}\n", level="info")
            else:
                logger.log(f"\nAssistant [{response_model}]: {content}", level="assistant")
                logger.log(f"  {format_turn_usage(response.usage)}\n", level="info")

            if recovery_cleanup and recovery.had_failure:
                removed = ctx.drop_failed_tool_messages(start_index=turn_start_index + 1)
                if removed > 0:
                    logger.log(
                        f"  [recovery] Cleaned {removed} failed intermediate messages from context.\n",
                        level="system",
                    )
            if domain_adapter is not None:
                domain_adapter.try_save_history(content, ctx, logger)
            if queued_input_while_waiting:
                logger.log("  [preempt] Current response delivered; switching to queued input.\n", level="system")

            outcome.content = content
            outcome.response_model = response_model
            outcome.steps = step_idx + 1
            outcome.tool_called = did_call_tool
            outcome.had_failure = recovery.had_failure
            break
        except Exception as exc:
            logger.log(f"\nError: {exc}\n", level="error")
            ctx.pop_last_message()
            outcome.steps = step_idx + 1
            break
    else:
        # Exhausted max_agent_steps
        help_msg = (
            f"I reached the max autonomous step limit ({max_agent_steps}) for this turn "
            "and may be stuck. Please provide guidance, constraints, or the next action."
        )
        if recovery.failures:
            help_msg += f" Latest failure: {recovery.failures[-1]}"
        ctx.add_assistant_message(help_msg)
        logger.log(f"\nAssistant: {help_msg}\n", level="assistant")
        outcome.content = help_msg
        outcome.response_model = model
        outcome.steps = max_agent_steps
        outcome.tool_called = did_call_tool
        outcome.had_failure = recovery.had_failure

    return outcome


def chat(
    client: "OpenAI",
    model: str,
    max_tokens: int,
    max_agent_steps: int,
    safe_shell: bool,
    recovery_cleanup: bool,
    preempt_shell_kill: bool,
    initial_message: str | None,
    compact_client: "OpenAI" | None,
    compact_model: str | None,
    logger,
    command_registry,
    domain_adapter,
    skills_root: Path,
    no_truncate: bool = False,
    headless: bool = False,
) -> BatchTurnResult | None:
    set_shell_safety(safe_shell)
    set_shell_interrupt_on_preempt(preempt_shell_kill)
    set_no_truncate(no_truncate)

    system_prompt = domain_adapter.compose_system_prompt()
    ctx = ContextManager(system_prompt=system_prompt, max_tokens=max_tokens)
    all_tool_schemas = get_all_tool_schemas()
    tool_schema_map = get_tool_schema_map()
    skills = load_skills(skills_root)

    default_selected_skills = select_skills("", skills, pinned_on=set())
    default_tool_schemas = resolve_active_tool_schemas(default_selected_skills, tool_schema_map, all_tool_schemas)
    default_skill_prompt = build_skill_prompt(default_selected_skills)
    default_skill_names = [skill.name for skill in default_selected_skills]

    runtime_state = {
        "verbose": False,
        "no_truncate": no_truncate,
        "safe_shell": safe_shell,
        "recovery_cleanup": recovery_cleanup,
        "skills": skills,
        "pinned_skills": set(),
        "active_skill_names": default_skill_names,
        "active_tool_schemas": default_tool_schemas,
        "active_skill_prompt": default_skill_prompt,
        "task_dir": domain_adapter.task_dir,
        "preempt_shell_kill": preempt_shell_kill,
        "compact_client": compact_client,
        "compact_model": compact_model,
        "model": model,
        "fork_result_queue": Queue(),
        "fork_job_counter": 0,
        "fork_results": {},
        "fork_threads": [],
    }

    logger.log("ICT Agent batch turn started." if headless else "ICT Agent ready.")
    logger.log(f"Model:        {model}")
    logger.log(f"Workspace:    {domain_adapter.workspace_root}" if headless else f"Workspace: {domain_adapter.workspace_root}")
    logger.log(f"Context:      {max_tokens:,} tokens")
    logger.log(
        f"Max steps:    {max_agent_steps if max_agent_steps > 0 else 'unlimited'}"
        if headless else
        f"Max autonomous steps per turn: {max_agent_steps if max_agent_steps > 0 else 'unlimited'}"
    )
    if not headless:
        logger.log(f"Tools loaded: {', '.join(tool['function']['name'] for tool in all_tool_schemas)}")
        if skills:
            logger.log(f"Skills loaded: {', '.join(sorted(skills.keys()))}")
        logger.log(f"Safe shell mode: {'on' if safe_shell else 'off'}")
        logger.log(f"Preempt shell-kill: {'on' if preempt_shell_kill else 'off'}")
        logger.log(f"Recovery cleanup: {'on' if recovery_cleanup else 'off'}")
        logger.log(f"Compact model: {compact_model or model}")
        if no_truncate:
            logger.log("No truncate: on (full output)")
    # System prompt preview (full when no_truncate, else first 12 lines)
    _preview = ctx.system_prompt.strip()
    _lines = _preview.split("\n")
    if no_truncate or (len(_lines) <= 12 and len(_preview) <= 800):
        _sp_preview = _preview
    else:
        _sp_preview = "\n".join(_lines[:12])
        if len(_lines) > 12:
            _sp_preview += f"\n... ({len(_lines) - 12} more lines, {len(_preview):,} chars total)"
    logger.log(f"System prompt:\n---\n{_sp_preview}\n---")
    if not headless:
        if domain_adapter.history_prompt:
            logger.log("History: loaded previous implementation for reference")
        if domain_adapter.task_context_source:
            logger.log(f"Task context: loaded from {domain_adapter.task_context_source}")
        if logger.is_live_session():
            logger.log("Send messages via: ict-agent send <message>")
        logger.log("Type /help for commands, 'quit' or Ctrl+C to exit.\n")

    # --- Headless mode: single turn, no stdin, return BatchTurnResult ---
    if headless:
        user_input = (initial_message or "").strip()
        if not user_input:
            return BatchTurnResult(
                assistant_content="", response_model=model, steps=0,
                tool_called=False, had_failure=False, error="Empty user input",
                ctx_messages=list(ctx.messages), token_usage=_token_usage_from_ctx(ctx),
            )
        try:
            set_current_runtime(ctx=ctx, runtime_state=runtime_state, client=client, logger=logger)
            outcome = _run_single_turn(
                client=client, model=model, ctx=ctx,
                runtime_state=runtime_state, logger=logger,
                user_input=user_input,
                tool_schema_map=tool_schema_map, all_tool_schemas=all_tool_schemas,
                max_agent_steps=max_agent_steps,
                no_truncate=no_truncate, recovery_cleanup=recovery_cleanup,
                use_streaming=False, user_queue=None,
            )
            return BatchTurnResult(
                assistant_content=outcome.content,
                response_model=outcome.response_model,
                steps=outcome.steps,
                tool_called=outcome.tool_called,
                had_failure=outcome.had_failure,
                ctx_messages=list(ctx.messages),
                token_usage=_token_usage_from_ctx(ctx),
            )
        except Exception as exc:
            logger.log(f"\nError: {exc}\n", level="error")
            return BatchTurnResult(
                assistant_content="", response_model=model, steps=0,
                tool_called=False, had_failure=False, error=str(exc),
                ctx_messages=list(ctx.messages), token_usage=_token_usage_from_ctx(ctx),
            )
        finally:
            set_autonomous_turn(False)
            clear_preempt_request()
            clear_current_runtime()

    # --- Interactive mode ---
    user_queue: Queue[tuple[str, str]] = Queue()
    reader_stop_event = threading.Event()
    reader_thread = InputReaderThread(user_queue, reader_stop_event)
    reader_thread.start()
    pending_input = initial_message

    try:
        while True:
            if pending_input is not None:
                user_input = pending_input
                pending_input = None
            else:
                logger.log(">>> Ready for input.")
                logger.print_user_prompt()
                # Poll with short timeout so we can drain fork results and print them while waiting
                event, user_input = None, None
                while True:
                    item = dequeue_user_input_with_timeout(user_queue, timeout=1.0)
                    if item is None:
                        # No input; drain any completed fork results, inject into ctx, and print to terminal
                        n_before = len(ctx.messages)
                        drain_fork_results(ctx, runtime_state)
                        for msg in ctx.messages[n_before:]:
                            if msg.get("role") == "assistant":
                                content = (msg.get("content") or "").strip()
                                if content.startswith("[subagent "):
                                    logger.log("\n[Subagent result]\n" + content + "\n", level="info")
                        continue
                    event, user_input = item
                    if event in ("eof", "interrupt"):
                        break
                    if event == "input" and user_input:
                        break
                    # empty line, keep waiting
                    continue
                logger.reset_style()
                if event in ("eof", "interrupt"):
                    logger.log("\nBye!")
                    break

            if not user_input:
                continue
            logger.log(f"You: {user_input}", level="user")
            if user_input.lower() in ("quit", "exit", "q"):
                logger.log("Bye!", level="info")
                break
            normalized_command = normalize_command_input(user_input)
            if normalized_command.startswith("/"):
                handled = command_registry.dispatch(
                    normalized_command,
                    CommandContext(
                        client=client,
                        ctx=ctx,
                        runtime_state=runtime_state,
                        logger=logger,
                        domain_adapter=domain_adapter,
                    ),
                )
                if not handled:
                    logger.log("\nUnknown command. Type /help.\n", level="error")
                continue

            drain_fork_results(ctx, runtime_state)
            set_current_runtime(ctx=ctx, runtime_state=runtime_state, client=client, logger=logger)

            outcome = _run_single_turn(
                client=client, model=model, ctx=ctx,
                runtime_state=runtime_state, logger=logger,
                user_input=user_input,
                tool_schema_map=tool_schema_map, all_tool_schemas=all_tool_schemas,
                max_agent_steps=max_agent_steps,
                no_truncate=no_truncate, recovery_cleanup=recovery_cleanup,
                use_streaming=True, user_queue=user_queue,
                domain_adapter=domain_adapter,
                compact_client=compact_client, compact_model=compact_model,
            )

            set_autonomous_turn(False)
            clear_preempt_request()
            if outcome.preempted:
                pending_input = outcome.pending_input
                continue
    finally:
        set_autonomous_turn(False)
        clear_preempt_request()
        reader_stop_event.set()
        reader_thread.join(timeout=1.0)
    return None

