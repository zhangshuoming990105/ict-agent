"""Conversation compaction helpers."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


COMPACTOR_SYSTEM_PROMPT_LOW = """\
You are a conversation compressor. Given older conversation history, produce a \
shorter version that preserves essential information for future turns.

Rules:
1. Output MUST be a valid JSON array of messages: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}].
2. Keep multi-turn structure as user/assistant pairs; do not output one big paragraph.
3. For user messages: keep short prompts verbatim; summarize long prompts to intent + constraints.
4. For assistant messages: keep deliverables only (final answer, code, commands, file paths, key facts).
5. Preserve critical exact fields when present: file paths, symbols, commands, versions, numeric results.
6. Remove chain-of-thought style reasoning, retries, failed attempts, and repetitive explanations.
7. For tool sequences, convert to concise result-oriented assistant messages.
8. Merge adjacent messages of same role when it improves clarity.
9. These are OLD turns only. Compress old code/tool outputs aggressively.
10. Assume recent turns are preserved elsewhere; avoid repeating recent details.
11. Write in the same language as source conversation.
12. Output ONLY JSON array, no markdown, no commentary.
"""

COMPACTOR_SYSTEM_PROMPT_HIGH = """\
You are an aggressive conversation compressor. Given older conversation history, produce the \
most compact possible summary that still lets future turns proceed correctly.

Rules:
1. Output MUST be a valid JSON array of messages: [{"role":"user","content":"..."},{"role":"assistant","content":"..."}].
2. Collapse the entire history into as few user/assistant pairs as possible (ideally 1-3 pairs).
3. Preserve ONLY hard facts: exact file paths, function names, shell commands, numeric results, and error messages.
4. Discard reasoning, verbose tool outputs, and intermediate retries.
5. Write in the same language as source conversation.
6. Output ONLY JSON array, no markdown, no commentary.
"""

COMPACTOR_SYSTEM_PROMPT = COMPACTOR_SYSTEM_PROMPT_LOW

_MAX_OUTPUT_TOKENS = {
    "low": 2400,
    "high": 2000,
}
_TOKENS_PER_INPUT_MESSAGE = 30
_BASE_MESSAGE_THRESHOLD = 20


def compact_messages(
    client: "OpenAI",
    model: str,
    messages_to_compact: list[dict],
    level: str = "low",
    max_output_tokens: int | None = None,
) -> list[dict] | None:
    """Compact older messages using the configured model."""
    if len(messages_to_compact) < 4:
        return None

    level = level if level in ("low", "high") else "low"
    system_prompt = (
        COMPACTOR_SYSTEM_PROMPT_HIGH
        if level == "high"
        else COMPACTOR_SYSTEM_PROMPT_LOW
    )

    if max_output_tokens is not None:
        budget = max_output_tokens
    else:
        base = _MAX_OUTPUT_TOKENS[level]
        message_count = len(messages_to_compact)
        if message_count > _BASE_MESSAGE_THRESHOLD:
            budget = base + (message_count - _BASE_MESSAGE_THRESHOLD) * _TOKENS_PER_INPUT_MESSAGE
        else:
            budget = base

    conversation_text = _format_messages_for_compaction(messages_to_compact)

    def call_once() -> list[dict] | None:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text},
            ],
            max_tokens=budget,
            temperature=0.0,
        )
        choice = response.choices[0]
        raw = choice.message.content or ""
        if choice.finish_reason == "length" and raw:
            raw = _repair_truncated_json(raw)
        result = _parse_compacted_output(raw)
        if result is None:
            out_tokens = getattr(response.usage, "completion_tokens", "?")
            print(
                "  (Compaction parse failed. "
                f"finish_reason={choice.finish_reason}, "
                f"output_tokens={out_tokens}, budget={budget})"
            )
        return result

    max_retries = 2
    for attempt in range(max_retries):
        try:
            return call_once()
        except Exception as exc:
            err = str(exc)
            is_rate_limit = "429" in err or "rate limit" in err.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait_seconds = 10 * (attempt + 1)
                print(
                    "  (Compaction rate-limited, "
                    f"retrying in {wait_seconds}s... attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait_seconds)
                continue
            print(f"  (Compaction failed: {exc})")
            return None
    return None


def _repair_truncated_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]).rstrip("`").strip()
    last_brace = text.rfind("}")
    if last_brace == -1:
        return raw
    candidate = text[: last_brace + 1].rstrip().rstrip(",")
    if not candidate.lstrip().startswith("["):
        candidate = "[" + candidate
    return candidate + "]"


def _format_messages_for_compaction(messages: list[dict]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "") or ""
        if role == "tool":
            parts.append(f"[TOOL RESULT: {msg.get('name', 'unknown_tool')}]\n{content}")
        elif role == "assistant" and not content and msg.get("tool_calls"):
            calls_desc = []
            for tool_call in msg["tool_calls"]:
                fn = tool_call.get("function", {})
                calls_desc.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
            parts.append(f"ASSISTANT: [Called tools: {', '.join(calls_desc)}]")
        else:
            parts.append(f"{role.upper()}: {content}")
    return "\n\n".join(parts)


def _parse_compacted_output(raw: str) -> list[dict] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None

    result: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            result.append({"role": role, "content": content})
    return result if len(result) >= 2 else None
