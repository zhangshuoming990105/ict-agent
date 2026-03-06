"""Core interactive agent loop."""

from __future__ import annotations

from queue import Queue
import re
import threading
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
    to_pending_input_from_preempt_event,
)
from ict_agent.skills import build_skill_prompt, load_skills, select_skills
from ict_agent.tools import execute_tool, get_all_tool_schemas, get_tool_schema_map, set_shell_safety


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


def start_async_model_call(
    client: "OpenAI",
    model: str,
    request_messages: list[dict],
    active_tool_schemas: list[dict],
) -> AsyncModelCall:
    call = AsyncModelCall()

    def worker() -> None:
        try:
            call.response = client.chat.completions.create(
                model=model,
                messages=request_messages,
                tools=active_tool_schemas,
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


def resolve_active_tool_schemas(selected_skills, tool_schema_map: dict[str, dict], all_tool_schemas: list[dict]) -> list[dict]:
    selected_tool_names = [
        tool_name
        for skill in selected_skills
        for tool_name in skill.tools
        if tool_name in tool_schema_map
    ]
    if not selected_tool_names:
        return all_tool_schemas
    return [tool_schema_map[name] for name in unique_preserve_order(selected_tool_names)]


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


def process_tool_calls(response, ctx: ContextManager, logger) -> ToolExecutionOutcome:
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
        display = result if len(result) <= 2000 else result[:1000] + "\n...(truncated)...\n" + result[-500:]
        logger.log(f"  <- Result: {display}", level="result")
        ctx.add_tool_result(tool_call.id, name, result)
        if is_tool_failure(name, result):
            failures.append(summarize_failure(name, result))
            failure_kinds.append(classify_cuda_failure(name, result))
    return ToolExecutionOutcome(called=True, failures=failures, failure_kinds=failure_kinds)


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
) -> None:
    set_shell_safety(safe_shell)
    set_shell_interrupt_on_preempt(preempt_shell_kill)

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
    }

    logger.log("ICT Agent ready.")
    logger.log(f"Model:        {model}")
    logger.log(f"Context:      {max_tokens:,} tokens")
    logger.log(f"Tools loaded: {', '.join(tool['function']['name'] for tool in all_tool_schemas)}")
    if skills:
        logger.log(f"Skills loaded: {', '.join(sorted(skills.keys()))}")
    logger.log(f"Safe shell mode: {'on' if safe_shell else 'off'}")
    logger.log(f"Preempt shell-kill: {'on' if preempt_shell_kill else 'off'}")
    logger.log(f"Recovery cleanup: {'on' if recovery_cleanup else 'off'}")
    logger.log(f"Max autonomous steps per turn: {max_agent_steps}")
    logger.log(f"Compact model: {compact_model or model}")
    logger.log(f"Workspace: {domain_adapter.workspace_root}")
    if domain_adapter.history_prompt:
        logger.log("History: loaded previous implementation for reference")
    if domain_adapter.task_context_source:
        logger.log(f"Task context: loaded from {domain_adapter.task_context_source}")
    logger.log("Type /help for commands, 'quit' to exit.\n")

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
                event, text = dequeue_user_input_blocking(user_queue)
                logger.reset_style()
                if event in ("eof", "interrupt"):
                    logger.log("\nBye!")
                    break
                user_input = text

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

            turn_start_index = len(ctx.messages)
            ctx.add_user_message(user_input)

            selected_skills = select_skills(user_input, runtime_state["skills"], runtime_state["pinned_skills"])
            maybe_extend_skills_for_continuation(user_input, selected_skills, runtime_state)
            selected_skill_names = [skill.name for skill in selected_skills]
            active_tool_schemas = resolve_active_tool_schemas(selected_skills, tool_schema_map, all_tool_schemas)
            active_skill_prompt = build_skill_prompt(selected_skills)
            runtime_state["active_skill_names"] = selected_skill_names
            runtime_state["active_tool_schemas"] = active_tool_schemas
            runtime_state["active_skill_prompt"] = active_skill_prompt

            if selected_skill_names:
                logger.log(f"  [skills] active: {', '.join(selected_skill_names)}", level="debug")

            overhead_tokens = ctx.estimate_tokens(str(active_tool_schemas)) + ctx.estimate_tokens(active_skill_prompt)
            if ctx.needs_compaction(overhead_tokens=overhead_tokens):
                logger.log("  (Context approaching limit, auto-compacting...)", level="system")
                do_compact(
                    client,
                    runtime_state["model"],
                    ctx,
                    logger,
                    current_overhead_tokens=overhead_tokens,
                    level="low",
                    compact_client=compact_client,
                    compact_model=compact_model,
                )

            # Autonomy suppression disabled; multiagent will handle orchestration.
            likely_action_intent = False
            autonomy_nudge = ""
            did_call_tool = False
            recovery = RecoveryState()
            preempted = False
            set_autonomous_turn(True)

            for step_idx in range(max_agent_steps):
                try:
                    pending_from_preempt = to_pending_input_from_preempt_event(
                        dequeue_user_input_nowait(user_queue),
                        ctx,
                    )
                    if pending_from_preempt:
                        pending_input = pending_from_preempt
                        preempted = True
                        clear_preempt_request()
                        logger.log(
                            "  [preempt] "
                            + (
                                "Received terminal interrupt signal; finishing current turn."
                                if pending_input == "quit"
                                else "Received user input during autonomous loop; switching turns."
                            ),
                            level="system",
                        )
                        break

                    request_messages = list(ctx.messages)
                    if autonomy_nudge:
                        request_messages.append({"role": "system", "content": autonomy_nudge})
                    if active_skill_prompt:
                        request_messages.append({"role": "system", "content": active_skill_prompt})

                    response_model = runtime_state["model"]
                    async_call = start_async_model_call(
                        client=client,
                        model=response_model,
                        request_messages=request_messages,
                        active_tool_schemas=active_tool_schemas,
                    )
                    queued_input_while_waiting = False
                    while not async_call.done.wait(0.1):
                        if pending_input is None:
                            pending_from_preempt = to_pending_input_from_preempt_event(
                                dequeue_user_input_nowait(user_queue),
                                ctx,
                            )
                            if pending_from_preempt:
                                pending_input = pending_from_preempt
                                preempted = True
                                queued_input_while_waiting = True
                                clear_preempt_request()
                                logger.log(
                                    "  [preempt] "
                                    + (
                                        "Queued terminal interrupt while waiting model response; "
                                        "will switch turns after the current response."
                                        if pending_input == "quit"
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
                    ctx.record_usage(response.usage, overhead_tokens=overhead_tokens)

                    tool_outcome = process_tool_calls(response, ctx, logger)
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

                    content = response.choices[0].message.content or ""
                    should_continue = False
                    if likely_action_intent and step_idx < max_agent_steps - 1:
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
                    logger.log(f"\nAssistant [{response_model}]: {content}", level="assistant")
                    logger.log(f"  {format_turn_usage(response.usage)}\n", level="info")
                    if runtime_state.get("recovery_cleanup", True) and recovery.had_failure:
                        removed = ctx.drop_failed_tool_messages(start_index=turn_start_index + 1)
                        if removed > 0:
                            logger.log(
                                f"  [recovery] Cleaned {removed} failed intermediate messages from context.\n",
                                level="system",
                            )
                    domain_adapter.try_save_history(content, ctx, logger)
                    if queued_input_while_waiting:
                        logger.log("  [preempt] Current response delivered; switching to queued input.\n", level="system")
                    break
                except Exception as exc:
                    logger.log(f"\nError: {exc}\n", level="error")
                    ctx.pop_last_message()
                    break
            else:
                help_msg = (
                    f"I reached the max autonomous step limit ({max_agent_steps}) for this turn "
                    "and may be stuck. Please provide guidance, constraints, or the next action."
                )
                if recovery.failures:
                    help_msg += f" Latest failure: {recovery.failures[-1]}"
                ctx.add_assistant_message(help_msg)
                logger.log(f"\nAssistant: {help_msg}\n", level="assistant")

            set_autonomous_turn(False)
            clear_preempt_request()
            if preempted:
                continue
    finally:
        set_autonomous_turn(False)
        clear_preempt_request()
        reader_stop_event.set()
        reader_thread.join(timeout=1.0)
