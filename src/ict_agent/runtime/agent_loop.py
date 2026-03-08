"""Core interactive agent loop."""

from __future__ import annotations

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
from ict_agent.skills import build_skill_prompt, load_skills, select_skills
from ict_agent.skills import SkillSpec
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
            # Log request size for 400 debugging (e.g. gpt-oss vs mco-4 tool_calls strictness)
            try:
                n_msgs = len(request_messages)
                n_tools = len(active_tool_schemas)
                err_str = str(exc)
                if "400" in err_str or "Bad Request" in err_str:
                    import logging
                    logging.getLogger(__name__).warning(
                        "API 400 on chat.completions.create: model=%s messages=%s tools=%s err=%s",
                        model, n_msgs, n_tools, err_str[:200],
                    )
            except Exception:  # noqa: S110
                pass
        finally:
            call.done.set()

    threading.Thread(target=worker, daemon=True).start()
    return call


@dataclass
class BatchTurnResult:
    assistant_content: str
    response_model: str
    steps: int
    tool_called: bool
    had_failure: bool
    error: str = ""
    ctx_messages: list[dict] | None = None


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
) -> str:
    """Run a fork (Agent as Skill) sub-agent: isolated context, skill's tools only, single task.

    return_mode: "final" = return only the last assistant reply (default; saves main context).
                 "full_context" = return entire subagent conversation as one blob (for handoff).
    Returns the final assistant message content, or full context string, or an error message.
    """
    fork_log = _ForkLogger(logger, f"[fork:{skill.name}]")
    fork_log.log(f"Starting fork task: {task[:200]}{'...' if len(task) > 200 else ''}", level="system")
    ctx = ContextManager(system_prompt=skill.instructions or "Complete the user's task.", max_tokens=max_tokens)
    ctx.add_user_message(task)
    active_tool_schemas = resolve_active_tool_schemas([skill], tool_schema_map, all_tool_schemas)
    if not active_tool_schemas:
        return "Error: fork skill has no tools available (check skill's tools list and tool registry)."

    last_content = ""
    for step_idx in range(max_steps):
        try:
            request_messages = list(ctx.messages)
            async_call = start_async_model_call(
                client=client,
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

            tool_outcome = process_tool_calls(response, ctx, fork_log)
            if tool_outcome.called:
                continue

            content = (response.choices[0].message.content or "").strip()
            last_content = content
            ctx.add_assistant_message(content)
            fork_log.log(f"Completed in {step_idx + 1} step(s).", level="system")
            fork_log.log(content[:500] + ("..." if len(content) > 500 else ""), level="assistant")
            if return_mode == "full_context":
                return _serialize_fork_context(ctx.messages)
            return content
        except Exception as exc:
            err_msg = f"Error in fork step {step_idx + 1}: {exc}"
            fork_log.log(err_msg, level="error")
            return err_msg

    msg = (
        f"Fork reached max steps ({max_steps}) without final reply. "
        + (f"Last content: {last_content[:300]}..." if len(last_content) > 300 else f"Last content: {last_content}")
    )
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
    thread = threading.Thread(
        target=_run_fork_in_thread,
        args=(job_id, skill, task, client, model, tool_schema_map, all_tool_schemas, logger, result_queue, return_mode),
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
        "fork_result_queue": Queue(),
        "fork_job_counter": 0,
        "fork_results": {},
        "fork_threads": [],
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
            turn_start_index = len(ctx.messages)
            ctx.add_user_message(user_input)

            selected_skills = select_skills(user_input, runtime_state["skills"], runtime_state["pinned_skills"])
            maybe_extend_skills_for_continuation(user_input, selected_skills, runtime_state)
            selected_skill_names = [skill.name for skill in selected_skills]
            active_tool_schemas = resolve_active_tool_schemas(selected_skills, tool_schema_map, all_tool_schemas)
            for fork_tool_name in ("fork_subagent", "get_subagent_result"):
                if fork_tool_name in tool_schema_map and not any(
                    t.get("function", {}).get("name") == fork_tool_name for t in active_tool_schemas
                ):
                    active_tool_schemas = active_tool_schemas + [tool_schema_map[fork_tool_name]]
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
                    model_wait_deadline = time.monotonic() + 90  # 90s max per turn (avoid indefinite hang)
                    while not async_call.done.wait(0.1):
                        if time.monotonic() > model_wait_deadline:
                            async_call.error = TimeoutError("Model call timed out after 90s")
                            async_call.done.set()
                            break
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


def run_batch_turn(
    client: "OpenAI",
    model: str,
    user_input: str,
    max_tokens: int,
    max_agent_steps: int,
    safe_shell: bool,
    recovery_cleanup: bool,
    preempt_shell_kill: bool,
    compact_client: "OpenAI" | None,
    compact_model: str | None,
    logger,
    command_registry,
    domain_adapter,
    skills_root: Path,
) -> BatchTurnResult:
    del command_registry  # Batch mode does not dispatch slash commands.

    set_shell_safety(safe_shell)
    set_shell_interrupt_on_preempt(preempt_shell_kill)

    system_prompt = domain_adapter.compose_system_prompt()
    ctx = ContextManager(system_prompt=system_prompt, max_tokens=max_tokens)
    all_tool_schemas = get_all_tool_schemas()
    tool_schema_map = get_tool_schema_map()
    skills = load_skills(skills_root)

    default_selected_skills = select_skills("", skills, pinned_on=set())
    default_tool_schemas = resolve_active_tool_schemas(
        default_selected_skills, tool_schema_map, all_tool_schemas
    )
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
        "fork_result_queue": Queue(),
        "fork_job_counter": 0,
        "fork_results": {},
        "fork_threads": [],
    }

    logger.log("ICT Agent batch turn started.")
    logger.log(f"Model:        {model}")
    logger.log(f"Workspace:    {domain_adapter.workspace_root}")
    logger.log(f"Context:      {max_tokens:,} tokens")
    logger.log(f"Max steps:    {max_agent_steps}")

    if not user_input.strip():
        return BatchTurnResult(
            assistant_content="",
            response_model=model,
            steps=0,
            tool_called=False,
            had_failure=False,
            error="Empty user input",
            ctx_messages=list(ctx.messages),
        )

    try:
        set_current_runtime(ctx=ctx, runtime_state=runtime_state, client=client, logger=logger)
        turn_start_index = len(ctx.messages)
        ctx.add_user_message(user_input)

        selected_skills = select_skills(
            user_input, runtime_state["skills"], runtime_state["pinned_skills"]
        )
        active_tool_schemas = resolve_active_tool_schemas(
            selected_skills, tool_schema_map, all_tool_schemas
        )
        for fork_tool_name in ("fork_subagent", "get_subagent_result"):
            if fork_tool_name in tool_schema_map and not any(
                t.get("function", {}).get("name") == fork_tool_name
                for t in active_tool_schemas
            ):
                active_tool_schemas = active_tool_schemas + [tool_schema_map[fork_tool_name]]
        active_skill_prompt = build_skill_prompt(selected_skills)
        runtime_state["active_skill_names"] = [skill.name for skill in selected_skills]
        runtime_state["active_tool_schemas"] = active_tool_schemas
        runtime_state["active_skill_prompt"] = active_skill_prompt

        overhead_tokens = ctx.estimate_tokens(str(active_tool_schemas)) + ctx.estimate_tokens(
            active_skill_prompt
        )
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

        likely_action_intent = has_action_intent(user_input)
        autonomy_nudge = ""
        did_call_tool = False
        recovery = RecoveryState()
        set_autonomous_turn(True)

        for step_idx in range(max_agent_steps):
            request_messages = list(ctx.messages)
            if autonomy_nudge:
                request_messages.append({"role": "system", "content": autonomy_nudge})
            if active_skill_prompt:
                request_messages.append({"role": "system", "content": active_skill_prompt})

            response_model = runtime_state["model"]
            response = client.chat.completions.create(
                model=response_model,
                messages=request_messages,
                tools=active_tool_schemas,
            )
            ctx.record_usage(response.usage, overhead_tokens=overhead_tokens)

            tool_outcome = process_tool_calls(response, ctx, logger)
            if tool_outcome.called:
                did_call_tool = True
                if tool_outcome.failures:
                    recovery.record_failures(
                        tool_outcome.failures, tool_outcome.failure_kinds
                    )
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
                    logger.log(
                        "  [autonomy] Suppressed no-tool reply; continuing...",
                        level="system",
                    )
                    autonomy_nudge = AUTONOMY_NUDGE_TEXT
                    should_continue = True
                elif recovery.unresolved_failure:
                    logger.log(
                        "  [recovery] Suppressed unresolved-failure reply; continuing...",
                        level="system",
                    )
                    autonomy_nudge = build_recovery_nudge(recovery)
                    should_continue = True
                elif is_procedural_confirmation(content):
                    logger.log(
                        "  [autonomy] Suppressed procedural-confirmation reply; continuing...",
                        level="system",
                    )
                    autonomy_nudge = AUTONOMY_NUDGE_TEXT
                    should_continue = True
            if should_continue:
                continue

            ctx.add_assistant_message(content)
            logger.log(f"\nAssistant [{response_model}]: {content}", level="assistant")
            logger.log(f"  {format_turn_usage(response.usage)}\n", level="info")
            if recovery_cleanup and recovery.had_failure:
                removed = ctx.drop_failed_tool_messages(start_index=turn_start_index + 1)
                if removed > 0:
                    logger.log(
                        f"  [recovery] Cleaned {removed} failed intermediate messages from context.\n",
                        level="system",
                    )
            return BatchTurnResult(
                assistant_content=content,
                response_model=response_model,
                steps=step_idx + 1,
                tool_called=did_call_tool,
                had_failure=recovery.had_failure,
                ctx_messages=list(ctx.messages),
            )

        help_msg = (
            f"I reached the max autonomous step limit ({max_agent_steps}) for this turn "
            "and may be stuck. Please provide guidance, constraints, or the next action."
        )
        if recovery.failures:
            help_msg += f" Latest failure: {recovery.failures[-1]}"
        ctx.add_assistant_message(help_msg)
        logger.log(f"\nAssistant [{model}]: {help_msg}", level="assistant")
        return BatchTurnResult(
            assistant_content=help_msg,
            response_model=model,
            steps=max_agent_steps,
            tool_called=did_call_tool,
            had_failure=recovery.had_failure,
            error="max_agent_steps_reached",
            ctx_messages=list(ctx.messages),
        )
    except Exception as exc:
        logger.log(f"\nError: {exc}\n", level="error")
        return BatchTurnResult(
            assistant_content="",
            response_model=model,
            steps=0,
            tool_called=False,
            had_failure=False,
            error=str(exc),
            ctx_messages=list(ctx.messages),
        )
    finally:
        set_autonomous_turn(False)
        clear_preempt_request()
        clear_current_runtime()
