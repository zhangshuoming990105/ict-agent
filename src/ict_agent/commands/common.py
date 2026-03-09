"""Common slash commands shared across domains."""

from __future__ import annotations

import json

from ict_agent.compactor import compact_messages
from ict_agent.context import ContextManager
from ict_agent.tools import (
    get_all_tool_schemas,
    get_shell_policy_snapshot,
    get_tool_schema_map,
    set_shell_safety,
)


def format_turn_usage(usage) -> str:
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0
    # Anthropic: cache_read_input_tokens / cache_creation_input_tokens
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    # OpenAI: prompt_tokens_details.cached_tokens
    if not cache_read:
        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd:
            cache_read = getattr(ptd, "cached_tokens", 0) or 0
    base = f"[tokens: prompt={prompt:,}, completion={completion:,}, total={total:,}"
    if cache_read or cache_write:
        base += f", cache_read={cache_read:,}, cache_write={cache_write:,}"
    base += "]"
    return base


def estimate_schema_tokens(ctx: ContextManager, tool_schemas: list[dict]) -> int:
    return ctx.estimate_tokens(json.dumps(tool_schemas, ensure_ascii=False))


def estimate_skill_tokens(ctx: ContextManager, skill_prompt: str) -> int:
    if not skill_prompt:
        return 0
    return ctx.estimate_tokens(skill_prompt) + 4


def render_token_report(
    ctx: ContextManager,
    tool_schemas: list[dict],
    skill_prompt: str,
    verbose: bool = False,
    active_model: str = "",
) -> str:
    schema_tokens = estimate_schema_tokens(ctx, tool_schemas)
    skill_tokens = estimate_skill_tokens(ctx, skill_prompt)
    diag = ctx.get_token_diagnostics(
        schema_tokens_estimate=schema_tokens,
        skill_tokens_estimate=skill_tokens,
    )
    utilization = diag["effective"] / ctx.max_tokens * 100
    if not verbose:
        lines = [
            *(["Model:        " + active_model] if active_model else []),
            f"Messages:     {len(ctx.messages)} ({len(ctx.messages) - 1} excluding system)",
            f"Requests:     {ctx.stats.total_requests}",
            "",
            "Cumulative API-reported usage:",
            f"  Prompt:     {ctx.stats.total_prompt_tokens:,} tokens",
            f"  Completion: {ctx.stats.total_completion_tokens:,} tokens",
            f"  Total:      {ctx.stats.total_tokens:,} tokens",
            "",
            "Context accounting:",
            f"  Managed:    ~{diag['structured']:,} tokens",
            f"  Overhead:   ~{diag['overhead_estimate']:,} tokens",
            f"  Effective:  ~{diag['effective']:,} tokens",
            f"  Limit:      {ctx.max_tokens:,} tokens",
            f"  Usage:      {utilization:.1f}%",
        ]
        return "\n".join(lines)
    lines = [
        f"Messages:     {len(ctx.messages)} ({len(ctx.messages) - 1} excluding system)",
        f"Requests:     {ctx.stats.total_requests}",
        "",
        "Cumulative API-reported usage:",
        f"  Prompt:     {ctx.stats.total_prompt_tokens:,} tokens",
        f"  Completion: {ctx.stats.total_completion_tokens:,} tokens",
        f"  Total:      {ctx.stats.total_tokens:,} tokens",
        "",
        "Current context (verbose):",
        f"  Managed:    ~{diag['structured']:,} tokens",
        f"  Content:    ~{diag['content_only']:,} tokens",
        f"  Schemas:    ~{diag['schema_estimate']:,} tokens",
        f"  Skills:     ~{diag['skill_estimate']:,} tokens",
        f"  Overhead:   ~{diag['overhead_estimate']:,} tokens",
        f"  Effective:  ~{diag['effective']:,} tokens",
        f"  Hidden:     ~{diag['hidden_overhead_estimate']:,} tokens",
        f"  Limit:      {ctx.max_tokens:,} tokens",
        f"  Usage:      {utilization:.1f}%",
    ]
    return "\n".join(lines)


def do_compact(
    client,
    model: str,
    ctx: ContextManager,
    logger,
    current_overhead_tokens: int = 0,
    level: str = "low",
    compact_client=None,
    compact_model: str | None = None,
) -> None:
    keep_recent = 6
    droppable = len(ctx.messages) - 1 - keep_recent
    if droppable < 4:
        logger.log(
            f"  Not enough old messages to compact ({len(ctx.messages) - 1} total, need >{keep_recent + 3}).\n",
            level="system",
        )
        return
    active_client = compact_client if compact_client is not None else client
    active_model = compact_model if compact_model else model
    before = ctx.get_context_tokens(overhead_tokens=current_overhead_tokens)
    before_local = ctx.estimate_messages_tokens_structured()
    old_messages = ctx.messages[1:-keep_recent]
    logger.log(f"  Compacting {len(old_messages)} old messages (level={level}, model={active_model})...", level="system")
    compacted = compact_messages(active_client, active_model, old_messages, level=level)
    if not compacted:
        logger.log("  Compaction failed - context unchanged.\n", level="error")
        return
    candidate_local = ctx.estimate_messages_tokens_structured(
        [ctx.messages[0]] + compacted + ctx.messages[-keep_recent:]
    )
    if candidate_local >= before_local:
        logger.log("  Compacted version is not smaller - skipped.\n", level="system")
        return
    replaced, new_count = ctx.apply_compacted_messages(compacted, keep_recent)
    after = ctx.get_context_tokens(overhead_tokens=current_overhead_tokens)
    logger.log(
        f"  Compressed {replaced} old messages -> {new_count} summary message(s) "
        f"(+ {keep_recent} recent kept). Context: ~{before:,} -> ~{after:,} tokens\n",
        level="success",
    )


SLASH_COMMANDS_HELP = """Available commands:
  /tokens       - Show token usage statistics
  /history      - Show all messages with token estimates (compact)
  /debug        - Show full context with readable summary
  /debug raw    - Show raw OpenAI messages array
  /model <name> - Switch the active LLM model for subsequent turns
  /set-model <name> - Switch the active LLM model for subsequent turns
  /compact [low|high] - Smart context compaction
  /skills       - Show loaded skills and currently active skills
  /skill <name> on|off - Pin or unpin a skill
  /run <skill> <task> - Run a fork skill synchronously; result injected when done
  /fork <skill> [full_context] <task> - Start fork; optional full_context returns entire subagent log
  /fork-status   - Show whether any fork subagent threads are still running
  /fork-wait [sec] - Wait for all fork subagents to finish (default 60s)
  /verbose      - Show or set verbose token diagnostics: /verbose on|off
  /shell-safe   - Toggle safe shell mode: /shell-safe on|off
  /shell-policy - Show current shell allowlist/denylist
  /preempt      - Show preemption settings
  /preempt shell-kill on|off - Toggle killing long shell commands on preempt
  /recovery     - Show or set recovery cleanup: /recovery on|off
  /workspace    - Show current task workspace info
  /clear        - Reset conversation history
  /help         - Show this help message"""


def handle_common_command(command: str, cmd_ctx) -> bool:
    cmd = command.strip().lower()
    logger = cmd_ctx.logger
    runtime_state = cmd_ctx.runtime_state
    ctx = cmd_ctx.ctx
    if cmd == "/help":
        logger.log(f"\n{SLASH_COMMANDS_HELP}\n")
        return True
    if cmd == "/tokens":
        logger.log(
            "\n"
            + render_token_report(
                ctx,
                runtime_state["active_tool_schemas"],
                runtime_state["active_skill_prompt"],
                verbose=runtime_state["verbose"],
                active_model=runtime_state.get("model", ""),
            )
            + "\n"
        )
        return True
    if cmd == "/history":
        logger.log(f"\n{ctx.format_history()}\n")
        return True
    if cmd in ("/debug", "/debug raw"):
        logger.log(f"\n{ctx.format_raw() if cmd == '/debug raw' else ctx.format_debug()}\n")
        return True
    if cmd == "/model":
        logger.log(f"\nCurrent model: {runtime_state['model']}\nUsage: /model <model-name>\n")
        return True
    if cmd.startswith("/model ") or cmd.startswith("/set-model"):
        parts = command.strip().split(None, 1)
        if len(parts) < 2 or not parts[1].strip():
            logger.log(
                f"\nUsage: /model <model-name>\n"
                f"Alias: /set-model <model-name>\n"
                f"Current model: {runtime_state['model']}\n"
            )
            return True
        new_model = parts[1].strip()
        old_model = runtime_state["model"]
        runtime_state["model"] = new_model
        logger.log(
            f"\nModel switched: {old_model} -> {new_model}\n"
            "Conversation history is preserved; new model will see all previous context.\n",
            level="success",
        )
        return True
    if cmd.startswith("/compact"):
        parts = command.strip().split()
        level = "low"
        if len(parts) >= 2:
            arg = parts[1].lower()
            if arg not in ("low", "high"):
                logger.log("\nUsage: /compact [low|high]  (default: low)\n", level="error")
                return True
            level = arg
        overhead_tokens = estimate_schema_tokens(
            ctx, runtime_state["active_tool_schemas"]
        ) + estimate_skill_tokens(ctx, runtime_state["active_skill_prompt"])
        do_compact(
            cmd_ctx.client,
            runtime_state["model"],
            ctx,
            logger,
            current_overhead_tokens=overhead_tokens,
            level=level,
            compact_client=runtime_state.get("compact_client"),
            compact_model=runtime_state.get("compact_model"),
        )
        return True
    if cmd == "/skills":
        all_skill_names = sorted(runtime_state["skills"].keys())
        active = runtime_state.get("active_skill_names", [])
        pinned = sorted(runtime_state["pinned_skills"])
        logger.log("\nSkills:")
        logger.log(f"  Loaded:  {', '.join(all_skill_names) if all_skill_names else '(none)'}")
        logger.log(f"  Active:  {', '.join(active) if active else '(none)'}")
        logger.log(f"  Pinned:  {', '.join(pinned) if pinned else '(none)'}\n")
        return True
    if cmd.startswith("/skill "):
        parts = cmd.split()
        if len(parts) != 3:
            logger.log("\nUsage: /skill <name> on|off\n")
            return True
        _, name, mode = parts
        if name not in runtime_state["skills"]:
            logger.log(f"\nUnknown skill: {name}\n", level="error")
            return True
        if mode == "on":
            runtime_state["pinned_skills"].add(name)
            logger.log(f"\nPinned skill: {name}\n", level="success")
        elif mode == "off":
            runtime_state["pinned_skills"].discard(name)
            logger.log(f"\nUnpinned skill: {name}\n", level="success")
        else:
            logger.log("\nUsage: /skill <name> on|off\n", level="error")
        return True
    if command.strip().lower().startswith("/run "):
        rest = command.strip()[len("/run ") :].strip()
        if not rest:
            logger.log("\nUsage: /run <skill-name> <task>\n", level="error")
            return True
        parts = rest.split(None, 1)
        skill_name = parts[0].strip()
        task = parts[1].strip() if len(parts) > 1 else ""
        if not task:
            logger.log("\nUsage: /run <skill-name> <task>  (task cannot be empty)\n", level="error")
            return True
        skills = runtime_state.get("skills", {})
        if skill_name not in skills:
            logger.log(f"\nUnknown skill: {skill_name}. Use /skills to list.\n", level="error")
            return True
        skill = skills[skill_name]
        if skill.context_mode != "fork":
            logger.log(
                f"\nSkill '{skill_name}' is not a fork skill (context: fork). Only fork skills can be run with /run.\n",
                level="error",
            )
            return True
        tool_schema_map = get_tool_schema_map()
        all_tool_schemas = get_all_tool_schemas()
        client = cmd_ctx.client
        model = runtime_state.get("model", "")
        if not model:
            logger.log("\nNo model set in runtime state.\n", level="error")
            return True
        from ict_agent.runtime.agent_loop import run_fork_skill

        logger.log(f"\nRunning fork skill '{skill_name}' with task: {task[:100]}{'...' if len(task) > 100 else ''}\n")
        result = run_fork_skill(
            client=client,
            model=model,
            skill=skill,
            task=task,
            tool_schema_map=tool_schema_map,
            all_tool_schemas=all_tool_schemas,
            logger=logger,
        )
        # Preserve user's command in context; subagent result as assistant with [subagent xxx job_id=0] (sync)
        ctx.add_user_message(command)
        ctx.add_assistant_message(f"[subagent {skill_name} job_id=0] {result}")
        logger.log(f"\nFork result injected into conversation. Next turn the assistant will see it.\n", level="success")
        return True
    if command.strip().lower().startswith("/fork "):
        rest = command.strip()[len("/fork ") :].strip()
        if not rest:
            logger.log("\nUsage: /fork <skill-name> [full_context] <task>\n", level="error")
            return True
        parts = rest.split(None, 1)
        skill_name = parts[0].strip()
        remainder = parts[1].strip() if len(parts) > 1 else ""
        if not remainder:
            logger.log("\nUsage: /fork <skill-name> [full_context] <task>  (task cannot be empty)\n", level="error")
            return True
        return_mode = "final"
        if remainder.lower().startswith("full_context "):
            return_mode = "full_context"
            task = remainder[len("full_context ") :].strip()
        else:
            task = remainder
        if not task:
            logger.log("\nUsage: /fork <skill-name> [full_context] <task>  (task cannot be empty)\n", level="error")
            return True
        skills = runtime_state.get("skills", {})
        if skill_name not in skills:
            logger.log(f"\nUnknown skill: {skill_name}. Use /skills to list.\n", level="error")
            return True
        skill = skills[skill_name]
        if skill.context_mode != "fork":
            logger.log(
                f"\nSkill '{skill_name}' is not a fork skill (context: fork). Only fork skills can be run with /fork.\n",
                level="error",
            )
            return True
        client = cmd_ctx.client
        model = runtime_state.get("model", "")
        if not model:
            logger.log("\nNo model set in runtime state.\n", level="error")
            return True
        from ict_agent.runtime.agent_loop import start_async_fork

        job_id = start_async_fork(runtime_state, client, model, skill, task, logger, return_mode=return_mode)
        task_preview = task[:80] + "..." if len(task) > 80 else task
        # Preserve user's command; record system flow so context shows what happened
        ctx.add_user_message(command)
        ctx.add_assistant_message(
            f"[system] Async fork started job_id={job_id} for skill {skill_name} (task: {task_preview}). "
            "Result will appear as [subagent ...] when ready."
        )
        logger.log(f"\nAsync fork started job_id={job_id}. Result will be injected next turn.\n", level="success")
        return True
    if cmd.strip().lower() == "/fork-status":
        from ict_agent.runtime.agent_loop import get_fork_threads_status

        running = get_fork_threads_status(runtime_state)
        if not running:
            logger.log("\nNo fork subagents running; all finished.\n", level="success")
        else:
            lines = [f"  job_id={e['job_id']} skill={e['skill_name']}" for e in running]
            logger.log(f"\nFork subagents still running ({len(running)}):\n" + "\n".join(lines) + "\n", level="info")
        return True
    if cmd.strip().lower().startswith("/fork-wait"):
        from ict_agent.runtime.agent_loop import get_fork_threads_status, wait_for_fork_threads

        parts = cmd.strip().split()
        timeout_sec = 60.0
        if len(parts) >= 2:
            try:
                timeout_sec = float(parts[1])
            except ValueError:
                pass
        if wait_for_fork_threads(runtime_state, timeout_sec=timeout_sec):
            logger.log("\nAll fork subagents finished.\n", level="success")
        else:
            running = get_fork_threads_status(runtime_state)
            logger.log(f"\nTimeout: {len(running)} fork subagent(s) still running. Use /fork-status to list.\n", level="error")
        return True
    if cmd.startswith("/verbose"):
        parts = cmd.split()
        if len(parts) == 1:
            logger.log(f"\nVerbose token diagnostics: {'on' if runtime_state['verbose'] else 'off'}\n")
            return True
        arg = parts[1]
        if arg in ("on", "true", "1"):
            runtime_state["verbose"] = True
            logger.log("\nVerbose token diagnostics: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["verbose"] = False
            logger.log("\nVerbose token diagnostics: off\n")
            return True
        logger.log("\nUsage: /verbose on|off\n")
        return True
    if cmd.startswith("/shell-safe"):
        parts = cmd.split()
        if len(parts) == 1:
            logger.log(f"\nSafe shell mode: {'on' if runtime_state['safe_shell'] else 'off'}\n")
            return True
        arg = parts[1]
        if arg in ("on", "true", "1"):
            runtime_state["safe_shell"] = True
            set_shell_safety(True)
            logger.log("\nSafe shell mode: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["safe_shell"] = False
            set_shell_safety(False)
            logger.log("\nSafe shell mode: off\n")
            return True
        logger.log("\nUsage: /shell-safe on|off\n")
        return True
    if cmd == "/shell-policy":
        snapshot = get_shell_policy_snapshot()
        logger.log("\nShell policy:")
        logger.log(f"  safe_mode: {snapshot['safe_mode']}")
        logger.log(f"  policy_file: {snapshot['policy_file']}")
        logger.log(f"  allowlist: {len(snapshot['allowlist'])} entries")
        logger.log(f"  denylist: {len(snapshot['denylist'])} entries")
        if snapshot["allowlist"]:
            logger.log("  allowlist entries:")
            for entry in snapshot["allowlist"]:
                logger.log(f"    - {entry}")
        if snapshot["denylist"]:
            logger.log("  denylist entries:")
            for entry in snapshot["denylist"]:
                logger.log(f"    - {entry}")
        logger.log()
        return True
    if cmd == "/preempt":
        mode = "on" if runtime_state.get("preempt_shell_kill", False) else "off"
        logger.log("\nPreemption settings:")
        logger.log("  queue: enabled")
        logger.log("  soft_preempt: enabled")
        logger.log(f"  shell_kill_on_preempt: {mode}\n")
        return True
    if cmd.startswith("/preempt shell-kill"):
        return cmd_ctx.domain_adapter.handle_preempt_shell_kill(command, cmd_ctx)
    if cmd.startswith("/recovery"):
        parts = cmd.split()
        if len(parts) == 1:
            logger.log(
                "\nRecovery cleanup:\n"
                f"  remove failed intermediate traces after success: "
                f"{'on' if runtime_state.get('recovery_cleanup', True) else 'off'}\n"
            )
            return True
        arg = parts[1]
        if arg in ("on", "true", "1"):
            runtime_state["recovery_cleanup"] = True
            logger.log("\nRecovery cleanup: on\n")
            return True
        if arg in ("off", "false", "0"):
            runtime_state["recovery_cleanup"] = False
            logger.log("\nRecovery cleanup: off\n")
            return True
        logger.log("\nUsage: /recovery on|off\n")
        return True
    if cmd == "/workspace":
        logger.log(f"\n{cmd_ctx.domain_adapter.workspace_summary()}\n")
        return True
    if cmd == "/clear":
        ctx.clear()
        logger.log("\nConversation cleared.\n")
        return True
    return False


def register_common_commands(registry) -> None:
    registry.register(handle_common_command)
