"""CUDA-specific slash commands."""

from __future__ import annotations

from ict_agent.commands.common import estimate_schema_tokens, estimate_skill_tokens
from ict_agent.tools import set_shell_safety, set_workspace_root


TASK_COMMANDS_HELP = """Available task commands:
  /task         - Show current task context status
  /task load X  - Load or switch task and reset conversation
  /task reload  - Reload current task context and reset conversation
  /task inject  - Inject current task markdown into the current conversation
  /workspace    - Show current task workspace info
  /preempt shell-kill on|off - Toggle killing long shell commands on preempt
"""


def handle_cuda_command(command: str, cmd_ctx) -> bool:
    cmd = command.strip().lower()
    adapter = cmd_ctx.domain_adapter
    logger = cmd_ctx.logger
    ctx = cmd_ctx.ctx
    runtime_state = cmd_ctx.runtime_state

    if cmd == "/task":
        logger.log("\nTask context:")
        logger.log(f"  task_dir: {adapter.task_dir if adapter.task_dir else '(none loaded)'}")
        logger.log(f"  workspace: {adapter.workspace_root}")
        logger.log(f"  task_md: {adapter.task_context_source if adapter.task_context_source else '(none)'}")
        logger.log("  tip: /task load <specifier>  |  /task inject\n")
        return True

    if cmd.startswith("/task load"):
        spec = command.strip()[len("/task load") :].strip()
        if not spec:
            logger.log("\nUsage: /task load <specifier>\n")
            return True
        try:
            adapter.load_task(spec)
            set_workspace_root(adapter.workspace_root)
        except Exception as exc:
            logger.log(f"\nFailed to load task: {exc}\n")
            return True
        ctx.system_prompt = adapter.compose_system_prompt()
        ctx.clear()
        logger.log("\nTask loaded:")
        logger.log(f"  task_dir: {adapter.task_dir}")
        logger.log(f"  workspace: {adapter.workspace_root}")
        logger.log(f"  history: {'loaded' if adapter.history_prompt else 'none'}")
        logger.log(f"  task_md: {adapter.task_context_source if adapter.task_context_source else 'none'}")
        logger.log(f"\n{adapter.workspace_summary()}\n")
        runtime_state["task_dir"] = adapter.task_dir
        return True

    if cmd == "/task reload":
        if not adapter.task_dir:
            logger.log("\nNo active task. Use /task load <specifier> first.\n")
            return True
        try:
            adapter.reload_task_context()
        except Exception as exc:
            logger.log(f"\nFailed to reload task context: {exc}\n")
            return True
        ctx.system_prompt = adapter.compose_system_prompt()
        ctx.clear()
        logger.log("\nTask context reloaded and conversation reset.")
        logger.log(f"  history: {'loaded' if adapter.history_prompt else 'none'}")
        logger.log(f"  task_md: {adapter.task_context_source if adapter.task_context_source else 'none'}\n")
        return True

    if cmd == "/task inject":
        if not adapter.task_dir:
            logger.log("\nNo active task. Use /task load <specifier> first.\n")
            return True
        injected = adapter.inject_task_context()
        if not injected:
            logger.log("\nNo task.md/TASK.md found (or file is empty).\n")
            return True
        ctx.messages.append(
            {
                "role": "system",
                "content": (
                    "## Runtime Task Context Injection\n"
                    "Use this newly injected task context for subsequent turns.\n\n"
                    + injected
                ),
            }
        )
        logger.log(f"\nInjected task context from: {adapter.task_context_source}\n")
        return True

    if cmd.startswith("/preempt shell-kill"):
        return adapter.handle_preempt_shell_kill(command, cmd_ctx)

    return False


def register_cuda_commands(registry) -> None:
    registry.register(handle_cuda_command)
