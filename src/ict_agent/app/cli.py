"""CLI entrypoint for the refactored agent."""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
import os
import sys
from pathlib import Path

from ict_agent.app.bootstrap import create_command_registry, create_domain_adapter, create_logger
from ict_agent.app.live_session import (
    cmd_paths as live_cmd_paths,
    cmd_send as live_cmd_send,
    cmd_start as live_cmd_start,
    cmd_status as live_cmd_status,
    cmd_stop as live_cmd_stop,
    get_state_paths as live_get_state_paths,
)
from ict_agent.app.config import AppConfig
from ict_agent.commands.common import register_common_commands
from ict_agent.llm import SUPPORTED_PROVIDERS, create_client, get_provider_help_text, list_models
from ict_agent.runtime.agent_loop import chat
from ict_agent.tools import set_gpu_auto, set_gpu_device, set_sandbox_enabled, set_workspace_root


DEFAULT_MAX_TOKENS = 128_000
DEFAULT_MAX_AGENT_STEPS = 30


def _run_uniopbench(root_dir: Path) -> int:
    cli_path = root_dir / "task" / "uniopbench" / "cli.py"
    spec = importlib.util.spec_from_file_location("uniopbench_cli", cli_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load uniopbench CLI from {cli_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.run(sys.argv)

task_table = {
    "uniopbench": _run_uniopbench,
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ICT Agent")
    parser.add_argument(
        "--provider",
        type=str,
        default="ksyun",
        choices=SUPPORTED_PROVIDERS,
        help="LLM provider to use (default: ksyun). Use 'auto' to prefer ksyun and fall back to infini.",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="Show available provider options and exit",
    )
    parser.add_argument("--list-models", action="store_true", help="List models and exit")
    parser.add_argument("--model", type=str, default=None, help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Context window size")
    parser.add_argument(
        "--max-agent-steps",
        type=int,
        default=DEFAULT_MAX_AGENT_STEPS,
        help="Max autonomous LLM/tool rounds per user turn",
    )
    parser.add_argument("--task", type=str, default=None, help="Task specifier")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks and exit")
    parser.add_argument("--workdir", type=str, default=None, help="Task working directory")
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for logs and live-session state (default: ICT_AGENT_SESSION_ID env or 0)",
    )
    parser.add_argument("--safe-shell", action="store_true", help="Enable safe shell mode")
    parser.add_argument(
        "--keep-recovery-trace",
        action="store_true",
        help="Keep failed intermediate traces in context after successful task completion",
    )
    parser.add_argument(
        "--preempt-shell-kill",
        action="store_true",
        help="Kill running shell commands when preempt input arrives",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="GPU selection: auto, none, or explicit index like 1",
    )
    parser.add_argument(
        "--compact-model",
        type=str,
        default=None,
        help="Model to use for compaction (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Enable process-level sandbox for shell commands (requires bubblewrap on Linux or sandbox-exec on macOS)",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Disable all truncation: full system prompt, full tool results (read/write/list/grep), no large-output persist",
    )
    # System prompt customization (Claude Code style)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Replace entire system prompt with custom text",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=None,
        help="Load system prompt from file, replacing the default",
    )
    parser.add_argument(
        "--append-system-prompt",
        type=str,
        default=None,
        help="Append text to the default system prompt",
    )
    parser.add_argument(
        "--append-system-prompt-file",
        type=str,
        default=None,
        help="Append file contents to the default system prompt",
    )
    # Initial user message (Claude Code style)
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Initial user message (first turn)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Load initial user message from file",
    )
    parser.add_argument(
        "input_positional",
        nargs="?",
        default=None,
        help="Initial user message (positional, e.g. ict-agent 'fix the bug')",
    )
    return parser


def _resolve_initial_message(
    task_initial: str | None,
    args: argparse.Namespace,
) -> str | None:
    """Resolve initial user message from CLI args, task, or stdin."""
    if getattr(args, "input", None):
        return args.input
    if getattr(args, "input_file", None):
        path = Path(args.input_file)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path.read_text(encoding="utf-8", errors="replace").strip() or None
    if getattr(args, "input_positional", None):
        return args.input_positional
    if task_initial:
        return task_initial
    # Live session: stdin is FIFO, never EOF; let chat loop read line-by-line instead of read()
    # live_session.sh sets ICT_AGENT_SESSION_TTL; ict-agent start sets ICT_AGENT_LIVE_LOG
    if os.environ.get("ICT_AGENT_LIVE_LOG") or os.environ.get("ICT_AGENT_SESSION_TTL"):
        return None
    # When stdin is piped (not a TTY), use it as initial message
    if not sys.stdin.isatty():
        try:
            text = sys.stdin.read().strip()
            return text or None
        except (EOFError, OSError):
            pass
    return None


LIVE_COMMANDS = ("start", "send", "status", "stop", "paths")

# Flags (from build_parser + _live_dispatch) that consume the next arg as a value.
# Used by _find_live_command to avoid misdetecting e.g. ``--input start`` as a subcommand.
_VALUE_FLAGS = frozenset({
    "--provider", "--model", "--max-tokens", "--max-agent-steps", "--task",
    "--workdir", "--session-id", "--gpu", "--compact-model",
    "--system-prompt", "--system-prompt-file",
    "--append-system-prompt", "--append-system-prompt-file",
    "--input", "-i", "--input-file", "--ttl",
})


def _find_live_command(argv: list[str]) -> tuple[str | None, list[str]]:
    """Locate a live-session subcommand in *argv* regardless of flag ordering.

    Returns ``(command, remaining_argv)`` or ``(None, argv)`` if not found.
    Correctly skips values of known flags so ``--input start`` is not misdetected.
    """
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("-"):
            # --flag=value doesn't consume next arg; --flag value does
            if "=" not in arg and arg in _VALUE_FLAGS:
                skip_next = True
            continue
        if arg in LIVE_COMMANDS:
            return arg, argv[:i] + argv[i + 1:]
    return None, argv


def _live_dispatch(cmd: str, argv: list[str]) -> int:
    root_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(prog=f"ict-agent {cmd}")
    parser.add_argument("--session-id", type=str, default=os.getenv("ICT_AGENT_SESSION_ID", "0"))
    parser.add_argument("--ttl", type=int, default=3600)
    args, unknown = parser.parse_known_args(argv)
    session_id = (args.session_id or "0").strip() or "0"
    paths = live_get_state_paths(root_dir, session_id)
    if cmd == "start":
        return live_cmd_start(root_dir, session_id, args.ttl, unknown)
    if cmd == "send":
        return live_cmd_send(paths, " ".join(unknown).strip() if unknown else "")
    if cmd == "status":
        return live_cmd_status(paths, session_id)
    if cmd == "stop":
        return live_cmd_stop(paths, session_id)
    if cmd == "paths":
        return live_cmd_paths(paths, session_id)
    return 1


def main() -> int:
    argv = sys.argv[1:]
    live_cmd, remaining = _find_live_command(argv)
    if live_cmd is not None:
        return _live_dispatch(live_cmd, remaining)
    parser = build_parser()
    args, _ = parser.parse_known_args()

    # Validate mutually exclusive system prompt flags
    if args.system_prompt is not None and args.system_prompt_file is not None:
        print("Error: --system-prompt and --system-prompt-file are mutually exclusive.", file=sys.stderr)
        return 1

    root_dir = Path(__file__).resolve().parents[3]
    session_id = (
        getattr(args, "session_id", None)
        or os.getenv("ICT_AGENT_SESSION_ID")
        or "0"
    )
    session_id = (session_id or "0").strip() or "0"
    domain_adapter = create_domain_adapter(root_dir)

    if args.list_tasks:
        print(domain_adapter.list_tasks())
        return 0

    if args.task in task_table:
        return task_table[args.task](root_dir)

    if args.list_providers:
        print(get_provider_help_text())
        return 0

    client, provider_name, provider_model, base_url = create_client(args.provider)
    model = args.model or provider_model
    if args.list_models:
        list_models(client)
        return 0

    compact_model = args.compact_model or "gpt-oss-120b"
    if args.sandbox:
        from ict_agent.sandbox import sandbox_backend
        backend = sandbox_backend()
        set_sandbox_enabled(True)
        if backend == "none":
            import platform
            platform_name = platform.system()
            hint = "apt install bubblewrap" if platform_name == "Linux" else "sandbox-exec is built into macOS"
            print(
                f"Warning: --sandbox enabled but no sandbox backend found on {platform_name}.\n"
                f"  Install hint: {hint}\n"
                f"  Falling back to unsandboxed execution.\n"
            )
    gpu_flag = (args.gpu or "").strip().lower()
    if gpu_flag == "auto":
        set_gpu_auto(True)
    elif gpu_flag and gpu_flag != "none":
        set_gpu_device(args.gpu.strip())

    task_dir = None
    task_initial_message = None
    if args.task:
        domain_adapter.load_task(args.task, args.workdir)
        set_workspace_root(domain_adapter.workspace_root)
        task_dir = domain_adapter.task_dir
        task_initial_message = domain_adapter.initial_message
    else:
        workspace = Path(args.workdir or os.getcwd()).resolve()
        set_workspace_root(workspace)
        domain_adapter.workspace_root = workspace

    # Apply system prompt overrides (Claude Code style)
    if args.system_prompt_file is not None:
        path = Path(args.system_prompt_file)
        if not path.is_file():
            raise FileNotFoundError(f"System prompt file not found: {path}")
        domain_adapter.system_prompt_override = path.read_text(encoding="utf-8", errors="replace")
    elif args.system_prompt is not None:
        domain_adapter.system_prompt_override = args.system_prompt

    append_parts: list[str] = []
    if args.append_system_prompt_file is not None:
        path = Path(args.append_system_prompt_file)
        if not path.is_file():
            raise FileNotFoundError(f"Append system prompt file not found: {path}")
        append_parts.append(path.read_text(encoding="utf-8", errors="replace"))
    if args.append_system_prompt is not None:
        append_parts.append(args.append_system_prompt)
    if append_parts:
        domain_adapter.append_system_prompt = "\n\n".join(p.strip() for p in append_parts if p.strip())

    initial_message = _resolve_initial_message(task_initial_message, args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_parent = (task_dir / "logs") if task_dir else (root_dir / "logs")
    log_dir = log_parent / f"session_{session_id}"
    config = AppConfig(
        provider=provider_name,
        model=model,
        max_tokens=args.max_tokens,
        max_agent_steps=max(1, args.max_agent_steps),
        safe_shell=args.safe_shell,
        recovery_cleanup=not args.keep_recovery_trace,
        preempt_shell_kill=args.preempt_shell_kill,
        compact_model=compact_model,
    )
    logger = create_logger(log_dir / f"{timestamp}.log")
    command_registry = create_command_registry(domain_adapter)

    if task_dir:
        logger.log(
            f"Setting up workspace: task={task_dir}, workdir={domain_adapter.workspace_root}, gpu={args.gpu}"
        )
        logger.log(f"Workspace ready:\n{domain_adapter.workspace_summary()}")
        if gpu_flag == "auto":
            logger.log(f"GPU status: {domain_adapter.gpu_status_summary()}")
        if domain_adapter.history_prompt:
            logger.log(f"Loaded history from {task_dir / 'history'}")
        if domain_adapter.task_context_source:
            logger.log(f"Loaded task context from {domain_adapter.task_context_source}")
    else:
        logger.log("No task specified. Chat mode enabled.")
        logger.log("Use /task load <specifier> to load a task and initialize workdir.")
    logger.log(f"Provider:     {config.provider}")
    logger.log(f"Base URL:     {base_url}")
    logger.log(f"Session ID:   {session_id}")
    logger.log()

    try:
        chat(
            client=client,
            model=config.model,
            max_tokens=config.max_tokens,
            max_agent_steps=config.max_agent_steps,
            safe_shell=config.safe_shell,
            recovery_cleanup=config.recovery_cleanup,
            preempt_shell_kill=config.preempt_shell_kill,
            initial_message=initial_message,
            compact_client=client,
            compact_model=config.compact_model,
            logger=logger,
            command_registry=command_registry,
            domain_adapter=domain_adapter,
            skills_root=root_dir / "skills",
            no_truncate=args.no_truncate,
        )
    except KeyboardInterrupt:
        logger.log("\nBye!")
        return 130
    finally:
        logger.log(f"\nLog saved to: {log_dir / f'{timestamp}.log'}")
        logger.close()
    return 0
