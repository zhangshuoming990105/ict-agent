"""CLI entrypoint for the refactored agent."""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
import os
import sys
from pathlib import Path

from ict_agent.app.bootstrap import create_command_registry, create_domain_adapter, create_logger
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
    return parser


def main() -> int:
    parser = build_parser()
    args, _ = parser.parse_known_args()

    root_dir = Path(__file__).resolve().parents[3]
    session_id = (os.getenv("ICT_AGENT_SESSION_ID") or "0").strip() or "0"
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
            sys = platform.system()
            hint = "apt install bubblewrap" if sys == "Linux" else "sandbox-exec is built into macOS"
            print(
                f"Warning: --sandbox enabled but no sandbox backend found on {sys}.\n"
                f"  Install hint: {hint}\n"
                f"  Falling back to unsandboxed execution.\n"
            )
    gpu_flag = (args.gpu or "").strip().lower()
    if gpu_flag == "auto":
        set_gpu_auto(True)
    elif gpu_flag and gpu_flag != "none":
        set_gpu_device(args.gpu.strip())

    task_dir = None
    initial_message = None
    if args.task:
        domain_adapter.load_task(args.task, args.workdir)
        set_workspace_root(domain_adapter.workspace_root)
        task_dir = domain_adapter.task_dir
        initial_message = domain_adapter.initial_message
    else:
        workspace = Path(args.workdir or os.getcwd()).resolve()
        set_workspace_root(workspace)
        domain_adapter.workspace_root = workspace

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
        )
    except KeyboardInterrupt:
        logger.log("\nBye!")
        return 130
    finally:
        logger.log(f"\nLog saved to: {log_dir / f'{timestamp}.log'}")
        logger.close()
    return 0
