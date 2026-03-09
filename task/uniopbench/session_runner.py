"""Launch a UniOpBench-specific live chat session using the normal ict-agent CLI runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from ict_agent.app.bootstrap import create_command_registry, create_domain_adapter, create_logger
from ict_agent.llm import create_client
from ict_agent.runtime.agent_loop import chat
from ict_agent.tools import set_workspace_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniOpBench live session runner")
    parser.add_argument("--workspace", required=True, help="Workspace root for this operator session")
    parser.add_argument("--system-prompt-file", required=True, help="Injected system prompt file")
    parser.add_argument("--log-path", required=True, help="Agent log file path")
    parser.add_argument("--provider", default="auto", help="LLM provider")
    parser.add_argument("--model", required=True, help="LLM model name")
    parser.add_argument("--compact-model", default=None, help="Compaction model")
    parser.add_argument("--max-tokens", type=int, default=128_000, help="Conversation context window")
    parser.add_argument("--max-agent-steps", type=int, default=30, help="Max autonomous steps per turn")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    system_prompt_path = Path(args.system_prompt_file).resolve()
    log_path = Path(args.log_path).resolve()

    client, provider_name, default_model, base_url = create_client(args.provider)
    model_name = args.model or default_model
    compact_model = args.compact_model or model_name

    domain_adapter = create_domain_adapter(ROOT)
    domain_adapter.workspace_root = workspace
    domain_adapter.task_prompt = (
        "## Runtime Task Context Injection\n"
        "Use this task context for the current UniOpBench operator session.\n\n"
        + system_prompt_path.read_text(encoding="utf-8")
    )
    domain_adapter.task_context_source = str(system_prompt_path)
    set_workspace_root(workspace)

    logger = create_logger(log_path)
    command_registry = create_command_registry(domain_adapter)

    logger.log("Setting up workspace for UniOpBench live session.")
    logger.log(f"Workspace ready:\n{domain_adapter.workspace_summary()}")
    logger.log(f"Provider:     {provider_name}")
    logger.log(f"Base URL:     {base_url}")
    logger.log(f"Model:        {model_name}")
    logger.log()

    try:
        chat(
            client=client,
            model=model_name,
            max_tokens=max(1, args.max_tokens),
            max_agent_steps=args.max_agent_steps,
            safe_shell=False,
            recovery_cleanup=True,
            preempt_shell_kill=False,
            initial_message=None,
            compact_client=client,
            compact_model=compact_model,
            logger=logger,
            command_registry=command_registry,
            domain_adapter=domain_adapter,
            skills_root=ROOT / "skills",
        )
    finally:
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
