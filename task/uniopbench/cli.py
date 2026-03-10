"""UniOpBench task CLI. Parses its own args and delegates to the orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_orchestrator():
    orchestrator_path = Path(__file__).resolve().parent / "orchestrator.py"
    spec = importlib.util.spec_from_file_location("uniopbench_orchestrator", orchestrator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load orchestrator from {orchestrator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniOpBench experiment runner")
    parser.add_argument(
        "--task",
        choices=["uniopbench"],
        default="uniopbench",
        help="Task (always uniopbench when invoked from this module)",
    )
    parser.add_argument(
        "--config",
        help="Override task config path (default: task/uniopbench/task.yaml)",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run id; defaults to a timestamp",
    )
    parser.add_argument(
        "--operators",
        help="Optional comma-separated operator override list",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create prompts and run layout without calling the LLM",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a prior run id",
    )
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Disable truncation in agent output (full system prompt, full tool results)",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, _ = parser.parse_known_args(argv or sys.argv[1:])
    orchestrator = _load_orchestrator()
    return orchestrator.run_uniopbench_task(args)
