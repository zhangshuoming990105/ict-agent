import argparse
import importlib.util
import sys
from pathlib import Path


def _load_uniopbench_orchestrator():
    repo_root = Path(__file__).resolve().parent
    orchestrator_path = repo_root / "task" / "uniopbench" / "orchestrator.py"
    spec = importlib.util.spec_from_file_location("uniopbench_orchestrator", orchestrator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load orchestrator from {orchestrator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Repository-level experiment entrypoint")
    parser.add_argument(
        "--task",
        choices=["uniopbench"],
        required=True,
        help="Task to execute",
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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task != "uniopbench":
        raise ValueError(f"Unknown task: {args.task}")
    orchestrator = _load_uniopbench_orchestrator()
    return orchestrator.run_uniopbench_task(args)


if __name__ == "__main__":
    sys.exit(main())
