import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(relative_path: str, module_name: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_optimize_parser_accepts_resume():
    cli_module = _load_module("task/uniopbench/cli.py", "test_uniopbench_cli_resume")

    parser = cli_module.build_parser()
    args = parser.parse_args(["optimize", "--resume"])

    assert args.subcommand == "optimize"
    assert args.resume is True


def test_optimize_requires_resume_for_existing_run(tmp_path, monkeypatch):
    orchestrator = _load_module(
        "task/uniopbench/orchestrator.py",
        "test_uniopbench_orchestrator_existing_run",
    )

    task_config = SimpleNamespace(
        experiment=SimpleNamespace(name="exp"),
        operators=[],
    )
    monkeypatch.setattr(orchestrator, "load_task_config", lambda *args, **kwargs: task_config)
    monkeypatch.setattr(orchestrator, "runs_root", lambda _name: tmp_path)

    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()

    args = SimpleNamespace(
        config=None,
        operators=None,
        no_truncate=False,
        dry_run=False,
        target_speedup=1.0,
        rounds=1,
        max_version=None,
        ref_impl=None,
        run_id="resume_run",
        resume=False,
    )

    with pytest.raises(FileExistsError):
        orchestrator.run_optimize_task(args)


def test_optimize_resume_skips_recorded_operators(tmp_path, monkeypatch):
    orchestrator = _load_module(
        "task/uniopbench/orchestrator.py",
        "test_uniopbench_orchestrator_resume_skip",
    )

    task_config = SimpleNamespace(
        experiment=SimpleNamespace(name="exp"),
        operators=["activation/relu", "conv/depthwiseconv"],
    )
    monkeypatch.setattr(orchestrator, "load_task_config", lambda *args, **kwargs: task_config)
    monkeypatch.setattr(orchestrator, "runs_root", lambda _name: tmp_path)

    source_dir = tmp_path / "operator_source"
    source_dir.mkdir()
    (source_dir / "test.py").write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(orchestrator, "operator_scaffold_dir", lambda _operator: source_dir)
    monkeypatch.setattr(orchestrator, "supports_variants", lambda _path: False)
    monkeypatch.setattr(orchestrator, "build_prompt", lambda *args, **kwargs: ("system", "user"))

    copy_calls: list[Path] = []

    def fake_copy_operator_tree(_src: Path, dst: Path) -> None:
        copy_calls.append(dst)

    monkeypatch.setattr(orchestrator, "copy_operator_tree", fake_copy_operator_tree)

    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "operators": {
                    "activation__relu": {
                        "operator": "activation/relu",
                        "status": "passed",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        config=None,
        operators=None,
        no_truncate=False,
        dry_run=True,
        target_speedup=1.0,
        rounds=1,
        max_version=None,
        ref_impl=None,
        run_id="resume_run",
        resume=True,
    )

    rc = orchestrator.run_optimize_task(args)

    assert rc == 0
    assert copy_calls == [run_dir / "operators" / "conv__depthwiseconv" / "artifact"]

    updated_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(updated_summary["operators"]) == {
        "activation__relu",
        "conv__depthwiseconv",
    }
    assert updated_summary["operators"]["activation__relu"]["status"] == "passed"
    assert updated_summary["operators"]["conv__depthwiseconv"]["status"] == "dry_run"
