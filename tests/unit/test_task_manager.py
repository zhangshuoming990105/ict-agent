from pathlib import Path

from ict_agent.domains.cuda import task_manager


def test_resolve_local_example_task():
    task_path = task_manager.resolve_task_path("example_axpby")
    assert task_path.name == "example_axpby"
    assert (task_path / "model.py").is_file()


def test_load_task_prompt_uses_local_template():
    prompt, src = task_manager.load_task_prompt(task_manager._task_dir() / "example_axpby")
    assert "Task Context" in prompt
    assert src is not None


def test_workspace_summary_for_missing_workspace():
    assert task_manager.workspace_summary(Path("/tmp/does-not-exist-for-ict-agent")) == "(no active workspace)"
