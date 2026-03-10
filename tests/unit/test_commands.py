from pathlib import Path
from queue import Queue
from types import SimpleNamespace

from ict_agent.commands.common import handle_common_command
from ict_agent.context import ContextManager
from ict_agent.skills import load_skills


class DummyLogger:
    def __init__(self):
        self.lines = []

    def log(self, msg="", **kwargs):
        self.lines.append(msg)


class DummyDomainAdapter:
    def __init__(self):
        self.workspace_root = "/tmp/workspace"

    def workspace_summary(self):
        return "workspace-summary"

    def handle_preempt_shell_kill(self, command, cmd_ctx):
        cmd_ctx.runtime_state["preempt_shell_kill"] = True
        cmd_ctx.logger.log("preempt-shell-kill")
        return True


def test_set_model_command_updates_runtime_state():
    logger = DummyLogger()
    runtime_state = {
        "active_tool_schemas": [],
        "active_skill_prompt": "",
        "verbose": False,
        "model": "mco-4",
        "safe_shell": False,
        "skills": {},
        "pinned_skills": set(),
    }
    command_context = SimpleNamespace(
        client=None,
        ctx=ContextManager("system"),
        runtime_state=runtime_state,
        logger=logger,
        domain_adapter=DummyDomainAdapter(),
    )
    handled = handle_common_command("/set-model gpt-oss-120b", command_context)
    assert handled is True
    assert runtime_state["model"] == "gpt-oss-120b"
    assert any("Model switched" in line for line in logger.lines)


def _make_run_command_context(runtime_state_overrides=None):
    skills_root = Path(__file__).resolve().parents[2] / "skills"
    skills = load_skills(skills_root)
    runtime_state = {
        "active_tool_schemas": [],
        "active_skill_prompt": "",
        "verbose": False,
        "model": "mco-4",
        "safe_shell": False,
        "skills": skills,
        "pinned_skills": set(),
    }
    if runtime_state_overrides:
        runtime_state.update(runtime_state_overrides)
    logger = DummyLogger()
    ctx = ContextManager("system")
    return SimpleNamespace(
        client=None,
        ctx=ctx,
        runtime_state=runtime_state,
        logger=logger,
        domain_adapter=DummyDomainAdapter(),
    ), logger, ctx


def test_run_command_unknown_skill():
    cmd_ctx, logger, _ = _make_run_command_context()
    handled = handle_common_command("/run nosuchskill list files", cmd_ctx)
    assert handled is True
    assert any("Unknown skill" in line for line in logger.lines or [])


def test_run_command_non_fork_skill_rejected():
    cmd_ctx, logger, _ = _make_run_command_context()
    handled = handle_common_command("/run core list files", cmd_ctx)
    assert handled is True
    assert any("not a fork skill" in line or "context: fork" in line for line in logger.lines or [])


def test_run_command_empty_task_rejected():
    cmd_ctx, logger, _ = _make_run_command_context()
    handled = handle_common_command("/run scout", cmd_ctx)
    assert handled is True
    assert any("task cannot be empty" in line or "Usage" in line for line in logger.lines or [])


def test_fork_command_starts_async_and_injects_message():
    cmd_ctx, logger, ctx = _make_run_command_context({
        "fork_result_queue": Queue(),
        "fork_job_counter": 0,
        "fork_results": {},
    })
    handled = handle_common_command("/fork scout list Python files in src", cmd_ctx)
    assert handled is True
    assert any("Async fork started" in line for line in logger.lines or [])
    assert any("job_id=" in line for line in logger.lines or [])
    assert len(ctx.messages) >= 2
    # User's command and assistant [system] flow message
    assert any(m.get("role") == "user" and "/fork scout" in (m.get("content") or "") for m in ctx.messages)
    assert any("[system]" in (m.get("content") or "") and "Async fork started" in (m.get("content") or "") for m in ctx.messages)
